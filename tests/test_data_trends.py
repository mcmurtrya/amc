"""Tests for the Google Trends as-pulled archiver (collector 3). No network.

Fixtures in ``tests/fixtures/trends/`` are real response bytes captured live on
2026-07-12 for the frozen ``sell_side_v1`` term set (5 terms, geo=US,
timeframe="today 5-y", tz=0):

- ``explore_sell_side_v1.txt``   — the full /trends/api/explore response,
  verbatim (anti-JSON prefix included).
- ``multiline_sell_side_v1.txt`` — the /trends/api/widgetdata/multiline
  response trimmed from 262 to 24 timeline points (first 12 + last 12,
  including the trailing ``isPartial`` week) and re-serialized compactly with
  the same ``)]}',`` prefix; point contents are untouched.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pytest

from metals.data.migrations import runner
from metals.data.trends import (
    BACKOFF_SLEEP_S,
    REALTIME_WINDOW_DAYS,
    SOURCE_TAG,
    _request_with_backoff,
    collect_group,
    extract_timeseries_widget,
    load_term_groups,
    parse_timeline,
    strip_antijson_prefix,
    upsert_search_interest,
    widget_terms,
)

FIXTURES = Path(__file__).parent / "fixtures" / "trends"
TERMS = ["sell gold", "cash for gold", "gold price", "sell silver", "coin shop near me"]
# Naive-UTC stand-in for the actual capture moment of the fixtures.
PULLED_AT = datetime(2026, 7, 12, 18, 10, 0)
N_POINTS = 24  # trimmed fixture: first 12 + last 12 weekly points


def _explore_raw() -> str:
    return (FIXTURES / "explore_sell_side_v1.txt").read_text(encoding="utf-8")


def _multiline_raw() -> str:
    return (FIXTURES / "multiline_sell_side_v1.txt").read_text(encoding="utf-8")


def _parsed_fixture_df():
    payload = json.loads(strip_antijson_prefix(_multiline_raw()))
    return parse_timeline(
        payload,
        terms=TERMS,
        geo="US",
        resolution="WEEK",
        pulled_at=PULLED_AT,
        request_params={"terms": TERMS, "geo": "US", "timeframe": "today 5-y", "tz": 0},
    )


# ---------------------------------------------------------------------------
# Anti-JSON prefix
# ---------------------------------------------------------------------------


def test_strip_antijson_prefix_on_both_real_responses():
    """Both endpoints' real bytes parse to JSON after stripping the prefix."""
    explore = json.loads(strip_antijson_prefix(_explore_raw()))
    assert "widgets" in explore
    multiline = json.loads(strip_antijson_prefix(_multiline_raw()))
    assert "timelineData" in multiline["default"]


def test_strip_antijson_prefix_raises_without_prefix():
    with pytest.raises(ValueError, match="anti-JSON prefix"):
        strip_antijson_prefix('{"default": {}}')


def test_strip_antijson_prefix_raises_on_prefix_only():
    with pytest.raises(ValueError, match="no payload"):
        strip_antijson_prefix(")]}'\n")


# ---------------------------------------------------------------------------
# Explore parsing
# ---------------------------------------------------------------------------


def test_extract_timeseries_widget_from_real_explore():
    payload = json.loads(strip_antijson_prefix(_explore_raw()))
    widget = extract_timeseries_widget(payload)
    assert widget["token"]
    assert widget["request"]["resolution"] == "WEEK"
    assert widget_terms(widget) == TERMS


def test_extract_timeseries_widget_raises_when_absent():
    payload = json.loads(strip_antijson_prefix(_explore_raw()))
    payload["widgets"] = [w for w in payload["widgets"] if w.get("id") != "TIMESERIES"]
    with pytest.raises(ValueError, match="no TIMESERIES widget"):
        extract_timeseries_widget(payload)


# ---------------------------------------------------------------------------
# Timeline parsing
# ---------------------------------------------------------------------------


def test_parse_timeline_shape_and_columns():
    df = _parsed_fixture_df()
    assert len(df) == N_POINTS * len(TERMS)
    assert list(df.columns) == [
        "pulled_at",
        "geo",
        "term",
        "period_start",
        "period_end",
        "value",
        "request_params",
        "source",
        "is_realtime",
    ]
    assert (df["source"] == SOURCE_TAG).all()
    assert (df["geo"] == "US").all()
    assert set(df["term"]) == set(TERMS)
    assert df["value"].between(0, 100).all()
    for raw in df["request_params"]:
        assert json.loads(raw)["timeframe"] == "today 5-y"


def test_parse_timeline_utc_epochs_and_weekly_periods():
    """Point times are epoch seconds -> UTC dates; a WEEK period spans 7 days."""
    df = _parsed_fixture_df()
    # First fixture point: epoch 1625961600 == 2021-07-11 00:00:00 UTC (Sunday).
    first = df.iloc[0]
    assert first["period_start"] == datetime(2021, 7, 11)
    assert first["period_end"] == datetime(2021, 7, 17)
    # Last fixture point: epoch 1783814400 == 2026-07-12 UTC, the partial week.
    last = df.iloc[-1]
    assert last["period_start"] == datetime(2026, 7, 12)
    assert last["period_end"] == datetime(2026, 7, 18)
    assert (df["period_end"] - df["period_start"]).dt.days.eq(6).all()
    assert (df["pulled_at"] == PULLED_AT).all()


def test_parse_timeline_normalizes_aware_pulled_at_to_naive_utc():
    payload = json.loads(strip_antijson_prefix(_multiline_raw()))
    df = parse_timeline(
        payload,
        terms=TERMS,
        geo="US",
        resolution="WEEK",
        pulled_at=PULLED_AT.replace(tzinfo=UTC),
        request_params={},
    )
    assert df["pulled_at"].dt.tz is None
    assert (df["pulled_at"] == PULLED_AT).all()


def test_is_realtime_window():
    """Realtime iff period_end within 14 days of pulled_at; earlier rows are
    setup-time context, permanently False."""
    df = _parsed_fixture_df()
    realtime_ends = set(df.loc[df["is_realtime"], "period_end"].dt.date.astype(str))
    # Weeks ending 2026-07-04, 2026-07-11 and the partial week ending
    # 2026-07-18 are within the window of the 2026-07-12 pull.
    assert realtime_ends == {"2026-07-04", "2026-07-11", "2026-07-18"}
    # The week ending 2026-06-27 is 15 days before the pull -> not realtime.
    boundary = df[df["period_end"] == datetime(2026, 6, 27)]
    assert not boundary.empty
    assert not boundary["is_realtime"].any()
    # 2021 setup-time history is all non-realtime.
    old = df[df["period_start"] == datetime(2021, 7, 11)]
    assert not old["is_realtime"].any()


def _day_payload(days):
    """Minimal multiline payload: one DAY-resolution point per date."""
    points = [
        {
            "time": str(int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp())),
            "value": [1] * len(TERMS),
        }
        for d in days
    ]
    return {"default": {"timelineData": points}}


def test_is_realtime_exact_boundary_through_parse_timeline():
    """A period_end exactly REALTIME_WINDOW_DAYS before the pull is realtime;
    one day beyond the window is permanently False."""
    at_boundary = PULLED_AT.date() - timedelta(days=REALTIME_WINDOW_DAYS)
    beyond = at_boundary - timedelta(days=1)
    df = parse_timeline(
        _day_payload([beyond, at_boundary]),
        terms=TERMS,
        geo="US",
        resolution="DAY",  # period_end == period_start, so the boundary is exact
        pulled_at=PULLED_AT,
        request_params={},
    )
    boundary_rows = df[df["period_end"].dt.date == at_boundary]
    assert len(boundary_rows) == len(TERMS)
    assert boundary_rows["is_realtime"].all()
    beyond_rows = df[df["period_end"].dt.date == beyond]
    assert len(beyond_rows) == len(TERMS)
    assert not beyond_rows["is_realtime"].any()


def test_parse_timeline_raises_on_missing_timeline():
    with pytest.raises(ValueError, match="timelineData"):
        parse_timeline(
            {"default": {}},
            terms=TERMS,
            geo="US",
            resolution="WEEK",
            pulled_at=PULLED_AT,
            request_params={},
        )


def test_parse_timeline_raises_on_value_count_mismatch():
    """A value array that doesn't match the term list must raise, never
    misattribute columns."""
    payload = json.loads(strip_antijson_prefix(_multiline_raw()))
    with pytest.raises(ValueError, match="schema drift"):
        parse_timeline(
            payload,
            terms=TERMS + ["extra term"],
            geo="US",
            resolution="WEEK",
            pulled_at=PULLED_AT,
            request_params={},
        )


def test_parse_timeline_raises_on_unknown_resolution():
    payload = json.loads(strip_antijson_prefix(_multiline_raw()))
    with pytest.raises(ValueError, match="resolution"):
        parse_timeline(
            payload,
            terms=TERMS,
            geo="US",
            resolution="EIGHT_MINUTE",
            pulled_at=PULLED_AT,
            request_params={},
        )


def test_truncated_response_raises():
    """A truncated body must raise (json error is a ValueError), not yield rows."""
    with pytest.raises(ValueError):
        json.loads(strip_antijson_prefix(_multiline_raw()[:100]))


# ---------------------------------------------------------------------------
# Transport: 429 backoff
# ---------------------------------------------------------------------------


class _Resp:
    def __init__(self, status_code: int, text: str = ""):
        self.status_code = status_code
        self.text = text


class _StubSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[tuple] = []

    def request(self, method, url, params=None, timeout=None):
        self.calls.append((method, url, params))
        return self._responses.pop(0)


def test_backoff_retries_once_on_429_then_succeeds():
    session = _StubSession([_Resp(429), _Resp(200, "ok-body")])
    slept: list[float] = []
    out = _request_with_backoff(session, "GET", "http://x", {}, sleep=slept.append)
    assert out == "ok-body"
    assert slept == [BACKOFF_SLEEP_S]
    assert len(session.calls) == 2


def test_backoff_raises_after_second_429():
    """One retry only — after that the collector must raise, not gap silently."""
    session = _StubSession([_Resp(429), _Resp(429)])
    with pytest.raises(RuntimeError, match="429"):
        _request_with_backoff(session, "GET", "http://x", {}, sleep=lambda _s: None)
    assert len(session.calls) == 2


def test_non_200_raises_immediately_without_retry():
    session = _StubSession([_Resp(500)])
    with pytest.raises(RuntimeError, match="500"):
        _request_with_backoff(session, "GET", "http://x", {}, sleep=lambda _s: None)
    assert len(session.calls) == 1


def test_empty_body_raises():
    session = _StubSession([_Resp(200, "")])
    with pytest.raises(RuntimeError, match="empty"):
        _request_with_backoff(session, "GET", "http://x", {}, sleep=lambda _s: None)


# ---------------------------------------------------------------------------
# collect_group wiring (stubbed session, real fixture bytes)
# ---------------------------------------------------------------------------


def _sell_side_group() -> dict:
    return {
        "name": "sell_side_v1",
        "geo": "US",
        "timeframe": "today 5-y",
        "terms": list(TERMS),
    }


def test_collect_group_end_to_end_on_fixture_bytes():
    session = _StubSession([_Resp(200, _explore_raw()), _Resp(200, _multiline_raw())])
    df = collect_group(
        session, _sell_side_group(), tz=0, pulled_at=PULLED_AT, sleep=lambda _s: None
    )
    assert len(df) == N_POINTS * len(TERMS)
    params = json.loads(df["request_params"].iloc[0])
    assert params["group"] == "sell_side_v1"
    assert params["terms"] == TERMS
    assert params["tz"] == 0
    assert params["widget_request"]["resolution"] == "WEEK"
    # Both endpoints were hit: explore first, multiline second.
    assert [c[1].rsplit("/", 1)[-1] for c in session.calls] == ["explore", "multiline"]
    tok = json.loads(session.calls[0][2]["req"])
    assert [item["keyword"] for item in tok["comparisonItem"]] == TERMS
    assert session.calls[1][2]["token"]


def test_collect_group_raises_on_term_mismatch():
    """If the widget answers for different terms, refuse the whole pull."""
    session = _StubSession([_Resp(200, _explore_raw()), _Resp(200, _multiline_raw())])
    group = _sell_side_group()
    group["terms"] = list(reversed(group["terms"]))
    with pytest.raises(ValueError, match="misattributed"):
        collect_group(session, group, tz=0, pulled_at=PULLED_AT, sleep=lambda _s: None)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def test_load_term_groups_real_config():
    groups = load_term_groups()
    names = [g["name"] for g in groups]
    assert "sell_side_v1" in names
    sell_side = groups[names.index("sell_side_v1")]
    assert sell_side["terms"] == TERMS
    assert sell_side["geo"] == "US"
    assert sell_side["timeframe"] == "today 5-y"


def test_load_term_groups_raises_on_missing_terms(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("groups:\n  - name: g1\n    geo: US\n    timeframe: today 5-y\n")
    with pytest.raises(ValueError, match="terms"):
        load_term_groups(bad)


def test_load_term_groups_raises_on_too_many_terms(tmp_path):
    terms = "".join(f"      - t{i}\n" for i in range(6))
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        f"groups:\n  - name: g1\n    geo: US\n    timeframe: today 5-y\n    terms:\n{terms}"
    )
    with pytest.raises(ValueError, match="at most"):
        load_term_groups(bad)


# ---------------------------------------------------------------------------
# Upsert (temp DB)
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db(monkeypatch, tmp_path):
    db_file = tmp_path / "t.duckdb"
    monkeypatch.setenv("METALS_DB_PATH", str(db_file))
    runner.apply_migrations(verbose=False)
    return db_file


def test_upsert_idempotent_and_typed(tmp_db):
    df = _parsed_fixture_df()
    assert upsert_search_interest(df) == len(df)
    assert upsert_search_interest(df) == len(df)  # second run: same keys, no dupes

    conn = duckdb.connect(str(tmp_db), read_only=True)
    try:
        n, n_realtime = conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN is_realtime THEN 1 ELSE 0 END) FROM search_interest"
        ).fetchone()
        assert n == len(df)
        assert n_realtime == int(df["is_realtime"].sum())
        row = conn.execute(
            """
            SELECT period_start, period_end, value, source,
                   json_extract_string(request_params, '$.geo')
            FROM search_interest
            WHERE term = 'gold price' AND is_realtime
            ORDER BY period_start DESC LIMIT 1
            """
        ).fetchone()
        assert row is not None
        period_start, period_end, value, source, geo = row
        assert str(period_start) == "2026-07-12"  # DATE column round-trips
        assert str(period_end) == "2026-07-18"
        assert value == 15  # as returned in the fixture's partial week
        assert source == SOURCE_TAG
        assert geo == "US"
    finally:
        conn.close()


def test_upsert_empty_frame_writes_nothing(tmp_db):
    import pandas as pd

    assert upsert_search_interest(pd.DataFrame()) == 0
