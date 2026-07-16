"""Tests for the Google Trends CSV importer (collector 3). No network.

The collector was rewritten 2026-07-16 from a live scraper into an importer of
Google's sanctioned manual export (journal.md). Fixtures under
``tests/fixtures/trends/`` are realistic ``multiTimeline.csv`` files for the frozen
``sell_side_v1`` basket (5 terms, geo=US, weekly), including the UTF-8 BOM a fresh
browser download carries and the literal ``<1`` cell Trends emits for sub-1
interest:

- ``multiTimeline_sell_side_v1.csv`` — weekly, 6 rows (2 setup-time 2021 weeks +
  4 recent 2026 weeks, the last a trailing partial week), two ``<1`` cells.
- ``multiTimeline_monthly.csv``      — the low-volume monthly fallback layout.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from metals.data.migrations import runner
from metals.data.trends import (
    REALTIME_WINDOW_DAYS,
    SOURCE_TAG,
    ExportFrame,
    TrendsImportError,
    build_rows,
    get_group,
    load_term_groups,
    parse_multitimeline,
    read_export_csv,
    reconcile_terms,
    refresh,
    upsert_search_interest,
)

FIXTURES = Path(__file__).parent / "fixtures" / "trends"
WEEKLY_CSV = FIXTURES / "multiTimeline_sell_side_v1.csv"
MONTHLY_CSV = FIXTURES / "multiTimeline_monthly.csv"
TERMS = ["sell gold", "cash for gold", "gold price", "sell silver", "coin shop near me"]
# Fixed naive-UTC stand-in for the moment the fixture was "downloaded".
PULLED_AT = datetime(2026, 7, 16, 12, 0, 0)
N_WEEKS = 6

EXPECTED_COLUMNS = [
    "pulled_at",
    "geo",
    "term",
    "period_start",
    "period_end",
    "value",
    "value_lt1",
    "request_params",
    "source",
    "is_realtime",
]


def _group() -> dict:
    return {"name": "sell_side_v1", "geo": "US", "timeframe": "today 5-y", "terms": list(TERMS)}


def _df(pulled_at: datetime = PULLED_AT) -> pd.DataFrame:
    exp = read_export_csv(WEEKLY_CSV)
    return build_rows(exp, geo="US", pulled_at=pulled_at, request_params={"group": "sell_side_v1"})


# ---------------------------------------------------------------------------
# CSV structural parse
# ---------------------------------------------------------------------------


def test_parse_multitimeline_structure_and_bom():
    exp = read_export_csv(WEEKLY_CSV)
    assert exp.resolution_token == "Week"
    assert exp.resolution == "WEEK"
    assert exp.geo_label == "United States"
    assert exp.terms == TERMS  # order preserved from the header
    assert len(exp.periods) == N_WEEKS
    # BOM is stripped: the title line is clean, no stray U+FEFF.
    assert exp.title_line == "Category: All categories"
    assert "﻿" not in (exp.title_line or "")
    assert exp.header_line.startswith("Week,sell gold: (United States)")


def test_parse_monthly_resolution_and_period_bounds():
    exp = read_export_csv(MONTHLY_CSV)
    assert exp.resolution == "MONTH"
    df = build_rows(exp, geo="US", pulled_at=PULLED_AT, request_params={})
    may = df[(df["term"] == "sell gold") & (df["period_start"] == datetime(2026, 5, 1))].iloc[0]
    assert may["period_end"] == datetime(2026, 5, 31)  # month-end via _period_end


def test_parse_raises_without_resolution_header():
    with pytest.raises(TrendsImportError, match="no Day/Week/Month header row"):
        parse_multitimeline("Category: All categories\n\nfoo,bar\n2026-01-01,1\n")


def test_parse_raises_on_bad_column_header():
    text = "Category: All categories\n\nWeek,sell gold\n2026-07-05,5\n"  # missing ': (geo)'
    with pytest.raises(TrendsImportError, match="format drift"):
        parse_multitimeline(text)


def test_parse_raises_on_ragged_row():
    text = (
        "Category: All categories\n\n"
        "Week,sell gold: (United States),gold price: (United States)\n"
        "2026-07-05,5\n"  # only one value for two term columns
    )
    with pytest.raises(TrendsImportError, match="ragged"):
        parse_multitimeline(text)


def test_parse_raises_on_unparseable_date():
    text = "Category: All categories\n\nWeek,sell gold: (United States)\nnot-a-date,5\n"
    with pytest.raises(TrendsImportError, match="unparseable"):
        parse_multitimeline(text)


def test_parse_tolerates_excel_trailing_commas_and_missing_bom():
    # Excel re-save: no BOM, trailing commas padding the short preamble lines.
    text = "Category: All categories,\n,\nWeek,sell gold: (United States)\n2026-07-05,7\n"
    exp = parse_multitimeline(text)
    assert exp.terms == ["sell gold"]
    assert exp.periods[0][0] == date(2026, 7, 5)


# ---------------------------------------------------------------------------
# The <1 gotcha: distinct from 0, never dropped
# ---------------------------------------------------------------------------


def test_sub_one_stored_as_zero_with_flag_never_dropped():
    df = _df()
    # No row is dropped despite two '<1' cells: 6 weeks x 5 terms.
    assert len(df) == N_WEEKS * len(TERMS)
    sub = df[df["value_lt1"]]
    assert len(sub) == 2  # exactly the two '<1' cells in the fixture
    assert (sub["value"] == 0).all()  # stored as 0, not dropped, not int-crashed
    # A true 0 would carry value_lt1 == False; the flag is the only distinguisher.
    assert not df.loc[~df["value_lt1"], "value_lt1"].any()


def test_value_out_of_range_rejected():
    exp = read_export_csv(WEEKLY_CSV)
    bad = ExportFrame(
        resolution_token="Week",
        resolution="WEEK",
        geo_label="United States",
        terms=list(TERMS),
        header_line=exp.header_line,
        title_line=exp.title_line,
        periods=[(date(2026, 7, 5), ["150", "1", "1", "1", "1"])],  # 150 > 100
    )
    with pytest.raises(TrendsImportError, match="outside the 0-100"):
        build_rows(bad, geo="US", pulled_at=PULLED_AT, request_params={})


def test_unknown_value_token_rejected():
    exp = read_export_csv(WEEKLY_CSV)
    bad = ExportFrame(
        resolution_token="Week",
        resolution="WEEK",
        geo_label="United States",
        terms=list(TERMS),
        header_line=exp.header_line,
        title_line=exp.title_line,
        periods=[(date(2026, 7, 5), ["N/A", "1", "1", "1", "1"])],
    )
    with pytest.raises(TrendsImportError, match="neither an integer"):
        build_rows(bad, geo="US", pulled_at=PULLED_AT, request_params={})


# ---------------------------------------------------------------------------
# Row assembly: shape, periods, is_realtime, tz
# ---------------------------------------------------------------------------


def test_build_rows_shape_and_columns():
    df = _df()
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) == N_WEEKS * len(TERMS)
    assert (df["source"] == SOURCE_TAG).all()
    assert (df["geo"] == "US").all()  # config code stored, not the CSV label
    assert set(df["term"]) == set(TERMS)
    assert df["value"].between(0, 100).all()


def test_weekly_periods_span_seven_days():
    df = _df()
    assert (df["period_end"] - df["period_start"]).dt.days.eq(6).all()
    first = df[df["period_start"] == datetime(2021, 7, 18)].iloc[0]
    assert first["period_end"] == datetime(2021, 7, 24)


def test_is_realtime_window():
    """Realtime iff period_end within 14 days of pulled_at; earlier weeks are
    setup-time context, permanently False."""
    df = _df()
    realtime_ends = set(df.loc[df["is_realtime"], "period_end"].dt.date.astype(str))
    assert realtime_ends == {"2026-07-04", "2026-07-11", "2026-07-18"}
    # Week ending 2026-06-27 is 19 days before the 2026-07-16 pull -> not realtime.
    boundary = df[df["period_end"] == datetime(2026, 6, 27)]
    assert not boundary.empty
    assert not boundary["is_realtime"].any()
    # 2021 setup-time history is all non-realtime.
    old = df[df["period_start"] == datetime(2021, 7, 18)]
    assert not old["is_realtime"].any()


def test_is_realtime_exact_boundary():
    """period_end exactly REALTIME_WINDOW_DAYS before the pull is realtime; one day
    beyond is permanently False. DAY resolution makes period_end == period_start."""
    at_boundary = PULLED_AT.date() - timedelta(days=REALTIME_WINDOW_DAYS)
    beyond = at_boundary - timedelta(days=1)
    exp = ExportFrame(
        resolution_token="Day",
        resolution="DAY",
        geo_label="United States",
        terms=["sell gold"],
        header_line="Day,sell gold: (United States)",
        title_line=None,
        periods=[(beyond, ["1"]), (at_boundary, ["1"])],
    )
    df = build_rows(exp, geo="US", pulled_at=PULLED_AT, request_params={})
    assert df.loc[df["period_end"].dt.date == at_boundary, "is_realtime"].all()
    assert not df.loc[df["period_end"].dt.date == beyond, "is_realtime"].any()


def test_build_rows_normalizes_aware_pulled_at_to_naive_utc():
    df = build_rows(
        read_export_csv(WEEKLY_CSV),
        geo="US",
        pulled_at=PULLED_AT.replace(tzinfo=UTC),
        request_params={},
    )
    assert df["pulled_at"].dt.tz is None
    assert (df["pulled_at"] == PULLED_AT).all()


# ---------------------------------------------------------------------------
# Term / geo reconciliation
# ---------------------------------------------------------------------------


def test_reconcile_ok_on_frozen_basket():
    violations: list[str] = []
    reconcile_terms(read_export_csv(WEEKLY_CSV), _group(), violations)
    assert violations == []


def test_reconcile_flags_missing_term():
    exp = read_export_csv(WEEKLY_CSV)
    trimmed = ExportFrame(
        resolution_token=exp.resolution_token,
        resolution=exp.resolution,
        geo_label=exp.geo_label,
        terms=TERMS[:-1],  # drop "coin shop near me"
        header_line=exp.header_line,
        title_line=exp.title_line,
        periods=[(s, c[:-1]) for s, c in exp.periods],
    )
    violations: list[str] = []
    reconcile_terms(trimmed, _group(), violations)
    assert any("missing frozen term" in v and "coin shop near me" in v for v in violations)


def test_reconcile_flags_unexpected_term():
    exp = read_export_csv(WEEKLY_CSV)
    extra = ExportFrame(
        resolution_token=exp.resolution_token,
        resolution=exp.resolution,
        geo_label=exp.geo_label,
        terms=[*TERMS, "buy platinum"],
        header_line=exp.header_line,
        title_line=exp.title_line,
        periods=[(s, [*c, "1"]) for s, c in exp.periods],
    )
    violations: list[str] = []
    reconcile_terms(extra, _group(), violations)
    assert any("not in group" in v and "buy platinum" in v for v in violations)


def test_reconcile_flags_geo_mismatch():
    exp = read_export_csv(WEEKLY_CSV)
    worldwide = ExportFrame(
        resolution_token=exp.resolution_token,
        resolution=exp.resolution,
        geo_label="Worldwide",
        terms=list(TERMS),
        header_line=exp.header_line,
        title_line=exp.title_line,
        periods=exp.periods,
    )
    violations: list[str] = []
    reconcile_terms(worldwide, _group(), violations)
    assert any("geography" in v and "Worldwide" in v for v in violations)


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


def test_get_group_returns_named_group():
    grp = get_group("sell_side_v1")
    assert grp["terms"] == TERMS


def test_get_group_raises_on_unknown():
    with pytest.raises(ValueError, match="not found"):
        get_group("no_such_group")


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
# Upsert + refresh (temp DB)
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db(monkeypatch, tmp_path):
    db_file = tmp_path / "t.duckdb"
    monkeypatch.setenv("METALS_DB_PATH", str(db_file))
    runner.apply_migrations(verbose=False)
    return db_file


def test_upsert_idempotent_and_typed(tmp_db):
    df = _df()
    assert upsert_search_interest(df) == len(df)
    assert upsert_search_interest(df) == len(df)  # same keys, no dupes

    conn = duckdb.connect(str(tmp_db), read_only=True)
    try:
        n, n_realtime = conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN is_realtime THEN 1 ELSE 0 END) FROM search_interest"
        ).fetchone()
        assert n == len(df)
        assert n_realtime == int(df["is_realtime"].sum())
        # New imported rows are usable (never auto-quarantined).
        assert conn.execute(
            "SELECT COUNT(*) FROM search_interest WHERE quarantine_reason IS NULL"
        ).fetchone()[0] == len(df)
        # The '<1' cell for the peak-week low-volume term round-trips as value 0 + flag.
        row = conn.execute(
            """
            SELECT value, value_lt1 FROM search_interest
            WHERE term = 'coin shop near me' AND period_start = DATE '2026-07-05'
            """
        ).fetchone()
        assert row == (0, True)
    finally:
        conn.close()


def test_upsert_empty_frame_writes_nothing(tmp_db):
    assert upsert_search_interest(pd.DataFrame()) == 0


def test_refresh_imports_fixture_end_to_end(tmp_db):
    summary = refresh(WEEKLY_CSV, group="sell_side_v1", pulled_at=PULLED_AT)
    assert summary["rows_written"] == N_WEEKS * len(TERMS)
    assert summary["sub_one_rows"] == 2
    assert summary["realtime_rows"] == 15  # 3 realtime weeks x 5 terms
    assert summary["period_range"] == ["2021-07-18", "2026-07-18"]

    conn = duckdb.connect(str(tmp_db), read_only=True)
    try:
        raw = conn.execute("SELECT request_params FROM search_interest LIMIT 1").fetchone()[0]
        params = json.loads(raw)
        assert params["acquisition"] == "manual_csv_export"
        assert params["timeframe_source"] == "config"
        assert params["source_file"] == WEEKLY_CSV.name
    finally:
        conn.close()


def test_refresh_default_pulled_at_does_not_overstate_realtime(tmp_db):
    """Omitting --pulled-at uses import-now; import is at/after the true download,
    so the default can only demote freshness, never inflate it."""
    summary = refresh(WEEKLY_CSV, group="sell_side_v1")  # pulled_at defaults to now
    # 'now' is well after 2026-07-16, so at most the same realtime rows, never more.
    assert summary["realtime_rows"] <= 15


def test_refresh_rejects_wrong_basket(tmp_db, tmp_path):
    bad = tmp_path / "wrong.csv"
    bad.write_text(
        "Category: All categories\n\nWeek,platinum price: (United States)\n2026-07-05,50\n",
        encoding="utf-8",
    )
    with pytest.raises(TrendsImportError, match="missing frozen term"):
        refresh(bad, group="sell_side_v1")
