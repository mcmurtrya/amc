"""Macro consensus capture from the ForexFactory weekly calendar feed.

Phase 7.1 collector 5d (``plans/phase_7_amc_program.md``,
``results/amc_data_acquisition_program.md``). Captures the pre-release
consensus ("forecast") and prior print for the USD Consumer Price Index and
Employment Situation releases into the append-only ``macro_consensus`` table.

Clean capture is the entire point of this collector: consensus numbers are
hindsight-sensitive — vendors regenerate them after the fact (the FXMacroData
refutation in ``results/amc_paid_data_review.md``). **Only rows with
``is_realtime = TRUE`` (``pulled_at < release_utc``) are training-grade**;
everything else is second-class context under the 7.7 purchased-history gate.
Each pull is a new observation: ``pulled_at`` is part of the primary key and
rows are never updated (``INSERT ... ON CONFLICT DO NOTHING``).

Feed facts (verified 2026-07-12): the feed exposes exactly
title/country/date/impact/forecast/previous. It does **not** carry an
``actual`` field even after release (checked against a Wayback snapshot taken
the day after the 2026-05-12 CPI print), so ``actual`` stays NULL from this
source; first-print actuals come from ALFRED in the 7.8 analysis. Only a
this-week feed exists — lastweek/nextweek variants return 404 — so each
``refresh()`` captures the current calendar week (Sun–Sat, US-eastern feed
times). Run it at least once early in each release week.

Units: percent fields (``cpi_mom``, ``cpi_yoy``, ``core_cpi_mom``,
``unemployment_rate``, ``ahe_mom``) are stored in percentage points
(0.2 == "0.2%"); ``nfp_change_k`` is thousands of jobs (185.0 == "185K").

Run as:
    uv run python -m metals.data.consensus
"""

from __future__ import annotations

import argparse
import re
from datetime import UTC, datetime

import pandas as pd
import requests

from metals.data.db import connection

SOURCE_TAG = "forexfactory"
FEED_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
USER_AGENT = "AMCResearchCollector/0.1 (internal research)"

# Every calendar item must carry these keys; anything less is schema drift.
EXPECTED_KEYS = frozenset({"title", "country", "date", "impact", "forecast", "previous"})

# Exact feed titles -> (release_type, normalized field). Matching must stay
# exact: "ADP Non-Farm Employment Change" is a different, out-of-scope event.
EVENT_FIELDS: dict[str, tuple[str, str]] = {
    "CPI m/m": ("CPI", "cpi_mom"),
    "CPI y/y": ("CPI", "cpi_yoy"),
    "Core CPI m/m": ("CPI", "core_cpi_mom"),
    "Non-Farm Employment Change": ("EMPSIT", "nfp_change_k"),
    "Unemployment Rate": ("EMPSIT", "unemployment_rate"),
    "Average Hourly Earnings m/m": ("EMPSIT", "ahe_mom"),
}

# Fields stored in percentage points; the one remaining field (nfp_change_k)
# is stored in thousands of jobs.
PERCENT_FIELDS = frozenset({"cpi_mom", "cpi_yoy", "core_cpi_mom", "unemployment_rate", "ahe_mom"})

COLUMNS = [
    "release_utc",
    "release_type",
    "field",
    "consensus",
    "previous",
    "actual",
    "consensus_source",
    "pulled_at",
    "is_realtime",
]

_VALUE_RE = re.compile(r"^(-?\d+(?:\.\d+)?)(%|K|M)?$")


def parse_value(raw: str | None, field: str) -> float | None:
    """Parse a feed value like ``"0.2%"`` or ``"185K"`` into the field's unit.

    Percent fields keep percentage points ("0.2%" -> 0.2) and must carry a
    ``%`` suffix; ``nfp_change_k`` is thousands ("185K" -> 185.0, "1.2M" ->
    1200.0). Empty/missing values (consensus not yet posted) return None;
    anything else unparseable raises — a changed unit is schema drift, not
    a value to guess at.
    """
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    match = _VALUE_RE.match(text)
    if match is None:
        raise ValueError(f"unparseable {field} value {raw!r} from {SOURCE_TAG}")
    number = float(match.group(1))
    suffix = match.group(2)
    if field in PERCENT_FIELDS:
        if suffix != "%":
            raise ValueError(f"expected percent for {field}, got {raw!r} from {SOURCE_TAG}")
        return number
    if suffix == "K":
        return number
    if suffix == "M":
        return number * 1000.0
    raise ValueError(f"expected K/M count for {field}, got {raw!r} from {SOURCE_TAG}")


def fetch_calendar(url: str = FEED_URL, timeout: int = 30) -> list[dict]:
    """Fetch the weekly calendar feed. Raises on HTTP error or empty payload."""
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    try:
        payload = resp.json()
    except ValueError as exc:
        raise ValueError(f"non-JSON response from {url}") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"empty or unexpected calendar payload from {url}")
    return payload


def parse_calendar(payload: list[dict], pulled_at: datetime) -> pd.DataFrame:
    """Normalize the feed into ``macro_consensus``-shaped rows.

    Filters to the in-scope USD releases (EVENT_FIELDS), converts the feed's
    offset datetimes to naive UTC, parses values, and stamps provenance.
    ``is_realtime`` is decided per row: ``pulled_at < release_utc``. A valid
    week with no in-scope events yields an empty frame (CPI/payrolls are
    monthly; most weeks have neither) — that is not a failure. Missing keys
    on any item, an offset-less datetime, or an unparseable value raise.
    """
    if not payload:
        raise ValueError("empty calendar payload — refusing to parse")
    pulled_ts = pd.Timestamp(pulled_at)
    if pulled_ts.tzinfo is not None:
        pulled_ts = pulled_ts.tz_convert("UTC").tz_localize(None)

    rows: list[dict] = []
    for item in payload:
        missing = EXPECTED_KEYS - item.keys()
        if missing:
            raise ValueError(
                f"calendar item missing keys {sorted(missing)} — feed schema drift: {item!r}"
            )
        if item["country"] != "USD":
            continue
        mapped = EVENT_FIELDS.get(item["title"])
        if mapped is None:
            continue
        release_type, field = mapped
        ts = pd.Timestamp(item["date"])
        if ts.tzinfo is None:
            raise ValueError(f"feed datetime lacks a UTC offset: {item['date']!r}")
        release_utc = ts.tz_convert("UTC").tz_localize(None)
        rows.append(
            {
                "release_utc": release_utc,
                "release_type": release_type,
                "field": field,
                "consensus": parse_value(item["forecast"], field),
                "previous": parse_value(item["previous"], field),
                "actual": None,  # the feed never publishes actuals; ALFRED does
                "consensus_source": SOURCE_TAG,
                "pulled_at": pulled_ts,
                "is_realtime": bool(pulled_ts < release_utc),
            }
        )
    df = pd.DataFrame(rows, columns=COLUMNS)
    for col in ("consensus", "previous", "actual"):
        df[col] = df[col].astype("float64")
    return df


def upsert_consensus(df: pd.DataFrame) -> int:
    """Append-only insert into ``macro_consensus``. Returns rows written.

    ``ON CONFLICT DO NOTHING``: re-inserting an identical pull is a no-op,
    while a later pull (new ``pulled_at``) lands as new observation rows.
    """
    if df.empty:
        return 0
    with connection() as conn:
        conn.register("incoming_consensus", df)
        result = conn.execute(
            """
            INSERT INTO macro_consensus
                (release_utc, release_type, field, consensus, previous, actual,
                 consensus_source, pulled_at, is_realtime)
            SELECT release_utc, release_type, field, consensus, previous, actual,
                   consensus_source, pulled_at, is_realtime
            FROM incoming_consensus
            ON CONFLICT (release_utc, release_type, field, consensus_source, pulled_at)
                DO NOTHING
            """
        ).fetchone()
        conn.unregister("incoming_consensus")
    return int(result[0]) if result else 0


def refresh(url: str = FEED_URL, timeout: int = 30) -> dict:
    """Pull the current-week calendar and append in-scope rows. Summary dict."""
    pulled_at = datetime.now(UTC).replace(tzinfo=None)
    payload = fetch_calendar(url, timeout=timeout)
    df = parse_calendar(payload, pulled_at=pulled_at)
    n = upsert_consensus(df)
    releases = sorted({f"{r.release_type} {r.release_utc.isoformat()}" for r in df.itertuples()})
    return {
        "rows_written": n,
        "rows_in_scope": int(len(df)),
        "events_in_feed": len(payload),
        "realtime_rows": int(df["is_realtime"].sum()) if not df.empty else 0,
        "releases": releases,
        "pulled_at": pulled_at.isoformat(),
        "source": SOURCE_TAG,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture USD CPI / Employment Situation consensus from ForexFactory."
    )
    parser.add_argument("--url", default=FEED_URL, help="Calendar feed URL.")
    args = parser.parse_args()
    summary = refresh(url=args.url)
    print(f"Feed events:     {summary['events_in_feed']}")
    print(f"In-scope rows:   {summary['rows_in_scope']}")
    print(f"Rows written:    {summary['rows_written']}")
    print(f"Real-time rows:  {summary['realtime_rows']}")
    print(f"Releases (UTC):  {summary['releases'] or '(none this week)'}")
    print(f"Pulled at (UTC): {summary['pulled_at']}")


if __name__ == "__main__":
    main()
