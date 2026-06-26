"""Macroeconomic series ingestion from FRED.

Requires FRED_API_KEY in the environment (load_dotenv'd by metals.data.db).
Run as:
    uv run python -m metals.data.fred --refresh
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd
from dotenv import load_dotenv

from metals.data.config import fred_series
from metals.data.db import connection

load_dotenv()
SOURCE_TAG = "fred"

# Approximate observations per year by reported frequency.
EXPECTED_PER_YEAR: dict[str, int] = {
    "daily": 252,
    "weekly": 52,
    "monthly": 12,
    "quarterly": 4,
    "annual": 1,
}


def _client():
    """Build the fredapi client (lazy import)."""
    from fredapi import Fred

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY not set. Add it to .env (copy .env.example to start).")
    return Fred(api_key=api_key)


def fetch_fred_series(
    series_ids: Iterable[str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """Pull one or more FRED series into a long DataFrame.

    Returns columns: timestamp_utc, series_id, value.
    Missing values are dropped - FRED uses '.' for missing observations.
    """
    fred = _client()
    frames: list[pd.DataFrame] = []
    for sid in series_ids:
        try:
            s = fred.get_series(sid, observation_start=start, observation_end=end)
        except ValueError as exc:
            print(f"WARNING: skipping FRED series {sid!r}: {exc}")
            continue
        if s is None or len(s) == 0:
            continue
        df = s.reset_index()
        df.columns = ["timestamp_utc", "value"]
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"]).dt.tz_localize(None)
        df["series_id"] = sid
        df = df.dropna(subset=["value"])
        frames.append(df[["timestamp_utc", "series_id", "value"]])
    if not frames:
        return pd.DataFrame(columns=["timestamp_utc", "series_id", "value"])
    return pd.concat(frames, ignore_index=True)


def coverage_report(
    df: pd.DataFrame,
    start: str,
    end: str | None,
    series_freq: dict[str, str] | None = None,
    min_coverage: float = 0.5,
) -> pd.DataFrame:
    """Per-series coverage audit.

    For each series in ``df`` (long format with series_id, value, timestamp_utc),
    compares observed row count to the count expected for the requested
    [start, end] window at the series' configured frequency. Series whose
    observed coverage is below ``min_coverage`` are flagged.

    Columns: series_id, freq, rows, first_obs, last_obs,
             expected_rows, coverage, flagged.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.utcnow().normalize()
    series_freq = series_freq or {}

    cols = [
        "series_id",
        "freq",
        "rows",
        "first_obs",
        "last_obs",
        "expected_rows",
        "coverage",
        "flagged",
    ]
    if df.empty:
        return pd.DataFrame(columns=cols)

    years = max((end_ts - start_ts).days / 365.25, 1e-9)
    rows_out: list[dict] = []
    for sid, g in df.groupby("series_id"):
        freq = series_freq.get(sid, "daily")
        per_year = EXPECTED_PER_YEAR.get(freq, EXPECTED_PER_YEAR["daily"])
        expected = max(int(round(per_year * years)), 1)
        rows = int(len(g))
        coverage = rows / expected
        rows_out.append(
            {
                "series_id": sid,
                "freq": freq,
                "rows": rows,
                "first_obs": g["timestamp_utc"].min(),
                "last_obs": g["timestamp_utc"].max(),
                "expected_rows": expected,
                "coverage": coverage,
                "flagged": bool(coverage < min_coverage),
            }
        )
    return pd.DataFrame(rows_out, columns=cols).sort_values("coverage").reset_index(drop=True)


def upsert_macro(df: pd.DataFrame) -> int:
    """Idempotent upsert into the macro table. Returns row count written."""
    if df.empty:
        return 0
    insert_df = df.copy()
    insert_df["source"] = SOURCE_TAG
    with connection() as conn:
        conn.register("incoming_macro", insert_df)
        conn.execute(
            """
            INSERT INTO macro (timestamp_utc, series_id, value, source)
            SELECT timestamp_utc, series_id, value, source
            FROM incoming_macro
            ON CONFLICT (timestamp_utc, series_id) DO UPDATE SET
                value  = EXCLUDED.value,
                source = EXCLUDED.source
            """
        )
        conn.unregister("incoming_macro")
    return len(insert_df)


def refresh(start: str | None = None, end: str | None = None) -> dict:
    """Pull every configured FRED series and upsert. Return a summary dict."""
    cfg = fred_series()
    series_ids = [row["id"] for row in cfg.get("series", [])]
    dr = cfg.get("date_range", {})
    start = start or dr.get("start") or "2007-01-01"
    end = end or dr.get("end") or datetime.now(timezone.utc).date().isoformat()

    df = fetch_fred_series(series_ids, start=start, end=end)
    n = upsert_macro(df)
    counts = df.groupby("series_id").size().to_dict() if not df.empty else {}
    series_freq = {row["id"]: row.get("freq", "daily") for row in cfg.get("series", [])}
    cov = coverage_report(df, start=start, end=end, series_freq=series_freq)
    return {
        "series_ids": series_ids,
        "rows_written": n,
        "date_range": [start, end],
        "rows_per_series": counts,
        "coverage": cov,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh FRED macro series.")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()
    summary = refresh(start=args.start, end=args.end)
    print(f"Series refreshed:  {len(summary['series_ids'])}")
    print(f"Rows written:      {summary['rows_written']}")
    print(f"Date range:        {summary['date_range']}")
    cov = summary["coverage"]
    if not cov.empty:
        print("\nCoverage audit:")
        hdr = f"  {'series_id':20s} {'freq':>8s} {'rows':>8s} {'expected':>10s} {'coverage':>10s}"
        print(hdr)
        for _, r in cov.iterrows():
            flag = "  <-- FLAGGED" if r["flagged"] else ""
            print(
                f"  {r['series_id']:20s} {r['freq']:>8s} {r['rows']:>8d} "
                f"{r['expected_rows']:>10d} {r['coverage']:>10.2%}{flag}"
            )
        flagged = cov[cov["flagged"]]
        if not flagged.empty:
            print(
                f"\nWARNING: {len(flagged)} series under {0.5:.0%} expected coverage. "
                f"Investigate before training."
            )


if __name__ == "__main__":
    main()
