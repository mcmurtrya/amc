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


def _client():
    """Build the fredapi client (lazy import)."""
    from fredapi import Fred

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY not set. Add it to .env (copy .env.example to start)."
        )
    return Fred(api_key=api_key)


def fetch_fred_series(
    series_ids: Iterable[str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """Pull one or more FRED series into a long DataFrame.

    Returns columns: timestamp_utc, series_id, value.
    Missing values are dropped — FRED uses '.' for missing observations.
    """
    fred = _client()
    frames: list[pd.DataFrame] = []
    for sid in series_ids:
        s = fred.get_series(sid, observation_start=start, observation_end=end)
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
    return {
        "series_ids": series_ids,
        "rows_written": n,
        "date_range": [start, end],
        "rows_per_series": counts,
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
    for sid, n in sorted(summary["rows_per_series"].items()):
        print(f"  {sid:20s} {n:>8d} rows")


if __name__ == "__main__":
    main()
