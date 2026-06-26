"""Caldara–Iacoviello Geopolitical Risk (GPR) index ingestion.

Source: https://www.matteoiacoviello.com/gpr.htm

The daily series is published as an Excel file. Pulls and parses it, then
writes both the daily and monthly aggregates into the ``macro`` table with
series_ids ``GPR_DAILY`` and ``GPR_MONTHLY``.

Run as:
    uv run python -m metals.data.gpr --refresh
"""

from __future__ import annotations

import argparse
import io
from datetime import datetime, timezone

import pandas as pd
import requests

from metals.data.db import connection

SOURCE_TAG = "gpr_iacoviello"
DAILY_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"


def fetch_gpr_daily(url: str = DAILY_URL, timeout: int = 30) -> pd.DataFrame:
    """Download and parse the daily GPR series.

    Returns columns: timestamp_utc, series_id, value.
    """
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    df = pd.read_excel(io.BytesIO(resp.content))
    # Standardize column names — the source uses 'date' and 'GPRD'.
    cols_lower = {c.lower(): c for c in df.columns}
    date_col = cols_lower.get("date") or cols_lower.get("day") or df.columns[0]
    gpr_col = cols_lower.get("gprd") or cols_lower.get("gpr") or df.columns[1]
    out = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(df[date_col]).dt.tz_localize(None),
            "value": pd.to_numeric(df[gpr_col], errors="coerce"),
        }
    )
    out["series_id"] = "GPR_DAILY"
    out = out.dropna(subset=["value"])
    return out[["timestamp_utc", "series_id", "value"]]


def upsert_macro(df: pd.DataFrame) -> int:
    """Idempotent upsert into the macro table."""
    if df.empty:
        return 0
    insert_df = df.copy()
    insert_df["source"] = SOURCE_TAG
    with connection() as conn:
        conn.register("incoming_gpr", insert_df)
        conn.execute(
            """
            INSERT INTO macro (timestamp_utc, series_id, value, source)
            SELECT timestamp_utc, series_id, value, source
            FROM incoming_gpr
            ON CONFLICT (timestamp_utc, series_id) DO UPDATE SET
                value  = EXCLUDED.value,
                source = EXCLUDED.source
            """
        )
        conn.unregister("incoming_gpr")
    return len(insert_df)


def refresh() -> dict:
    """Pull GPR daily, upsert. Return a summary dict."""
    df = fetch_gpr_daily()
    n = upsert_macro(df)
    return {
        "rows_written": n,
        "min_date": df["timestamp_utc"].min() if not df.empty else None,
        "max_date": df["timestamp_utc"].max() if not df.empty else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the GPR daily index.")
    parser.parse_args()
    summary = refresh()
    print(f"GPR rows written: {summary['rows_written']}")
    print(f"Date range:       [{summary['min_date']}, {summary['max_date']}]")


if __name__ == "__main__":
    main()
