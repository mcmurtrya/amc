"""Economic event ingestion.

Phase 2 step 2.1. Loads the curated FOMC calendar from
``configs/fomc_calendar.csv`` and upserts into the ``events`` table.

The calendar covers scheduled FOMC meetings plus intermeeting conference
calls that resulted in policy action (e.g. emergency rate cuts in March
2020). Cancelled meetings and purely procedural notation votes are
intentionally omitted — they aren't market-moving events in the way
scheduled or action-bearing meetings are.

Date convention: ``end_date`` is the canonical "FOMC day" (the date the
statement is released). For one-day meetings it equals the start date; for
multi-day meetings it is the second day.

Run as:
    uv run python -m metals.data.events
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from metals.data.db import connection

SOURCE_TAG = "federalreserve.gov"
DEFAULT_FOMC_CSV = Path(__file__).resolve().parents[3] / "configs" / "fomc_calendar.csv"


def load_fomc_csv(path: Path | str = DEFAULT_FOMC_CSV) -> pd.DataFrame:
    """Load the curated FOMC calendar into a clean DataFrame.

    Returns columns ready for the ``events`` table:
        timestamp_utc, event_type, event_id, metadata (JSON str), source.
    """
    df = pd.read_csv(path, dtype={"notes": "string"})
    df["timestamp_utc"] = pd.to_datetime(df["end_date"]).dt.tz_localize(None)
    df["event_type"] = "FOMC"
    df["event_id"] = "fomc_" + df["timestamp_utc"].dt.strftime("%Y-%m-%d")
    df["source"] = SOURCE_TAG

    def _meta(row: pd.Series) -> str:
        payload = {
            "is_scheduled": bool(row["is_scheduled"]),
            "is_multi_day": bool(row["is_multi_day"]),
            "has_press_conference": bool(row["has_press_conference"]),
            "meeting_kind": row["meeting_kind"],
        }
        note = row.get("notes")
        if isinstance(note, str) and note.strip():
            payload["notes"] = note.strip()
        return json.dumps(payload, sort_keys=True)

    df["metadata"] = df.apply(_meta, axis=1)
    return df[["timestamp_utc", "event_type", "event_id", "metadata", "source"]]


def upsert_events(df: pd.DataFrame) -> int:
    """Idempotent upsert into the ``events`` table. Returns rows written."""
    if df.empty:
        return 0
    with connection() as conn:
        conn.register("incoming_events", df)
        conn.execute(
            """
            INSERT INTO events
                (timestamp_utc, event_type, event_id, metadata, source)
            SELECT timestamp_utc, event_type, event_id, metadata, source
            FROM incoming_events
            ON CONFLICT (timestamp_utc, event_type, event_id) DO UPDATE SET
                metadata = EXCLUDED.metadata,
                source   = EXCLUDED.source
            """
        )
        conn.unregister("incoming_events")
    return int(len(df))


def refresh(
    path: Path | str = DEFAULT_FOMC_CSV,
    cutoff: str | None = None,
) -> dict:
    """Load + upsert FOMC events. ``cutoff`` (YYYY-MM-DD) drops future rows."""
    df = load_fomc_csv(path)
    if cutoff is None:
        cutoff = datetime.now(timezone.utc).date().isoformat()
    df = df[df["timestamp_utc"] <= pd.Timestamp(cutoff)]
    n = upsert_events(df)
    return {
        "rows_written": n,
        "min_date": df["timestamp_utc"].min().date().isoformat() if n else None,
        "max_date": df["timestamp_utc"].max().date().isoformat() if n else None,
        "cutoff": cutoff,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh FOMC events table.")
    parser.add_argument("--path", default=str(DEFAULT_FOMC_CSV), help="Path to FOMC calendar CSV.")
    parser.add_argument("--cutoff", default=None, help="YYYY-MM-DD; drop events after this date.")
    args = parser.parse_args()
    summary = refresh(path=args.path, cutoff=args.cutoff)
    print(f"FOMC rows written: {summary['rows_written']}")
    print(f"Date range:        [{summary['min_date']}, {summary['max_date']}]")
    print(f"Cutoff:            {summary['cutoff']}")


if __name__ == "__main__":
    main()
