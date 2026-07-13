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

Release-time convention (Phase 7.1 collector 5a): the optional
``release_time_et`` / ``presser_time_et`` columns carry the statement release
and press-conference start times as ET wall-clock ``HH:MM`` strings. Era
rules (per federalreserve.gov press releases monetary20110324a and
monetary20130313a): scheduled meetings released at 14:15 through the January
2013 meeting, except press-conference meetings April 2011 - December 2012 at
12:30 (presser 14:15); from the March 2013 meeting onward all statements at
14:00 with pressers at 14:30. Unscheduled actions carry their documented
announcement time or blank. CSVs without these columns still load (the keys
are simply absent from the metadata JSON).

Run as:
    uv run python -m metals.data.events
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Hashable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from metals.data.db import connection

SOURCE_TAG = "federalreserve.gov"
DEFAULT_FOMC_CSV = Path(__file__).resolve().parents[3] / "configs" / "fomc_calendar.csv"
REQUIRED_FOMC_COLUMNS = (
    "end_date",
    "is_scheduled",
    "is_multi_day",
    "has_press_conference",
    "meeting_kind",
)
_TIME_COLUMNS = ("release_time_et", "presser_time_et")
_TIME_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")


def _clean_time(row: pd.Series, column: str) -> str | None:
    """Validate an optional ET wall-clock cell; blank/absent -> None."""
    value = row.get(column)
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    if not _TIME_RE.match(value):
        raise ValueError(f"Malformed {column} {value!r} for event dated {row['end_date']!r}")
    return value


def load_fomc_csv(path: Path | str = DEFAULT_FOMC_CSV) -> pd.DataFrame:
    """Load the curated FOMC calendar into a clean DataFrame.

    Returns columns ready for the ``events`` table:
        timestamp_utc, event_type, event_id, metadata (JSON str), source.
    """
    dtypes: dict[Hashable, str] = {c: "string" for c in ("notes", *_TIME_COLUMNS)}
    df = pd.read_csv(path, dtype=dtypes)
    missing = [c for c in REQUIRED_FOMC_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"FOMC calendar {path!r} is missing required columns: {missing}")
    if df.empty:
        raise ValueError(f"FOMC calendar {path!r} contains no rows")
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
        for column in _TIME_COLUMNS:
            cleaned = _clean_time(row, column)
            if cleaned is not None:
                payload[column] = cleaned
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
        cutoff = datetime.now(UTC).date().isoformat()
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
