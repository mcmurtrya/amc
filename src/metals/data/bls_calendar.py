"""BLS release-calendar ingestion.

Phase 7.1 collector 5b. Loads the curated CPI and Employment Situation
release calendar from ``configs/bls_calendar.csv`` and upserts into the
``events`` table (event_type ``CPI`` / ``EMPSIT``).

The calendar was built from bls.gov "Schedule of Releases" pages (current
pages plus Internet Archive captures of the same pages for past years) and
cross-checked row-by-row against ALFRED's realized release dates, so delayed
or cancelled prints (e.g. the fall-2025 government shutdown) carry the date
the release actually hit, with the original schedule in ``notes``.

Date convention: ``release_date`` is the day the report is (or was) published;
the 08:30 ET wall-clock release time rides along in the metadata JSON as
``release_time_et``, together with the ``reference_period`` (YYYY-MM) the
print describes.

Run as:
    uv run python -m metals.data.bls_calendar
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Hashable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from metals.data.events import upsert_events

SOURCE_TAG = "bls.gov"
DEFAULT_BLS_CSV = Path(__file__).resolve().parents[3] / "configs" / "bls_calendar.csv"
RELEASE_TYPES = ("CPI", "EMPSIT")
REQUIRED_BLS_COLUMNS = ("release_date", "release_time_et", "release_type", "reference_period")
_TIME_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")
_PERIOD_RE = re.compile(r"^\d{4}-(?:0[1-9]|1[0-2])$")


def load_bls_csv(path: Path | str = DEFAULT_BLS_CSV) -> pd.DataFrame:
    """Load the curated BLS release calendar into a clean DataFrame.

    Returns columns ready for the ``events`` table:
        timestamp_utc, event_type, event_id, metadata (JSON str), source.

    Raises ``ValueError`` on schema drift (missing columns), an empty file,
    unknown release types, malformed times/periods, or duplicate releases —
    a silently short calendar is worse than a loud one.
    """
    dtypes: dict[Hashable, str] = {
        c: "string" for c in ("release_time_et", "reference_period", "notes")
    }
    df = pd.read_csv(path, dtype=dtypes)
    missing = [c for c in REQUIRED_BLS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"BLS calendar {path!r} is missing required columns: {missing}")
    if df.empty:
        raise ValueError(f"BLS calendar {path!r} contains no rows")
    bad_types = sorted(set(df["release_type"]) - set(RELEASE_TYPES))
    if bad_types:
        raise ValueError(f"BLS calendar {path!r} has unknown release_type values: {bad_types}")
    for column, pattern in (("release_time_et", _TIME_RE), ("reference_period", _PERIOD_RE)):
        values = df[column].fillna("")
        bad = sorted(values[~values.str.match(pattern)].unique())
        if bad:
            raise ValueError(f"BLS calendar {path!r} has malformed {column} values: {bad}")

    df["timestamp_utc"] = pd.to_datetime(df["release_date"]).dt.tz_localize(None)
    df["event_type"] = df["release_type"]
    df["event_id"] = (
        df["release_type"].str.lower() + "_" + df["timestamp_utc"].dt.strftime("%Y-%m-%d")
    )
    dupes = df.loc[df["event_id"].duplicated(), "event_id"].tolist()
    if dupes:
        raise ValueError(f"BLS calendar {path!r} has duplicate releases: {dupes}")
    df["source"] = SOURCE_TAG

    def _meta(row: pd.Series) -> str:
        payload = {
            "reference_period": row["reference_period"],
            "release_time_et": row["release_time_et"],
        }
        note = row.get("notes")
        if isinstance(note, str) and note.strip():
            payload["notes"] = note.strip()
        return json.dumps(payload, sort_keys=True)

    df["metadata"] = df.apply(_meta, axis=1)
    return df[["timestamp_utc", "event_type", "event_id", "metadata", "source"]]


def refresh(
    path: Path | str = DEFAULT_BLS_CSV,
    cutoff: str | None = None,
) -> dict:
    """Load + upsert BLS release events. ``cutoff`` (YYYY-MM-DD) drops future rows."""
    df = load_bls_csv(path)
    if cutoff is None:
        cutoff = datetime.now(UTC).date().isoformat()
    df = df[df["timestamp_utc"] <= pd.Timestamp(cutoff)]
    n = upsert_events(df)
    return {
        "rows_written": n,
        "rows_per_type": df.groupby("event_type").size().to_dict() if n else {},
        "min_date": df["timestamp_utc"].min().date().isoformat() if n else None,
        "max_date": df["timestamp_utc"].max().date().isoformat() if n else None,
        "cutoff": cutoff,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh BLS release events table.")
    parser.add_argument("--path", default=str(DEFAULT_BLS_CSV), help="Path to BLS calendar CSV.")
    parser.add_argument("--cutoff", default=None, help="YYYY-MM-DD; drop events after this date.")
    args = parser.parse_args()
    summary = refresh(path=args.path, cutoff=args.cutoff)
    print(f"BLS rows written:  {summary['rows_written']}")
    print(f"Rows per type:     {summary['rows_per_type']}")
    print(f"Date range:        [{summary['min_date']}, {summary['max_date']}]")
    print(f"Cutoff:            {summary['cutoff']}")


if __name__ == "__main__":
    main()
