"""Tests for the FOMC calendar loader."""

from __future__ import annotations

import io
import json

import pandas as pd

from metals.data.events import load_fomc_csv

CSV_FIXTURE = """end_date,is_scheduled,is_multi_day,has_press_conference,meeting_kind,notes
2019-01-30,true,true,true,regular,
2020-03-15,false,false,false,conference_call,emergency 100bp cut + QE
2020-04-29,true,true,true,regular,
"""


def _read_fixture() -> pd.DataFrame:
    return load_fomc_csv(io.StringIO(CSV_FIXTURE))


def test_load_fomc_csv_columns():
    df = _read_fixture()
    assert list(df.columns) == [
        "timestamp_utc",
        "event_type",
        "event_id",
        "metadata",
        "source",
    ]
    assert (df["event_type"] == "FOMC").all()
    assert (df["source"] == "federalreserve.gov").all()


def test_load_fomc_csv_event_id_is_stable():
    """event_id derives deterministically from the end date."""
    df = _read_fixture()
    assert df.loc[0, "event_id"] == "fomc_2019-01-30"
    assert df.loc[2, "event_id"] == "fomc_2020-04-29"


def test_load_fomc_csv_metadata_is_valid_json():
    df = _read_fixture()
    md = json.loads(df.loc[1, "metadata"])
    assert md["is_scheduled"] is False
    assert md["has_press_conference"] is False
    assert md["meeting_kind"] == "conference_call"
    assert md["notes"] == "emergency 100bp cut + QE"


def test_load_fomc_csv_omits_blank_notes():
    """A blank notes cell should not produce a 'notes' key in metadata."""
    df = _read_fixture()
    md = json.loads(df.loc[0, "metadata"])
    assert "notes" not in md
    assert md["is_scheduled"] is True
    assert md["has_press_conference"] is True


def test_load_fomc_csv_real_file_parses():
    """The committed configs/fomc_calendar.csv must parse without error and
    cover at least 2007-2025."""
    df = load_fomc_csv()
    assert len(df) > 150  # ~170 scheduled + a handful of unscheduled
    years = df["timestamp_utc"].dt.year
    assert years.min() <= 2007
    assert years.max() >= 2025
    # Every row must have valid JSON metadata.
    for s in df["metadata"]:
        json.loads(s)  # raises if invalid


def test_load_fomc_csv_dates_are_unique():
    """No two FOMC events should share an end date and event_id."""
    df = load_fomc_csv()
    assert df["event_id"].is_unique
