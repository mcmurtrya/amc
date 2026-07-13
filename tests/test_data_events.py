"""Tests for the FOMC calendar loader."""

from __future__ import annotations

import io
import json

import duckdb
import pandas as pd
import pytest

from metals.data.events import load_fomc_csv, refresh, upsert_events
from metals.data.migrations import runner

# Legacy layout without the release-time columns — must keep loading.
CSV_FIXTURE = """end_date,is_scheduled,is_multi_day,has_press_conference,meeting_kind,notes
2019-01-30,true,true,true,regular,
2020-03-15,false,false,false,conference_call,emergency 100bp cut + QE
2020-04-29,true,true,true,regular,
"""

# Current layout with release_time_et / presser_time_et (real rows).
CSV_FIXTURE_TIMES = (
    "end_date,is_scheduled,is_multi_day,has_press_conference,meeting_kind,"
    "release_time_et,presser_time_et,notes\n"
    "2011-04-27,true,true,true,regular,12:30,14:15,first Bernanke press conference\n"
    "2019-01-30,true,true,true,regular,14:00,14:30,\n"
    "2020-03-15,false,false,false,conference_call,17:00,,emergency 100bp cut + QE\n"
)


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


def test_load_fomc_csv_without_time_columns_omits_time_keys():
    """Legacy CSVs (no release-time columns) load with the keys absent."""
    df = _read_fixture()
    for s in df["metadata"]:
        md = json.loads(s)
        assert "release_time_et" not in md
        assert "presser_time_et" not in md


def test_load_fomc_csv_carries_release_times_into_metadata():
    df = load_fomc_csv(io.StringIO(CSV_FIXTURE_TIMES)).set_index("event_id")
    md = json.loads(df.loc["fomc_2011-04-27", "metadata"])
    assert md["release_time_et"] == "12:30"
    assert md["presser_time_et"] == "14:15"
    md = json.loads(df.loc["fomc_2019-01-30", "metadata"])
    assert md["release_time_et"] == "14:00"
    assert md["presser_time_et"] == "14:30"
    # blank presser cell -> key absent
    md = json.loads(df.loc["fomc_2020-03-15", "metadata"])
    assert md["release_time_et"] == "17:00"
    assert "presser_time_et" not in md


def test_load_fomc_csv_rejects_malformed_release_time():
    bad = CSV_FIXTURE_TIMES.replace("12:30", "2:30pm")
    with pytest.raises(ValueError, match="Malformed release_time_et"):
        load_fomc_csv(io.StringIO(bad))


def test_load_fomc_csv_rejects_missing_required_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        load_fomc_csv(io.StringIO("end_date,notes\n2019-01-30,\n"))


def test_load_fomc_csv_real_file_parses():
    """The committed configs/fomc_calendar.csv must parse without error and
    cover at least 2007-2026."""
    df = load_fomc_csv()
    assert len(df) > 175  # ~175 scheduled + a handful of unscheduled
    years = df["timestamp_utc"].dt.year
    assert years.min() <= 2007
    assert years.max() >= 2026
    # Every row must have valid JSON metadata.
    for s in df["metadata"]:
        json.loads(s)  # raises if invalid


def test_load_fomc_csv_dates_are_unique():
    """No two FOMC events should share an end date and event_id."""
    df = load_fomc_csv()
    assert df["event_id"].is_unique


def test_real_file_release_time_eras():
    """Spot-check the release-time era rules on the committed calendar
    (era boundaries per Fed press releases monetary20110324a / 20130313a)."""
    df = load_fomc_csv().set_index("event_id")
    md = {eid: json.loads(df.loc[eid, "metadata"]) for eid in df.index}
    assert md["fomc_2010-11-03"]["release_time_et"] == "14:15"  # pre-presser era
    assert md["fomc_2011-04-27"]["release_time_et"] == "12:30"  # presser meeting
    assert md["fomc_2011-04-27"]["presser_time_et"] == "14:15"
    assert md["fomc_2011-08-09"]["release_time_et"] == "14:15"  # 2011-12 non-presser
    assert md["fomc_2013-01-30"]["release_time_et"] == "14:15"  # pre-change (Mar 2013)
    assert md["fomc_2013-03-20"]["release_time_et"] == "14:00"  # modern era
    assert md["fomc_2013-03-20"]["presser_time_et"] == "14:30"
    assert md["fomc_2020-03-15"]["release_time_et"] == "17:00"  # documented emergency
    # every scheduled regular meeting carries a statement release time
    regular = [m for m in md.values() if m["meeting_kind"] == "regular"]
    assert regular and all("release_time_et" in m for m in regular)
    # modern presser meetings carry the 14:30 presser start
    assert md["fomc_2026-12-09"]["release_time_et"] == "14:00"
    assert md["fomc_2026-12-09"]["presser_time_et"] == "14:30"


def test_real_file_covers_all_2026_meetings():
    """The Fed's published 2026 schedule (fomccalendars.htm) is fully present."""
    df = load_fomc_csv()
    dates_2026 = set(
        df.loc[df["timestamp_utc"].dt.year == 2026, "timestamp_utc"].dt.strftime("%Y-%m-%d")
    )
    assert dates_2026 == {
        "2026-01-28",
        "2026-03-18",
        "2026-04-29",
        "2026-06-17",
        "2026-07-29",
        "2026-09-16",
        "2026-10-28",
        "2026-12-09",
    }


@pytest.fixture()
def tmp_db(monkeypatch, tmp_path):
    db_file = tmp_path / "t.duckdb"
    monkeypatch.setenv("METALS_DB_PATH", str(db_file))
    runner.apply_migrations(verbose=False)
    return db_file


def _count_events(db_file) -> int:
    conn = duckdb.connect(str(db_file), read_only=True)
    try:
        return conn.execute("SELECT count(*) FROM events").fetchone()[0]
    finally:
        conn.close()


def test_upsert_events_is_idempotent(tmp_db):
    df = load_fomc_csv(io.StringIO(CSV_FIXTURE_TIMES))
    assert upsert_events(df) == 3
    assert upsert_events(df) == 3
    assert _count_events(tmp_db) == 3  # ON CONFLICT updated, not duplicated


def test_refresh_cutoff_drops_future_rows(tmp_db):
    summary = refresh(path=io.StringIO(CSV_FIXTURE_TIMES), cutoff="2019-12-31")
    assert summary["rows_written"] == 2
    assert summary["max_date"] == "2019-01-30"
    assert _count_events(tmp_db) == 2
