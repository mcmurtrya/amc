"""Tests for the BLS release-calendar loader (no network; fixture-driven)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from metals.data.bls_calendar import load_bls_csv, refresh
from metals.data.migrations import runner

FIXTURE = Path(__file__).parent / "fixtures" / "bls_calendar" / "bls_calendar_excerpt.csv"

VALID_CSV = """release_date,release_time_et,release_type,reference_period,notes
2024-12-06,08:30,EMPSIT,2024-11,
2024-12-11,08:30,CPI,2024-11,
2025-10-24,08:30,CPI,2025-09,delayed by government shutdown; originally scheduled 2025-10-15
"""


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


def test_load_bls_csv_columns_and_tags():
    df = load_bls_csv(FIXTURE)
    assert list(df.columns) == [
        "timestamp_utc",
        "event_type",
        "event_id",
        "metadata",
        "source",
    ]
    assert set(df["event_type"]) == {"CPI", "EMPSIT"}
    assert (df["source"] == "bls.gov").all()


def test_load_bls_csv_event_id_derives_from_type_and_date():
    df = load_bls_csv(io.StringIO(VALID_CSV)).set_index("event_id")
    assert "empsit_2024-12-06" in df.index
    assert "cpi_2024-12-11" in df.index
    assert df.loc["cpi_2024-12-11", "event_type"] == "CPI"


def test_load_bls_csv_timestamps_are_naive_utc_midnight():
    df = load_bls_csv(FIXTURE)
    assert df["timestamp_utc"].dt.tz is None
    assert (df["timestamp_utc"].dt.time == pd.Timestamp("00:00").time()).all()
    assert df["timestamp_utc"].iloc[0] == pd.Timestamp("2024-12-06")


def test_load_bls_csv_metadata_carries_period_time_and_notes():
    df = load_bls_csv(FIXTURE).set_index("event_id")
    md = json.loads(df.loc["cpi_2025-10-24", "metadata"])
    assert md["reference_period"] == "2025-09"
    assert md["release_time_et"] == "08:30"
    assert md["notes"].startswith("delayed by government shutdown")
    # blank notes cells must not produce a 'notes' key
    md_clean = json.loads(df.loc["cpi_2024-12-11", "metadata"])
    assert md_clean == {"reference_period": "2024-11", "release_time_et": "08:30"}


def test_load_bls_csv_real_file_parses():
    """The committed configs/bls_calendar.csv must parse and cover 2015-2026."""
    df = load_bls_csv()
    assert len(df) >= 280  # ~143 CPI + ~143 EMPSIT
    years = df["timestamp_utc"].dt.year
    assert years.min() == 2015
    assert years.max() == 2026
    assert df["event_id"].is_unique
    for s in df["metadata"]:
        payload = json.loads(s)  # raises if invalid
        assert payload["release_time_et"] == "08:30"


def test_load_bls_csv_shutdown_gap_is_honest():
    """Fall 2025: no CPI/EMPSIT row may claim an October 2025 reference —
    those prints were never published standalone."""
    df = load_bls_csv()
    periods = {json.loads(s)["reference_period"] for s in df["metadata"]}
    assert "2025-09" in periods
    assert "2025-10" not in periods


def test_load_bls_csv_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        load_bls_csv(io.StringIO("release_date,release_type\n2024-12-11,CPI\n"))


def test_load_bls_csv_rejects_empty_file():
    header_only = "release_date,release_time_et,release_type,reference_period,notes\n"
    with pytest.raises(ValueError, match="no rows"):
        load_bls_csv(io.StringIO(header_only))


def test_load_bls_csv_rejects_unknown_release_type():
    bad = VALID_CSV.replace("EMPSIT", "PAYROLLS")
    with pytest.raises(ValueError, match="unknown release_type"):
        load_bls_csv(io.StringIO(bad))


def test_load_bls_csv_rejects_malformed_time_and_period():
    with pytest.raises(ValueError, match="malformed release_time_et"):
        load_bls_csv(io.StringIO(VALID_CSV.replace("08:30", "8.30am")))
    with pytest.raises(ValueError, match="malformed reference_period"):
        load_bls_csv(io.StringIO(VALID_CSV.replace("2025-09", "2025-13")))


def test_load_bls_csv_rejects_duplicate_releases():
    dup = VALID_CSV + "2025-10-24,08:30,CPI,2025-09,\n"
    with pytest.raises(ValueError, match="duplicate releases"):
        load_bls_csv(io.StringIO(dup))


def test_refresh_upsert_is_idempotent(tmp_db):
    first = refresh(path=FIXTURE, cutoff="2026-12-31")
    assert first["rows_written"] == 12
    assert _count_events(tmp_db) == 12
    second = refresh(path=FIXTURE, cutoff="2026-12-31")
    assert second["rows_written"] == 12
    assert _count_events(tmp_db) == 12  # ON CONFLICT updated, not duplicated


def test_refresh_cutoff_drops_future_rows(tmp_db):
    summary = refresh(path=FIXTURE, cutoff="2026-01-31")
    assert summary["rows_written"] == 9  # rows dated after 2026-01-31 dropped
    assert summary["max_date"] == "2026-01-13"
    assert summary["rows_per_type"] == {"CPI": 5, "EMPSIT": 4}
    assert _count_events(tmp_db) == 9
