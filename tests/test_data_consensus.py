"""Tests for the macro consensus collector (Phase 7.1 collector 5d).

Fixtures are real excerpts of https://nfs.faireconomy.media/ff_calendar_thisweek.json:
- ff_calendar_cpi_week.json    — live pull 2026-07-12 (CPI week of 2026-07-14)
- ff_calendar_empsit_week.json — Wayback snapshot 2026-07-01 (pre-release for
  the 2026-07-02 Employment Situation), plus real decoy rows: JPY/EUR
  "Unemployment Rate" (exact title, wrong country) and USD "ADP Non-Farm
  Employment Change" (near-name, out of scope).
No network anywhere in this file.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from metals.data.consensus import (
    COLUMNS,
    SOURCE_TAG,
    parse_calendar,
    parse_value,
    upsert_consensus,
)
from metals.data.migrations import runner

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "consensus"

# Naive UTC instants bracketing the fixtures' release times.
BEFORE_CPI = datetime(2026, 7, 12, 23, 0)  # CPI week releases at 2026-07-14 12:30 UTC
AFTER_CPI = datetime(2026, 7, 14, 13, 0)
BEFORE_EMPSIT = datetime(2026, 7, 1, 21, 0)  # EMPSIT releases at 2026-07-02 12:30 UTC


def _load(name: str) -> list[dict]:
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture()
def tmp_db(monkeypatch, tmp_path):
    db_file = tmp_path / "t.duckdb"
    monkeypatch.setenv("METALS_DB_PATH", str(db_file))
    runner.apply_migrations(verbose=False)
    return db_file


def _table_rows(db_file) -> list[tuple]:
    conn = duckdb.connect(str(db_file), read_only=True)
    try:
        return conn.execute(
            "SELECT * FROM macro_consensus ORDER BY release_utc, field, pulled_at"
        ).fetchall()
    finally:
        conn.close()


# ---------------------------------------------------------------- parsing


def test_parse_cpi_week_fixture():
    df = parse_calendar(_load("ff_calendar_cpi_week.json"), pulled_at=BEFORE_CPI)
    assert list(df.columns) == COLUMNS
    assert set(df["field"]) == {"cpi_mom", "cpi_yoy", "core_cpi_mom"}
    assert (df["release_type"] == "CPI").all()
    assert (df["consensus_source"] == SOURCE_TAG).all()
    by_field = df.set_index("field")
    assert by_field.loc["cpi_mom", "consensus"] == -0.1  # negative print parses
    assert by_field.loc["cpi_mom", "previous"] == 0.5
    assert by_field.loc["cpi_yoy", "consensus"] == 3.8
    assert by_field.loc["core_cpi_mom", "consensus"] == 0.2
    assert df["actual"].isna().all()  # the feed never carries actuals


def test_parse_empsit_week_fixture_scope_is_exact():
    df = parse_calendar(_load("ff_calendar_empsit_week.json"), pulled_at=BEFORE_EMPSIT)
    # Exactly the three USD Employment Situation fields: the USD "ADP Non-Farm
    # Employment Change" near-name and the JPY/EUR "Unemployment Rate" exact
    # titles must all be excluded.
    assert len(df) == 3
    assert set(df["field"]) == {"nfp_change_k", "unemployment_rate", "ahe_mom"}
    assert (df["release_type"] == "EMPSIT").all()
    by_field = df.set_index("field")
    assert by_field.loc["nfp_change_k", "consensus"] == 114.0  # "114K" -> thousands
    assert by_field.loc["nfp_change_k", "previous"] == 172.0
    assert by_field.loc["unemployment_rate", "consensus"] == 4.3
    assert by_field.loc["ahe_mom", "consensus"] == 0.3


def test_release_utc_converted_from_feed_offset():
    """2026-07-14T08:30:00-04:00 (US-eastern feed time) -> 12:30 naive UTC."""
    df = parse_calendar(_load("ff_calendar_cpi_week.json"), pulled_at=BEFORE_CPI)
    assert set(df["release_utc"]) == {pd.Timestamp("2026-07-14 12:30:00")}
    assert df["release_utc"].dt.tz is None
    assert set(df["pulled_at"]) == {pd.Timestamp(BEFORE_CPI)}


def test_is_realtime_true_before_release_false_after():
    payload = _load("ff_calendar_cpi_week.json")
    before = parse_calendar(payload, pulled_at=BEFORE_CPI)
    after = parse_calendar(payload, pulled_at=AFTER_CPI)
    assert before["is_realtime"].all()
    assert not after["is_realtime"].any()


def test_aware_pulled_at_normalized_to_naive_utc():
    """A tz-aware pulled_at compares correctly against the naive-UTC release."""
    aware = datetime(2026, 7, 14, 13, 0, tzinfo=UTC)  # 30 min after release
    df = parse_calendar(_load("ff_calendar_cpi_week.json"), pulled_at=aware)
    assert set(df["pulled_at"]) == {pd.Timestamp("2026-07-14 13:00:00")}
    assert not df["is_realtime"].any()


def test_week_without_in_scope_events_is_empty_not_an_error():
    payload = [d for d in _load("ff_calendar_cpi_week.json") if d["country"] != "USD"]
    df = parse_calendar(payload, pulled_at=BEFORE_CPI)
    assert df.empty
    assert list(df.columns) == COLUMNS
    assert upsert_consensus(df) == 0  # no DB touched for an empty frame


# ------------------------------------------------------- values and units


def test_parse_value_units():
    assert parse_value("0.2%", "cpi_mom") == 0.2
    assert parse_value("-0.1%", "cpi_mom") == -0.1
    assert parse_value("4.3%", "unemployment_rate") == 4.3
    assert parse_value("185K", "nfp_change_k") == 185.0
    assert parse_value("-33K", "nfp_change_k") == -33.0
    assert parse_value("1.2M", "nfp_change_k") == 1200.0
    assert parse_value("", "cpi_mom") is None  # consensus not yet posted
    assert parse_value(None, "nfp_change_k") is None


@pytest.mark.parametrize(
    ("raw", "field"),
    [
        ("abc", "cpi_mom"),
        ("0.2", "cpi_mom"),  # percent field without % — unit drift
        ("0.2%%", "cpi_mom"),
        ("185K", "cpi_mom"),  # count suffix on a percent field
        ("185", "nfp_change_k"),  # bare count — ambiguous unit
        ("185%", "nfp_change_k"),
        ("-132.8B", "nfp_change_k"),
    ],
)
def test_parse_value_malformed_raises(raw, field):
    with pytest.raises(ValueError):
        parse_value(raw, field)


# ------------------------------------------------------------ fail loudly


def test_empty_payload_raises():
    with pytest.raises(ValueError, match="empty"):
        parse_calendar([], pulled_at=BEFORE_CPI)


def test_missing_key_is_schema_drift_and_raises():
    payload = _load("ff_calendar_cpi_week.json")
    del payload[0]["forecast"]  # drift on ANY item must raise, in scope or not
    with pytest.raises(ValueError, match="schema drift"):
        parse_calendar(payload, pulled_at=BEFORE_CPI)


def test_offsetless_datetime_raises():
    payload = _load("ff_calendar_cpi_week.json")
    for item in payload:
        item["date"] = item["date"][:19]  # strip the -04:00 offset
    with pytest.raises(ValueError, match="offset"):
        parse_calendar(payload, pulled_at=BEFORE_CPI)


def test_malformed_in_scope_value_raises():
    payload = _load("ff_calendar_cpi_week.json")
    for item in payload:
        if item["title"] == "CPI m/m":
            item["forecast"] = "n/a"
    with pytest.raises(ValueError, match="cpi_mom"):
        parse_calendar(payload, pulled_at=BEFORE_CPI)


# ------------------------------------------------------------------ upsert


def test_upsert_idempotent_same_pull_counts_once(tmp_db):
    df = parse_calendar(_load("ff_calendar_cpi_week.json"), pulled_at=BEFORE_CPI)
    assert upsert_consensus(df) == 3
    assert upsert_consensus(df) == 0  # identical pull is a no-op
    rows = _table_rows(tmp_db)
    assert len(rows) == 3


def test_upsert_later_pull_appends_new_observations(tmp_db):
    payload = _load("ff_calendar_cpi_week.json")
    pre = parse_calendar(payload, pulled_at=BEFORE_CPI)
    post = parse_calendar(payload, pulled_at=AFTER_CPI)
    assert upsert_consensus(pre) == 3
    assert upsert_consensus(post) == 3  # new pulled_at -> new rows, never updates
    rows = _table_rows(tmp_db)
    assert len(rows) == 6
    conn = duckdb.connect(str(tmp_db), read_only=True)
    try:
        realtime = dict(
            conn.execute(
                "SELECT is_realtime, count(*) FROM macro_consensus GROUP BY is_realtime"
            ).fetchall()
        )
    finally:
        conn.close()
    assert realtime == {True: 3, False: 3}


def test_upsert_roundtrip_values_and_null_actual(tmp_db):
    df = parse_calendar(_load("ff_calendar_empsit_week.json"), pulled_at=BEFORE_EMPSIT)
    upsert_consensus(df)
    conn = duckdb.connect(str(tmp_db), read_only=True)
    try:
        row = conn.execute(
            """
            SELECT release_utc, release_type, consensus, previous, actual,
                   consensus_source, is_realtime
            FROM macro_consensus WHERE field = 'nfp_change_k'
            """
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    release_utc, release_type, consensus, previous, actual, source, is_realtime = row
    assert release_utc == datetime(2026, 7, 2, 12, 30)
    assert release_type == "EMPSIT"
    assert consensus == 114.0
    assert previous == 172.0
    assert actual is None
    assert source == SOURCE_TAG
    assert is_realtime is True
