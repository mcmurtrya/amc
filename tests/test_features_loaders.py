"""Tests for the Phase 5 read-side loaders (fomc_surprises, events, positioning).

Each test runs against a per-test temporary DuckDB (METALS_DB_PATH) with the
real migrations applied, then inserts a few rows directly.
"""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def tmp_db(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.duckdb")
        monkeypatch.setenv("METALS_DB_PATH", db_path)
        from metals.data.migrations.runner import apply_migrations

        apply_migrations(verbose=False)
        yield db_path


def test_load_fomc_surprises_indexed_and_sorted():
    from metals.data.db import connection
    from metals.features.loaders import load_fomc_surprises

    with connection() as conn:
        conn.execute(
            "INSERT INTO fomc_surprises (timestamp_utc, is_unscheduled, mps, mps_orth, source) "
            "VALUES (?,?,?,?,?)",
            [pd.Timestamp("2015-06-17"), False, -0.5, -0.4, "test"],
        )
        conn.execute(
            "INSERT INTO fomc_surprises (timestamp_utc, is_unscheduled, mps, mps_orth, source) "
            "VALUES (?,?,?,?,?)",
            [pd.Timestamp("2015-03-18"), False, 0.3, 0.25, "test"],
        )

    out = load_fomc_surprises()
    assert out.index.is_monotonic_increasing
    assert list(out.index) == [pd.Timestamp("2015-03-18"), pd.Timestamp("2015-06-17")]
    assert out.loc[pd.Timestamp("2015-06-17"), "mps_orth"] == pytest.approx(-0.4)
    assert "mps_orth" in out.columns


def test_load_fomc_surprises_empty_returns_empty_frame():
    from metals.features.loaders import load_fomc_surprises

    out = load_fomc_surprises()
    assert out.empty


def test_load_events_filters_by_type():
    from metals.data.db import connection
    from metals.features.loaders import load_events

    with connection() as conn:
        conn.execute(
            "INSERT INTO events (timestamp_utc, event_type, event_id, metadata, source) "
            "VALUES (?,?,?,?,?)",
            [pd.Timestamp("2020-01-29"), "FOMC", "fomc_2020-01-29", "{}", "test"],
        )
        conn.execute(
            "INSERT INTO events (timestamp_utc, event_type, event_id, metadata, source) "
            "VALUES (?,?,?,?,?)",
            [pd.Timestamp("2020-02-12"), "CPI", "cpi_2020-02-12", "{}", "test"],
        )

    fomc = load_events("FOMC")
    assert len(fomc) == 1
    assert fomc.iloc[0]["event_type"] == "FOMC"
    assert pd.api.types.is_datetime64_any_dtype(fomc["timestamp_utc"])
    assert len(load_events()) == 2


def test_load_positioning_single_metal_indexed_and_release_date_preserved():
    from metals.data.db import connection
    from metals.features.loaders import load_positioning

    friday = pd.Timestamp("2021-05-07")  # a Friday release date
    with connection() as conn:
        conn.execute(
            "INSERT INTO positioning (timestamp_utc, metal, managed_money_long, "
            "managed_money_short, open_interest, source) VALUES (?,?,?,?,?,?)",
            [friday, "gold", 200_000, 50_000, 500_000, "test"],
        )
        conn.execute(
            "INSERT INTO positioning (timestamp_utc, metal, managed_money_long, "
            "managed_money_short, open_interest, source) VALUES (?,?,?,?,?,?)",
            [friday, "silver", 80_000, 30_000, 150_000, "test"],
        )

    gold = load_positioning("gold")
    assert gold.index.name == "timestamp_utc"
    assert "metal" not in gold.columns
    assert list(gold.index) == [friday]  # Friday release date unchanged
    assert gold.loc[friday, "managed_money_long"] == 200_000

    long = load_positioning()
    assert set(long["metal"]) == {"gold", "silver"}
