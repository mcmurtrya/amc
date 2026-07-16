"""Tests for the ΔDGS2 FOMC surprise proxy (Phase 7.2). Seeds a small events +
macro(DGS2) + fomc_surprises fixture in a temp DB; no network."""

from __future__ import annotations

from datetime import datetime

import pytest

from metals.data import fomc_dgs2
from metals.data.db import connection
from metals.data.migrations import runner


@pytest.fixture()
def seeded_db(monkeypatch, tmp_path):
    monkeypatch.setenv("METALS_DB_PATH", str(tmp_path / "t.duckdb"))
    runner.apply_migrations(verbose=False)
    with connection() as conn:
        # DGS2 daily closes (a Fri->Mon gap sits between 02-02 and 02-05).
        dgs2 = [
            ("2024-01-30", 4.35),
            ("2024-01-31", 4.50),  # FOMC Wed -> prev 01-30 -> +15 bp
            ("2024-02-01", 4.40),
            ("2024-02-02", 4.45),  # Friday
            ("2024-02-05", 4.55),  # FOMC Mon -> prev is Fri 02-02 -> +10 bp
            ("2024-02-06", 4.50),
        ]
        for d, v in dgs2:
            conn.execute("INSERT INTO macro VALUES (?, 'DGS2', ?, 'fred')", [f"{d} 00:00:00", v])
        # FOMC events: two on DGS2 trading days, one on a Saturday (no DGS2 -> excluded).
        for d in ("2024-01-31", "2024-02-05", "2024-02-10"):
            conn.execute(
                "INSERT INTO events VALUES (?, 'FOMC', ?, ?, 'test')",
                [f"{d} 00:00:00", f"fomc_{d}", '{"is_scheduled": true}'],
            )
        # Bauer-Swanson surprises for the overlap validation.
        conn.execute(
            "INSERT INTO fomc_surprises VALUES (?, false, 0, 0, 0, ?, ?, 'test')",
            ["2024-01-31 00:00:00", 0.10, 0.08],  # same sign as +15 bp
        )
        conn.execute(
            "INSERT INTO fomc_surprises VALUES (?, false, 0, 0, 0, ?, ?, 'test')",
            ["2024-02-05 00:00:00", -0.05, -0.03],  # opposite sign to +10 bp
        )
    return tmp_path


def _rows() -> dict:
    with connection(read_only=True) as conn:
        return {
            str(r[0].date()): r
            for r in conn.execute(
                "SELECT timestamp_utc, delta_dgs2_bp, prev_trading_day, is_realtime "
                "FROM fomc_yield_surprises ORDER BY timestamp_utc"
            ).fetchall()
        }


def test_delta_and_prior_trading_day_alignment(seeded_db):
    summary = fomc_dgs2.refresh()
    assert summary["rows_written"] == 2  # the Saturday FOMC has no same-day DGS2
    rows = _rows()
    # Wednesday meeting: 4.50 - 4.35 = +15 bp, prior day is 01-30.
    assert rows["2024-01-31"][1] == pytest.approx(15.0)
    assert str(rows["2024-01-31"][2]) == "2024-01-30"
    # Monday meeting: prior DGS2 trading day is the FRIDAY (02-02), not calendar -1.
    assert rows["2024-02-05"][1] == pytest.approx(10.0)
    assert str(rows["2024-02-05"][2]) == "2024-02-02"  # weekend skipped correctly


def test_sign_convention_rise_is_hawkish(seeded_db):
    fomc_dgs2.refresh()
    rows = _rows()
    assert rows["2024-01-31"][1] > 0  # yield rose -> hawkish -> positive


def test_holiday_fomc_day_excluded_not_nulled(seeded_db):
    summary = fomc_dgs2.refresh()
    assert "2024-02-10" in summary["excluded_fomc_days"]  # Saturday, no Treasury trading
    assert "2024-02-10" not in _rows()


def test_backfill_is_not_realtime(seeded_db):
    # Refresh "today" (2026) is far past the 2024 meetings -> all retro.
    fomc_dgs2.refresh(pulled_at=datetime(2026, 7, 16))
    assert all(not r[3] for r in _rows().values())


def test_realtime_row_preserved_on_backfill(seeded_db):
    # A genuine meeting-evening capture with a pinned (as-known) value + realtime flag.
    with connection() as conn:
        conn.execute(
            "INSERT INTO fomc_yield_surprises VALUES "
            "(?, false, 9.99, 9.00, DATE '2024-01-30', 99.0, 'evening_capture', ?, true)",
            ["2024-01-31 00:00:00", "2024-01-31 19:00:00"],
        )
    fomc_dgs2.refresh(pulled_at=datetime(2026, 7, 16))  # backfill must not clobber it
    with connection(read_only=True) as conn:
        row = conn.execute(
            "SELECT delta_dgs2_bp, dgs2_release, source, is_realtime FROM fomc_yield_surprises "
            "WHERE timestamp_utc = '2024-01-31 00:00:00'"
        ).fetchone()
    assert row == (99.0, 9.99, "evening_capture", True)  # pinned realtime values survive


def test_idempotent_rerefresh(seeded_db):
    fomc_dgs2.refresh()
    first = _rows()
    fomc_dgs2.refresh()  # re-run: same keys, values updated in place, no duplicates
    assert _rows().keys() == first.keys()
    with connection(read_only=True) as conn:
        assert conn.execute("SELECT count(*) FROM fomc_yield_surprises").fetchone()[0] == 2


def test_validate_against_mps(seeded_db):
    fomc_dgs2.refresh()
    with connection(read_only=True) as conn:
        v = fomc_dgs2.validate_against_mps(conn)
    assert v["n_overlap"] == 2
    # +15 agrees with mps_orth +0.08, +10 disagrees with -0.03 -> 50% sign agreement.
    assert v["sign_agree_mps_orth"] == pytest.approx(0.5)
    assert -1.0 <= v["corr_mps"] <= 1.0
