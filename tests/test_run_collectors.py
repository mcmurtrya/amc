"""Tests for scripts/run_collectors.py — the Phase 7.1 scheduled entry point.

No real collector module is ever imported here: run-mode tests install fake
modules in ``sys.modules`` (importlib returns them without touching the
filesystem), and --check-gaps tests run against a temp DuckDB built by the
migrations runner.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pytest

from scripts import run_collectors as rc

# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------


def _install_fake_module(monkeypatch, dotted: str, refresh) -> None:
    """Register a fake collector module so importlib resolves it from sys.modules."""
    mod = types.ModuleType(dotted)
    mod.refresh = refresh  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, dotted, mod)


def _spec(name: str, module: str, cadence: int = 1, table: str = "t", **kwargs) -> rc.CollectorSpec:
    return rc.CollectorSpec(
        name=name,
        module=module,
        refresh_kwargs=kwargs,
        cadence_days=cadence,
        table=table,
        timestamp_col="pulled_at",
    )


@pytest.fixture()
def fake_run(monkeypatch, tmp_path):
    """Two-collector registry: alpha succeeds, beta raises. Returns (state_path, calls)."""
    calls: dict[str, dict] = {}

    def alpha_refresh(**kw):
        calls["alpha"] = kw
        return {"rows_written": 7, "detail": "ignored by the runner"}

    def beta_refresh(**kw):
        calls["beta"] = kw
        raise RuntimeError("layout changed: price node missing\nlong traceback detail")

    _install_fake_module(monkeypatch, "fake_mod_alpha", alpha_refresh)
    _install_fake_module(monkeypatch, "fake_mod_beta", beta_refresh)
    monkeypatch.setattr(
        rc,
        "REGISTRY",
        (
            _spec("alpha", "fake_mod_alpha", cadence=1, table="alpha_t", days=3),
            _spec("beta", "fake_mod_beta", cadence=7, table="beta_t"),
        ),
    )
    return tmp_path / "state.json", calls


@pytest.fixture()
def tmp_db(monkeypatch, tmp_path):
    """Fresh migrated DuckDB at a temp path (METALS_DB_PATH is honored by db.py).

    Also points rc.STATE_FILE at a nonexistent temp path so --check-gaps never
    reads the repo's real collector state during tests.
    """
    db_file = tmp_path / "t.duckdb"
    monkeypatch.setenv("METALS_DB_PATH", str(db_file))
    monkeypatch.setattr(rc, "STATE_FILE", tmp_path / "no_state.json")
    from metals.data.migrations.runner import apply_migrations

    apply_migrations(verbose=False)
    return db_file


def _seed(db_file: Path, sql: str, params: list) -> None:
    conn = duckdb.connect(str(db_file))
    try:
        conn.execute(sql, params)
    finally:
        conn.close()


def _utcnow_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _write_state(path: Path, name: str, last_success: datetime) -> None:
    """Write a state file recording one collector's last successful run."""
    entry = {
        "last_success_utc": last_success.isoformat(timespec="seconds"),
        "last_rows": 0,
        "last_error": None,
    }
    path.write_text(json.dumps({name: entry}, indent=2) + "\n")


# ---------------------------------------------------------------------------
# registry contract
# ---------------------------------------------------------------------------


def test_registry_matches_phase7_spec():
    entries = {s.name: s for s in rc.REGISTRY}
    expected = {
        "coin_premiums": ("metals.data.coin_premiums", 1, "coin_premiums"),
        "trends": ("metals.data.trends", 7, "search_interest"),
        "cme_daily": ("metals.data.cme_daily", 1, "cme_daily"),
        "jm_pgm": ("metals.data.jm_pgm", 7, "pgm_prices"),
        "consensus": ("metals.data.consensus", 1, "macro_consensus"),
    }
    assert set(entries) == set(expected)
    for name, (module, cadence, table) in expected.items():
        assert entries[name].module == module
        assert entries[name].cadence_days == cadence
        assert entries[name].table == table
        assert entries[name].timestamp_col == "pulled_at"


# ---------------------------------------------------------------------------
# run mode: isolation, exit codes, summary, state file
# ---------------------------------------------------------------------------


def test_run_isolates_failures_and_exits_1(fake_run, capsys):
    state_path, calls = fake_run
    assert rc.main(["--state-file", str(state_path)]) == 1
    out = capsys.readouterr().out
    # Both ran despite beta raising.
    assert calls["alpha"] == {"days": 3}
    assert calls["beta"] == {}
    # Aligned summary table with name, status, rows, error head.
    assert "collector" in out and "status" in out
    assert "ok" in out and "FAILED" in out
    assert "7" in out
    assert "RuntimeError: layout changed" in out
    assert "1 of 2 collector(s) FAILED" in out


def test_run_all_ok_exits_0(fake_run, monkeypatch, tmp_path, capsys):
    state_path, _ = fake_run
    monkeypatch.setattr(rc, "REGISTRY", (_spec("alpha", "fake_mod_alpha", days=3),))
    assert rc.main(["--state-file", str(state_path)]) == 0
    assert "succeeded" in capsys.readouterr().out


def test_state_file_records_success_and_failure(fake_run):
    state_path, _ = fake_run
    rc.main(["--state-file", str(state_path)])
    state = json.loads(state_path.read_text())
    assert set(state) == {"alpha", "beta"}
    alpha, beta = state["alpha"], state["beta"]
    assert alpha["last_rows"] == 7
    assert alpha["last_error"] is None
    # ISO UTC and recent.
    ts = datetime.fromisoformat(alpha["last_success_utc"])
    assert ts.utcoffset() == timedelta(0)
    assert abs((datetime.now(UTC) - ts).total_seconds()) < 300
    # Failure: no success recorded, error head keeps only the first line.
    assert beta["last_success_utc"] is None
    assert beta["last_rows"] is None
    assert beta["last_error"] == "RuntimeError: layout changed: price node missing"


def test_failure_preserves_previous_success_and_unrelated_entries(fake_run):
    state_path, _ = fake_run
    prior = {
        "beta": {
            "last_success_utc": "2026-07-01T00:00:00+00:00",
            "last_rows": 5,
            "last_error": None,
        },
        "gamma": {"last_success_utc": "2026-06-30T00:00:00+00:00", "last_rows": 1},
    }
    state_path.write_text(json.dumps(prior))
    rc.main(["--state-file", str(state_path)])
    state = json.loads(state_path.read_text())
    # beta failed this run: error is recorded, last success kept.
    assert state["beta"]["last_success_utc"] == "2026-07-01T00:00:00+00:00"
    assert state["beta"]["last_rows"] == 5
    assert "RuntimeError" in state["beta"]["last_error"]
    # Entries for collectors not in this run survive.
    assert state["gamma"] == prior["gamma"]


def test_only_and_skip_filters(fake_run, capsys):
    state_path, calls = fake_run
    assert rc.main(["--only", "alpha", "--state-file", str(state_path)]) == 0
    assert "beta" not in calls

    calls.clear()
    assert rc.main(["--skip", "alpha", "--state-file", str(state_path)]) == 1
    assert "alpha" not in calls
    assert "beta" in calls
    assert "alpha" not in capsys.readouterr().out.splitlines()[-3]


def test_unknown_collector_name_is_a_cli_error(fake_run):
    state_path, _ = fake_run
    with pytest.raises(SystemExit) as excinfo:
        rc.main(["--only", "nope", "--state-file", str(state_path)])
    assert excinfo.value.code == 2


def test_missing_module_is_isolated_not_fatal(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        rc, "REGISTRY", (_spec("ghost", "metals.data.definitely_not_a_module_xyz"),)
    )
    state_path = tmp_path / "state.json"
    assert rc.main(["--state-file", str(state_path)]) == 1
    assert "ModuleNotFoundError" in capsys.readouterr().out
    assert "ModuleNotFoundError" in json.loads(state_path.read_text())["ghost"]["last_error"]


def test_refresh_contract_violations_count_as_failures(monkeypatch, tmp_path):
    _install_fake_module(monkeypatch, "fake_mod_none", lambda **kw: None)
    _install_fake_module(monkeypatch, "fake_mod_norows", lambda **kw: {"rows": 3})
    monkeypatch.setattr(
        rc,
        "REGISTRY",
        (_spec("none", "fake_mod_none"), _spec("norows", "fake_mod_norows")),
    )
    state_path = tmp_path / "state.json"
    assert rc.main(["--state-file", str(state_path)]) == 1
    state = json.loads(state_path.read_text())
    assert "rows_written" in state["none"]["last_error"]
    assert "rows_written" in state["norows"]["last_error"]


def test_dry_run_lists_without_importing_or_writing(fake_run, monkeypatch, capsys):
    state_path, calls = fake_run
    imported: list[str] = []
    real_import = importlib.import_module

    def spy(name, package=None):
        imported.append(name)
        return real_import(name, package)

    monkeypatch.setattr(rc.importlib, "import_module", spy)
    assert rc.main(["--dry-run", "--state-file", str(state_path)]) == 0
    out = capsys.readouterr().out
    assert "fake_mod_alpha" in out and "fake_mod_beta" in out
    assert imported == [] and calls == {}
    assert not state_path.exists()


# ---------------------------------------------------------------------------
# --check-gaps mode
# ---------------------------------------------------------------------------


def test_check_gaps_fresh_stale_empty_and_missing(tmp_db, monkeypatch, capsys):
    now = _utcnow_naive()
    _seed(
        tmp_db,
        """
        INSERT INTO coin_premiums
            (pulled_at, dealer, product_id, metal, fine_troy_oz, source, is_realtime)
        VALUES (?, 'apmex', 'age_1oz', 'gold', 0.9675, 'test', true)
        """,
        [now - timedelta(hours=6)],
    )
    _seed(
        tmp_db,
        """
        INSERT INTO pgm_prices
            (price_date, metal, quote, price_usd_oz, source, pulled_at, is_realtime)
        VALUES ('2026-06-12', 'rhodium', 'ny_am', 5150.0, 'test', ?, false)
        """,
        [now - timedelta(days=30)],
    )
    monkeypatch.setattr(
        rc,
        "REGISTRY",
        (
            _spec("coin_premiums", "unused", cadence=1, table="coin_premiums"),
            _spec("jm_pgm", "unused", cadence=7, table="pgm_prices"),
            _spec("cme_daily", "unused", cadence=1, table="cme_daily"),  # migrated but empty
            _spec("ghost", "unused", cadence=1, table="no_such_table"),
        ),
    )
    assert rc.main(["--check-gaps"]) == 2
    lines = {ln.split()[0]: ln for ln in capsys.readouterr().out.splitlines() if ln.strip()}
    assert "fresh" in lines["coin_premiums"]
    assert "STALE" in lines["jm_pgm"]
    assert "STALE" in lines["cme_daily"] and "no rows" in lines["cme_daily"]
    assert "STALE" in lines["ghost"] and "does not exist" in lines["ghost"]

    # Restricted to the fresh collector, the audit passes.
    assert rc.main(["--check-gaps", "--only", "coin_premiums"]) == 0


def test_check_gaps_grace_day_boundary(tmp_db):
    """cadence + 1 grace day: 7.8d old is fresh at cadence 7; 2d1m old is stale at cadence 1."""
    now = datetime(2026, 7, 12, 12, 0, 0)
    _seed(
        tmp_db,
        """
        INSERT INTO pgm_prices
            (price_date, metal, quote, price_usd_oz, source, pulled_at, is_realtime)
        VALUES ('2026-07-04', 'iridium', 'ny_am', 4800.0, 'test', ?, true)
        """,
        [now - timedelta(days=7, hours=20)],
    )
    _seed(
        tmp_db,
        """
        INSERT INTO coin_premiums
            (pulled_at, dealer, product_id, metal, fine_troy_oz, source, is_realtime)
        VALUES (?, 'jmbullion', 'maple_1oz', 'gold', 1.0, 'test', true)
        """,
        [now - timedelta(days=2, minutes=1)],
    )
    within = rc.check_gaps([_spec("jm_pgm", "unused", cadence=7, table="pgm_prices")], now=now)
    assert not within[0].stale
    over = rc.check_gaps([_spec("coins", "unused", cadence=1, table="coin_premiums")], now=now)
    assert over[0].stale


def test_check_gaps_zero_row_collector_with_fresh_state_is_not_stale(
    tmp_db, tmp_path, monkeypatch, capsys
):
    """consensus writes zero rows in any week without a CPI/EMPSIT event, so a
    stale max(pulled_at) alone must not alarm while last_success_utc is recent."""
    now = _utcnow_naive()
    _seed(
        tmp_db,
        """
        INSERT INTO macro_consensus
            (release_utc, release_type, field, consensus, consensus_source,
             pulled_at, is_realtime)
        VALUES ('2026-06-10 12:30:00', 'CPI', 'cpi_mom', 0.2, 'test', ?, true)
        """,
        [now - timedelta(days=10)],  # far beyond cadence 1d + 1d grace
    )
    state_path = tmp_path / "state.json"
    _write_state(state_path, "consensus", datetime.now(UTC) - timedelta(hours=6))
    monkeypatch.setattr(
        rc, "REGISTRY", (_spec("consensus", "unused", cadence=1, table="macro_consensus"),)
    )
    assert rc.main(["--check-gaps", "--state-file", str(state_path)]) == 0
    out = capsys.readouterr().out
    assert "fresh" in out and "STALE" not in out
    assert "last successful run" in out


def test_check_gaps_dead_collector_trips_when_both_signals_stale(
    tmp_db, tmp_path, monkeypatch, capsys
):
    """A collector whose table AND last_success_utc are both old is genuinely dead."""
    now = _utcnow_naive()
    _seed(
        tmp_db,
        """
        INSERT INTO macro_consensus
            (release_utc, release_type, field, consensus, consensus_source,
             pulled_at, is_realtime)
        VALUES ('2026-06-10 12:30:00', 'CPI', 'cpi_mom', 0.2, 'test', ?, true)
        """,
        [now - timedelta(days=10)],
    )
    state_path = tmp_path / "state.json"
    _write_state(state_path, "consensus", datetime.now(UTC) - timedelta(days=10))
    monkeypatch.setattr(
        rc, "REGISTRY", (_spec("consensus", "unused", cadence=1, table="macro_consensus"),)
    )
    assert rc.main(["--check-gaps", "--state-file", str(state_path)]) == 2
    out = capsys.readouterr().out
    assert "STALE" in out
    assert "1 of 1 collector(s) STALE" in out


def test_check_gaps_empty_table_with_fresh_state_is_not_stale(tmp_db, tmp_path):
    """Migrated-but-empty table is healthy if the collector ran cleanly recently."""
    state_path = tmp_path / "state.json"
    _write_state(state_path, "consensus", datetime.now(UTC))
    results = rc.check_gaps(
        [_spec("consensus", "unused", cadence=1, table="macro_consensus")],
        state_path=state_path,
    )
    assert not results[0].stale
    assert "no rows yet" in results[0].detail


def test_check_gaps_missing_table_stays_stale_even_with_fresh_state(tmp_db, tmp_path):
    """A missing table means migrations were never applied — always an alarm."""
    state_path = tmp_path / "state.json"
    _write_state(state_path, "ghost", datetime.now(UTC))
    results = rc.check_gaps(
        [_spec("ghost", "unused", table="no_such_table")], state_path=state_path
    )
    assert results[0].stale
    assert "does not exist" in results[0].detail


def test_check_gaps_missing_db_file_marks_everything_stale(monkeypatch, tmp_path):
    monkeypatch.setenv("METALS_DB_PATH", str(tmp_path / "never_created.duckdb"))
    monkeypatch.setattr(rc, "STATE_FILE", tmp_path / "no_state.json")
    results = rc.check_gaps([_spec("coin_premiums", "unused")])
    assert all(r.stale for r in results)
    assert "database file missing" in results[0].detail
