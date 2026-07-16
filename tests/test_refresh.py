"""Tests for the metals.refresh orchestrator (Phase 6.10). No network — the core
sources' refresh() calls are stubbed via a fake importlib."""

from __future__ import annotations

import types

import pytest

from metals import refresh


def _fake_import(record: dict, failing: set[str] | None = None):
    """Return an import_module stand-in whose modules record/raise on refresh()."""
    failing = failing or set()

    def _import(dotted: str):
        name = dotted.rsplit(".", 1)[-1]
        mod = types.SimpleNamespace()

        def _refresh(*args, **kwargs):
            record[name] = {"args": args, "kwargs": kwargs}
            if name in failing:
                raise RuntimeError(f"{name} upstream is down")
            return {"rows_written": 1}

        mod.refresh = _refresh
        return mod

    return _import


def test_dry_run_lists_seven_core_sources_without_gdelt():
    results = refresh.refresh_all(dry_run=True)
    assert set(results) == set(refresh.CORE_NAMES)
    assert "gdelt" not in results
    assert all(status == "planned" for status, _ in results.values())


def test_with_gdelt_appends_gdelt():
    results = refresh.refresh_all(dry_run=True, with_gdelt=True)
    assert "gdelt" in results
    assert set(results) == set(refresh.CORE_NAMES) | {"gdelt"}


def test_naming_a_barred_collector_points_elsewhere():
    for name in ("coin_premiums", "consensus", "jm_pgm", "trends", "cme_daily", "amc_ledger"):
        with pytest.raises(refresh.RefreshSelectionError, match="Phase 7.1"):
            refresh.refresh_all(only={name}, dry_run=True)


def test_unknown_source_rejected():
    with pytest.raises(refresh.RefreshSelectionError, match="unknown source"):
        refresh.refresh_all(only={"nope"}, dry_run=True)


def test_runs_all_core_sources(monkeypatch):
    called: dict = {}
    monkeypatch.setattr(refresh.importlib, "import_module", _fake_import(called))
    results = refresh.refresh_all()
    assert set(called) == set(refresh.CORE_NAMES)  # each core refresh() was invoked
    assert all(status == "ok" for status, _ in results.values())


def test_failure_is_isolated_not_fatal(monkeypatch):
    called: dict = {}
    monkeypatch.setattr(refresh.importlib, "import_module", _fake_import(called, failing={"cot"}))
    results = refresh.refresh_all()
    assert results["cot"][0] == "error"
    # every other source still ran and succeeded despite cot failing
    assert results["prices"][0] == "ok"
    assert set(called) == set(refresh.CORE_NAMES)


def test_gdelt_without_dates_errors(monkeypatch):
    called: dict = {}
    monkeypatch.setattr(refresh.importlib, "import_module", _fake_import(called))
    results = refresh.refresh_all(only={"gdelt"}, with_gdelt=True)
    assert results["gdelt"][0] == "error"
    assert "requires --start" in results["gdelt"][1]


def test_gdelt_passes_dates_through(monkeypatch):
    called: dict = {}
    monkeypatch.setattr(refresh.importlib, "import_module", _fake_import(called))
    refresh.refresh_all(
        only={"gdelt"}, with_gdelt=True, gdelt_start="2015-01-01", gdelt_end="2015-02-01"
    )
    assert called["gdelt"]["args"] == ("2015-01-01", "2015-02-01")
    assert called["gdelt"]["kwargs"]["chunk_days"] == 7
