"""Tests for the metals.train orchestrator (Phase 6.10). Step execution is stubbed
so no real training runs; one test exercises the real subprocess runner on a
trivial command to prove the plumbing."""

from __future__ import annotations

import sys

import pytest

from metals import train as tr


def _runner(record: list, fail: set[str] | None = None):
    """A stub runner: train() passes each step's exact argv, so match on identity."""
    fail = fail or set()
    by_argv = {tuple(s.argv): s.name for s in tr.STEPS}

    def run(argv: list[str]) -> int:
        step = by_argv[tuple(argv)]
        record.append(step)
        return 1 if step in fail else 0

    return run


def test_dry_run_plan_is_cpu_only_in_order():
    results = tr.train(dry_run=True)
    assert list(results) == [s.name for s in tr.STEPS if not s.gpu]
    assert "phase3_embed" not in results  # the GPU stage is skipped by default
    assert all(v == "planned" for v in results.values())


def test_with_gpu_dry_run_includes_embed():
    results = tr.train(dry_run=True, with_gpu=True)
    assert "phase3_embed" in results


def test_all_steps_run_in_order(monkeypatch):
    ran: list = []
    results = tr.train(runner=_runner(ran), skip_preflight=True)
    cpu_steps = [s.name for s in tr.STEPS if not s.gpu]
    assert ran == cpu_steps  # exact dependency order, each once
    assert all(v == "ok" for v in results.values())


def test_stops_on_first_failure(monkeypatch):
    ran: list = []
    results = tr.train(runner=_runner(ran, fail={"phase5_causal"}), skip_preflight=True)
    assert results["phase5_causal"] == "failed"
    # everything after the failure is skipped, not run
    assert results["phase5_master"] == "skipped"
    assert "phase5_master" not in ran
    # everything before ran ok
    assert results["phase1_diagnose"] == "ok"


def test_keep_going_continues_past_failure():
    ran: list = []
    results = tr.train(
        runner=_runner(ran, fail={"phase5_causal"}), keep_going=True, skip_preflight=True
    )
    assert results["phase5_causal"] == "failed"
    assert results["phase5_master"] in ("ok", "failed")  # it still attempted
    assert "phase5_master" in ran


def test_only_selects_subset():
    ran: list = []
    tr.train(only={"phase6_holdout", "phase6_scenario"}, runner=_runner(ran), skip_preflight=True)
    assert ran == ["phase6_holdout", "phase6_scenario"]


def test_from_resumes_onward():
    ran: list = []
    tr.train(from_step="phase5_master", runner=_runner(ran), skip_preflight=True)
    assert ran == ["phase5_master", "phase6_holdout", "phase6_scenario"]


def test_unknown_step_rejected():
    with pytest.raises(ValueError, match="unknown step"):
        tr.train(only={"nope"}, runner=_runner([]), skip_preflight=True)


def test_with_gpu_without_cuda_refuses(monkeypatch):
    monkeypatch.setattr(tr, "cuda_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CUDA device"):
        tr.train(with_gpu=True, runner=_runner([]), skip_preflight=True)


def test_default_runner_spawns_a_real_subprocess():
    assert tr._default_runner([sys.executable, "-c", "import sys; sys.exit(0)"]) == 0
    assert tr._default_runner([sys.executable, "-c", "import sys; sys.exit(3)"]) == 3


def test_preflight_returns_a_list():
    # Against the real DB (prices/macro populated) this should be empty; the
    # contract is that it returns a list of blocking problems.
    assert isinstance(tr.preflight(), list)
