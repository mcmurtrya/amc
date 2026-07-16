"""Single-command model-retraining orchestrator (Phase 6.10 repro entry point).

    uv run python -m metals.train --dry-run     # show the ordered plan + gate decisions
    uv run python -m metals.train --all         # CPU steps in dependency order
    uv run python -m metals.train --with-gpu    # also the GPU Phase 3 embed stage
    uv run python -m metals.train --only phase6_holdout

A thin orchestrator: each step is an existing script/module run in its own
subprocess, so argv stays clean and a failure is caught by return code (no shared
interpreter state). Steps run in dependency order and STOP on the first failure by
default (a dependency chain — a bad causal step poisons the master table);
`--keep-going` overrides.

**The repro is Option C throughout** — tone/theme text features, NO neural
embeddings — which is exactly the configuration Phase 6 validated (embeddings and
regime/sentiment features hurt out-of-sample). So the CUDA embed stage is SKIPPED
by default and only runs under `--with-gpu` on a box with a visible GPU; `--all`
on this CPU box is honest and complete for what Phase 6 actually shipped.

Data prerequisites: a migrated, ingested DuckDB (run `python -m metals.refresh`
first; the Phase 3 steps additionally need the `headlines` corpus, i.e.
`--gdelt`). Preflight checks the numeric core (prices/macro) and refuses to start
against an empty DB.

Dependency order encoded below:
  phase1 -> phase3 (Option C clusters) -> {phase5 causal, subsample, svar}
  -> phase5 cate (needs phase3 clusters) -> phase5 master (needs causal+subsample)
  -> {phase6 holdout (needs phase3 text tables), phase6 scenario}.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

# cate_regimes.py hard-codes this model_version, so the Phase 3 cluster stage must
# produce it for the downstream CATE step to resolve (the most fragile clean-box link).
PHASE3_MODEL_VERSION = "phase3_optC_tone_lag1_2024split"

_PY = sys.executable


@dataclass(frozen=True)
class Step:
    name: str
    argv: list[str]  # run in a subprocess with this exact argv
    gpu: bool  # requires CUDA; skipped unless --with-gpu
    note: str


# Canonical dependency order. Everything is seed-pinned (LGBM seed, DML random_state,
# SVAR seed=42, clustering random_state=42, phase6 SEED=42) so a rerun reproduces.
STEPS: tuple[Step, ...] = (
    Step(
        "phase1_diagnose",
        [_PY, "scripts/phase1_diagnose.py"],
        False,
        "Phase 1 LightGBM vol diagnosis (5 feature sets x 4 metals) -> harness + md",
    ),
    Step(
        "phase3_embed",
        [
            _PY,
            "scripts/phase3_pipeline.py",
            "--only",
            "embed",
            "--model-version",
            PHASE3_MODEL_VERSION,
        ],
        True,  # CUDA-only; NOT consumed by the Option C path below — exploratory
        "Phase 3 neural embedding stage (CUDA). Not used by Option C; --with-gpu only.",
    ),
    Step(
        "phase3_clusters",
        [
            _PY,
            "scripts/phase3_pipeline.py",
            "--resume-from",
            "aggregate",
            "--no-text-embeddings",
            "--model-version",
            PHASE3_MODEL_VERSION,
        ],
        False,
        "Phase 3 Option C aggregate->cluster (CPU) -> cluster_assignments/centroids",
    ),
    Step(
        "phase5_causal",
        [_PY, "-m", "metals.models.causal"],
        False,
        "Phase 5 DoubleML scenario ATEs -> data/processed/double_ml_ates.parquet",
    ),
    Step(
        "phase5_subsample",
        [_PY, "scripts/phase5_subsample_stability.py"],
        False,
        "Phase 5 subsample stability -> results/phase5_subsample_stability.csv",
    ),
    Step(
        "phase5_svar",
        [_PY, "-m", "metals.models.svar"],
        False,
        "Phase 5 sign-restricted SVAR IRFs -> results/phase5_svar_irfs.csv",
    ),
    Step(
        "phase5_cate",
        [_PY, "scripts/phase5_cate_regimes.py"],
        False,
        "Phase 5 CATE by Phase 3 regime (needs phase3_clusters) -> csv",
    ),
    Step(
        "phase5_master",
        [_PY, "scripts/phase5_master_table.py"],
        False,
        "Phase 5 master table (needs causal + subsample) -> scenario_master.parquet",
    ),
    Step(
        "phase6_holdout",
        [_PY, "scripts/phase6_holdout.py"],
        False,
        "Phase 6 hold-out bake-off (needs phase3 text tables) -> holdout_metrics.csv",
    ),
    Step(
        "phase6_scenario",
        [_PY, "scripts/phase6_scenario_holdout.py"],
        False,
        "Phase 6 scenario sign-validation on the hold-out -> csv",
    ),
)
STEP_NAMES = tuple(s.name for s in STEPS)


def cuda_available() -> bool:
    """True if a CUDA device is visible. Import torch lazily; absent torch = no GPU."""
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def preflight() -> list[str]:
    """Return a list of blocking problems (empty = ok to start)."""
    problems: list[str] = []
    try:
        from metals.data.db import connection

        with connection(read_only=True) as conn:
            for table in ("prices", "macro"):
                try:
                    row = conn.execute(f"SELECT count(*) FROM {table}").fetchone()
                    n = row[0] if row else 0
                except Exception:
                    n = 0
                if not n:
                    problems.append(f"{table} is empty — run `python -m metals.refresh` first")
    except Exception as exc:
        problems.append(f"cannot open the DB ({exc}) — run migrations + metals.refresh")
    return problems


def _select(only: set[str] | None, from_step: str | None, with_gpu: bool) -> list[Step]:
    steps = list(STEPS)
    if from_step is not None:
        if from_step not in STEP_NAMES:
            raise ValueError(f"unknown --from step {from_step!r}; known: {', '.join(STEP_NAMES)}")
        steps = steps[STEP_NAMES.index(from_step) :]
    if only is not None:
        unknown = sorted(only - set(STEP_NAMES))
        if unknown:
            raise ValueError(
                f"unknown step(s): {', '.join(unknown)}; known: {', '.join(STEP_NAMES)}"
            )
        steps = [s for s in steps if s.name in only]
    if not with_gpu:
        steps = [s for s in steps if not s.gpu]
    return steps


Runner = Callable[[list[str]], int]


def _default_runner(argv: list[str]) -> int:
    return subprocess.run(argv, check=False).returncode


def train(
    *,
    only: set[str] | None = None,
    from_step: str | None = None,
    with_gpu: bool = False,
    keep_going: bool = False,
    dry_run: bool = False,
    runner: Runner = _default_runner,
    skip_preflight: bool = False,
) -> dict[str, str]:
    """Run the selected training steps in order. Returns {step: status}.

    status is "ok" | "failed" | "planned" | "skipped". Stops on first failure
    unless keep_going.
    """
    steps = _select(only, from_step, with_gpu)
    if with_gpu and any(s.gpu for s in steps) and not (dry_run or cuda_available()):
        raise RuntimeError(
            "--with-gpu was requested but no CUDA device is visible; the embed stage "
            "would fall back to a multi-day CPU encode. Run on a GPU box or drop --with-gpu."
        )
    if not dry_run and not skip_preflight:
        problems = preflight()
        if problems:
            raise RuntimeError("preflight failed:\n  - " + "\n  - ".join(problems))

    results: dict[str, str] = {}
    for step in steps:
        gpu = " [GPU]" if step.gpu else ""
        if dry_run:
            print(f"  {step.name:18}{gpu}  {' '.join(step.argv)}\n      {step.note}")
            results[step.name] = "planned"
            continue
        print(f"-> {step.name}{gpu}: {' '.join(step.argv)}", flush=True)
        code = runner(step.argv)
        if code == 0:
            results[step.name] = "ok"
        else:
            results[step.name] = "failed"
            print(f"   step {step.name} exited {code}", file=sys.stderr)
            if not keep_going:
                for later in steps[steps.index(step) + 1 :]:
                    results[later.name] = "skipped"
                break
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrain the Phase 1/3/5/6 models in order.")
    parser.add_argument("--all", action="store_true", help="Run all CPU steps (the default).")
    parser.add_argument("--with-gpu", action="store_true", help="Also run the CUDA embed stage.")
    parser.add_argument("--only", default=None, help="Comma-separated step names to run.")
    parser.add_argument(
        "--from", dest="from_step", default=None, help="Resume from this step onward."
    )
    parser.add_argument("--keep-going", action="store_true", help="Continue past a failed step.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the ordered plan; run nothing."
    )
    args = parser.parse_args(argv)

    only = {n.strip() for n in args.only.split(",") if n.strip()} if args.only else None
    try:
        results = train(
            only=only,
            from_step=args.from_step,
            with_gpu=args.with_gpu,
            keep_going=args.keep_going,
            dry_run=args.dry_run,
        )
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    if not args.dry_run:
        ok = sum(1 for s in results.values() if s == "ok")
        failed = [n for n, s in results.items() if s == "failed"]
        print(
            f"\n{ok}/{len(results)} step(s) ok"
            + (f"; FAILED: {', '.join(failed)}" if failed else "")
        )
        return 1 if failed else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
