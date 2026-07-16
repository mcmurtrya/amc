"""Scheduled entry point for the Phase 7.1 data collectors.

One cron / Task-Scheduler job runs this script; it lazy-imports each
registered collector module (contract: the module exposes
``refresh(**kwargs) -> dict`` with at least ``"rows_written"``), isolates
failures per collector so one broken scraper never blocks the rest, prints
an aligned summary table, and records per-collector state in
``data/raw/collector_state.json`` for fail-loud alerting.

Run as:
    uv run python scripts/run_collectors.py                     # run everything
    uv run python scripts/run_collectors.py --only trends,jm_pgm
    uv run python scripts/run_collectors.py --skip consensus
    uv run python scripts/run_collectors.py --dry-run           # list, run nothing
    uv run python scripts/run_collectors.py --check-gaps        # read-only staleness audit

Exit codes:
    0  everything succeeded (or nothing was stale in --check-gaps)
    1  at least one collector failed this run
    2  --check-gaps found at least one stale / empty / missing table
       (also argparse's own exit code for bad CLI arguments)

``--check-gaps`` opens the DuckDB store read-only and compares each
registry table's newest timestamp against the collector's expected cadence
plus a one-day grace window — the "missed day noticed the week it happens"
alarm from the acquisition program. Because some collectors (consensus)
legitimately write zero rows in weeks without a scheduled event, the audit
also consults ``last_success_utc`` in the state file: a collector is stale
only when BOTH signals are older than the allowance. Collector modules are
NEVER imported at module import time; only a plain run touches them, via
importlib.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))  # importable without an installed package

STATE_FILE = REPO_ROOT / "data" / "raw" / "collector_state.json"
GRACE_DAYS = 1  # allowance on top of each collector's cadence before it counts as stale
ERROR_HEAD_CHARS = 60  # error truncation width in the summary table


@dataclass(frozen=True)
class CollectorSpec:
    """One registry entry: how to run a collector and how to audit its table."""

    name: str
    module: str  # dotted path, imported lazily via importlib
    refresh_kwargs: dict[str, Any]
    cadence_days: int  # expected days between successful pulls
    table: str  # DuckDB table the collector writes
    timestamp_col: str  # column whose max() dates the newest pull


REGISTRY: tuple[CollectorSpec, ...] = (
    CollectorSpec(
        name="coin_premiums",
        module="metals.data.coin_premiums",
        refresh_kwargs={},
        cadence_days=1,
        table="coin_premiums",
        timestamp_col="pulled_at",
    ),
    # trends is deliberately absent (2026-07-16). Unlike cme_daily (barred at the
    # source), Google LICENSES Trends data but only via a sanctioned MANUAL CSV
    # export; the old scraper defeated a non-browser gate, which the ToS bar. So
    # metals.data.trends is now an operator-run CSV importer (like amc_ledger) whose
    # refresh() takes a file path and cannot be scheduled argless. It is NOT
    # backfillable — Trends rescales per request — so run it weekly by hand; every
    # skipped week is a lost snapshot. See journal.md 2026-07-16.
    # cme_daily is deliberately absent (2026-07-15). Its website source is barred by
    # CME's Data Terms of Use for AMC's use, so the scheduler must not keep attempting
    # it nightly. The series is backfillable via Databento, so nothing accrues in the
    # meantime and it needs no cadence entry until it is retargeted. See journal.md.
    CollectorSpec(
        name="jm_pgm",
        module="metals.data.jm_pgm",
        refresh_kwargs={},
        cadence_days=7,
        table="pgm_prices",
        timestamp_col="pulled_at",
    ),
    CollectorSpec(
        name="consensus",
        module="metals.data.consensus",
        refresh_kwargs={},
        cadence_days=1,
        table="macro_consensus",
        timestamp_col="pulled_at",
    ),
)


@dataclass
class RunResult:
    """Outcome of one collector invocation."""

    name: str
    ok: bool
    rows: int | None
    seconds: float
    error: str | None  # first line of the exception, None on success


@dataclass
class GapResult:
    """Outcome of one --check-gaps table audit."""

    name: str
    stale: bool
    detail: str


def _error_head(exc: BaseException, limit: int = 200) -> str:
    """First line of an exception, prefixed with its type, truncated."""
    msg = str(exc).splitlines()[0].strip() if str(exc) else ""
    head = f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__
    return head[:limit]


def run_one(spec: CollectorSpec) -> RunResult:
    """Import and run one collector, converting ANY exception into a failed result.

    Catch-and-continue is forbidden inside collectors — this is the single
    place where isolation happens, so a broken scraper cannot take down the
    nightly run for every other non-backfillable series.
    """
    t0 = time.monotonic()
    try:
        module = importlib.import_module(spec.module)
        refresh = getattr(module, "refresh", None)
        if not callable(refresh):
            raise AttributeError(f"{spec.module} does not expose a callable refresh()")
        summary = refresh(**spec.refresh_kwargs)
        if not isinstance(summary, dict) or "rows_written" not in summary:
            raise TypeError(
                f"{spec.module}.refresh() must return a dict containing 'rows_written', "
                f"got {type(summary).__name__}"
            )
        rows = int(summary["rows_written"])
    except Exception as exc:  # per-collector isolation: the runner owns failure handling
        return RunResult(spec.name, False, None, time.monotonic() - t0, _error_head(exc))
    return RunResult(spec.name, True, rows, time.monotonic() - t0, None)


def load_state(path: Path) -> dict[str, dict[str, Any]]:
    """Read the collector state file; a corrupt file is rebuilt (with a warning)."""
    if not path.exists():
        return {}
    try:
        state = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        print(f"WARNING: state file {path} is corrupt ({exc}); rebuilding from scratch.")
        return {}
    if not isinstance(state, dict):
        print(f"WARNING: state file {path} is not a JSON object; rebuilding from scratch.")
        return {}
    return state


def update_state(state: dict[str, dict[str, Any]], result: RunResult, now_utc_iso: str) -> None:
    """Fold one run result into the state mapping (in place).

    Success overwrites last_success_utc/last_rows and clears last_error;
    failure records last_error but preserves the last known success so the
    alerting wrapper can see how long a collector has been broken.
    """
    entry = dict(state.get(result.name, {}))
    if result.ok:
        entry["last_success_utc"] = now_utc_iso
        entry["last_rows"] = result.rows
        entry["last_error"] = None
    else:
        entry.setdefault("last_success_utc", None)
        entry.setdefault("last_rows", None)
        entry["last_error"] = result.error
    state[result.name] = entry


def write_state(path: Path, state: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def print_summary(results: Sequence[RunResult]) -> None:
    """Aligned per-collector table: name, status, rows, seconds, error head."""
    if not results:
        return
    name_w = max(len("collector"), max(len(r.name) for r in results))
    print(f"{'collector':<{name_w}}  {'status':<6}  {'rows':>8}  {'seconds':>8}  error")
    for r in results:
        status = "ok" if r.ok else "FAILED"
        rows = "-" if r.rows is None else str(r.rows)
        err = r.error or "-"
        if len(err) > ERROR_HEAD_CHARS:
            err = err[: ERROR_HEAD_CHARS - 3] + "..."
        print(f"{r.name:<{name_w}}  {status:<6}  {rows:>8}  {r.seconds:>8.2f}  {err}")


def _state_age_days(entry: dict[str, Any] | None, now: datetime) -> float | None:
    """Days since the collector's last recorded successful run; None if unknown.

    ``last_success_utc`` is written by run mode as an ISO-8601 UTC string; an
    absent, null, or unparseable value counts as "never succeeded".
    """
    if not isinstance(entry, dict):
        return None
    raw = entry.get("last_success_utc")
    if not isinstance(raw, str):
        return None
    try:
        ts = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if ts.tzinfo is not None:
        ts = ts.astimezone(UTC).replace(tzinfo=None)
    return (now - ts).total_seconds() / 86400.0


def check_gaps(
    specs: Sequence[CollectorSpec],
    now: datetime | None = None,
    state_path: Path | None = None,
) -> list[GapResult]:
    """Audit each collector read-only: has it produced anything within cadence?

    Two signals are consulted, and a collector is stale only when BOTH are
    older than ``cadence_days + GRACE_DAYS``:

    * the table's max(timestamp_col) — proof that rows actually landed, and
    * ``last_success_utc`` from the state file — proof the collector ran
      cleanly, which matters for collectors (consensus) whose ``refresh()``
      legitimately writes zero rows in any week without a CPI/EMPSIT event.

    A collector that runs successfully but writes nothing stays fresh via
    ``last_success_utc``; a dead collector trips both. A missing database
    file or a missing table still counts as stale unconditionally — nothing
    was ever collected, or migrations were never applied — exactly the
    silent gap this mode exists to catch. Timestamps in the DB are naive UTC.
    """
    from metals.data.db import connection, db_path

    if now is None:
        now = datetime.now(UTC).replace(tzinfo=None)
    state = load_state(state_path if state_path is not None else STATE_FILE)
    if not db_path().exists():
        return [
            GapResult(s.name, True, f"database file missing: {db_path()} — nothing ever collected")
            for s in specs
        ]
    results: list[GapResult] = []
    with connection(read_only=True) as conn:
        for spec in specs:
            allowed_days = spec.cadence_days + GRACE_DAYS
            success_age = _state_age_days(state.get(spec.name), now)
            state_fresh = success_age is not None and success_age <= allowed_days
            state_note = (
                "no successful run recorded in state"
                if success_age is None
                else f"last successful run {success_age:.1f}d ago"
            )
            (n_tables,) = conn.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
                [spec.table],
            ).fetchone()
            if not n_tables:
                results.append(
                    GapResult(
                        spec.name,
                        True,
                        f"table '{spec.table}' does not exist yet — collector has never run "
                        f"(apply migrations, then run it)",
                    )
                )
                continue
            (last,) = conn.execute(f"SELECT max({spec.timestamp_col}) FROM {spec.table}").fetchone()
            if last is None:
                if state_fresh:
                    results.append(
                        GapResult(
                            spec.name,
                            False,
                            f"table '{spec.table}' has no rows yet, but {state_note} — "
                            f"zero-row refreshes are healthy",
                        )
                    )
                else:
                    results.append(
                        GapResult(
                            spec.name,
                            True,
                            f"table '{spec.table}' has no rows yet — collector has never written",
                        )
                    )
                continue
            if isinstance(last, date) and not isinstance(last, datetime):
                last = datetime(last.year, last.month, last.day)
            age_days = (now - last).total_seconds() / 86400.0
            detail = (
                f"last {spec.timestamp_col}={last:%Y-%m-%d %H:%M} UTC, "
                f"age {age_days:.1f}d vs allowed {allowed_days}d "
                f"(cadence {spec.cadence_days}d + {GRACE_DAYS}d grace); {state_note}"
            )
            stale = age_days > allowed_days and not state_fresh
            results.append(GapResult(spec.name, stale, detail))
    return results


def print_gap_report(results: Sequence[GapResult]) -> None:
    if not results:
        return
    name_w = max(len("collector"), max(len(r.name) for r in results))
    print(f"{'collector':<{name_w}}  {'status':<6}  detail")
    for r in results:
        status = "STALE" if r.stale else "fresh"
        print(f"{r.name:<{name_w}}  {status:<6}  {r.detail}")


def _parse_names(raw: str, valid: set[str]) -> set[str]:
    names = {n.strip() for n in raw.split(",") if n.strip()}
    unknown = sorted(names - valid)
    if unknown:
        raise ValueError(
            f"unknown collector(s): {', '.join(unknown)}. Known: {', '.join(sorted(valid))}"
        )
    return names


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Phase 7.1 collectors (or audit their tables for gaps).",
    )
    parser.add_argument("--only", default=None, help="Comma-separated collector names to run.")
    parser.add_argument("--skip", default=None, help="Comma-separated collector names to skip.")
    parser.add_argument("--dry-run", action="store_true", help="List what would run, run nothing.")
    parser.add_argument(
        "--check-gaps",
        action="store_true",
        help="Read-only staleness audit of each collector's table; exit 2 if any is stale.",
    )
    parser.add_argument(
        "--state-file",
        default=None,
        help=f"Path of the JSON state file (default: {STATE_FILE}).",
    )
    args = parser.parse_args(argv)

    valid = {s.name for s in REGISTRY}
    try:
        only = _parse_names(args.only, valid) if args.only else None
        skip = _parse_names(args.skip, valid) if args.skip else set()
    except ValueError as exc:
        parser.error(str(exc))
    specs = [s for s in REGISTRY if (only is None or s.name in only) and s.name not in skip]
    if not specs:
        print("Nothing selected — every collector was filtered out.")
        return 0

    state_path = Path(args.state_file) if args.state_file else STATE_FILE

    if args.check_gaps:
        results = check_gaps(specs, state_path=state_path)
        print_gap_report(results)
        stale = [r for r in results if r.stale]
        if stale:
            print(f"\n{len(stale)} of {len(results)} collector(s) STALE.")
            return 2
        print(f"\nAll {len(results)} collector(s) fresh.")
        return 0

    if args.dry_run:
        name_w = max(len(s.name) for s in specs)
        print(f"Would run {len(specs)} collector(s):")
        for s in specs:
            kwargs = f" kwargs={s.refresh_kwargs}" if s.refresh_kwargs else ""
            print(
                f"  {s.name:<{name_w}}  {s.module}.refresh(){kwargs}  "
                f"cadence={s.cadence_days}d  gap check: {s.table}.{s.timestamp_col}"
            )
        return 0

    state = load_state(state_path)
    run_results: list[RunResult] = []
    for spec in specs:
        print(f"-> {spec.name} ({spec.module}) ...", flush=True)
        result = run_one(spec)
        run_results.append(result)
        update_state(state, result, datetime.now(UTC).isoformat(timespec="seconds"))
        write_state(state_path, state)  # written after every collector: crash-safe

    print()
    print_summary(run_results)
    failed = [r for r in run_results if not r.ok]
    if failed:
        print(f"\n{len(failed)} of {len(run_results)} collector(s) FAILED — see errors above.")
        return 1
    print(f"\nAll {len(run_results)} collector(s) succeeded. State: {state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
