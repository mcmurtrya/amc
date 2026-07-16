"""Single-command refresh of the Phases 0-6 research inputs (Phase 6.10 repro entry point).

    uv run python -m metals.refresh              # the 7 core sources
    uv run python -m metals.refresh --gdelt --start 2015-01-01 --end 2026-06-30
    uv run python -m metals.refresh --only prices,fred
    uv run python -m metals.refresh --dry-run

Orchestrates the licence-clean research sources, each of which exposes an
idempotent `refresh()` that upserts into the canonical DuckDB. Per-source failures
are isolated and reported (one dead upstream URL never aborts the batch).

**GDELT is opt-in** (`--gdelt`): it is the only expensive + billed source (a
TB-scale BigQuery scan), the only one needing `GOOGLE_APPLICATION_CREDENTIALS`, and
its `refresh()` requires an explicit date range — so pass `--start`/`--end`.

**The Phase 7.1 AMC collectors are NOT refreshed here.** They are separately
governed (`scripts/run_collectors.py`, plan §7.7) and, as of the 2026-07-16 ToU
audit, mostly barred or manual. Naming one prints where it actually lives instead
of running anything.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import traceback
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    name: str
    module: str  # dotted path, imported lazily
    needs: str  # one-line requirement note for --dry-run and errors


# Order is display-only — the sources write disjoint tables and have no data
# dependency on each other (verified 2026-07-16).
CORE_SOURCES: tuple[Source, ...] = (
    Source("prices", "metals.data.prices", "network (Yahoo Finance); no key; best-effort"),
    Source("fred", "metals.data.fred", "FRED_API_KEY"),
    Source("gpr", "metals.data.gpr", "network (Iacoviello GPR); no key"),
    Source("cot", "metals.data.cot", "network (CFTC); no key"),
    Source("fomc_surprises", "metals.data.fomc_surprises", "network (SF Fed); no key"),
    Source("events", "metals.data.events", "local configs/fomc_calendar.csv"),
    Source("bls_calendar", "metals.data.bls_calendar", "local configs/bls_calendar.csv"),
)
GDELT = Source(
    "gdelt",
    "metals.data.gdelt",
    "GOOGLE_APPLICATION_CREDENTIALS + BigQuery (billed); requires --start/--end",
)
CORE_NAMES = tuple(s.name for s in CORE_SOURCES)

# Phase 7.1 collectors — never auto-refreshed by this orchestrator. Value = where
# the source actually lives now (ToU audit 2026-07-16, journal.md).
PHASE7_COLLECTORS: dict[str, str] = {
    "coin_premiums": "BARRED (APMEX/JM Bullion ToU); pending Greysheet licence — see licensing/",
    "consensus": "BARRED (Fair Economy ToU); pending consent — see licensing/",
    "jm_pgm": "BARRED (Johnson Matthey ToU); pending consent — see licensing/",
    "trends": "manual CSV import: uv run python -m metals.data.trends <multiTimeline.csv>",
    "cme_daily": "retired (CME licensed-not-scraped); retarget to Databento",
    "amc_ledger": "manual, local-only: python -m metals.data.amc_ledger <export.csv> --table ...",
}


class RefreshSelectionError(ValueError):
    """A requested source name is not a refreshable core source."""


def _resolve(only: set[str] | None, skip: set[str], with_gdelt: bool) -> list[Source]:
    """Resolve the source list, rejecting Phase 7.1 collector names with a pointer."""
    requested = (only or set()) | skip
    barred = sorted(requested & set(PHASE7_COLLECTORS))
    if barred:
        lines = "\n".join(f"  - {n}: {PHASE7_COLLECTORS[n]}" for n in barred)
        raise RefreshSelectionError(
            "these are Phase 7.1 AMC collectors, not refreshed by metals.refresh:\n"
            + lines
            + "\nUse scripts/run_collectors.py (paused) or the module directly."
        )
    valid = set(CORE_NAMES) | {"gdelt"}
    unknown = sorted(requested - valid)
    if unknown:
        raise RefreshSelectionError(
            f"unknown source(s): {', '.join(unknown)}. Known: {', '.join(sorted(valid))}"
        )
    pool = [*CORE_SOURCES]
    if with_gdelt or (only and "gdelt" in only):
        pool.append(GDELT)
    return [s for s in pool if (only is None or s.name in only) and s.name not in skip]


def refresh_all(
    *,
    only: set[str] | None = None,
    skip: set[str] | None = None,
    with_gdelt: bool = False,
    gdelt_start: str | None = None,
    gdelt_end: str | None = None,
    gdelt_chunk_days: int = 7,
    dry_run: bool = False,
) -> dict[str, tuple[str, object]]:
    """Refresh the selected core sources. Returns {name: (status, summary_or_error)}.

    status is "ok" | "error" | "planned". Failures are isolated per source.
    """
    sources = _resolve(only, skip or set(), with_gdelt)
    results: dict[str, tuple[str, object]] = {}
    for src in sources:
        if dry_run:
            print(f"  {src.name:16} {src.module}.refresh()   [{src.needs}]")
            results[src.name] = ("planned", src.needs)
            continue
        print(f"-> {src.name} ({src.module}) ...", flush=True)
        try:
            module = importlib.import_module(src.module)
            if src.name == "gdelt":
                if not gdelt_start or not gdelt_end:
                    raise ValueError("gdelt requires --start and --end (a TB-scale billed scan)")
                summary = module.refresh(gdelt_start, gdelt_end, chunk_days=gdelt_chunk_days)
            else:
                summary = module.refresh()
            results[src.name] = ("ok", summary)
            rows = summary.get("rows_written") if isinstance(summary, dict) else None
            print(f"   ok{'' if rows is None else f' — {rows} rows'}")
        except Exception as exc:  # isolate: one bad source must not abort the batch
            results[src.name] = ("error", str(exc))
            print(f"   FAILED: {exc}", file=sys.stderr)
            traceback.print_exc()
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh the Phases 0-6 research inputs.")
    parser.add_argument("--only", default=None, help="Comma-separated source names to run.")
    parser.add_argument("--skip", default=None, help="Comma-separated source names to skip.")
    parser.add_argument(
        "--gdelt", action="store_true", help="Also run GDELT (needs --start/--end)."
    )
    parser.add_argument("--start", default=None, help="GDELT start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="GDELT end date (YYYY-MM-DD).")
    parser.add_argument("--chunk-days", type=int, default=7, help="GDELT week chunk size.")
    parser.add_argument("--dry-run", action="store_true", help="List the plan; run nothing.")
    args = parser.parse_args(argv)

    only = {n.strip() for n in args.only.split(",") if n.strip()} if args.only else None
    skip = {n.strip() for n in args.skip.split(",") if n.strip()} if args.skip else set()
    try:
        results = refresh_all(
            only=only,
            skip=skip,
            with_gdelt=args.gdelt,
            gdelt_start=args.start,
            gdelt_end=args.end,
            gdelt_chunk_days=args.chunk_days,
            dry_run=args.dry_run,
        )
    except RefreshSelectionError as exc:
        parser.error(str(exc))

    failed = [n for n, (status, _) in results.items() if status == "error"]
    if not args.dry_run:
        ok = sum(1 for _, (s, _) in results.items() if s == "ok")
        print(
            f"\n{ok}/{len(results)} source(s) refreshed"
            + (f"; FAILED: {', '.join(failed)}" if failed else "")
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
