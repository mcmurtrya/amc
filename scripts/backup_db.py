"""Backup for the AMC metals DuckDB store.

The store is ~54 GB, but that mass is the static GDELT ``headlines`` corpus
(already mirrored in the pre-collector Windows copy). What is *irreplaceable*
is a few kilobytes of daily collector captures — retail coin premiums, as-pulled
search interest, pre-release macro consensus, CME TradeDate figures that age off
the endpoint in ~5 days, JM realtime forward rows — plus AMC's own ledger. So
this script has two legs:

    --tables   Export the small append-only capture + ledger tables to a
               timestamped Parquet snapshot. Fast, safe (read-only connection),
               meant to run *daily*. This is the leg that protects the data
               that can never be re-pulled.
    --full     Snapshot the entire DuckDB file (checkpoint, then a consistent
               single-file copy taken while holding the writer lock). Heavy;
               meant to run *weekly*.

Destination is ``$AMC_BACKUP_DIR`` (default ``/mnt/c/Users/mcmur/amc-backups``).

AMC's ledger tables (``amc_*``) are LOCAL-ONLY per CLAUDE.md. They are included
only when the destination is on the local machine; ``--exclude-ledger`` forces
them out and is REQUIRED for any off-machine / cloud target.

Run as:
    uv run python scripts/backup_db.py --tables            # daily leg
    uv run python scripts/backup_db.py --full              # weekly leg
    uv run python scripts/backup_db.py --tables --full     # both
    uv run python scripts/backup_db.py --tables --dest /mnt/d/amc --exclude-ledger

Exit codes:
    0  requested leg(s) succeeded
    1  a leg failed (unreachable destination, lock contention, low disk, ...)
    2  bad CLI arguments (argparse)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]

# Non-backfillable / expensive-to-refetch capture tables (small; safe to copy daily).
CAPTURE_TABLES = (
    "coin_premiums",
    "search_interest",
    "macro_consensus",
    "cme_daily",
    "pgm_prices",
    "events",
    "fomc_surprises",
)
# AMC's own books — LOCAL-ONLY; only ever written to a same-machine destination.
LEDGER_TABLES = (
    "amc_scrap_lots",
    "amc_coin_trades",
    "amc_till_daily",
)

DEFAULT_DEST = "/mnt/c/Users/mcmur/amc-backups"
RETAIN_SNAPSHOTS = 14  # daily table snapshots to keep
RETAIN_FULL = 4  # weekly full-file snapshots to keep
RW_OPEN_RETRIES = 6  # collectors hold the writer lock for seconds; retry around them
RW_OPEN_WAIT_S = 10


def db_path() -> Path:
    """Canonical DuckDB path, honoring ``METALS_DB_PATH`` (mirrors data/db.py)."""
    override = os.getenv("METALS_DB_PATH")
    if override:
        return Path(override)
    return REPO_ROOT / "data" / "processed" / "metals.duckdb"


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def existing_tables(con: duckdb.DuckDBPyConnection, wanted: tuple[str, ...]) -> list[str]:
    """Filter ``wanted`` down to tables that actually exist in the store."""
    present = {row[0] for row in con.execute("SELECT table_name FROM duckdb_tables()").fetchall()}
    return [t for t in wanted if t in present]


def prune(directory: Path, keep: int) -> list[str]:
    """Keep the ``keep`` newest entries (names sort by date); remove the rest."""
    if not directory.exists():
        return []
    entries = sorted(p for p in directory.iterdir() if not p.name.startswith("."))
    removed: list[str] = []
    for old in entries[:-keep] if keep > 0 else entries:
        if old.is_dir():
            shutil.rmtree(old)
        else:
            old.unlink()
        removed.append(old.name)
    return removed


def require_dest(dest: Path) -> None:
    """Fail loud if the destination mount is missing (e.g. /mnt/c not mounted)."""
    mount_root = dest
    while not mount_root.exists() and mount_root != mount_root.parent:
        mount_root = mount_root.parent
    if not mount_root.exists():
        raise RuntimeError(f"backup destination root does not exist: {dest}")
    dest.mkdir(parents=True, exist_ok=True)


def backup_tables(dest: Path, include_ledger: bool) -> None:
    """Export the small capture (+ ledger) tables to a timestamped Parquet dir."""
    wanted = CAPTURE_TABLES + (LEDGER_TABLES if include_ledger else ())
    snap_root = dest / "snapshots"
    snap_root.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    final = snap_root / stamp
    tmp = snap_root / f".{stamp}.partial"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    manifest: dict[str, int] = {}
    con = duckdb.connect(str(db_path()), read_only=True)
    try:
        tables = existing_tables(con, wanted)
        for table in tables:
            out = tmp / f"{table}.parquet"
            con.execute(f"COPY (SELECT * FROM {table}) TO '{out}' (FORMAT PARQUET)")
            manifest[table] = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    finally:
        con.close()

    (tmp / "MANIFEST.json").write_text(
        json.dumps(
            {
                "captured_at_utc": datetime.now(UTC).isoformat(),
                "source_db": str(db_path()),
                "ledger_included": include_ledger,
                "rows": manifest,
            },
            indent=2,
        )
    )
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)

    total = sum(manifest.values())
    print(
        f"[tables] {len(manifest)} tables, {total:,} rows -> {final}"
        f"  (ledger {'included' if include_ledger else 'EXCLUDED'})"
    )
    removed = prune(snap_root, RETAIN_SNAPSHOTS)
    if removed:
        print(f"[tables] pruned {len(removed)} old snapshot(s): {', '.join(removed)}")


def _connect_rw_with_retry(path: Path) -> duckdb.DuckDBPyConnection:
    """Acquire the single writer lock, retrying around brief collector runs."""
    last: Exception | None = None
    for attempt in range(1, RW_OPEN_RETRIES + 1):
        try:
            return duckdb.connect(str(path), read_only=False)
        except duckdb.IOException as exc:  # held by a collector; wait and retry
            last = exc
            if attempt < RW_OPEN_RETRIES:
                print(f"[full] DB locked (attempt {attempt}), retrying in {RW_OPEN_WAIT_S}s...")
                time.sleep(RW_OPEN_WAIT_S)
    raise RuntimeError(f"could not acquire DB writer lock after {RW_OPEN_RETRIES} tries: {last}")


def backup_full(dest: Path) -> None:
    """Consistent snapshot of the whole DuckDB file.

    Checkpoint to fold the WAL into the main file, then copy that file while the
    writer lock is still held (no collector can mutate it mid-copy). The temp
    lands on the destination filesystem so the final ``os.replace`` is atomic.
    """
    src = db_path()
    if not src.exists():
        raise RuntimeError(f"source DB not found: {src}")
    full_root = dest / "full"
    full_root.mkdir(parents=True, exist_ok=True)

    src_size = src.stat().st_size
    free = shutil.disk_usage(full_root).free
    if free < int(src_size * 1.1):
        raise RuntimeError(
            f"insufficient space at {full_root}: need ~{src_size / 1e9:.1f} GB, "
            f"have {free / 1e9:.1f} GB free"
        )

    stamp = utc_stamp()
    final = full_root / f"metals-{stamp}.duckdb"
    tmp = full_root / f".metals-{stamp}.partial"

    con = _connect_rw_with_retry(src)
    try:
        con.execute("CHECKPOINT")  # committed data now fully in the main file
        t0 = time.monotonic()
        shutil.copy2(src, tmp)  # lock held: no writer can intervene
        elapsed = time.monotonic() - t0
    finally:
        con.close()
    os.replace(tmp, final)

    print(
        f"[full] {src_size / 1e9:.1f} GB -> {final}  "
        f"({elapsed:.0f}s, {src_size / 1e9 / max(elapsed, 1):.2f} GB/s)"
    )
    removed = prune(full_root, RETAIN_FULL)
    if removed:
        print(f"[full] pruned {len(removed)} old snapshot(s): {', '.join(removed)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tables", action="store_true", help="Daily capture/ledger Parquet leg.")
    parser.add_argument("--full", action="store_true", help="Weekly full-DB snapshot leg.")
    parser.add_argument(
        "--dest", default=None, help=f"Destination dir (default ${'AMC_BACKUP_DIR'})."
    )
    parser.add_argument(
        "--exclude-ledger",
        action="store_true",
        help="Never write amc_* ledger tables (REQUIRED for any off-machine target).",
    )
    args = parser.parse_args(argv)

    if not (args.tables or args.full):
        parser.error("choose at least one leg: --tables and/or --full")

    dest = Path(args.dest or os.getenv("AMC_BACKUP_DIR") or DEFAULT_DEST)
    include_ledger = not args.exclude_ledger

    try:
        require_dest(dest)
        if args.tables:
            backup_tables(dest, include_ledger)
        if args.full:
            backup_full(dest)
    except Exception as exc:  # fail loud, non-zero for the systemd alert
        print(f"backup FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
