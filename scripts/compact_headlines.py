"""Reclaim space in metals.duckdb by rebuilding a compacted copy.

Why this exists
---------------
Migration 005 drops the redundant ``headline`` column (a byte-for-byte copy of
``article_url``) from the schema, but DuckDB does **not** shrink the data file in
place when a column is dropped -- the dead bytes linger in the existing file.
This script rebuilds a fresh database with the current (slim) schema and streams
every table's data into it, producing a densely-packed file with the dropped
column's storage actually reclaimed.

Safety
------
* The source database is opened **read-only** and never mutated.
* By default the compacted file is written to ``<db>.compact`` next to the
  original and the script prints swap instructions -- nothing is overwritten.
* ``--replace`` atomically moves the compacted file over the original after
  verifying that every source table's row count is preserved, keeping a
  timestamped ``.bak`` of the original unless ``--no-backup`` is given.

Run locally (the OneDrive-synced DB is too slow to rebuild from a sandbox):

    uv run python scripts/compact_headlines.py            # build <db>.compact
    uv run python scripts/compact_headlines.py --replace  # build + swap in place
    uv run python scripts/compact_headlines.py --db path/to/other.duckdb
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))  # so we can import the harness schema
MIGRATIONS_DIR = REPO_ROOT / "src" / "metals" / "data" / "migrations"
DEFAULT_DB = REPO_ROOT / "data" / "processed" / "metals.duckdb"
TRACKING_TABLE = "_schema_migrations"


def _human(nbytes: int) -> str:
    val = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if val < 1024 or unit == "TB":
            return f"{val:.2f} {unit}"
        val /= 1024
    return f"{val:.2f} TB"


def build_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Recreate the canonical schema: migrations + harness-created tables.

    The migration files cover prices/macro/events/positioning/headlines/
    fomc_surprises/run_feature_importances. The eval harness creates ``runs``
    and ``run_predictions`` lazily (with their primary keys), so we invoke its
    schema initializer too -- otherwise those tables would be missing from the
    rebuilt database.
    """
    con.execute(
        f"CREATE TABLE IF NOT EXISTS {TRACKING_TABLE} ("
        "  migration_id VARCHAR PRIMARY KEY,"
        "  applied_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    for sql_path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        con.execute(sql_path.read_text())
        con.execute(
            f"INSERT INTO {TRACKING_TABLE}(migration_id) VALUES (?)",
            [sql_path.stem],
        )
    from metals.eval.harness import _ensure_schema  # noqa: WPS433

    _ensure_schema(con)


def _tables_in(con: duckdb.DuckDBPyConnection, database: str) -> list[str]:
    rows = con.execute(
        "SELECT table_name FROM duckdb_tables() "
        "WHERE database_name = ? AND schema_name = 'main' "
        "ORDER BY table_name",
        [database],
    ).fetchall()
    return [r[0] for r in rows]


def _columns(con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    return [r[1] for r in con.execute(f'PRAGMA table_info("{table}")').fetchall()]


def _src_columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    rows = con.execute(
        "SELECT column_name FROM duckdb_columns() "
        "WHERE database_name = 'src' AND schema_name = 'main' AND table_name = ?",
        [table],
    ).fetchall()
    return {r[0] for r in rows}


def compact(source: Path, out: Path, *, force: bool) -> dict:
    if source.resolve() == out.resolve():
        raise SystemExit("Source and output paths must differ.")
    if not source.exists():
        raise SystemExit(f"Source database not found: {source}")
    if out.exists():
        if not force:
            raise SystemExit(f"Output already exists: {out} (use --force to overwrite)")
        out.unlink()

    out.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(out))
    con.execute("PRAGMA threads=4")
    build_schema(con)
    con.execute(f"ATTACH '{source}' AS src (READ_ONLY)")

    target_db = con.execute("SELECT current_database()").fetchone()[0]
    target_tables = set(_tables_in(con, target_db))

    # Iterate SOURCE tables so nothing in the source is ever silently dropped.
    summary: dict[str, dict] = {}
    for t in _tables_in(con, "src"):
        if t == TRACKING_TABLE:
            continue  # target already has its own fresh migration records
        if t not in target_tables:
            # Not defined by migrations or the harness; clone structure+data
            # verbatim (note: constraints/indexes are not reproduced).
            print(f"  ! {t}: not in canonical schema, cloning verbatim")
            con.execute(f'CREATE TABLE "{t}" AS SELECT * FROM src."{t}"')
            target_tables.add(t)
        else:
            cols = [c for c in _columns(con, t) if c in _src_columns(con, t)]
            collist = ", ".join(f'"{c}"' for c in cols)
            con.execute(f'INSERT INTO "{t}" ({collist}) SELECT {collist} FROM src."{t}"')
        n_out = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        n_src = con.execute(f'SELECT COUNT(*) FROM src."{t}"').fetchone()[0]
        ok = n_out == n_src
        summary[t] = {"src": n_src, "out": n_out, "ok": ok}
        print(f"  + {t:<24} {n_out:>12,} rows  [{'OK' if ok else 'MISMATCH'}]")

    con.execute("CHECKPOINT")
    con.execute("DETACH src")
    con.close()

    if not all(v["ok"] for v in summary.values()):
        raise SystemExit(
            f"Row-count mismatch detected; refusing to proceed. Inspect {out} and re-run."
        )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--db", type=Path, default=DEFAULT_DB, help=f"Source DuckDB (default: {DEFAULT_DB})."
    )
    ap.add_argument("--out", type=Path, default=None, help="Output path (default: <db>.compact).")
    ap.add_argument(
        "--replace",
        action="store_true",
        help="Swap the compacted file over the original after verifying.",
    )
    ap.add_argument(
        "--no-backup",
        action="store_true",
        help="With --replace, do not keep a .bak of the original.",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite an existing output file.")
    args = ap.parse_args()

    source = args.db
    out = args.out or source.with_suffix(source.suffix + ".compact")

    size_before = source.stat().st_size if source.exists() else 0
    print(f"Compacting {source}")
    print(f"  -> {out}")
    print(f"  source size: {_human(size_before)}\n")

    compact(source, out, force=args.force)

    size_after = out.stat().st_size
    saved = size_before - size_after
    pct = (saved / size_before * 100) if size_before else 0.0
    print(f"\n  compacted size: {_human(size_after)}")
    print(f"  reclaimed:      {_human(saved)}  ({pct:.1f}%)")

    if args.replace:
        if not args.no_backup:
            bak = source.with_suffix(source.suffix + f".bak-{time.strftime('%Y%m%d_%H%M%S')}")
            os.replace(source, bak)
            print(f"\n  backed up original -> {bak}")
        else:
            source.unlink()
        os.replace(out, source)
        print(f"  swapped compacted file into place: {source}")
        print("  Done. Delete the .bak once you've confirmed everything works.")
    else:
        print("\n  Dry-run complete (no files swapped). To put it in place either:")
        print("    - re-run with --replace, or")
        print(f"    - manually back up {source} and rename {out} -> {source}")


if __name__ == "__main__":
    main()
