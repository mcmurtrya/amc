"""Export / re-import the eval-harness records (Phase 6.10 repro package).

The harness tables (`runs`, `run_predictions`, `run_feature_importances`) live only
inside the 54 GB DuckDB, and two of the three are created LAZILY by the harness —
NOT by a migration — so rebuilding the DB from migrations alone loses every run
record (CLAUDE.md sharp edge). They are also irreplaceable: re-deriving them means
re-running every Phase 1/3/5/6 model. But they are tiny (~1.3 MB as ZSTD Parquet),
so they travel with the repo instead of the database.

    export  (default)  read-only COPY of the three tables → results/harness_export/
                       *.parquet + a manifest.json (row counts, UTC time, git hash).
    --load             re-import those Parquet files into the DB the current
                       connection points at, creating the tables first (the
                       harness's own schema helpers) and upserting on each PK so a
                       re-load is a no-op.

Run as:
    uv run python scripts/export_harness.py                 # export to results/
    uv run python scripts/export_harness.py --load          # import back
    uv run python scripts/export_harness.py --dest /tmp/h   # custom directory
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from metals.data.db import connection
from metals.eval.harness import _ensure_importance_schema, _ensure_schema, _git_hash

DEFAULT_DEST = Path(__file__).resolve().parents[1] / "results" / "harness_export"
HARNESS_TABLES = ("runs", "run_predictions", "run_feature_importances")
# PK columns per table — used for idempotent re-load (ON CONFLICT DO NOTHING).
PRIMARY_KEYS: dict[str, str] = {
    "runs": "run_id",
    "run_predictions": "run_id, timestamp_utc, ticker, horizon",
    "run_feature_importances": "run_id, split_id, feature_name, importance_type",
}


def export(dest: Path = DEFAULT_DEST) -> dict:
    """Copy the three harness tables to ZSTD Parquet under ``dest``. Returns a summary."""
    dest.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with connection(read_only=True) as conn:
        for table in HARNESS_TABLES:
            counts[table] = int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
            out = dest / f"{table}.parquet"
            conn.execute(
                f"COPY (SELECT * FROM {table}) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)"
            )
    manifest = {
        "exported_at_utc": datetime.now(UTC).isoformat(),
        "source_git_hash": _git_hash(),
        "row_counts": counts,
        "tables": list(HARNESS_TABLES),
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load(src: Path = DEFAULT_DEST) -> dict:
    """Re-import the exported Parquet into the current DB, upserting on each PK.

    Creates the tables first via the harness's own schema helpers, because
    `runs`/`run_predictions` are otherwise created lazily and a fresh migrated DB
    does not have them.
    """
    manifest_path = src / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest.json under {src} — nothing to load")
    counts: dict[str, int] = {}
    with connection() as conn:
        _ensure_schema(conn)
        _ensure_importance_schema(conn)
        for table in HARNESS_TABLES:
            parquet = src / f"{table}.parquet"
            if not parquet.exists():
                raise FileNotFoundError(f"missing {parquet}")
            conn.execute(
                f"INSERT INTO {table} SELECT * FROM read_parquet('{parquet}') "
                f"ON CONFLICT ({PRIMARY_KEYS[table]}) DO NOTHING"
            )
            counts[table] = int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
    return {"loaded_into_row_counts": counts, "source": str(src)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", action="store_true", help="Re-import instead of export.")
    parser.add_argument("--dest", default=str(DEFAULT_DEST), help="Parquet directory.")
    args = parser.parse_args()
    path = Path(args.dest)
    if args.load:
        summary = load(path)
        print("Loaded harness records; table totals now:")
        for table, n in summary["loaded_into_row_counts"].items():
            print(f"  {table:26} {n:>8,}")
    else:
        summary = export(path)
        print(f"Exported to {path} (git {summary['source_git_hash']}):")
        for table, n in summary["row_counts"].items():
            size = (path / f"{table}.parquet").stat().st_size
            print(f"  {table:26} {n:>8,} rows  {size / 1024:8.1f} KB")


if __name__ == "__main__":
    main()
