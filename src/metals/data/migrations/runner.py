"""Apply SQL migrations to the canonical DuckDB store.

Usage:
    uv run python -m metals.data.migrations.runner

Migrations live as ``NNN_*.sql`` files in this directory and are applied in
lexicographic order. Applied migrations are recorded in the
``_schema_migrations`` tracking table so subsequent runs are idempotent.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from metals.data.db import connection

MIGRATIONS_DIR = Path(__file__).resolve().parent
TRACKING_TABLE = "_schema_migrations"


def _ensure_tracking_table(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TRACKING_TABLE} (
            migration_id    VARCHAR PRIMARY KEY,
            applied_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def _applied_migrations(conn: duckdb.DuckDBPyConnection) -> set[str]:
    rows = conn.execute(f"SELECT migration_id FROM {TRACKING_TABLE}").fetchall()
    return {row[0] for row in rows}


def _record_migration(conn: duckdb.DuckDBPyConnection, migration_id: str) -> None:
    conn.execute(
        f"INSERT INTO {TRACKING_TABLE}(migration_id) VALUES (?)",
        [migration_id],
    )


def apply_migrations(verbose: bool = True) -> list[str]:
    """Apply all unapplied migrations in lexicographic order.

    Returns the list of migration IDs that were applied in this invocation.
    """
    applied_now: list[str] = []
    with connection() as conn:
        _ensure_tracking_table(conn)
        already = _applied_migrations(conn)
        for sql_path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            mig_id = sql_path.stem
            if mig_id in already:
                if verbose:
                    print(f"  - {mig_id} (already applied)")
                continue
            sql = sql_path.read_text()
            conn.execute(sql)
            _record_migration(conn, mig_id)
            applied_now.append(mig_id)
            if verbose:
                print(f"  + applied {mig_id}")
    return applied_now


def main() -> None:
    print("Applying migrations...")
    applied = apply_migrations()
    if applied:
        print(f"Applied {len(applied)} migration(s).")
    else:
        print("No new migrations to apply.")


if __name__ == "__main__":
    main()
