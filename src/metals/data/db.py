"""DuckDB connection helpers.

The canonical metals DuckDB lives at ``<repo>/data/processed/metals.duckdb``
unless overridden by the ``METALS_DB_PATH`` environment variable.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb
from dotenv import load_dotenv

load_dotenv()


def _repo_root() -> Path:
    """Resolve the repository root from this file's location."""
    return Path(__file__).resolve().parents[3]


def db_path() -> Path:
    """Return the canonical DuckDB path, honoring ``METALS_DB_PATH`` override."""
    override = os.getenv("METALS_DB_PATH")
    if override:
        return Path(override)
    db_dir = _repo_root() / "data" / "processed"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "metals.duckdb"


def get_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open a new DuckDB connection to the canonical store.

    Parameters
    ----------
    read_only : bool
        Open in read-only mode. Useful for parallel readers and for guarding
        against accidental writes in notebooks.
    """
    return duckdb.connect(str(db_path()), read_only=read_only)


@contextmanager
def connection(read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    """Context-managed DuckDB connection that closes on exit."""
    conn = get_connection(read_only=read_only)
    try:
        yield conn
    finally:
        conn.close()
