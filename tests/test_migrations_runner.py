"""Migration runner behavior across the three real-world DB states.

The 005→006 rename of the phase3-artifacts migration means live DBs exist in
three states, and `apply_migrations` must upgrade all of them cleanly:

1. fresh        — nothing applied (new checkout / compact rebuild)
2. laptop-state — 001..005_drop applied, no Phase 3 tables, no 007 columns
3. server-state — as (2) plus the artifacts migration applied *under its old
                  stem* `005_phase3_artifacts`: tables exist, tracking holds a
                  stale id the runner no longer knows, 006/007 unapplied
"""

from __future__ import annotations

import duckdb
import pytest

from metals.data.migrations import runner

PHASE3_TABLES = {
    "daily_text_features",
    "daily_topic_prevalence",
    "cluster_assignments",
    "cluster_centroids",
}


@pytest.fixture()
def tmp_db(monkeypatch, tmp_path):
    db_file = tmp_path / "migrations.duckdb"
    monkeypatch.setenv("METALS_DB_PATH", str(db_file))
    return db_file


def _tables(db_file) -> set[str]:
    conn = duckdb.connect(str(db_file), read_only=True)
    try:
        return {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
    finally:
        conn.close()


def _headline_columns(db_file) -> set[str]:
    conn = duckdb.connect(str(db_file), read_only=True)
    try:
        return {r[0] for r in conn.execute("DESCRIBE headlines").fetchall()}
    finally:
        conn.close()


def _apply_manually(db_file, *, through_stem: str, record_overrides: dict[str, str]):
    """Replay migration files with stem <= through_stem, recording each under
    its stem unless overridden — lets tests reconstruct historical states."""
    conn = duckdb.connect(str(db_file))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS _schema_migrations (
                migration_id VARCHAR PRIMARY KEY,
                applied_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        for sql_path in sorted(runner.MIGRATIONS_DIR.glob("*.sql")):
            if sql_path.stem > through_stem:
                continue
            conn.execute(sql_path.read_text())
            recorded = record_overrides.get(sql_path.stem, sql_path.stem)
            conn.execute("INSERT INTO _schema_migrations(migration_id) VALUES (?)", [recorded])
    finally:
        conn.close()


def test_fresh_db_applies_everything_in_lexicographic_order(tmp_db):
    applied = runner.apply_migrations(verbose=False)
    assert applied == sorted(applied)
    # Containment, not position: future migrations (008+) must not break this.
    assert {"006_phase3_artifacts", "007_headlines_page_title"} <= set(applied)
    assert PHASE3_TABLES <= _tables(tmp_db)
    assert {"page_title", "src_lang"} <= _headline_columns(tmp_db)
    # Idempotent: a second run applies nothing.
    assert runner.apply_migrations(verbose=False) == []


def test_laptop_state_upgrade_applies_006_and_007(tmp_db):
    _apply_manually(tmp_db, through_stem="005_zzz", record_overrides={})
    assert not PHASE3_TABLES & _tables(tmp_db)
    assert "page_title" not in _headline_columns(tmp_db)

    applied = runner.apply_migrations(verbose=False)
    # First two upgrades from this state are 006+007; 008+ may follow later.
    assert applied[:2] == ["006_phase3_artifacts", "007_headlines_page_title"]
    assert PHASE3_TABLES <= _tables(tmp_db)
    assert {"page_title", "src_lang"} <= _headline_columns(tmp_db)


def test_server_state_upgrade_reruns_renamed_006_as_noop(tmp_db):
    # The artifacts file (now 006_*) originally ran under the stem
    # 005_phase3_artifacts: tables exist, tracking holds only the stale id.
    _apply_manually(
        tmp_db,
        through_stem="006_phase3_artifacts",
        record_overrides={"006_phase3_artifacts": "005_phase3_artifacts"},
    )
    assert PHASE3_TABLES <= _tables(tmp_db)
    assert "page_title" not in _headline_columns(tmp_db)

    applied = runner.apply_migrations(verbose=False)
    # 006 re-runs over the existing tables (IF NOT EXISTS no-op), 007 lands.
    assert applied[:2] == ["006_phase3_artifacts", "007_headlines_page_title"]
    assert PHASE3_TABLES <= _tables(tmp_db)
    assert {"page_title", "src_lang"} <= _headline_columns(tmp_db)
    assert runner.apply_migrations(verbose=False) == []
