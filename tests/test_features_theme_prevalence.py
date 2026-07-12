"""Tests for the themes-via-SQL topic-prevalence path (Phase 3 default).

These exercise the stable theme->topic_id mapping and the streaming DuckDB
aggregation that replaces BERTopic, against a small in-memory DuckDB.
"""

from __future__ import annotations

import duckdb
import pytest

from metals.features.topics import (
    TOPIC_THEMES,
    compute_theme_prevalence,
    theme_topic_map,
)


def test_theme_topic_map_is_indexed_and_unique():
    tmap = theme_topic_map()
    assert tmap["ECON_CENTRALBANK"] == 0
    assert list(tmap.values()) == list(range(len(TOPIC_THEMES)))
    assert len(set(tmap.values())) == len(tmap)


def test_topic_themes_match_config():
    """The curated topic set must equal configs/gdelt_themes.yaml."""
    from metals.data.gdelt import load_themes

    assert set(TOPIC_THEMES) == set(load_themes())


def _toy_db() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE headlines (timestamp_utc TIMESTAMP, themes VARCHAR)")
    rows = [
        ("2022-01-01 09:00:00", '["ECON_INFLATION","WB_442_INFLATION"]'),
        ("2022-01-01 15:00:00", '["ECON_INFLATION"]'),
        ("2022-01-02 12:00:00", '["SANCTIONS"]'),
        ("2022-01-02 13:00:00", "[]"),  # tagged with nothing -> day_total only
        ("2022-01-02 14:00:00", None),  # null themes        -> day_total only
        ("2022-01-03 08:00:00", '["NOT_A_CURATED_THEME"]'),  # outside set -> ignored
    ]
    con.executemany("INSERT INTO headlines VALUES (?, ?)", rows)
    return con


def test_compute_theme_prevalence_shares_are_correct():
    con = _toy_db()
    prev = compute_theme_prevalence(conn=con)
    tmap = theme_topic_map()
    prev["d"] = prev["timestamp_utc"].dt.strftime("%Y-%m-%d")
    got = {(r.d, r.topic_id): r.prevalence for r in prev.itertuples()}

    # day 1: 2 articles, ECON_INFLATION in both (2/2), WB_442_INFLATION in one (1/2)
    assert got[("2022-01-01", tmap["ECON_INFLATION"])] == pytest.approx(1.0)
    assert got[("2022-01-01", tmap["WB_442_INFLATION"])] == pytest.approx(0.5)
    # day 2: 3 articles total (incl empty + null), SANCTIONS in one -> 1/3
    assert got[("2022-01-02", tmap["SANCTIONS"])] == pytest.approx(1 / 3)


def test_compute_theme_prevalence_ignores_uncurated_themes():
    con = _toy_db()
    prev = compute_theme_prevalence(conn=con)
    # 2022-01-03's only theme is outside TOPIC_THEMES -> no rows for that day
    days = set(prev["timestamp_utc"].dt.strftime("%Y-%m-%d"))
    assert "2022-01-03" not in days


def test_compute_theme_prevalence_respects_date_window():
    con = _toy_db()
    prev = compute_theme_prevalence(conn=con, start="2022-01-02", end="2022-01-02")
    days = set(prev["timestamp_utc"].dt.strftime("%Y-%m-%d"))
    assert days == {"2022-01-02"}


def test_compute_theme_prevalence_empty_returns_schema():
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE headlines (timestamp_utc TIMESTAMP, themes VARCHAR)")
    prev = compute_theme_prevalence(conn=con)
    assert list(prev.columns) == ["timestamp_utc", "topic_id", "prevalence"]
    assert prev.empty
