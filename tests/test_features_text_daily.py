"""Tests for daily text-feature aggregation."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from metals.features.text_daily import (
    MARKET,
    METALS,
    THEME_TO_METALS,
    _parse_themes_field,
    aggregate_daily,
    metals_for_themes,
)


@pytest.fixture(autouse=True)
def tmp_db(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.duckdb")
        monkeypatch.setenv("METALS_DB_PATH", db_path)
        yield db_path


def test_parse_themes_field_json_list():
    assert _parse_themes_field('["ECON_GOLDPRICE","ECON_INFLATION"]') == \
        ["ECON_GOLDPRICE", "ECON_INFLATION"]


def test_parse_themes_field_comma_string():
    assert _parse_themes_field("ECON_GOLDPRICE,ECON_INFLATION") == \
        ["ECON_GOLDPRICE", "ECON_INFLATION"]


def test_parse_themes_field_empty_and_none():
    assert _parse_themes_field(None) == []
    assert _parse_themes_field("") == []
    assert _parse_themes_field(float("nan")) == []


def test_parse_themes_field_passes_list_through():
    assert _parse_themes_field(["A", "B"]) == ["A", "B"]


def test_metals_for_themes_gold_specific():
    assert metals_for_themes(["ECON_GOLDPRICE"]) == {"gold"}


def test_metals_for_themes_industry_wide():
    # Monetary-policy themes apply to every metal.
    assert metals_for_themes(["ECON_CENTRALBANK"]) == set(METALS)
    assert metals_for_themes(["ECON_INFLATION"]) == set(METALS)


def test_metals_for_themes_unknown_drops():
    assert metals_for_themes(["NOT_A_REAL_THEME"]) == set()


def test_theme_to_metals_covers_all_in_config():
    """All themes in configs/gdelt_themes.yaml should map to >= 1 metal."""
    from metals.data.gdelt import load_themes
    themes = load_themes()
    missing = [t for t in themes if t not in THEME_TO_METALS]
    assert not missing, f"Themes without metal mapping: {missing}"


def _toy_headlines() -> pd.DataFrame:
    return pd.DataFrame([
        {"timestamp_utc": pd.Timestamp("2024-01-15 09:00:00"),
         "headline_id": "h1", "source": "reuters.com",
         "themes_list": ["ECON_GOLDPRICE"],
         "tone_overall": -1.0, "tone_positive": 0.0, "tone_negative": 1.0,
         "article_url": "https://r/1"},
        {"timestamp_utc": pd.Timestamp("2024-01-15 12:00:00"),
         "headline_id": "h2", "source": "bloomberg.com",
         "themes_list": ["ECON_CENTRALBANK"],
         "tone_overall": 0.5, "tone_positive": 1.0, "tone_negative": 0.5,
         "article_url": "https://b/2"},
        {"timestamp_utc": pd.Timestamp("2024-01-16 08:00:00"),
         "headline_id": "h3", "source": "kitco.com",
         "themes_list": ["ECON_GOLDPRICE", "ECON_INFLATION"],
         "tone_overall": 0.2, "tone_positive": 0.5, "tone_negative": 0.3,
         "article_url": "https://k/3"},
    ])


def test_aggregate_daily_collapses_to_market():
    """The per-metal axis is collapsed to one shared 'market' row/day. Both h1 and
    h2 carry a known theme, so 2024-01-15 has 2 articles; 2024-01-16 has 1 (h3)."""
    out = aggregate_daily(_toy_headlines())
    assert set(out["metal"].unique()) == {MARKET}
    by_day = out.set_index("timestamp_utc")["n_articles"]
    assert by_day[pd.Timestamp("2024-01-15")] == 2
    assert by_day[pd.Timestamp("2024-01-16")] == 1


def test_aggregate_daily_drops_unknown_theme_articles():
    """Articles whose themes map to no metal (no known theme) are excluded from
    the market count, matching the prior per-metal behaviour."""
    df = pd.DataFrame([
        {"timestamp_utc": pd.Timestamp("2024-02-01 10:00:00"), "headline_id": "x1",
         "source": "s", "themes_list": ["ECON_INFLATION"],
         "tone_overall": 1.0, "tone_positive": 1.0, "tone_negative": 0.0,
         "article_url": "u1"},
        {"timestamp_utc": pd.Timestamp("2024-02-01 11:00:00"), "headline_id": "x2",
         "source": "s", "themes_list": ["NOT_A_REAL_THEME"],
         "tone_overall": -5.0, "tone_positive": 0.0, "tone_negative": 5.0,
         "article_url": "u2"},
    ])
    out = aggregate_daily(df)
    row = out[out["timestamp_utc"] == pd.Timestamp("2024-02-01")].iloc[0]
    assert row["metal"] == MARKET
    assert row["n_articles"] == 1                      # x2 dropped (unknown theme)
    assert row["mean_tone_overall"] == pytest.approx(1.0)


def test_aggregate_daily_tone_means_correctly():
    out = aggregate_daily(_toy_headlines())
    row = out[(out["timestamp_utc"] == pd.Timestamp("2024-01-15")) &
              (out["metal"] == MARKET)].iloc[0]
    # Day 2024-01-15 market has h1 (overall=-1.0) and h2 (overall=0.5). Mean = -0.25
    assert row["mean_tone_overall"] == pytest.approx(-0.25)


def test_aggregate_daily_empty_returns_schema():
    out = aggregate_daily(pd.DataFrame())
    assert out.empty
    assert set(out.columns) >= {"timestamp_utc", "metal", "n_articles",
                                "mean_tone_overall", "mean_embedding"}


def test_aggregate_daily_with_embeddings_writes_dispersion_and_centroid():
    headlines = _toy_headlines()
    # Three rows; provide a 4-dim embedding per row.
    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)
    out = aggregate_daily(headlines, embeddings=embeddings)
    row = out[(out["timestamp_utc"] == pd.Timestamp("2024-01-15")) &
              (out["metal"] == MARKET)].iloc[0]
    assert row["mean_embedding"] is not None
    assert row["mean_embedding"].shape == (4,)
    # Dispersion is in [0, 2] for cosine distance from centroid
    assert 0.0 <= row["embedding_dispersion"] <= 2.0


def test_upsert_and_load_daily_round_trip():
    from metals.data.migrations.runner import apply_migrations
    from metals.features.text_daily import load_daily, upsert_daily
    apply_migrations(verbose=False)
    headlines = _toy_headlines()
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    agg = aggregate_daily(headlines, embeddings=embeddings)
    n = upsert_daily(agg)
    assert n == len(agg)
    back = load_daily(metal=MARKET)
    assert not back.empty
    assert set(back["metal"].unique()) == {MARKET}
    sample = back.iloc[0]["mean_embedding"]
    assert sample is None or sample.shape == (3,)
