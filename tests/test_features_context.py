"""Tests for the daily contextual feature builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metals.features.context import (
    ContextConfig,
    _pca_reduce,
    _stack_embeddings,
    build_context,
)


def _toy_prices(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    cols = ["GC=F", "SI=F", "PL=F", "PA=F"]
    rets = rng.normal(0, 0.01, (n, len(cols)))
    return pd.DataFrame(1000 * np.exp(np.cumsum(rets, axis=0)),
                        index=idx, columns=cols)


def _toy_macro(idx: pd.DatetimeIndex, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(idx)
    return pd.DataFrame({
        "DGS10":    3.0 + rng.normal(0, 0.05, n).cumsum() * 0.005,
        "DGS2":     2.0 + rng.normal(0, 0.05, n).cumsum() * 0.005,
        "T10YIE":   2.3 + rng.normal(0, 0.02, n).cumsum() * 0.003,
        "T5YIE":    2.5 + rng.normal(0, 0.02, n).cumsum() * 0.003,
        "DTWEXBGS": 100 + rng.normal(0, 0.3, n).cumsum() * 0.05,
        "VIXCLS":   18 + rng.normal(0, 1.5, n),
        "BAA10Y":   4.0 + rng.normal(0, 0.15, n),
        "GPR_DAILY": 100 + rng.normal(0, 25, n),
    }, index=idx)


def test_stack_embeddings_handles_none_rows():
    df = pd.DataFrame({"mean_embedding": [np.array([1.0, 2.0]), None, np.array([3.0, 4.0])]})
    out = _stack_embeddings(df, dim_hint=2)
    assert out.shape == (3, 2)
    assert (out[1] == 0).all()


def test_pca_reduce_basic():
    rng = np.random.default_rng(0)
    matrix = rng.normal(0, 1, (50, 16)).astype(np.float32)
    reduced, pca = _pca_reduce(matrix, n_components=4)
    assert reduced.shape == (50, 4)
    # whitened PCA components have unit variance per dimension
    assert np.abs(reduced.var(axis=0) - 1.0).max() < 0.2


def test_pca_reduce_caps_at_min_dim():
    matrix = np.eye(4, 6).astype(np.float32)
    reduced, pca = _pca_reduce(matrix, n_components=20)  # asks for too many
    assert reduced.shape[1] == min(20, matrix.shape[1], matrix.shape[0] - 1)


def test_build_context_minimal_smoke():
    prices = _toy_prices()
    macro = _toy_macro(prices.index)
    ctx, artifacts = build_context(
        prices=prices, macro_wide=macro,
        config=ContextConfig(target_metal="gold"),
    )
    assert not ctx.empty
    assert ctx.index.equals(prices.index)
    assert "real_yield_10y" in ctx.columns
    assert "GC=F_ret_5d" in ctx.columns
    assert "GC=F_rvol_20d" in ctx.columns


def test_build_context_with_text_features_adds_pca_columns():
    prices = _toy_prices(n=200)
    macro = _toy_macro(prices.index)
    # Build synthetic text_daily for gold on 100 of the 200 days, with a
    # 32-dim embedding each.
    rng = np.random.default_rng(2)
    n_text = 100
    sub_idx = prices.index[50:50 + n_text]
    text = pd.DataFrame({
        "timestamp_utc": sub_idx,
        "metal": ["gold"] * n_text,
        "n_articles": rng.integers(5, 50, n_text),
        "embedding_dispersion": rng.uniform(0.1, 0.4, n_text),
        "mean_embedding": [rng.normal(0, 1, 32).astype(np.float32) for _ in range(n_text)],
        "mean_tone_overall": rng.normal(0, 1, n_text),
        "mean_tone_positive": rng.uniform(0, 2, n_text),
        "mean_tone_negative": rng.uniform(0, 2, n_text),
    })
    ctx, artifacts = build_context(
        prices=prices, macro_wide=macro, text_daily=text,
        config=ContextConfig(target_metal="gold", embedding_pca_dims=8),
    )
    assert "n_articles" in ctx.columns
    assert "embedding_dispersion" in ctx.columns
    pca_cols = [c for c in ctx.columns if c.startswith("text_pca_")]
    assert 1 <= len(pca_cols) <= 8
    assert "text_pca" in artifacts


def test_build_context_rejects_unknown_target_metal():
    prices = _toy_prices(n=100)
    macro = _toy_macro(prices.index)
    with pytest.raises(ValueError, match="Unknown target_metal"):
        build_context(prices=prices, macro_wide=macro,
                      config=ContextConfig(target_metal="lead"))


def test_build_context_topic_prevalence_passes_through():
    prices = _toy_prices(n=100)
    macro = _toy_macro(prices.index)
    topic_wide = pd.DataFrame({
        "topic_0": np.random.uniform(0, 1, len(prices)),
        "topic_1": np.random.uniform(0, 1, len(prices)),
    }, index=prices.index)
    ctx, _ = build_context(prices=prices, macro_wide=macro,
                           topic_prevalence=topic_wide)
    assert "topic_0" in ctx.columns and "topic_1" in ctx.columns
