"""Tests for Phase 8 Stage-A view assembly (metals.features.ssl_views)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from metals.features.ssl_views import (
    TrainOnlyImputer,
    is_text_column,
    partition_columns,
    split_views,
)


def _context_frame() -> pd.DataFrame:
    idx = pd.date_range("2015-01-01", periods=6, freq="B")
    arange = np.arange(6, dtype=float)
    return pd.DataFrame(
        {
            "GC=F_ret_5d": arange,
            "GC=F_rvol_20d": arange,
            "tips_10y_level": arange,
            "cot_managed_money_z": arange,
            "n_articles": arange,
            "mean_tone_overall": arange,
            "mean_tone_positive": arange,
            "topic_0": arange,
            "topic_9": arange,
            "text_pca_0": arange,
        },
        index=idx,
    )


def test_is_text_column_classifies_both_views() -> None:
    for c in [
        "n_articles",
        "mean_tone_overall",
        "mean_tone_negative",
        "embedding_dispersion",
        "topic_0",
        "topic_13",
        "text_pca_0",
        "text_pca_15",
    ]:
        assert is_text_column(c), c
    for c in ["GC=F_ret_5d", "GC=F_rvol_20d", "cot_managed_money_z", "tips_10y_level"]:
        assert not is_text_column(c), c


def test_partition_and_split_views() -> None:
    ctx = _context_frame()
    price_cols, text_cols = partition_columns(ctx)
    assert set(text_cols) == {
        "n_articles",
        "mean_tone_overall",
        "mean_tone_positive",
        "topic_0",
        "topic_9",
        "text_pca_0",
    }
    assert set(price_cols) == {
        "GC=F_ret_5d",
        "GC=F_rvol_20d",
        "tips_10y_level",
        "cot_managed_money_z",
    }
    z_p, z_t = split_views(ctx)
    assert list(z_p.columns) == price_cols
    assert list(z_t.columns) == text_cols


def test_imputer_uses_train_prefix_mean_not_full_sample() -> None:
    # Train rows 0..3 have mean 1.5; the NaN sits in the test region (row 5).
    # A global mean-fill would incorporate row 4's value of 100 — a leak.
    df = pd.DataFrame(
        {"a": [0.0, 1.0, 2.0, 3.0, 100.0, np.nan]},
        index=pd.date_range("2015-01-01", periods=6, freq="B"),
    )
    imp = TrainOnlyImputer.fit(df, np.array([0, 1, 2, 3]))
    assert imp.fill_values["a"] == 1.5
    out = imp.transform(df)
    assert out["a"].iloc[5] == 1.5


def test_imputer_all_nan_train_column_falls_back_to_zero() -> None:
    df = pd.DataFrame(
        {"a": [np.nan, np.nan, 5.0]},
        index=pd.date_range("2015-01-01", periods=3, freq="B"),
    )
    imp = TrainOnlyImputer.fit(df, np.array([0, 1]))
    assert imp.fill_values["a"] == 0.0
    assert imp.transform(df)["a"].iloc[0] == 0.0
