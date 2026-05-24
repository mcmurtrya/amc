"""Tests for the leakage guard utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metals.features.leakage import (
    LeakageError,
    assert_chronological,
    assert_features_have_history,
    assert_target_strictly_future,
)


def test_chronological_ok():
    df = pd.DataFrame({"x": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3))
    assert_chronological(df)  # should not raise


def test_chronological_rejects_unsorted():
    df = pd.DataFrame(
        {"x": [1, 2]},
        index=pd.to_datetime(["2020-01-02", "2020-01-01"]),
    )
    with pytest.raises(LeakageError, match="increasing"):
        assert_chronological(df)


def test_chronological_rejects_duplicates():
    df = pd.DataFrame(
        {"x": [1, 2]},
        index=pd.to_datetime(["2020-01-01", "2020-01-01"]),
    )
    with pytest.raises(LeakageError, match="duplicate"):
        assert_chronological(df)


def test_target_strictly_future_ok():
    idx = pd.date_range("2020-01-01", periods=10)
    feats = pd.DataFrame({"x": np.arange(10)}, index=idx)
    src = pd.Series(np.arange(10, dtype=float), index=idx)
    target = src.shift(-3)  # last 3 are NaN
    assert_target_strictly_future(feats, target, target_horizon=3)


def test_target_strictly_future_rejects_unshifted():
    idx = pd.date_range("2020-01-01", periods=10)
    feats = pd.DataFrame({"x": np.arange(10)}, index=idx)
    target = pd.Series(np.arange(10, dtype=float), index=idx)
    with pytest.raises(LeakageError, match="strictly future"):
        assert_target_strictly_future(feats, target, target_horizon=3)


def test_target_horizon_must_be_positive():
    idx = pd.date_range("2020-01-01", periods=10)
    feats = pd.DataFrame({"x": np.arange(10)}, index=idx)
    target = pd.Series(np.arange(10, dtype=float), index=idx)
    with pytest.raises(LeakageError, match=">= 1"):
        assert_target_strictly_future(feats, target, target_horizon=0)


def test_features_warmup_ok():
    idx = pd.date_range("2020-01-01", periods=20)
    feats = pd.DataFrame({"x": [np.nan] * 5 + list(range(15))}, index=idx)
    assert_features_have_history(feats, min_warmup=5)


def test_features_warmup_rejects_full_data():
    idx = pd.date_range("2020-01-01", periods=20)
    feats = pd.DataFrame({"x": np.arange(20, dtype=float)}, index=idx)
    with pytest.raises(LeakageError, match="warmup"):
        assert_features_have_history(feats, min_warmup=5)
