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
    with pytest.raises(LeakageError, match="overlaps the present"):
        assert_target_strictly_future(feats, target, target_horizon=3)


def test_target_strictly_future_rejects_window_overlap():
    """For a window-valued target with width w and horizon h, the caller must
    pass min_nan_tail = h + w - 1. A target whose window overlaps the present
    (e.g. trailing realised vol shifted by only h) must be caught."""
    idx = pd.date_range("2020-01-01", periods=30)
    feats = pd.DataFrame({"x": np.arange(30)}, index=idx)
    src = pd.Series(np.arange(30, dtype=float), index=idx)
    # Shifted by only target_horizon=5 — the realised-vol bug. With w=20 the
    # window spans [t-14, t+5], so the leakage guard with min_nan_tail=24
    # must reject it because only 5 trailing rows are NaN.
    target_buggy = src.shift(-5)
    with pytest.raises(LeakageError, match="overlaps the present"):
        assert_target_strictly_future(
            feats, target_buggy, target_horizon=5, min_nan_tail=5 + 20 - 1
        )


def test_target_strictly_future_accepts_full_forward_window():
    idx = pd.date_range("2020-01-01", periods=30)
    feats = pd.DataFrame({"x": np.arange(30)}, index=idx)
    src = pd.Series(np.arange(30, dtype=float), index=idx)
    target_ok = src.shift(-(5 + 20 - 1))  # 24 NaN tail rows
    assert_target_strictly_future(feats, target_ok, target_horizon=5, min_nan_tail=5 + 20 - 1)


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
