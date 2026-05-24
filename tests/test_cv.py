"""Tests for the walk-forward cross-validation utility."""

from __future__ import annotations

import pandas as pd
import pytest

from metals.eval.cv import check_no_leakage, walk_forward_splits


def _daily_index(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="D")


def test_basic_splits_produced():
    idx = _daily_index("2010-01-01", "2024-12-31")
    splits = list(
        walk_forward_splits(
            idx,
            train_start="2010-01-01",
            val_days=180,
            test_days=180,
            step_days=180,
            min_train_days=5 * 365,
        )
    )
    assert len(splits) > 0
    for s in splits:
        assert len(s.train_idx) > 0
        assert len(s.val_idx) > 0
        assert len(s.test_idx) > 0


def test_no_internal_leakage():
    idx = _daily_index("2010-01-01", "2024-12-31")
    splits = list(
        walk_forward_splits(
            idx,
            train_start="2010-01-01",
            val_days=180,
            test_days=180,
            step_days=180,
            min_train_days=5 * 365,
        )
    )
    check_no_leakage(splits)


def test_train_window_expands_each_split():
    idx = _daily_index("2010-01-01", "2024-12-31")
    splits = list(
        walk_forward_splits(
            idx,
            train_start="2010-01-01",
            val_days=180,
            test_days=180,
            step_days=180,
            min_train_days=5 * 365,
        )
    )
    prev = -1
    for s in splits:
        assert len(s.train_idx) > prev, "train window should expand each split"
        prev = len(s.train_idx)


def test_unsorted_timestamps_raises():
    bad = pd.DatetimeIndex(["2020-01-02", "2020-01-01"])
    with pytest.raises(ValueError, match="sorted"):
        list(walk_forward_splits(bad, train_start="2020-01-01"))


def test_duplicate_timestamps_raises():
    dup = pd.DatetimeIndex(["2020-01-01", "2020-01-01"])
    with pytest.raises(ValueError, match="unique"):
        list(walk_forward_splits(dup, train_start="2020-01-01"))


def test_max_splits_caps_yield():
    idx = _daily_index("2010-01-01", "2024-12-31")
    splits = list(
        walk_forward_splits(
            idx,
            train_start="2010-01-01",
            val_days=180,
            test_days=180,
            step_days=180,
            min_train_days=5 * 365,
            max_splits=3,
        )
    )
    assert len(splits) == 3


def test_insufficient_history_returns_no_splits():
    idx = _daily_index("2023-01-01", "2024-01-01")
    splits = list(
        walk_forward_splits(
            idx,
            train_start="2023-01-01",
            min_train_days=5 * 365,
        )
    )
    assert splits == []


def test_train_test_strictly_chronological():
    idx = _daily_index("2010-01-01", "2024-12-31")
    splits = list(
        walk_forward_splits(
            idx,
            train_start="2010-01-01",
            val_days=180,
            test_days=180,
            step_days=180,
            min_train_days=5 * 365,
        )
    )
    for s in splits:
        assert s.train_idx.max() < s.val_idx.min()
        assert s.val_idx.max() < s.test_idx.min()
