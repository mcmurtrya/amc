"""Walk-forward cross-validation utilities.

Walk-forward CV is the only acceptable validation scheme for this project.
Each split has three windows in strict chronological order: train, val, test.
The training window expands each step; val and test slide forward by
``step_days``.

Within any single split, train, val, and test indices are disjoint. Across
splits, the test of split *i* may legitimately appear in train of split
*i+1* — that is the design of expanding-window walk-forward CV, not a leak.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Split:
    """A single walk-forward split."""

    split_id: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_splits(
    timestamps: pd.DatetimeIndex | pd.Series | np.ndarray,
    train_start: str | pd.Timestamp,
    val_days: int = 180,
    test_days: int = 180,
    step_days: int = 180,
    min_train_days: int = 5 * 365,
    max_splits: int | None = None,
) -> Iterator[Split]:
    """Yield expanding-window walk-forward splits.

    Each split layout:

        train : [train_start, train_end)
        val   : [train_end,   val_end)
        test  : [val_end,     test_end)

    Parameters
    ----------
    timestamps : DatetimeIndex or Series of datetimes
        Sorted unique timestamps in the dataset.
    train_start : Timestamp
        First permissible date in any training set.
    val_days, test_days, step_days : int
        Calendar-day windows for validation, test, and step between splits.
    min_train_days : int
        Minimum training span before yielding the first split. Default 5 years.
    max_splits : int, optional
        Cap the number of splits yielded.
    """
    ts = pd.DatetimeIndex(timestamps)
    if len(ts) == 0:
        return
    if not ts.is_monotonic_increasing:
        raise ValueError("walk_forward_splits: timestamps must be sorted ascending")
    if ts.has_duplicates:
        raise ValueError("walk_forward_splits: timestamps must be unique")

    train_start = pd.Timestamp(train_start)
    last = ts[-1]

    train_end = train_start + pd.Timedelta(days=min_train_days)
    split_id = 0

    while True:
        val_end = train_end + pd.Timedelta(days=val_days)
        test_end = val_end + pd.Timedelta(days=test_days)

        if test_end > last + pd.Timedelta(days=1):
            return

        train_mask = (ts >= train_start) & (ts < train_end)
        val_mask = (ts >= train_end) & (ts < val_end)
        test_mask = (ts >= val_end) & (ts < test_end)

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        test_idx = np.where(test_mask)[0]

        if len(train_idx) and len(val_idx) and len(test_idx):
            yield Split(
                split_id=split_id,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                train_start=ts[train_idx[0]],
                train_end=train_end,
                val_end=val_end,
                test_end=test_end,
            )
            split_id += 1
            if max_splits is not None and split_id >= max_splits:
                return

        train_end = train_end + pd.Timedelta(days=step_days)


def check_no_leakage(splits: list[Split]) -> None:
    """Assert within-split disjointness of train, val, test indices.

    Raises ``AssertionError`` if any split contains overlapping windows.
    Does not check cross-split overlap because it is expected by design.
    """
    for s in splits:
        train_set = set(s.train_idx.tolist())
        val_set = set(s.val_idx.tolist())
        test_set = set(s.test_idx.tolist())
        if train_set & val_set:
            raise AssertionError(f"Split {s.split_id}: train and val overlap")
        if train_set & test_set:
            raise AssertionError(f"Split {s.split_id}: train and test overlap")
        if val_set & test_set:
            raise AssertionError(f"Split {s.split_id}: val and test overlap")
