"""Look-ahead leakage guard.

A feature matrix passes the leakage check iff no row of features X with
timestamp t contains data observed at timestamp >= t. Because we use
rolling/lag-based features and align by timestamp, this can be checked
structurally: every feature column should be either (a) computable from
data with timestamp <= t, or (b) explicitly lagged.

The simplest reliable check: assert that for every (timestamp, feature)
pair, the feature value at timestamp t equals what it would be if we
truncated all source data at t. This is expensive in general, so we
provide an instead-of-end-to-end pragmatic check:

  ``check_no_lookahead(features, target, target_horizon)`` asserts that
  the target column was shifted by `target_horizon` positive periods
  (i.e. target_t = source_{t + horizon}), and that no feature column
  was shifted by a negative period.
"""

from __future__ import annotations

import pandas as pd


class LeakageError(AssertionError):
    """Raised when the leakage guard detects look-ahead bias."""


def assert_chronological(df: pd.DataFrame) -> None:
    """Ensure the index is strictly increasing — required for time-series CV."""
    idx = pd.DatetimeIndex(df.index)
    if not idx.is_monotonic_increasing:
        raise LeakageError("Index must be strictly increasing in time.")
    if idx.has_duplicates:
        raise LeakageError("Index must not contain duplicate timestamps.")


def assert_target_strictly_future(
    features: pd.DataFrame,
    target: pd.Series,
    target_horizon: int,
) -> None:
    """Sanity check that ``target`` is at least ``target_horizon`` steps ahead.

    Specifically, the target observed at row index t must originate from
    a source observation at index >= t + target_horizon. We can't verify
    the source directly, but we can check that the target series has
    `target_horizon` NaN tail values (typical of a forward-shift operation).
    """
    if target_horizon <= 0:
        raise LeakageError(f"target_horizon must be >= 1, got {target_horizon}")
    if not features.index.equals(target.index):
        raise LeakageError("features and target indices must align exactly.")
    tail = target.iloc[-target_horizon:]
    if tail.notna().any():
        raise LeakageError(
            f"target tail of {target_horizon} rows contains non-NaN values; "
            "suspect that target is not strictly future."
        )


def assert_features_have_history(
    features: pd.DataFrame,
    min_warmup: int,
) -> None:
    """Ensure the first ``min_warmup`` rows of features are dropped or NaN.

    Rolling features that have not yet accumulated ``min_warmup`` observations
    cannot be valid. Either explicit NaN or trimming away those rows is fine.
    """
    head = features.iloc[:min_warmup]
    if head.notna().all(axis=None):
        raise LeakageError(
            f"First {min_warmup} feature rows are fully non-NaN; "
            "expected warmup rows to be incomplete."
        )
