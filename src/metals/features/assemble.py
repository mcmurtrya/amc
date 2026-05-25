"""Feature-matrix assembly.

Given price and macro data, produce an (X, y) pair suitable for ML training.
The target is configurable: realized volatility (the Phase 1 default) or
forward return. All target construction goes through ``shift_target`` to
keep the look-ahead direction explicit and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from metals.features.leakage import (
    assert_chronological,
    assert_features_have_history,
    assert_target_strictly_future,
)
from metals.features.macro import compute_macro_features
from metals.features.returns import (
    compute_log_returns,
    compute_max_drawdown,
    compute_realized_skew_kurt,
    compute_realized_vol,
)
from metals.features.spreads import (
    compute_log_spread_changes,
    compute_ratios,
    compute_spread_zscores,
)


@dataclass(frozen=True)
class FeatureMatrix:
    """A bundled (X, y) plus diagnostic metadata."""

    X: pd.DataFrame
    y: pd.Series
    target_name: str
    target_horizon: int
    feature_names: list[str]


def build_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Build all price-derived features for a wide price frame."""
    ret_1 = compute_log_returns(prices, horizons=(1,))
    # Strip "_ret_1d" suffix so we have a clean 1-day return frame keyed by ticker
    ret_1d_only = ret_1.rename(columns=lambda c: c.replace("_ret_1d", ""))
    rvol = compute_realized_vol(ret_1d_only, windows=(5, 20, 60))
    skew_kurt = compute_realized_skew_kurt(ret_1d_only, window=20)
    mdd = compute_max_drawdown(prices, window=60)
    multi_ret = compute_log_returns(prices, horizons=(1, 5, 20))

    ratios = compute_ratios(prices)
    ratio_changes = compute_log_spread_changes(ratios, horizons=(1, 5, 20))
    ratio_z = compute_spread_zscores(ratios, window=252)

    return pd.concat([multi_ret, rvol, skew_kurt, mdd, ratio_changes, ratio_z], axis=1)


def shift_target(series: pd.Series, horizon: int) -> pd.Series:
    """Shift a series ``horizon`` steps into the future.

    After shift, value at row t == source value at row t + horizon. The last
    ``horizon`` rows will be NaN, which is the structural signature checked by
    ``assert_target_strictly_future``.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    return series.shift(-horizon)


def build_feature_matrix(
    prices: pd.DataFrame,
    macro_wide: pd.DataFrame,
    target_ticker: str,
    target_kind: str = "realized_vol",
    target_horizon: int = 5,
    realized_vol_window: int = 20,
    min_warmup: int = 252,
) -> FeatureMatrix:
    """Assemble the (X, y) frame for a given metal and target spec.

    Parameters
    ----------
    prices : DataFrame
        Wide price frame (timestamp index, ticker columns) — usually adj_close.
    macro_wide : DataFrame
        Wide macro frame from ``load_macro``.
    target_ticker : str
        Which asset to target, e.g. ``"GC=F"`` for gold futures.
    target_kind : str
        ``"realized_vol"`` or ``"return"``.
    target_horizon : int
        Number of trading days ahead.
    realized_vol_window : int
        Lookback window when computing the realized-vol target.
    min_warmup : int
        Number of initial rows expected to contain NaN warmup data.
    """
    assert_chronological(prices)
    assert_chronological(macro_wide)
    if target_ticker not in prices.columns:
        raise ValueError(f"{target_ticker!r} not present in prices columns.")

    # Align macro to price index, forward-fill at use time only (not at ingest)
    macro_aligned = macro_wide.reindex(prices.index).ffill()

    price_feats = build_price_features(prices)
    macro_feats = compute_macro_features(macro_aligned)
    X = pd.concat([price_feats, macro_feats], axis=1)

    # Build target
    target_returns_1d = compute_log_returns(prices, horizons=(1,))
    ret_col = f"{target_ticker}_ret_1d"
    if target_kind == "realized_vol":
        ANN = float(np.sqrt(252))
        realized = (
            target_returns_1d[ret_col]
            .rolling(window=realized_vol_window, min_periods=realized_vol_window)
            .std() * ANN
        )
        # Target window starts target_horizon days ahead and spans
        # realized_vol_window days: [t+h, t+h+w-1]. Equivalently, the
        # trailing-window realized vol at row (t+h+w-1) shifted back to t.
        shift_steps = target_horizon + realized_vol_window - 1
        y = realized.shift(-shift_steps)
        target_name = f"{target_ticker}_rvol_{realized_vol_window}d_fwd{target_horizon}"
        nan_tail = shift_steps
    elif target_kind == "return":
        y = shift_target(target_returns_1d[ret_col], target_horizon)
        target_name = f"{target_ticker}_ret_1d_fwd{target_horizon}"
        nan_tail = target_horizon
    else:
        raise ValueError(f"Unknown target_kind: {target_kind!r}")

    # Align X and y on common index
    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]

    # Leakage guards
    assert_features_have_history(X, min_warmup=min_warmup)
    assert_target_strictly_future(X, y, target_horizon=target_horizon, min_nan_tail=nan_tail)

    return FeatureMatrix(
        X=X,
        y=y,
        target_name=target_name,
        target_horizon=target_horizon,
        feature_names=list(X.columns),
    )
