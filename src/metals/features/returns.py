"""Return and volatility feature engineering.

All functions expect a wide DataFrame indexed by ``timestamp_utc`` with one
column per asset. Returned frames preserve that shape with appropriate
column-suffix conventions (e.g. ``GC=F_ret_5d``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANNUALIZATION = 252  # trading days per year


def compute_log_returns(
    prices: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 20),
) -> pd.DataFrame:
    """Log returns over each horizon, computed as log(P_t / P_{t-h}).

    Non-positive prices (e.g. WTI on 2020-04-20) are masked to NaN before the
    log, so the resulting return cell is NaN rather than -inf or a warning.

    Output column convention: ``{ticker}_ret_{h}d``.
    """
    out: dict[str, pd.Series] = {}
    log_p = np.log(prices.where(prices > 0))
    for h in horizons:
        diffs = log_p - log_p.shift(h)
        for col in prices.columns:
            out[f"{col}_ret_{h}d"] = diffs[col]
    return pd.DataFrame(out, index=prices.index)


def compute_realized_vol(
    returns_1d: pd.DataFrame,
    windows: tuple[int, ...] = (5, 20, 60),
    annualize: bool = True,
) -> pd.DataFrame:
    """Rolling realized volatility over each window of one-day log returns.

    ``returns_1d`` must be a wide frame of one-day log returns (columns are
    asset names). Output column convention: ``{asset}_rvol_{w}d``.
    """
    factor = float(np.sqrt(ANNUALIZATION)) if annualize else 1.0
    out: dict[str, pd.Series] = {}
    for w in windows:
        rolled = returns_1d.rolling(window=w, min_periods=w).std() * factor
        for col in returns_1d.columns:
            out[f"{col}_rvol_{w}d"] = rolled[col]
    return pd.DataFrame(out, index=returns_1d.index)


def compute_realized_skew_kurt(
    returns_1d: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """Rolling skewness and excess kurtosis of one-day returns.

    Output columns: ``{asset}_skew_{w}d``, ``{asset}_kurt_{w}d``.
    """
    skew = returns_1d.rolling(window=window, min_periods=window).skew()
    kurt = returns_1d.rolling(window=window, min_periods=window).kurt()
    out: dict[str, pd.Series] = {}
    for col in returns_1d.columns:
        out[f"{col}_skew_{window}d"] = skew[col]
        out[f"{col}_kurt_{window}d"] = kurt[col]
    return pd.DataFrame(out, index=returns_1d.index)


def compute_max_drawdown(
    prices: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """Rolling max-drawdown over a trailing window.

    Drawdown is defined as (P_t / rolling_max - 1), so values are in [-1, 0].

    Output column convention: ``{asset}_maxdd_{w}d``.
    """
    out: dict[str, pd.Series] = {}
    for col in prices.columns:
        p = prices[col].where(prices[col] > 0)
        roll_max = p.rolling(window=window, min_periods=window).max()
        dd = p / roll_max - 1.0
        out[f"{col}_maxdd_{window}d"] = dd
    return pd.DataFrame(out, index=prices.index)
