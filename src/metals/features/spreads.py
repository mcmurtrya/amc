"""Cross-asset spread and ratio features.

Spread features capture relationships between metals and between metals and
related macro/industrial assets. They tend to be more stationary than the
underlying levels and so are useful as ML inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Default spread pairs. Tickers reference the canonical universe (yfinance).
DEFAULT_SPREADS: tuple[tuple[str, str, str], ...] = (
    ("GC=F", "SI=F", "Au_Ag"),
    ("PL=F", "PA=F", "Pt_Pd"),
    ("GC=F", "HG=F", "Au_Cu"),
    ("GC=F", "CL=F", "Au_Oil"),
)


def compute_ratios(
    prices: pd.DataFrame,
    pairs: tuple[tuple[str, str, str], ...] = DEFAULT_SPREADS,
) -> pd.DataFrame:
    """Compute numerator/denominator ratios for each configured pair.

    Output column convention: ``{name}_ratio`` (e.g. ``Au_Ag_ratio``).
    Missing tickers are skipped silently with a recorded NaN column.
    """
    out: dict[str, pd.Series] = {}
    for num, den, name in pairs:
        if num in prices.columns and den in prices.columns:
            out[f"{name}_ratio"] = prices[num] / prices[den]
        else:
            out[f"{name}_ratio"] = pd.Series(np.nan, index=prices.index)
    return pd.DataFrame(out, index=prices.index)


def compute_log_spread_changes(
    ratios: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 20),
) -> pd.DataFrame:
    """Log change of each ratio over each horizon.

    Output column convention: ``{ratio_name}_logchg_{h}d``.
    """
    log_r = np.log(ratios)
    out: dict[str, pd.Series] = {}
    for h in horizons:
        diff = log_r - log_r.shift(h)
        for col in ratios.columns:
            base = col.replace("_ratio", "")
            out[f"{base}_logchg_{h}d"] = diff[col]
    return pd.DataFrame(out, index=ratios.index)


def compute_spread_zscores(
    ratios: pd.DataFrame,
    window: int = 252,
) -> pd.DataFrame:
    """Rolling z-score of each ratio over the trailing window."""
    out: dict[str, pd.Series] = {}
    for col in ratios.columns:
        mean = ratios[col].rolling(window=window, min_periods=window).mean()
        std = ratios[col].rolling(window=window, min_periods=window).std()
        base = col.replace("_ratio", "")
        out[f"{base}_z_{window}d"] = (ratios[col] - mean) / std
    return pd.DataFrame(out, index=ratios.index)
