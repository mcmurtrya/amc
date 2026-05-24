"""Macro feature engineering.

Inputs are a wide DataFrame indexed by ``timestamp_utc`` with one column per
FRED series ID (as loaded by ``metals.features.loaders.load_macro``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a column if present, else an all-NaN series aligned to the index."""
    if col in df.columns:
        return df[col]
    return pd.Series(np.nan, index=df.index, name=col)


def compute_macro_features(
    macro_wide: pd.DataFrame,
    change_horizons: tuple[int, ...] = (5, 20),
    rank_window: int = 252,
) -> pd.DataFrame:
    """Compute a panel of derived macro features.

    Features (each suffixed with the appropriate horizon where relevant):
      real_yield_10y, real_yield_chg_{h}d
      breakeven_5y_chg_{h}d
      dxy_chg_{h}d, dxy_pctile_{rank_window}d
      vix_chg_{h}d, vix_pctile_{rank_window}d
      yield_curve_slope (DGS10 - DGS2), and its change
      hy_oas_chg_{h}d
      gpr_chg_{h}d, gpr_pctile_{rank_window}d
    """
    df = macro_wide.copy()
    out = pd.DataFrame(index=df.index)

    # Real yield (nominal 10Y minus 10Y breakeven)
    real_yield_10y = _safe(df, "DGS10") - _safe(df, "T10YIE")
    out["real_yield_10y"] = real_yield_10y

    for h in change_horizons:
        out[f"real_yield_chg_{h}d"] = real_yield_10y - real_yield_10y.shift(h)
        out[f"breakeven_5y_chg_{h}d"] = _safe(df, "T5YIE") - _safe(df, "T5YIE").shift(h)
        out[f"dxy_chg_{h}d"] = _safe(df, "DTWEXBGS") - _safe(df, "DTWEXBGS").shift(h)
        out[f"vix_chg_{h}d"] = _safe(df, "VIXCLS") - _safe(df, "VIXCLS").shift(h)
        out[f"hy_oas_chg_{h}d"] = _safe(df, "BAMLH0A0HYM2") - _safe(df, "BAMLH0A0HYM2").shift(h)
        out[f"gpr_chg_{h}d"] = _safe(df, "GPR_DAILY") - _safe(df, "GPR_DAILY").shift(h)

    # Yield curve slope and change
    slope = _safe(df, "DGS10") - _safe(df, "DGS2")
    out["yield_curve_slope"] = slope
    out["yield_curve_slope_chg_5d"] = slope - slope.shift(5)

    # Trailing-window percentile ranks for level features
    def pctile(s: pd.Series, window: int) -> pd.Series:
        return s.rolling(window=window, min_periods=window).rank(pct=True)

    out[f"dxy_pctile_{rank_window}d"] = pctile(_safe(df, "DTWEXBGS"), rank_window)
    out[f"vix_pctile_{rank_window}d"] = pctile(_safe(df, "VIXCLS"), rank_window)
    out[f"gpr_pctile_{rank_window}d"] = pctile(_safe(df, "GPR_DAILY"), rank_window)

    return out
