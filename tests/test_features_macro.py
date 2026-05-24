"""Tests for macro feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from metals.features.macro import compute_macro_features


def _toy_macro(n: int = 400, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "DGS10": 3.0 + rng.normal(0, 0.1, n).cumsum() * 0.01,
            "DGS2": 2.0 + rng.normal(0, 0.1, n).cumsum() * 0.01,
            "T10YIE": 2.3 + rng.normal(0, 0.05, n).cumsum() * 0.005,
            "T5YIE": 2.5 + rng.normal(0, 0.05, n).cumsum() * 0.005,
            "DTWEXBGS": 100 + rng.normal(0, 0.5, n).cumsum() * 0.1,
            "VIXCLS": 18 + rng.normal(0, 2, n),
            "BAMLH0A0HYM2": 4.0 + rng.normal(0, 0.2, n),
            "GPR_DAILY": 100 + rng.normal(0, 30, n),
        },
        index=idx,
    )


def test_macro_features_columns_present():
    m = _toy_macro()
    f = compute_macro_features(m)
    expected = [
        "real_yield_10y",
        "real_yield_chg_5d",
        "real_yield_chg_20d",
        "yield_curve_slope",
        "yield_curve_slope_chg_5d",
        "dxy_chg_5d",
        "vix_chg_5d",
        "gpr_chg_5d",
        "dxy_pctile_252d",
        "vix_pctile_252d",
        "gpr_pctile_252d",
    ]
    for c in expected:
        assert c in f.columns


def test_real_yield_matches_definition():
    m = _toy_macro()
    f = compute_macro_features(m)
    expected = m["DGS10"] - m["T10YIE"]
    assert (f["real_yield_10y"] - expected).abs().max() < 1e-9


def test_handles_missing_series_with_nan():
    m = _toy_macro().drop(columns=["BAMLH0A0HYM2", "GPR_DAILY"])
    f = compute_macro_features(m)
    assert f["hy_oas_chg_5d"].isna().all()
    assert f["gpr_chg_5d"].isna().all()


def test_pctile_in_zero_one():
    m = _toy_macro(n=600)
    f = compute_macro_features(m, rank_window=252)
    valid = f["dxy_pctile_252d"].dropna()
    assert (valid >= 0).all() and (valid <= 1).all()
