"""Tests for end-to-end feature-matrix assembly."""

from __future__ import annotations

import numpy as np
import pandas as pd

from metals.features.assemble import build_feature_matrix, shift_target


def _toy_prices(n: int = 600, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    cols = ["GC=F", "SI=F", "PL=F", "PA=F", "HG=F", "CL=F"]
    rets = rng.normal(0, 0.01, (n, len(cols)))
    levels = 1000 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(levels, index=idx, columns=cols)


def _toy_macro(idx: pd.DatetimeIndex, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(idx)
    return pd.DataFrame(
        {
            "DGS10": 3.0 + rng.normal(0, 0.05, n).cumsum() * 0.01,
            "DGS2": 2.0 + rng.normal(0, 0.05, n).cumsum() * 0.01,
            "T10YIE": 2.3 + rng.normal(0, 0.02, n).cumsum() * 0.005,
            "T5YIE": 2.5 + rng.normal(0, 0.02, n).cumsum() * 0.005,
            "DTWEXBGS": 100 + rng.normal(0, 0.3, n).cumsum() * 0.05,
            "VIXCLS": 18 + rng.normal(0, 1.5, n),
            "BAMLH0A0HYM2": 4.0 + rng.normal(0, 0.15, n),
            "GPR_DAILY": 100 + rng.normal(0, 25, n),
        },
        index=idx,
    )


def test_shift_target_forward():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    out = shift_target(s, horizon=2)
    assert pd.isna(out.iloc[-1]) and pd.isna(out.iloc[-2])
    assert out.iloc[0] == 3.0


def test_build_feature_matrix_returns_dataclass_with_expected_shape():
    prices = _toy_prices()
    macro = _toy_macro(prices.index)
    fm = build_feature_matrix(
        prices=prices,
        macro_wide=macro,
        target_ticker="GC=F",
        target_kind="realized_vol",
        target_horizon=5,
    )
    assert fm.X.shape[0] == fm.y.shape[0]
    assert fm.target_horizon == 5
    assert "GC=F" in fm.target_name
    # Some realistic count of features given our pipeline
    assert len(fm.feature_names) > 30


def test_target_realized_vol_is_strictly_future():
    prices = _toy_prices()
    macro = _toy_macro(prices.index)
    fm = build_feature_matrix(
        prices=prices, macro_wide=macro,
        target_ticker="GC=F", target_kind="realized_vol", target_horizon=5,
    )
    assert fm.y.iloc[-5:].isna().all()


def test_target_return_works():
    prices = _toy_prices()
    macro = _toy_macro(prices.index)
    fm = build_feature_matrix(
        prices=prices, macro_wide=macro,
        target_ticker="GC=F", target_kind="return", target_horizon=5,
    )
    assert fm.y.iloc[-5:].isna().all()
    assert "ret" in fm.target_name


def test_build_rejects_unknown_target_ticker():
    prices = _toy_prices()
    macro = _toy_macro(prices.index)
    import pytest
    with pytest.raises(ValueError, match="not present"):
        build_feature_matrix(
            prices=prices, macro_wide=macro,
            target_ticker="XYZ", target_kind="realized_vol", target_horizon=5,
        )
