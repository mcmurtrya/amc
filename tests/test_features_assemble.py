"""Tests for end-to-end feature-matrix assembly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
            "BAA10Y": 4.0 + rng.normal(0, 0.15, n),
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
    """For realized-vol target with window w and horizon h, the target's
    source window is [t+h, t+h+w-1], so the last (h+w-1) rows must be NaN."""
    prices = _toy_prices()
    macro = _toy_macro(prices.index)
    h, w = 5, 20
    fm = build_feature_matrix(
        prices=prices, macro_wide=macro,
        target_ticker="GC=F", target_kind="realized_vol",
        target_horizon=h, realized_vol_window=w,
    )
    expected_nan_tail = h + w - 1
    assert fm.y.iloc[-expected_nan_tail:].isna().all()
    # And the row just before the tail must be defined.
    assert pd.notna(fm.y.iloc[-(expected_nan_tail + 1)])


def test_target_realized_vol_does_not_peek_at_past():
    """Regression for the original window-overlap bug. Construct a price path
    that is flat except for a one-time step at ``spike_idx`` (so only one
    nonzero return, at ``spike_idx``). The forward target at t=spike_idx
    measures vol over [spike_idx+h, spike_idx+h+w-1], which is past the spike,
    so y[spike_idx] must be 0. The old buggy target (trailing window ending at
    t+h) would have included the spike return and produced a nonzero value."""
    prices = _toy_prices(n=400)
    spike_idx = 200
    levels = np.full(len(prices), 1000.0)
    levels[spike_idx:] = 1050.0  # one-time price step → exactly one nonzero return
    prices["GC=F"] = levels
    macro = _toy_macro(prices.index)
    h, w = 5, 20
    fm = build_feature_matrix(
        prices=prices, macro_wide=macro,
        target_ticker="GC=F", target_kind="realized_vol",
        target_horizon=h, realized_vol_window=w,
    )
    assert fm.y.iloc[spike_idx] == pytest.approx(0.0, abs=1e-9), (
        f"y at t={spike_idx} should be 0 (spike is in the past of the forward "
        f"window) but is {fm.y.iloc[spike_idx]} — looks like the target is "
        f"peeking at returns observed at or before t."
    )


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
    with pytest.raises(ValueError, match="not present"):
        build_feature_matrix(
            prices=prices, macro_wide=macro,
            target_ticker="XYZ", target_kind="realized_vol", target_horizon=5,
        )
