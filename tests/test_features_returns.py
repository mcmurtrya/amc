"""Tests for return and volatility feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metals.features.returns import (
    ANNUALIZATION,
    compute_log_returns,
    compute_max_drawdown,
    compute_realized_skew_kurt,
    compute_realized_vol,
)


def _toy_prices(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rets = rng.normal(loc=0.0, scale=0.01, size=(n, 2))
    levels = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(levels, index=idx, columns=["A", "B"])


def test_log_returns_shape_and_naming():
    p = _toy_prices()
    r = compute_log_returns(p, horizons=(1, 5))
    assert r.shape == (len(p), 4)
    assert set(r.columns) == {"A_ret_1d", "A_ret_5d", "B_ret_1d", "B_ret_5d"}


def test_log_returns_nan_warmup():
    p = _toy_prices()
    r = compute_log_returns(p, horizons=(5,))
    assert r["A_ret_5d"].iloc[:5].isna().all()
    assert r["A_ret_5d"].iloc[5:].notna().all()


def test_log_returns_match_manual():
    p = _toy_prices(n=10)
    r = compute_log_returns(p, horizons=(1,))
    manual = np.log(p["A"].iloc[1] / p["A"].iloc[0])
    assert r["A_ret_1d"].iloc[1] == pytest.approx(manual)


def test_realized_vol_annualization():
    p = _toy_prices(n=300, seed=42)
    r1 = compute_log_returns(p, horizons=(1,)).rename(columns=lambda c: c.replace("_ret_1d", ""))
    rv_ann = compute_realized_vol(r1, windows=(20,), annualize=True)
    rv_raw = compute_realized_vol(r1, windows=(20,), annualize=False)
    # ratio between annualized and raw should equal sqrt(252)
    ratio = (rv_ann["A_rvol_20d"] / rv_raw["A_rvol_20d"]).dropna().iloc[-1]
    assert ratio == pytest.approx(np.sqrt(ANNUALIZATION), rel=1e-6)


def test_realized_skew_kurt_columns():
    p = _toy_prices()
    r1 = compute_log_returns(p, horizons=(1,)).rename(columns=lambda c: c.replace("_ret_1d", ""))
    sk = compute_realized_skew_kurt(r1, window=20)
    assert set(sk.columns) == {"A_skew_20d", "A_kurt_20d", "B_skew_20d", "B_kurt_20d"}


def test_max_drawdown_within_bounds():
    p = _toy_prices(n=300)
    mdd = compute_max_drawdown(p, window=60)
    valid = mdd.dropna()
    assert (valid <= 0).all().all()
    assert (valid >= -1).all().all()


def test_max_drawdown_zero_for_monotone_rising():
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    p = pd.DataFrame({"A": np.arange(100, dtype=float) + 1.0}, index=idx)
    mdd = compute_max_drawdown(p, window=20)
    assert (mdd.dropna()["A_maxdd_20d"] == 0.0).all()


def test_log_returns_handles_nonpositive_price():
    """A non-positive price (cf. WTI on 2020-04-20) must produce NaN returns,
    not -inf or a propagated negative value."""
    p = _toy_prices(n=10)
    p.iloc[5, 0] = -37.63  # Simulate WTI going negative.
    p.iloc[6, 0] = 0.0  # And a hard zero immediately after.
    r = compute_log_returns(p, horizons=(1,))
    # Returns into the bad rows (t=5,6) and out of them (t=6,7) must be NaN.
    assert pd.isna(r["A_ret_1d"].iloc[5])
    assert pd.isna(r["A_ret_1d"].iloc[6])
    assert pd.isna(r["A_ret_1d"].iloc[7])
    # Returns elsewhere must remain finite.
    finite = r["A_ret_1d"].dropna()
    assert np.isfinite(finite).all()
    # The clean column B should be untouched.
    assert r["B_ret_1d"].iloc[1:].notna().all()


def test_log_returns_does_not_warn_on_nonpositive_price():
    """Regression: np.log must not emit RuntimeWarning on the masked path."""
    import warnings

    p = _toy_prices(n=10)
    p.iloc[5, 0] = -1.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        compute_log_returns(p, horizons=(1, 5))  # would raise if warning fired


def test_max_drawdown_masks_nonpositive():
    """Drawdown must stay in [-1, 0] even when the input has a negative price."""
    p = _toy_prices(n=80)
    p.iloc[40, 0] = -10.0
    mdd = compute_max_drawdown(p, window=20)
    valid = mdd["A_maxdd_20d"].dropna()
    assert (valid <= 0).all()
    assert (valid >= -1).all()
