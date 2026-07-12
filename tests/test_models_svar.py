"""Tests for the sign-restricted SVAR (plan 5.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metals.models.svar import (
    SIGN_RESTRICTIONS,
    VARIABLES,
    build_svar_data,
    estimate_svar,
    fit_var_ols,
    irf_from_var,
    match_shocks,
    select_lag_bic,
)


def _simulate_var(t: int, a1: np.ndarray, sigma: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = a1.shape[0]
    chol = np.linalg.cholesky(sigma)
    y = np.zeros((t, n))
    for i in range(1, t):
        y[i] = a1 @ y[i - 1] + chol @ rng.standard_normal(n)
    return y


A1 = np.diag([0.5, 0.3, 0.2, 0.4])
SIGMA = np.eye(4) * 0.01


def test_fit_var_ols_recovers_coefficients():
    y = _simulate_var(4000, A1, SIGMA, seed=1)
    fit = fit_var_ols(y, 1)
    a1_hat = np.asarray(fit["B"])[1:, :].T
    assert np.allclose(a1_hat, A1, atol=0.05)


def test_select_lag_bic_picks_one_for_var1():
    y = _simulate_var(3000, A1, SIGMA, seed=2)
    assert select_lag_bic(y, max_lag=4) == 1


def test_irf_shapes_and_decay():
    y = _simulate_var(2000, A1, SIGMA, seed=3)
    fit = fit_var_ols(y, 1)
    impact = np.linalg.cholesky(np.asarray(fit["Sigma"]))
    irf = irf_from_var(np.asarray(fit["B"]), impact, 20, 4, 1)
    assert irf.shape == (21, 4, 4)
    # Stationary VAR: responses decay toward zero.
    assert np.abs(irf[20]).max() < np.abs(irf[0]).max()


def test_match_shocks_accepts_and_rejects():
    good = np.eye(4)
    # Column signs arranged so each restricted shock has a valid signed column.
    good[:, 0] = [1.0, 1.0, -0.5, -1.0]  # real_yield pattern (+, +, -0, -)
    good[:, 1] = [-1.0, 0.3, -1.0, 1.0]  # risk_aversion pattern (-, ?, -, +)
    good[:, 2] = [0.2, 1.0, 0.1, -1.0]  # usd pattern (?, +, ?, -)
    assign = match_shocks(good, SIGN_RESTRICTIONS)
    assert assign is not None
    cols = {c for c, _ in assign.values()}
    assert len(cols) == len(SIGN_RESTRICTIONS)  # distinct columns

    bad = np.ones((4, 4))  # all-positive: real_yield's gold "-" can never hold
    # A global flip makes every cell negative — then dxy "+" fails; reject.
    assert match_shocks(bad, {"real_yield": {"gold": "-", "dxy": "+"}}) is None


def test_estimate_svar_end_to_end_small():
    y = _simulate_var(1500, A1, SIGMA, seed=4)
    idx = pd.bdate_range("2015-01-01", periods=1500)
    data = pd.DataFrame(y, columns=list(VARIABLES), index=idx)
    res = estimate_svar(data, lag=1, horizons=10, n_target=25, max_draws=5000, seed=0)
    assert res.n_accepted > 0
    for shock in SIGN_RESTRICTIONS:
        assert res.irfs[shock].shape[1:] == (11, 4)
    q = res.quantiles("real_yield", "gold")
    assert list(q.columns) == ["q16", "q50", "q84"]
    assert (q["q16"] <= q["q84"]).all()
    # Impact restriction respected in every accepted draw: gold impact < 0.
    assert (res.irfs["real_yield"][:, 0, VARIABLES.index("gold")] < 0).all()


def test_build_svar_data_requires_columns():
    idx = pd.bdate_range("2020-01-01", periods=50)
    prices = pd.DataFrame({"GC=F": np.linspace(1800, 1850, 50)}, index=idx)
    macro = pd.DataFrame(index=idx)
    with pytest.raises(ValueError):
        build_svar_data(prices, macro)
