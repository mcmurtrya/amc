"""Tests for the Jordà local-projection estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metals.models.lp import LPResult, cumulative_log_returns, local_projection


def test_cumulative_log_returns_sums_forward_window():
    """cumulative_log_returns(r, h)[t] = r[t+1] + ... + r[t+h]. Last h rows NaN."""
    idx = pd.date_range("2020-01-01", periods=6)
    r = pd.Series([0.01, 0.02, -0.01, 0.03, 0.005, -0.02], index=idx)
    c2 = cumulative_log_returns(r, horizon=2)
    # at t=0: r[1] + r[2] = 0.02 - 0.01 = 0.01
    # at t=3: r[4] + r[5] = 0.005 - 0.02 = -0.015
    assert c2.iloc[0] == pytest.approx(0.01)
    assert c2.iloc[3] == pytest.approx(-0.015)
    # last 2 rows must be NaN
    assert pd.isna(c2.iloc[4])
    assert pd.isna(c2.iloc[5])


def test_cumulative_log_returns_requires_positive_horizon():
    r = pd.Series([0.0, 0.0])
    with pytest.raises(ValueError, match=">= 1"):
        cumulative_log_returns(r, horizon=0)


def _simulate_treatment_and_returns(
    n: int,
    treatment_prob: float,
    impulse_lags: dict[int, float],
    noise_sd: float,
    seed: int,
) -> tuple[pd.Series, pd.Series]:
    """Build (treatment, returns) where the true causal effect is known.

    impulse_lags maps lag k -> coefficient on treatment_{t-k}. So returns
    have a known cumulative response which the LP should recover.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=n)
    treatment = pd.Series(rng.binomial(1, treatment_prob, n), index=idx, name="treatment")
    r = pd.Series(rng.normal(0.0, noise_sd, n), index=idx)
    for lag, beta in impulse_lags.items():
        r = r + beta * treatment.shift(lag).fillna(0)
    return treatment, r


def test_local_projection_recovers_known_irf():
    """Truth: r_t = 0.010 * D_{t-1} + 0.005 * D_{t-3} + noise.

    Cumulative IRF should give:
       beta_1 ≈ 0.010
       beta_3 ≈ 0.015   (0.010 from h=1 + 0.005 from h=3)
       beta_5 ≈ 0.015   (no more effects beyond h=3)
    """
    treatment, r = _simulate_treatment_and_returns(
        n=2500,
        treatment_prob=0.10,
        impulse_lags={1: 0.010, 3: 0.005},
        noise_sd=0.01,
        seed=42,
    )
    out = local_projection(r, treatment, horizons=(1, 3, 5))
    assert isinstance(out, LPResult)
    irf = out.irf.set_index("horizon")
    assert irf.loc[1, "beta"] == pytest.approx(0.010, abs=0.003)
    assert irf.loc[3, "beta"] == pytest.approx(0.015, abs=0.003)
    assert irf.loc[5, "beta"] == pytest.approx(0.015, abs=0.003)
    # Effects at h=1 and h=3 should be statistically significant.
    assert abs(irf.loc[1, "t_stat"]) > 2.0
    assert abs(irf.loc[3, "t_stat"]) > 2.0


def test_local_projection_zero_effect_in_ci():
    """No causal effect → 95% CI should contain zero at every horizon."""
    treatment, r = _simulate_treatment_and_returns(
        n=1500,
        treatment_prob=0.10,
        impulse_lags={},
        noise_sd=0.01,
        seed=7,
    )
    out = local_projection(r, treatment, horizons=(1, 5, 20))
    for _, row in out.irf.iterrows():
        assert row["ci_low"] < 0 < row["ci_high"], (
            f"h={row['horizon']}: zero must lie in [{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
        )


def test_local_projection_with_controls_preserves_treatment_sign():
    """Adding a control should not flip the recovered treatment effect."""
    rng = np.random.default_rng(11)
    n = 2000
    idx = pd.date_range("2010-01-01", periods=n)
    treatment = pd.Series(rng.binomial(1, 0.15, n), index=idx, name="hawkish")
    ctrl = pd.Series(rng.normal(0, 1, n), index=idx, name="ctrl")
    r = (
        0.008 * treatment.shift(1).fillna(0)
        + 0.002 * ctrl
        + pd.Series(rng.normal(0, 0.01, n), index=idx)
    )
    out = local_projection(
        r,
        treatment,
        controls=pd.DataFrame({"ctrl": ctrl}),
        horizons=(1,),
    )
    row = out.irf.iloc[0]
    assert row["beta"] > 0
    assert row["beta"] == pytest.approx(0.008, abs=0.003)
    assert out.treatment_name == "hawkish"
    assert out.control_names == ["ctrl"]


def test_local_projection_rejects_index_mismatch():
    idx1 = pd.date_range("2020-01-01", periods=10)
    idx2 = pd.date_range("2020-01-02", periods=10)
    r = pd.Series(np.zeros(10), index=idx1)
    t = pd.Series(np.zeros(10), index=idx2)
    with pytest.raises(ValueError, match="share an index"):
        local_projection(r, t)


def test_local_projection_rejects_control_index_mismatch():
    idx = pd.date_range("2020-01-01", periods=200)
    bad_idx = pd.date_range("2020-01-02", periods=200)
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 200), index=idx)
    t = pd.Series(np.random.default_rng(1).binomial(1, 0.1, 200), index=idx)
    bad_ctrl = pd.DataFrame({"c": np.zeros(200)}, index=bad_idx)
    with pytest.raises(ValueError, match="same index"):
        local_projection(r, t, controls=bad_ctrl)


def test_local_projection_handles_too_few_observations():
    """Splits with <30 obs return NaN beta and the n_obs count, not a crash."""
    idx = pd.date_range("2020-01-01", periods=20)
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 20), index=idx)
    t = pd.Series(np.random.default_rng(1).binomial(1, 0.1, 20), index=idx)
    out = local_projection(r, t, horizons=(1,))
    row = out.irf.iloc[0]
    assert pd.isna(row["beta"])
    assert row["n_obs"] < 30
