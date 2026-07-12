"""Tests for the Phase 5 DoubleML causal estimator.

Synthetic designs with a *known* ATE (and known confounding) so the debiased
estimator must recover the truth where a naive difference-in-means would not.
Gated on doubleml/lightgbm being importable (they are core deps).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metals.models.causal import (
    _ATE_COLUMNS,
    CausalResult,
    estimate_ate,
    estimate_scenarios,
    placebo_pvalue,
)

pytest.importorskip("doubleml")
pytest.importorskip("lightgbm")


def _make_irm_data(n: int, tau: float, seed: int):
    """Confounded binary-treatment data with a known ATE = tau.

    x1 drives BOTH the propensity and the outcome, so a naive mean difference is
    biased; the debiased ATE should recover tau.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=n)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    p = 1.0 / (1.0 + np.exp(-(0.8 * x1 - 0.5 * x2)))
    d = rng.binomial(1, p)
    y = tau * d + 0.6 * x1 - 0.4 * x2 + rng.normal(0, 0.5, n)
    outcome = pd.Series(y, index=idx, name="y")
    treatment = pd.Series(d, index=idx, name="d")
    confounders = pd.DataFrame({"x1": x1, "x2": x2}, index=idx)
    return outcome, treatment, confounders


def test_estimate_ate_recovers_known_effect():
    outcome, treatment, conf = _make_irm_data(2000, tau=0.5, seed=0)
    res = estimate_ate(outcome, treatment, conf, n_folds=5, learner_params={"n_estimators": 100})
    assert res["ate"] == pytest.approx(0.5, abs=0.15)
    assert res["ci_low"] < 0.5 < res["ci_high"]
    assert res["n_treated"] > 0 and res["n_control"] > 0
    assert res["n_obs"] == 2000


def test_estimate_ate_zero_effect_ci_contains_zero():
    outcome, treatment, conf = _make_irm_data(2000, tau=0.0, seed=1)
    res = estimate_ate(outcome, treatment, conf, learner_params={"n_estimators": 100})
    assert res["ci_low"] < 0 < res["ci_high"]


def test_estimate_ate_degenerate_returns_nan():
    idx = pd.date_range("2010-01-01", periods=50)
    outcome = pd.Series(np.random.default_rng(0).normal(size=50), index=idx)
    treatment = pd.Series(np.zeros(50, dtype=int), index=idx)  # no treated units
    conf = pd.DataFrame({"x": np.zeros(50)}, index=idx)
    res = estimate_ate(outcome, treatment, conf)
    assert np.isnan(res["ate"])
    assert res["n_treated"] == 0


def test_estimate_ate_rejects_index_mismatch():
    idx1 = pd.date_range("2010-01-01", periods=10)
    idx2 = pd.date_range("2010-01-02", periods=10)
    o = pd.Series(np.zeros(10), index=idx1)
    t = pd.Series(np.zeros(10), index=idx2)
    c = pd.DataFrame({"x": np.zeros(10)}, index=idx1)
    with pytest.raises(ValueError, match="share an index"):
        estimate_ate(o, t, c)


def test_placebo_pvalue_in_range_for_null_effect():
    outcome, treatment, conf = _make_irm_data(1500, tau=0.0, seed=2)
    res = estimate_ate(outcome, treatment, conf, learner_params={"n_estimators": 60})
    out = placebo_pvalue(
        outcome,
        treatment,
        conf,
        res["ate"],
        n_trials=8,
        seed=2026,
        n_folds=3,
        learner_params={"n_estimators": 60},
    )
    assert 0.0 <= out["placebo_pvalue"] <= 1.0
    assert out["n_valid"] >= 1


def test_placebo_zero_trials_is_nan():
    outcome, treatment, conf = _make_irm_data(400, tau=0.1, seed=3)
    out = placebo_pvalue(outcome, treatment, conf, 0.1, n_trials=0)
    assert np.isnan(out["placebo_pvalue"])
    assert out["n_valid"] == 0.0


def test_estimate_cate_runs():
    pytest.importorskip("econml")
    from metals.models.causal import estimate_cate

    outcome, treatment, conf = _make_irm_data(1200, tau=0.4, seed=4)
    mods = pd.DataFrame({"m": np.random.default_rng(9).normal(size=1200)}, index=outcome.index)
    out = estimate_cate(outcome, treatment, conf, mods, n_estimators=100, min_obs=300)
    assert np.isfinite(out["cate_mean"])
    assert len(out["cate"]) == out["n_obs"]
    assert 0.0 <= out["frac_positive"] <= 1.0


def test_estimate_scenarios_table_shape():
    idx = pd.bdate_range("2010-01-01", periods=900)
    rng = np.random.default_rng(0)
    dxy = pd.Series(100 + np.cumsum(rng.normal(0, 0.3, 900)), index=idx)
    macro = pd.DataFrame(
        {
            "DTWEXBGS": dxy,
            "VIXCLS": 18 + rng.normal(0, 1, 900),
            "DGS10": 2.5 + rng.normal(0, 0.05, 900),
            "T10YIE": 2.0 + rng.normal(0, 0.05, 900),
        },
        index=idx,
    )
    prices = pd.DataFrame({"GC=F": 1800 + np.cumsum(rng.normal(0, 5, 900))}, index=idx)
    from metals.features.scenarios import ScenarioSpec

    dxy_spec = ScenarioSpec(
        id="dxy_up",
        name="x",
        definition_type="event",
        economic_family="usd",
        available=True,
        source_table="macro",
        source_field="DTWEXBGS",
        transform="pct_change",
        periods=5,
        rule="sigma_high",
        pct=None,
        k=1.5,
    )
    res = estimate_scenarios(
        [dxy_spec],
        prices=prices,
        macro=macro,
        metals=("GC=F",),
        horizons=(1, 5),
        placebo_trials=0,
        learner_params={"n_estimators": 60},
    )
    assert isinstance(res, CausalResult)
    assert list(res.ate.columns) == list(_ATE_COLUMNS)
    assert set(res.ate["horizon"]) == {1, 5}
    assert (res.ate["metal"] == "GC=F").all()
    assert len(res.ate) == 2
