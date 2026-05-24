"""Tests for spread and ratio feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metals.features.spreads import (
    compute_log_spread_changes,
    compute_ratios,
    compute_spread_zscores,
)


def _toy_prices(n: int = 300, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    cols = ["GC=F", "SI=F", "PL=F", "PA=F", "HG=F", "CL=F"]
    rets = rng.normal(loc=0.0, scale=0.01, size=(n, len(cols)))
    levels = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(levels, index=idx, columns=cols)


def test_ratios_named_and_finite():
    p = _toy_prices()
    r = compute_ratios(p)
    assert "Au_Ag_ratio" in r.columns
    assert "Pt_Pd_ratio" in r.columns
    assert "Au_Cu_ratio" in r.columns
    assert "Au_Oil_ratio" in r.columns
    assert r.dropna().notna().all().all()


def test_ratio_value_matches_division():
    p = _toy_prices()
    r = compute_ratios(p)
    manual = p["GC=F"].iloc[10] / p["SI=F"].iloc[10]
    assert r["Au_Ag_ratio"].iloc[10] == pytest.approx(manual)


def test_missing_ticker_yields_nan_column():
    p = _toy_prices().drop(columns=["CL=F"])
    r = compute_ratios(p)
    assert r["Au_Oil_ratio"].isna().all()


def test_log_spread_changes_warmup():
    p = _toy_prices()
    r = compute_ratios(p)
    lc = compute_log_spread_changes(r, horizons=(5,))
    assert lc["Au_Ag_logchg_5d"].iloc[:5].isna().all()
    assert lc["Au_Ag_logchg_5d"].iloc[5:].notna().any()


def test_zscores_centered_at_zero_on_stationary():
    idx = pd.date_range("2020-01-01", periods=600, freq="B")
    rng = np.random.default_rng(7)
    stationary = pd.DataFrame(
        {"Au_Ag_ratio": rng.normal(50, 2, 600)},
        index=idx,
    )
    z = compute_spread_zscores(stationary, window=252)
    valid = z.dropna()
    assert abs(valid["Au_Ag_z_252d"].mean()) < 0.5
