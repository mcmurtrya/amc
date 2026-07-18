"""Tests for the Phase 8 low-rank probing playbook (metals.eval.probes)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metals.eval.probes import (
    block_bootstrap_ci,
    block_permutation_pvalue,
    incremental_ic,
    information_coefficient,
    linear_probe,
    residualize_on,
)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    return information_coefficient(a, b, method="pearson")


def test_information_coefficient_perfect_and_degenerate() -> None:
    x = np.arange(10, dtype=float)
    assert information_coefficient(x, 2 * x) == pytest.approx(1.0)
    assert np.isnan(information_coefficient([1.0], [1.0]))


def test_residualize_on_is_fit_on_train_only() -> None:
    rng = np.random.default_rng(0)
    n = 200
    c = rng.normal(size=n)
    y = 3.0 * c + rng.normal(size=n) * 0.01
    resid = residualize_on(y, pd.DataFrame({"c": c}), np.arange(0, 150))
    # y is ~3c; a train-fit slope of 3 leaves near-zero residual on the held-out tail.
    assert np.nanstd(resid[150:]) < 0.1


def test_incremental_ic_detects_and_rejects() -> None:
    rng = np.random.default_rng(1)
    n = 500
    price_factor = rng.normal(size=n)
    news_factor = rng.normal(size=n)
    y = price_factor + news_factor + 0.1 * rng.normal(size=n)
    x_price = pd.DataFrame({"pf": price_factor, "noise": rng.normal(size=n)})
    tr, te = np.arange(0, 350), np.arange(350, 500)

    news_real = news_factor + 0.1 * rng.normal(size=n)
    assert incremental_ic(y, news_real, x_price, tr, te) > 0.3

    news_redundant = 2.0 * price_factor + 0.01 * rng.normal(size=n)
    assert abs(incremental_ic(y, news_redundant, x_price, tr, te)) < 0.2


def test_linear_probe_regression_recovers_signal() -> None:
    rng = np.random.default_rng(2)
    n = 400
    z = rng.normal(size=(n, 3))
    y = z[:, 0] * 2.0 + rng.normal(size=n) * 0.1
    out = linear_probe(
        z, y, np.arange(0, 250), np.arange(250, 320), np.arange(320, 400), task="reg"
    )
    assert out["test_ic"] > 0.8


def test_linear_probe_binary_auc() -> None:
    rng = np.random.default_rng(3)
    n = 400
    z = rng.normal(size=(n, 3))
    y = z[:, 0] + rng.normal(size=n) * 0.5
    out = linear_probe(
        z, y, np.arange(0, 250), np.arange(250, 320), np.arange(320, 400), task="bin"
    )
    assert out["test_auc"] > 0.7
    assert 0.0 <= out["base_rate"] <= 1.0


def test_block_permutation_separates_signal_from_noise() -> None:
    rng = np.random.default_rng(4)
    n = 300
    x = rng.normal(size=n)
    y_corr = x + rng.normal(size=n) * 0.2
    y_indep = rng.normal(size=n)
    p_corr = block_permutation_pvalue(_pearson, x, y_corr, block_len=10, n_perm=200)
    p_indep = block_permutation_pvalue(_pearson, x, y_indep, block_len=10, n_perm=200)
    assert p_corr < 0.05 < p_indep


def test_block_bootstrap_ci_brackets_point() -> None:
    rng = np.random.default_rng(5)
    n = 300
    x = rng.normal(size=n)
    y = x + rng.normal(size=n) * 0.3
    point = _pearson(x, y)
    lo, hi = block_bootstrap_ci(_pearson, x, y, block_len=10, n_boot=300)
    assert lo < point < hi
    assert lo > 0.0
