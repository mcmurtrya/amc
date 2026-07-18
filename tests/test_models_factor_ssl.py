"""Tests for the Phase 8 Stage-A joint factorization (metals.models.factor_ssl)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metals.models import factor_ssl
from metals.models.factor_ssl import (
    FactorSSLConfig,
    canonical_correlations,
    fit_factor_ssl,
    load_factor_ssl,
    save_factor_ssl,
    transform,
)


def _planted(t: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two views sharing one latent factor ``f`` (visible in p0 and t0)."""
    rng = np.random.default_rng(seed)
    f = rng.normal(size=t)
    idx = pd.date_range("2015-01-01", periods=t, freq="B")
    z_p = pd.DataFrame(
        {
            "p0": f + 0.1 * rng.normal(size=t),
            "p1": 0.5 * f + rng.normal(size=t),
            "p2": rng.normal(size=t),
        },
        index=idx,
    )
    z_t = pd.DataFrame(
        {"t0": f + 0.1 * rng.normal(size=t), "t1": rng.normal(size=t)},
        index=idx,
    )
    return z_p, z_t


def test_fit_transform_shapes_and_recovers_shared_axis() -> None:
    z_p, z_t = _planted()
    train_idx = np.arange(0, 300)
    model = fit_factor_ssl(z_p, z_t, train_idx, FactorSSLConfig(k_price=3, k_text=2, n_canonical=1))
    z = transform(model, z_p, z_t)
    assert z.shape == (400, 2)
    assert list(z.columns) == ["u_0", "v_0"]
    cc_train = canonical_correlations(model, z_p.iloc[train_idx], z_t.iloc[train_idx])
    assert cc_train[0] > 0.7  # recovers the planted shared latent


def test_fit_is_train_only() -> None:
    z_p, z_t = _planted()
    train_idx = np.arange(0, 300)
    model = fit_factor_ssl(z_p, z_t, train_idx, FactorSSLConfig(k_price=3, k_text=2))
    # The scaler mean must be the TRAIN-block mean, never the full-sample mean.
    np.testing.assert_allclose(
        model.scaler_price.mean_,
        z_p.iloc[train_idx].to_numpy().mean(axis=0),
        rtol=1e-9,
    )


def test_component_capping_never_exceeds_rank() -> None:
    z_p, z_t = _planted()
    model = fit_factor_ssl(
        z_p, z_t, np.arange(0, 300), FactorSSLConfig(k_price=50, k_text=50, n_canonical=10)
    )
    assert model.n_components <= 2  # z_t has only 2 features


def test_save_load_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(factor_ssl, "MODEL_DIR", tmp_path)
    z_p, z_t = _planted()
    model = fit_factor_ssl(
        z_p,
        z_t,
        np.arange(0, 300),
        FactorSSLConfig(k_price=3, k_text=2, n_canonical=1),
        model_version="unit_test",
    )
    save_factor_ssl(model)
    loaded = load_factor_ssl("unit_test")
    pd.testing.assert_frame_equal(transform(model, z_p, z_t), transform(loaded, z_p, z_t))


def test_finite_contract_rejects_nan_in_train() -> None:
    z_p, z_t = _planted()
    z_t = z_t.copy()
    z_t.iloc[5, 0] = np.nan  # row 5 is inside the train prefix
    with pytest.raises(ValueError, match="NaN"):
        fit_factor_ssl(z_p, z_t, np.arange(0, 300), FactorSSLConfig(k_price=3, k_text=2))
