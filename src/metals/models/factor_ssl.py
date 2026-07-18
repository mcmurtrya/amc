"""Classical low-rank joint factorization for Phase 8 Stage A (``LRJ-Metals``).

Phase 8 §2.1 / §3.3. Two views of each trading day — View A (price/macro/COT,
as-of close, unlagged) and View B (news, lagged one trading day) — are each
reduced to their own **train-fitted whitened principal components**, then joined
by ``PLSCanonical`` (covariance-maximising; preferred over the numerically
unstable ``CCA`` at these dimensions) fit on the **training prefix only**. The
frozen representation ``Z`` is the concatenation of the price-side and news-side
canonical scores (``u_*`` and ``v_*``), one row per trading day.

This is genuinely self-supervised: no labels touch the factorization; it learns
the shared price<->news latent purely from cross-view covariance. It adds no
torch — everything here is sklearn, mirroring the fit / save / load idiom of
:mod:`metals.models.clustering`.

Contract: :func:`fit_factor_ssl` and :func:`transform` require **finite** inputs.
Impute the missing-news-day NaNs with
:class:`metals.features.ssl_views.TrainOnlyImputer` (fit on the fold's
``train_idx``) before calling, exactly as the per-fold protocol specifies. The
canonical correlations must be read on **test** rows (:func:`canonical_correlations`),
never on train, where they are maximised by construction.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parents[3] / "data" / "processed" / "factor_ssl"


@dataclass(frozen=True)
class FactorSSLConfig:
    """Hyperparameters for the joint factorization.

    ``k_price`` / ``k_text`` are the per-view whitened-PCA ranks (capped at what
    the train fold can estimate); ``n_canonical`` is the number of canonical
    pairs kept (K ≈ 2-4 on a few-hundred effective samples).
    """

    k_price: int = 12
    k_text: int = 8
    n_canonical: int = 3
    seed: int = 42


@dataclass
class FactorSSL:
    """A fitted joint factorization: per-view scaler + whitened PCA, plus PLS."""

    config: FactorSSLConfig
    # sklearn estimators have no usable static types; ``Any`` mirrors the
    # ``ClusterPipeline`` convention (the pickled objects carry real runtime types).
    scaler_price: Any
    pca_price: Any
    scaler_text: Any
    pca_text: Any
    pls: Any
    feature_names_price: list[str]
    feature_names_text: list[str]
    n_components: int
    model_version: str
    fit_at: str


def _finite(frame: pd.DataFrame, name: str) -> np.ndarray:
    """Return ``frame`` as a float array, raising if it is not fully finite."""
    arr = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(
            f"{name} contains NaN/inf; impute with TrainOnlyImputer before factorizing."
        )
    return arr


def _cap_components(requested: int, n_rows: int, n_features: int) -> int:
    """Clamp a requested component count to what the block can estimate."""
    return max(1, min(requested, n_features, n_rows - 1))


def _utcnow_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _model_version(prefix: str = "phase8") -> str:
    return f"{prefix}_{datetime.utcnow():%Y%m%d_%H%M%S}"


def fit_factor_ssl(
    z_p: pd.DataFrame,
    z_t: pd.DataFrame,
    train_idx: np.ndarray | Sequence[int],
    config: FactorSSLConfig | None = None,
    model_version: str | None = None,
) -> FactorSSL:
    """Fit the joint factorization on the training rows only.

    ``z_p`` / ``z_t`` are the (imputed, finite) full-length views; only rows in
    ``train_idx`` are used to fit the scalers, the whitened PCAs, and the PLS.
    """
    from sklearn.cross_decomposition import PLSCanonical
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    cfg = config or FactorSSLConfig()
    tr = np.asarray(train_idx, dtype=int)
    if tr.size == 0:
        raise ValueError("fit_factor_ssl: empty train_idx.")
    if not z_p.index.equals(z_t.index):
        raise ValueError("fit_factor_ssl: z_p and z_t must share an index.")

    xp = _finite(z_p.iloc[tr], "z_p")
    xt = _finite(z_t.iloc[tr], "z_t")

    k_p = _cap_components(cfg.k_price, xp.shape[0], xp.shape[1])
    k_t = _cap_components(cfg.k_text, xt.shape[0], xt.shape[1])

    scaler_p = StandardScaler().fit(xp)
    pca_p = PCA(n_components=k_p, whiten=True, random_state=cfg.seed).fit(scaler_p.transform(xp))
    scaler_t = StandardScaler().fit(xt)
    pca_t = PCA(n_components=k_t, whiten=True, random_state=cfg.seed).fit(scaler_t.transform(xt))

    p_p = pca_p.transform(scaler_p.transform(xp))
    p_t = pca_t.transform(scaler_t.transform(xt))

    k = max(1, min(cfg.n_canonical, k_p, k_t))
    pls = PLSCanonical(n_components=k).fit(p_p, p_t)

    return FactorSSL(
        config=cfg,
        scaler_price=scaler_p,
        pca_price=pca_p,
        scaler_text=scaler_t,
        pca_text=pca_t,
        pls=pls,
        feature_names_price=[str(c) for c in z_p.columns],
        feature_names_text=[str(c) for c in z_t.columns],
        n_components=k,
        model_version=model_version or _model_version(),
        fit_at=_utcnow_iso(),
    )


def _project(
    model: FactorSSL, z_p: pd.DataFrame, z_t: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    xp = z_p.reindex(columns=model.feature_names_price, fill_value=0.0)
    xt = z_t.reindex(columns=model.feature_names_text, fill_value=0.0)
    p_p = model.pca_price.transform(model.scaler_price.transform(_finite(xp, "z_p")))
    p_t = model.pca_text.transform(model.scaler_text.transform(_finite(xt, "z_t")))
    u, v = model.pls.transform(p_p, p_t)
    return np.asarray(u), np.asarray(v)


def transform(model: FactorSSL, z_p: pd.DataFrame, z_t: pd.DataFrame) -> pd.DataFrame:
    """Project rows onto the frozen canonical axes → ``Z`` of shape ``(T, 2K)``.

    Columns ``u_0..u_{K-1}`` are the price-side scores, ``v_0..v_{K-1}`` the
    news-side scores. Row index is preserved from ``z_p``.
    """
    if not z_p.index.equals(z_t.index):
        raise ValueError("transform: z_p and z_t must share an index.")
    u, v = _project(model, z_p, z_t)
    cols = [f"u_{j}" for j in range(model.n_components)]
    cols += [f"v_{j}" for j in range(model.n_components)]
    z = np.concatenate([u, v], axis=1)
    return pd.DataFrame(z, index=z_p.index, columns=cols)


def canonical_correlations(model: FactorSSL, z_p: pd.DataFrame, z_t: pd.DataFrame) -> np.ndarray:
    """Per-component canonical correlation ρ_j between ``u_j`` and ``v_j``.

    Report this on **test** rows: on train rows PLS maximises it by construction.
    """
    u, v = _project(model, z_p, z_t)
    cors = np.empty(model.n_components, dtype=float)
    for j in range(model.n_components):
        uj, vj = u[:, j], v[:, j]
        if uj.std() == 0 or vj.std() == 0:
            cors[j] = np.nan
        else:
            cors[j] = float(np.corrcoef(uj, vj)[0, 1])
    return cors


def save_factor_ssl(model: FactorSSL) -> Path:
    """Persist a fitted factorization (pickle + human-readable JSON sidecar)."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{model.model_version}.pkl"
    with path.open("wb") as f:
        pickle.dump(model, f)
    sidecar = MODEL_DIR / f"{model.model_version}.json"
    with sidecar.open("w") as f:
        json.dump(
            {
                "model_version": model.model_version,
                "fit_at": model.fit_at,
                "config": asdict(model.config),
                "n_components": model.n_components,
                "n_features_price": len(model.feature_names_price),
                "n_features_text": len(model.feature_names_text),
            },
            f,
            indent=2,
        )
    return path


def load_factor_ssl(model_version: str) -> FactorSSL:
    path = MODEL_DIR / f"{model_version}.pkl"
    with path.open("rb") as f:
        return pickle.load(f)
