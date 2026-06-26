"""Daily contextual feature vector for Phase 3 step 3.9.

The contextual vector is a single row per (date, target_metal) that bundles
every input the clustering pipeline cares about:

    - macro state                (TIPS, DXY, VIX, GPR — levels and changes)
    - recent returns / vol       (5- and 20-day)
    - text mean embedding         (PCA-reduced, fit on the train window only)
    - topic prevalences          (wide vector; themes-via-SQL by default)
    - COT positioning             (z-scored over 1y)

This is the input to UMAP + HDBSCAN in ``metals.models.clustering``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from metals.features.leakage import assert_chronological
from metals.features.macro import compute_macro_features


@dataclass(frozen=True)
class ContextConfig:
    """How the contextual vector is constructed."""

    target_metal: str = "gold"
    embedding_pca_dims: int = 16
    rank_window: int = 252


def _pca_fit_transform(
    matrix: np.ndarray,
    fit_mask: np.ndarray,
    n_components: int,
    seed: int = 42,
) -> tuple[np.ndarray, object]:
    """Whitening PCA fit on ``matrix[fit_mask]`` and applied to all of ``matrix``.

    Fitting on a strict prefix (the clustering train window) and transforming the
    full series keeps the projection free of look-ahead: the centring mean, the
    components, and the whitening scale are all derived from training rows only.
    Returns the reduced matrix (all rows) and the fitted estimator.
    """
    from sklearn.decomposition import PCA

    fit_rows = matrix[fit_mask]
    n_components = min(n_components, matrix.shape[1], max(1, fit_rows.shape[0] - 1))
    pca = PCA(n_components=n_components, whiten=True, random_state=seed)
    pca.fit(fit_rows)
    return pca.transform(matrix), pca


def _pca_reduce(matrix: np.ndarray, n_components: int, seed: int = 42) -> tuple[np.ndarray, object]:
    """Whitening PCA fit and applied on the whole matrix (single in-sample block).

    The leakage-safe path in :func:`build_context` uses :func:`_pca_fit_transform`
    with an explicit train mask; this wrapper is for callers that legitimately
    have only one in-sample block (and for unit tests).
    """
    return _pca_fit_transform(matrix, np.ones(matrix.shape[0], dtype=bool), n_components, seed)


def _stack_embeddings(df: pd.DataFrame, dim_hint: int | None = None) -> np.ndarray:
    """Stack a ``mean_embedding`` column of ndarrays into a 2D matrix."""
    rows = []
    for v in df["mean_embedding"]:
        if v is None:
            rows.append(np.zeros(dim_hint or 1, dtype=np.float32))
        else:
            rows.append(np.asarray(v, dtype=np.float32))
    if not rows:
        return np.zeros((0, dim_hint or 1), dtype=np.float32)
    if dim_hint is None:
        # All non-None rows are assumed to share dimension.
        non_zero = [r for r in rows if r.size > 1]
        if non_zero:
            dim_hint = non_zero[0].size
            rows = [r if r.size == dim_hint else np.zeros(dim_hint, dtype=np.float32) for r in rows]
    return np.vstack(rows).astype(np.float32)


def build_context(
    prices: pd.DataFrame,
    macro_wide: pd.DataFrame,
    cot_positioning: pd.DataFrame | None = None,
    text_daily: pd.DataFrame | None = None,
    topic_prevalence: pd.DataFrame | None = None,
    pca_fit_until: str | pd.Timestamp | None = None,
    config: ContextConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Assemble the daily contextual feature DataFrame plus fit artifacts.

    Parameters
    ----------
    prices : DataFrame
        Wide price frame indexed by ``timestamp_utc``.
    macro_wide : DataFrame
        Wide macro frame, will be reindexed to ``prices.index`` and ffilled.
    cot_positioning : DataFrame, optional
        Long-format COT data with at minimum ``timestamp_utc``, ``metal``,
        and a ``net_managed_money`` column. Optional — silently zero-filled
        if missing.
    text_daily : DataFrame, optional
        From ``metals.features.text_daily.load_daily()``. Optional.
    topic_prevalence : DataFrame, optional
        Wide topic prevalence from ``load_topic_prevalence_wide()``. Optional.
    pca_fit_until : str or Timestamp, optional
        Train-window boundary for the text-embedding PCA. The projection is fit
        only on rows with timestamp <= this date and then applied to the whole
        series, so ``text_pca_*`` carries no look-ahead. Pass the same date used
        as the clustering ``train_until``. When ``None`` the PCA is fit on the
        full sample (in-sample only — leaks future covariance; avoid for any
        walk-forward run).
    config : ContextConfig

    Returns
    -------
    (context_df, artifacts)
        ``context_df`` indexed by ``timestamp_utc`` with all input features.
        ``artifacts`` carries the fitted PCA so the same projection can be
        re-applied at inference time.
    """
    cfg = config or ContextConfig()

    # 1) Macro features (already daily-aligned via reindex + ffill in compute_macro)
    macro_aligned = macro_wide.reindex(prices.index).ffill()
    macro_feats = compute_macro_features(macro_aligned)

    # 2) Recent returns / vol on the target metal
    if cfg.target_metal == "gold":
        ticker = "GC=F"
    elif cfg.target_metal == "silver":
        ticker = "SI=F"
    elif cfg.target_metal == "platinum":
        ticker = "PL=F"
    elif cfg.target_metal == "palladium":
        ticker = "PA=F"
    else:
        raise ValueError(f"Unknown target_metal {cfg.target_metal!r}.")
    if ticker not in prices.columns:
        raise ValueError(f"{ticker} not present in prices.")
    log_p = np.log(prices[ticker].astype(float))
    own_state = pd.DataFrame(index=prices.index)
    own_state[f"{ticker}_ret_5d"]  = log_p - log_p.shift(5)
    own_state[f"{ticker}_ret_20d"] = log_p - log_p.shift(20)
    ret_1d = log_p - log_p.shift(1)
    own_state[f"{ticker}_rvol_20d"] = ret_1d.rolling(20, min_periods=20).std() * float(np.sqrt(252))

    # 3) Text features for this metal: mean embedding PCA + dispersion + count
    artifacts: dict[str, object] = {}
    text_part = pd.DataFrame(index=prices.index)
    if text_daily is not None and not text_daily.empty:
        # Text is a single shared daily 'market' news-state (the per-metal text
        # axis is redundant — see results/phase3_gdelt_data_assessment.md §1/§7);
        # use it for every target metal, exactly as topic prevalence is shared.
        from metals.features.text_daily import MARKET
        shared = text_daily[text_daily["metal"] == MARKET]
        if shared.empty and text_daily["metal"].nunique() == 1:
            shared = text_daily  # single-series frame under a non-'market' label
        sub = shared.set_index("timestamp_utc")
        sub.index = pd.to_datetime(sub.index)
        sub = sub.reindex(prices.index)
        text_part["n_articles"] = sub["n_articles"].fillna(0).astype(float)
        text_part["embedding_dispersion"] = sub["embedding_dispersion"]
        # Reduce mean_embedding -> PCA(d). Fit the projection on the train window
        # only (rows up to ``pca_fit_until``) and apply it to every row, so the
        # text_pca_* columns carry no look-ahead; fitting on the full sample
        # would leak future covariance into past coordinates.
        present = sub["mean_embedding"].notna()
        if present.any():
            matrix = _stack_embeddings(sub[present])
            present_idx = present[present].index
            if pca_fit_until is not None:
                fit_mask = np.asarray(present_idx <= pd.Timestamp(pca_fit_until))
            else:
                fit_mask = np.ones(len(present_idx), dtype=bool)
            if int(fit_mask.sum()) >= 2:
                reduced, pca = _pca_fit_transform(matrix, fit_mask, cfg.embedding_pca_dims)
                for k in range(reduced.shape[1]):
                    col = pd.Series(np.nan, index=prices.index)
                    col.loc[present_idx] = reduced[:, k]
                    text_part[f"text_pca_{k}"] = col
                artifacts["text_pca"] = pca

    # 4) Topic prevalences
    topic_part = pd.DataFrame(index=prices.index)
    if topic_prevalence is not None and not topic_prevalence.empty:
        tp = topic_prevalence.reindex(prices.index).fillna(0.0)
        topic_part = tp

    # 5) COT positioning z-scores
    cot_part = pd.DataFrame(index=prices.index)
    if cot_positioning is not None and not cot_positioning.empty:
        sub = cot_positioning[cot_positioning["metal"] == cfg.target_metal]
        if not sub.empty and "net_managed_money" in sub.columns:
            sub = sub[["timestamp_utc", "net_managed_money"]].copy()
            sub["timestamp_utc"] = pd.to_datetime(sub["timestamp_utc"])
            sub = sub.set_index("timestamp_utc").sort_index()
            sub = sub.reindex(prices.index).ffill()
            roll = sub["net_managed_money"]
            mean = roll.rolling(cfg.rank_window, min_periods=cfg.rank_window).mean()
            std = roll.rolling(cfg.rank_window, min_periods=cfg.rank_window).std()
            cot_part["cot_managed_money_z"] = (roll - mean) / std

    parts = [macro_feats, own_state, text_part, topic_part, cot_part]
    context = pd.concat([p for p in parts if not p.empty], axis=1)
    # Clustering has no forward target, but the context index must still be
    # strictly increasing and unique for walk-forward training to be honest.
    assert_chronological(context)
    return context, artifacts
