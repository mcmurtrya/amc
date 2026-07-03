"""Per-fold regime features from the Phase 3 clustering, leak-free.

The cluster→forward-vol lift experiment (results/phase3_cluster_lift_design.md)
needs, for every walk-forward split, regime features whose generating process
saw nothing past the split's train window. :func:`build_regime_features` is
that unit of work: fit UMAP + HDBSCAN on context rows strictly before a
``boundary``, assign *all* rows through one uniform ``approximate_predict``
path, and return ONLY model-ready columns (safe to concat into a feature
matrix verbatim — raw cluster ids are deliberately not returned, because
HDBSCAN ids are arbitrary and fold-local and must never reach a model as an
ordinal):

- ``regime_confidence``  HDBSCAN assignment strength in [0, 1]
- ``regime_<cid>`` / ``regime_noise``  one-hot over the train-fit cluster ids
- ``regime_target_mean`` (optional) train-window mean of a target per cluster —
  a target encoding, so it is **purged**: only train rows whose index is
  before ``boundary - target_purge_days`` contribute, because forward-looking
  targets near the boundary peek past it. Size the purge with
  :func:`purge_days_for`, per target.

``boundary`` is EXCLUSIVE (rows ``< boundary`` are train), matching
``eval.cv.Split.train_end`` semantics. NB ``build_context``'s ``pca_fit_until``
mask is INCLUSIVE — callers building an embedding context per fold must pass
the last trading day strictly before the boundary, not the boundary itself.
Cluster ids are only comparable within one call — HDBSCAN labels are not
stable across fits, so every fold gets its own one-hot vocabulary and its own
downstream model fit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from metals.features.leakage import assert_chronological
from metals.models.clustering import ClusteringConfig, assign_clusters, fit_clustering


def purge_days_for(target_horizon: int, vol_window: int, slack_days: int = 10) -> int:
    """Calendar-day purge covering a forward realized-vol target's look-ahead.

    The target at row t uses returns through trading day
    ``t + target_horizon + vol_window - 1``; converting trading to calendar
    days at 7/5 and adding ``slack_days`` for holidays gives a purge that is
    sufficient for any (h, w) — e.g. 44 for the (5, 20) primary target and 65
    for the (20, 20) secondary, where a flat 45 would silently under-purge.
    """
    trading = target_horizon + vol_window - 1
    return math.ceil(trading * 7 / 5) + slack_days


@dataclass(frozen=True)
class RegimeFeatureConfig:
    """How per-fold regime features are constructed."""

    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    # Purge window for the target encoding (calendar days). Must exceed the
    # target's forward look — derive it from the target spec with
    # ``purge_days_for`` rather than trusting this default (which covers the
    # pre-registered h=5, w=20 primary target only).
    target_purge_days: int = 45


def build_regime_features(
    context: pd.DataFrame,
    boundary: str | pd.Timestamp,
    target: pd.Series | None = None,
    config: RegimeFeatureConfig | None = None,
) -> pd.DataFrame:
    """Fit clustering on ``context`` rows strictly before ``boundary``; emit features.

    Parameters
    ----------
    context : DataFrame
        The contextual feature frame from ``build_context`` (rows with NaN are
        dropped here, mirroring the cluster stage). Must be chronological.
        Callers building context with a PCA must pass the same boundary as
        ``pca_fit_until`` upstream.
    boundary : Timestamp
        Exclusive end of the training window (``Split.train_end``). Rows at or
        after the boundary are never seen by the clustering fit or the target
        encoding.
    target : Series, optional
        Forward-looking target aligned to ``context``'s index. When given, a
        ``regime_target_mean`` column is added (purged; see module docstring).
    config : RegimeFeatureConfig

    Returns
    -------
    DataFrame indexed like ``context.dropna()``.
    """
    cfg = config or RegimeFeatureConfig()
    boundary = pd.Timestamp(boundary)

    rows = context.dropna()
    assert_chronological(rows)
    train = rows.loc[rows.index < boundary]
    if train.empty:
        raise ValueError(f"build_regime_features: no context rows before {boundary}.")

    pipeline = fit_clustering(train, config=cfg.clustering)
    assigned = assign_clusters(pipeline, rows).set_index("timestamp_utc")
    # Fold-local ids stay internal: they are arbitrary HDBSCAN labels and must
    # only reach the model through the one-hots / encoding below.
    ids = assigned["cluster_id"].astype(int)

    out = pd.DataFrame(index=rows.index)
    out["regime_confidence"] = assigned["confidence"].astype(float)

    # One-hot vocabulary comes from the train fit only; approximate_predict can
    # produce exactly these ids or -1, so unseen columns cannot appear at test.
    train_ids = sorted({int(x) for x in pipeline.hdbscan_model.labels_} | {-1})
    for cid in train_ids:
        name = "regime_noise" if cid == -1 else f"regime_{cid}"
        out[name] = (ids == cid).astype(float)

    if target is not None:
        purge_cut = boundary - pd.Timedelta(days=cfg.target_purge_days)
        enc = target.reindex(rows.index).loc[rows.index < purge_cut]
        enc_ids = ids.loc[enc.index]
        means = enc.groupby(enc_ids).mean()
        fallback = float(enc.mean())
        out["regime_target_mean"] = ids.map(means).fillna(fallback).astype(float)

    return out
