"""View A / View B assembly for the Phase 8 low-rank joint factorization.

Phase 8 §3.2. The classical joint factorization (``metals.models.factor_ssl``)
needs two views of each trading day:

- **View A** (``Z_p``): the price / macro / COT state — as-of close, **unlagged**.
- **View B** (``Z_t``): the news state — article count, tone means, topic
  prevalences (and, once the embedding backfill lands, embedding dispersion and
  the ``text_pca_*`` block) — **already lagged one trading day** inside
  ``build_context``.

We reuse :func:`metals.features.context.build_context` rather than re-deriving
features, because it already bakes in (a) the one-trading-day text lag
(``context.py`` line 183) and (b) the train-only embedding PCA (``pca_fit_until``).
This module only (a) partitions its columns into the two views and (b) supplies
the **train-prefix-only** imputer for missing-news days: PLS/CCA cannot ingest
NaN, and a global mean-fill would leak future information into past rows.

First cut (Phase 8 §7 step 1): call with ``include_embeddings=False`` so View B
is tone + article count + topic prevalences only — the channels populated today
(``mean_embedding`` is still all-NULL, so dispersion and ``text_pca_*`` are
absent until ``scripts/materialize_day_embeddings.py`` runs).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from metals.features.context import ContextConfig, build_context

# Exact-match text (View B) columns emitted by ``build_context``. Everything not
# matched here or by ``_TEXT_PREFIXES`` is price/macro/COT (View A).
_TEXT_EXACT: frozenset[str] = frozenset(
    {
        "n_articles",
        "mean_tone_overall",
        "mean_tone_positive",
        "mean_tone_negative",
        "embedding_dispersion",
    }
)
# ``text_pca_<k>`` (embedding PCA) and ``topic_<id>`` (topic prevalence) blocks.
_TEXT_PREFIXES: tuple[str, ...] = ("text_pca_", "topic_")


def is_text_column(name: str) -> bool:
    """True iff ``name`` is a View B (news) column of a ``build_context`` frame."""
    return name in _TEXT_EXACT or name.startswith(_TEXT_PREFIXES)


def partition_columns(context: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split a context frame's columns into ``(price_cols, text_cols)``."""
    price_cols = [c for c in context.columns if not is_text_column(str(c))]
    text_cols = [c for c in context.columns if is_text_column(str(c))]
    return price_cols, text_cols


def split_views(context: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition a context frame into ``(Z_p, Z_t)`` — View A and View B."""
    price_cols, text_cols = partition_columns(context)
    return context[price_cols].copy(), context[text_cols].copy()


@dataclass
class TrainOnlyImputer:
    """Per-column NaN fill whose fill values are estimated on the train prefix.

    Fitting on ``train_idx`` only is what keeps missing-news-day imputation
    leakage-free: a global mean-fill would let a future no-news day's neighbours
    set a past row's value. Columns that are entirely NaN on the train prefix
    fall back to ``0.0``.
    """

    fill_values: pd.Series

    @classmethod
    def fit(cls, frame: pd.DataFrame, train_idx: np.ndarray | Sequence[int]) -> TrainOnlyImputer:
        train = frame.iloc[np.asarray(train_idx, dtype=int)]
        means = train.mean(axis=0, skipna=True)
        means = means.reindex(frame.columns).fillna(0.0).astype(float)
        return cls(fill_values=means)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        fills = self.fill_values.reindex(frame.columns).fillna(0.0)
        return frame.fillna(fills)


def assemble_views(
    prices: pd.DataFrame,
    macro_wide: pd.DataFrame,
    cot_positioning: pd.DataFrame | None = None,
    text_daily: pd.DataFrame | None = None,
    topic_prevalence: pd.DataFrame | None = None,
    *,
    train_end: str | pd.Timestamp | None,
    target_metal: str = "gold",
    include_embeddings: bool = False,
    rank_window: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build ``(Z_p, Z_t, artifacts)`` for one walk-forward fold.

    ``train_end`` is the fold's ``Split.train_end`` and is passed straight to
    ``build_context`` as ``pca_fit_until`` so the (optional) embedding PCA is fit
    on the train prefix only. The returned views are **raw** — still carrying
    warmup / missing-news NaNs; the caller fits a :class:`TrainOnlyImputer` on the
    fold's ``train_idx`` and applies it before factorization.
    """
    cfg = ContextConfig(
        target_metal=target_metal,
        include_embeddings=include_embeddings,
        rank_window=rank_window,
    )
    context, artifacts = build_context(
        prices,
        macro_wide,
        cot_positioning=cot_positioning,
        text_daily=text_daily,
        topic_prevalence=topic_prevalence,
        pca_fit_until=train_end,
        config=cfg,
    )
    z_p, z_t = split_views(context)
    return z_p, z_t, artifacts
