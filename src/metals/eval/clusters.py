"""Cluster-analysis utilities for Phase 3 step 3.12.

For each cluster produced by ``metals.models.clustering``, compute:

- forward-return distributions per metal at horizons 1, 5, 20, 60 days
- dominant topics (top-N by mean prevalence across the cluster's days)
- representative date list and example headlines

The functions are pure — they take dataframes in, return dataframes out —
so they're easy to test on synthetic clusters and easy to feed into the
Phase 5 triangulation tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 20, 60)


def forward_returns(
    prices: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Per-ticker forward log returns at each horizon.

    Output columns: ``{ticker}_fwd_{h}d``. Indexed by ``timestamp_utc``.
    """
    log_p = np.log(prices.astype(float))
    out = {}
    for h in horizons:
        diff = log_p.shift(-h) - log_p
        for c in prices.columns:
            out[f"{c}_fwd_{h}d"] = diff[c]
    return pd.DataFrame(out, index=prices.index)


def cluster_forward_stats(
    assignments: pd.DataFrame,
    forward: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """For each (cluster_id, horizon, ticker), mean / std / hit-rate / n.

    Hit-rate is the fraction of cluster days with forward return > 0.

    ``assignments`` must contain ``timestamp_utc, cluster_id``.
    """
    if assignments.empty or forward.empty:
        return pd.DataFrame(columns=["cluster_id", "ticker", "horizon",
                                     "n", "mean", "std", "hit_rate"])
    df = assignments.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    df = df.set_index("timestamp_utc")
    joined = df.join(forward, how="inner")

    tickers = sorted({c.rsplit("_fwd_", 1)[0] for c in forward.columns if "_fwd_" in c})
    rows = []
    for cid, g in joined.groupby("cluster_id"):
        for tk in tickers:
            for h in horizons:
                col = f"{tk}_fwd_{h}d"
                if col not in g.columns:
                    continue
                vals = g[col].dropna()
                if vals.empty:
                    continue
                rows.append({
                    "cluster_id": int(cid),
                    "ticker":     tk,
                    "horizon":    h,
                    "n":          int(len(vals)),
                    "mean":       float(vals.mean()),
                    "std":        float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
                    "hit_rate":   float((vals > 0).mean()),
                })
    return pd.DataFrame(rows)


def dominant_topics(
    assignments: pd.DataFrame,
    topic_prevalence_wide: pd.DataFrame,
    top_k: int = 3,
) -> pd.DataFrame:
    """Per-cluster top-K topics by mean prevalence on cluster days.

    Returns ``cluster_id, rank, topic_col, mean_prevalence``.
    """
    if assignments.empty or topic_prevalence_wide.empty:
        return pd.DataFrame(columns=["cluster_id", "rank", "topic_col", "mean_prevalence"])

    df = assignments.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    df = df.set_index("timestamp_utc")
    joined = df.join(topic_prevalence_wide, how="inner")
    topic_cols = [c for c in topic_prevalence_wide.columns]

    rows = []
    for cid, g in joined.groupby("cluster_id"):
        means = g[topic_cols].mean().sort_values(ascending=False).head(top_k)
        for rank, (col, prev) in enumerate(means.items(), start=1):
            rows.append({
                "cluster_id": int(cid),
                "rank":       rank,
                "topic_col":  col,
                "mean_prevalence": float(prev),
            })
    return pd.DataFrame(rows)


def representative_dates(
    assignments: pd.DataFrame,
    confidence_col: str = "confidence",
    per_cluster: int = 10,
) -> pd.DataFrame:
    """Per-cluster top-N dates by HDBSCAN membership confidence.

    Useful for hand-labeling and for sourcing example headlines.
    """
    if assignments.empty:
        return assignments
    work = assignments.copy()
    work["timestamp_utc"] = pd.to_datetime(work["timestamp_utc"])
    if confidence_col not in work.columns:
        work[confidence_col] = 1.0
    out = (
        work.sort_values(["cluster_id", confidence_col], ascending=[True, False])
            .groupby("cluster_id", as_index=False)
            .head(per_cluster)
            .sort_values(["cluster_id", "timestamp_utc"])
    )
    return out[["cluster_id", "timestamp_utc", confidence_col]]


def example_headlines(
    rep_dates: pd.DataFrame,
    headlines: pd.DataFrame,
    per_date: int = 5,
) -> pd.DataFrame:
    """Pull headline samples for each representative (cluster, date) pair.

    ``headlines`` must include ``timestamp_utc`` (any timestamp granularity),
    ``article_url``, and either ``themes`` (JSON) or a ``themes_list`` column.
    """
    if rep_dates.empty or headlines.empty:
        return pd.DataFrame(columns=["cluster_id", "timestamp_utc", "article_url"])

    hl = headlines.copy()
    hl["timestamp_utc"] = pd.to_datetime(hl["timestamp_utc"]).dt.floor("D")
    rep_dates = rep_dates.copy()
    rep_dates["timestamp_utc"] = pd.to_datetime(rep_dates["timestamp_utc"]).dt.floor("D")

    joined = rep_dates.merge(hl, on="timestamp_utc", how="left", suffixes=("", "_hl"))
    if joined.empty:
        return pd.DataFrame(columns=["cluster_id", "timestamp_utc", "article_url"])

    out_cols = ["cluster_id", "timestamp_utc", "article_url"]
    if "source" in joined.columns:
        out_cols.insert(2, "source")
    sample = (
        joined.dropna(subset=["article_url"])
              .groupby(["cluster_id", "timestamp_utc"], as_index=False, group_keys=False)
              .head(per_date)
    )
    return sample[out_cols].reset_index(drop=True)


def cluster_summary(
    assignments: pd.DataFrame,
    forward: pd.DataFrame,
    topic_prevalence_wide: pd.DataFrame | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    top_topics: int = 3,
) -> dict[str, pd.DataFrame]:
    """One-call summary bundle for Phase 3 write-up consumption."""
    out = {
        "forward_stats": cluster_forward_stats(assignments, forward, horizons),
        "representative_dates": representative_dates(assignments),
    }
    if topic_prevalence_wide is not None:
        out["dominant_topics"] = dominant_topics(
            assignments, topic_prevalence_wide, top_k=top_topics,
        )
    return out
