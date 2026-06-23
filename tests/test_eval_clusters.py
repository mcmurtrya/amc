"""Tests for cluster-analysis utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metals.eval.clusters import (
    cluster_forward_stats,
    cluster_summary,
    dominant_topics,
    example_headlines,
    forward_returns,
    representative_dates,
)


def _toy_prices(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    cols = ["GC=F", "SI=F"]
    rets = rng.normal(0, 0.01, (n, len(cols)))
    return pd.DataFrame(1000 * np.exp(np.cumsum(rets, axis=0)),
                        index=idx, columns=cols)


def test_forward_returns_columns_and_warmup():
    p = _toy_prices()
    fr = forward_returns(p, horizons=(1, 5))
    assert {"GC=F_fwd_1d", "GC=F_fwd_5d", "SI=F_fwd_1d", "SI=F_fwd_5d"} <= set(fr.columns)
    assert fr["GC=F_fwd_5d"].iloc[-5:].isna().all()


def test_forward_returns_value_matches_manual():
    p = _toy_prices()
    fr = forward_returns(p, horizons=(5,))
    expected = float(np.log(p["GC=F"].iloc[10] / p["GC=F"].iloc[5]))
    assert fr["GC=F_fwd_5d"].iloc[5] == pytest.approx(expected)


def test_cluster_forward_stats_basic():
    p = _toy_prices(n=120)
    fr = forward_returns(p, horizons=(1, 5))
    # Alternate assignments between clusters 0 and 1
    assignments = pd.DataFrame({
        "timestamp_utc": p.index,
        "cluster_id":    [0 if i % 2 == 0 else 1 for i in range(len(p))],
    })
    stats = cluster_forward_stats(assignments, fr, horizons=(1, 5))
    assert {"cluster_id", "ticker", "horizon", "n", "mean", "std", "hit_rate"} <= set(stats.columns)
    assert set(stats["cluster_id"]) == {0, 1}
    assert set(stats["ticker"]) == {"GC=F", "SI=F"}
    assert (stats["hit_rate"] >= 0).all() and (stats["hit_rate"] <= 1).all()


def test_cluster_forward_stats_empty_inputs():
    out = cluster_forward_stats(pd.DataFrame(), pd.DataFrame())
    assert out.empty


def test_dominant_topics_returns_top_k():
    n = 50
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    tp = pd.DataFrame({
        "topic_0": [0.2] * n,
        "topic_1": [0.5] * n,
        "topic_2": [0.3] * n,
    }, index=idx)
    assignments = pd.DataFrame({
        "timestamp_utc": idx,
        "cluster_id":    [0] * n,
    })
    out = dominant_topics(assignments, tp, top_k=2)
    assert len(out) == 2
    assert out.iloc[0]["topic_col"] == "topic_1"
    assert out.iloc[0]["mean_prevalence"] == pytest.approx(0.5)


def test_representative_dates_sorts_by_confidence():
    n = 10
    assignments = pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-01-01", periods=n, freq="D"),
        "cluster_id":    [0] * n,
        "confidence":    np.linspace(0.1, 1.0, n),  # increasing
    })
    out = representative_dates(assignments, per_cluster=3)
    assert len(out) == 3
    # Highest-confidence dates are the last three in the toy series.
    assert out["confidence"].min() >= 0.7


def test_representative_dates_missing_confidence_column_defaults_to_1():
    assignments = pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-01-01", periods=5),
        "cluster_id":    [0, 0, 1, 1, 1],
    })
    out = representative_dates(assignments)
    assert "confidence" in out.columns
    assert (out["confidence"] == 1.0).all()


def test_example_headlines_joins_on_day():
    rep = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "cluster_id":    [0, 0],
        "confidence":    [1.0, 1.0],
    })
    hl = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(["2024-01-01 09:00", "2024-01-01 14:00",
                                         "2024-01-02 10:00", "2024-01-03 11:00"]),
        "article_url":   ["u1", "u2", "u3", "u4"],
        "source":        ["a", "b", "c", "d"],
    })
    out = example_headlines(rep, hl, per_date=2)
    assert len(out) == 3   # 2024-01-01: 2 articles + 2024-01-02: 1 article
    assert set(out["article_url"]) == {"u1", "u2", "u3"}


def test_cluster_summary_bundles_outputs():
    p = _toy_prices()
    fr = forward_returns(p, horizons=(1,))
    assignments = pd.DataFrame({
        "timestamp_utc": p.index,
        "cluster_id":    [0 if i < len(p) // 2 else 1 for i in range(len(p))],
    })
    summary = cluster_summary(assignments, fr, horizons=(1,))
    assert "forward_stats" in summary
    assert "representative_dates" in summary
    assert "dominant_topics" not in summary
