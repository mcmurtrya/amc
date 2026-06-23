"""Tests for the LLM cluster-labeling module.

No live API calls: the Anthropic client is mocked via a thunk that returns
canned responses. Prompt construction and response parsing are tested
independently of the API call orchestration.
"""

from __future__ import annotations

import json
import os
import tempfile

import pandas as pd
import pytest

from metals.eval.cluster_labeling import (
    ClusterContext,
    ClusterLabel,
    build_cluster_context,
    build_labeling_prompt,
    label_all_clusters,
    label_cluster,
    parse_llm_response,
)


@pytest.fixture(autouse=True)
def tmp_db(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.duckdb")
        monkeypatch.setenv("METALS_DB_PATH", db_path)
        yield db_path


def _toy_context() -> ClusterContext:
    return ClusterContext(
        cluster_id=7,
        n_days=42,
        representative_dates=[pd.Timestamp("2022-03-08"), pd.Timestamp("2022-03-09")],
        example_headlines=[
            "2022-03-08: Russia sanctions tighten amid metals supply concerns",
            "2022-03-09: Gold rallies on safe-haven demand",
        ],
        dominant_topics=[("topic_5", 0.41), ("topic_2", 0.18)],
        mean_forward_returns={"GC=F_fwd_5d": 0.018, "SI=F_fwd_5d": 0.024},
    )


# ---------------------------------------------------------------------------
# Prompt and parser
# ---------------------------------------------------------------------------

def test_build_labeling_prompt_includes_required_fields():
    prompt = build_labeling_prompt(_toy_context())
    assert "Cluster ID: 7" in prompt
    assert "Days in cluster: 42" in prompt
    assert "2022-03-08" in prompt
    assert "GC=F_fwd_5d" in prompt
    assert "topic_5" in prompt


def test_parse_llm_response_plain_json():
    out = parse_llm_response(
        '{"label":"geopolitical-flight-to-safety",'
        '"description":"Russia/Ukraine sanctions","confidence":"high"}',
        cluster_id=7,
    )
    assert out.cluster_id == 7
    assert out.label == "geopolitical-flight-to-safety"
    assert out.confidence == "high"


def test_parse_llm_response_with_markdown_fences():
    raw = '```json\n{"label":"china-demand","description":"x","confidence":"medium"}\n```'
    out = parse_llm_response(raw, cluster_id=3)
    assert out.label == "china-demand"
    assert out.confidence == "medium"


def test_parse_llm_response_extracts_embedded_json():
    raw = "Sure, here you go:\n{\"label\":\"hawkish-fed\",\"description\":\"d\",\"confidence\":\"high\"}\nLet me know if..."
    out = parse_llm_response(raw, cluster_id=1)
    assert out.label == "hawkish-fed"


def test_parse_llm_response_unknown_confidence_falls_back_to_low():
    raw = '{"label":"x","description":"y","confidence":"super-high"}'
    out = parse_llm_response(raw, cluster_id=1)
    assert out.confidence == "low"


def test_parse_llm_response_raises_on_unrecoverable_garbage():
    with pytest.raises(ValueError, match="Could not extract JSON"):
        parse_llm_response("totally not json at all", cluster_id=1)


def test_parse_llm_response_normalizes_label_casing():
    raw = '{"label":"HAWKISH-FED","description":"d","confidence":"high"}'
    out = parse_llm_response(raw, cluster_id=1)
    assert out.label == "hawkish-fed"


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def test_build_cluster_context_assembles_all_fields():
    assignments = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "cluster_id":    [0, 0, 1],
        "confidence":    [0.9, 0.8, 0.7],
    })
    headlines = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(["2024-01-01 09:00", "2024-01-01 14:00",
                                         "2024-01-02 10:00"]),
        "article_url":   ["u1", "u2", "u3"],
        "headline":      ["h1", "h2", "h3"],
    })
    dominant_topics = pd.DataFrame({
        "cluster_id":      [0, 0, 1],
        "topic_col":       ["topic_3", "topic_7", "topic_2"],
        "mean_prevalence": [0.5, 0.3, 0.6],
    })
    forward_stats = pd.DataFrame({
        "cluster_id": [0, 0, 1],
        "ticker":     ["GC=F", "SI=F", "GC=F"],
        "horizon":    [5, 5, 5],
        "mean":       [0.01, 0.02, -0.005],
        "std":        [0.05, 0.07, 0.06],
        "hit_rate":   [0.6, 0.55, 0.45],
        "n":          [25, 25, 12],
    })
    ctx = build_cluster_context(
        cluster_id=0,
        assignments=assignments,
        headlines=headlines,
        dominant_topics=dominant_topics,
        forward_stats=forward_stats,
    )
    assert ctx.cluster_id == 0
    assert ctx.n_days == 2
    assert ("topic_3", 0.5) in ctx.dominant_topics
    assert ctx.mean_forward_returns["GC=F_fwd_5d"] == pytest.approx(0.01)
    # Headlines from both representative dates appear
    days_with_hl = {h.split(":")[0] for h in ctx.example_headlines}
    assert "2024-01-01" in days_with_hl


def test_build_cluster_context_handles_missing_optionals():
    assignments = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(["2024-01-01"]),
        "cluster_id":    [0],
    })
    ctx = build_cluster_context(
        cluster_id=0,
        assignments=assignments,
        headlines=None,
        dominant_topics=None,
        forward_stats=None,
    )
    assert ctx.example_headlines == []
    assert ctx.dominant_topics == []
    assert ctx.mean_forward_returns == {}


# ---------------------------------------------------------------------------
# End-to-end with mocked caller
# ---------------------------------------------------------------------------

def test_label_cluster_uses_caller_and_parses_response():
    canned = json.dumps({
        "label": "geopolitical-supply-shock",
        "description": "Russia sanctions hit Pd supply",
        "confidence": "high",
    })

    def fake_caller(system, user, model, max_tokens):
        # Spot-check we passed the system + prompt body through.
        assert "Cluster ID:" in user
        assert "JSON only" in system
        return canned

    out = label_cluster(_toy_context(), caller=fake_caller, max_retries=1)
    assert isinstance(out, ClusterLabel)
    assert out.label == "geopolitical-supply-shock"
    assert out.confidence == "high"


def test_label_cluster_retries_then_succeeds():
    calls = {"n": 0}

    def flaky_caller(system, user, model, max_tokens):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient")
        return '{"label":"x","description":"y","confidence":"high"}'

    out = label_cluster(_toy_context(), caller=flaky_caller,
                        max_retries=3, retry_delay_s=0.01)
    assert out.label == "x"
    assert calls["n"] == 2


def test_label_cluster_gives_up_after_retries():
    def always_fail(system, user, model, max_tokens):
        raise RuntimeError("api down")

    with pytest.raises(RuntimeError, match="failed after"):
        label_cluster(_toy_context(), caller=always_fail,
                      max_retries=2, retry_delay_s=0.01)


def test_label_all_clusters_iterates():
    canned = '{"label":"x","description":"y","confidence":"high"}'

    def caller(system, user, model, max_tokens):
        return canned

    outs = label_all_clusters([_toy_context(), _toy_context()], caller=caller)
    assert len(outs) == 2
    assert all(isinstance(o, ClusterLabel) for o in outs)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_upsert_and_load_labels_round_trip():
    from metals.data.migrations.runner import apply_migrations
    from metals.eval.cluster_labeling import load_labels, upsert_labels
    from metals.models.clustering import upsert_centroids
    import numpy as np

    apply_migrations(verbose=False)

    # Seed the centroids table so the UPDATE has rows to hit.
    centroids = pd.DataFrame([
        {"cluster_id": 0, "n_members": 25,
         "centroid": np.array([0.1, 0.2], dtype=np.float32), "centroid_dim": 2},
        {"cluster_id": 1, "n_members": 18,
         "centroid": np.array([0.5, 0.6], dtype=np.float32), "centroid_dim": 2},
    ])
    upsert_centroids(centroids, model_version="test_v1")

    labels = [
        ClusterLabel(cluster_id=0, label="hawkish-fed",
                     description="d0", confidence="high"),
        ClusterLabel(cluster_id=1, label="china-demand",
                     description="d1", confidence="medium"),
    ]
    n = upsert_labels(labels, model_version="test_v1")
    assert n == 2

    back = load_labels("test_v1")
    assert len(back) == 2
    by_id = back.set_index("cluster_id")
    assert by_id.loc[0, "label"] == "hawkish-fed"
    assert "llm:high" in by_id.loc[0, "label_source"]
    assert by_id.loc[1, "label"] == "china-demand"
