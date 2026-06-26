"""Tests for the clustering pipeline.

Pure-function logic (standardize, save/load round-trip, upsert wiring) is
exercised here. The UMAP + HDBSCAN fit/transform path is skipped if those
libraries aren't installed."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from metals.models.clustering import (
    ClusteringConfig,
    _standardize,
)


@pytest.fixture(autouse=True)
def tmp_db(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.duckdb")
        monkeypatch.setenv("METALS_DB_PATH", db_path)
        yield db_path


def test_standardize_zero_centers_and_unit_variance():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(5, 3, (300, 4)), columns=["a", "b", "c", "d"])
    Z, mean, std = _standardize(X)
    assert np.abs(Z.mean(axis=0)).max() < 1e-9
    assert np.abs(Z.std(axis=0) - 1.0).max() < 1e-9
    assert mean.shape == (4,) and std.shape == (4,)


def test_standardize_constant_column_does_not_divide_by_zero():
    X = pd.DataFrame({"a": np.ones(10), "b": np.arange(10).astype(float)})
    Z, mean, std = _standardize(X)
    # Constant column should be all zeros after centering
    assert (Z[:, 0] == 0).all()
    # Std for the variable column should be standardized normally
    assert np.abs(Z[:, 1].std() - 1.0) < 1e-6


def test_standardize_apply_with_explicit_params():
    rng = np.random.default_rng(1)
    X_train = pd.DataFrame(rng.normal(0, 1, (100, 3)), columns=["a", "b", "c"])
    _, mean, std = _standardize(X_train)
    X_new = pd.DataFrame(rng.normal(0, 1, (20, 3)), columns=["a", "b", "c"])
    Z, m2, s2 = _standardize(X_new, mean=mean, std=std)
    # Reused mean/std should be unchanged
    assert np.array_equal(m2, mean)
    assert np.array_equal(s2, std)


def test_clustering_config_defaults():
    cfg = ClusteringConfig()
    assert cfg.umap_n_components > 0
    assert cfg.hdbscan_min_cluster_size > 1
    assert cfg.random_state == 42


def test_fit_and_assign_clusters_on_synthetic_blobs():
    """End-to-end smoke test of the pipeline using sklearn blobs.
    Skipped if UMAP or HDBSCAN is not installed."""
    pytest.importorskip("umap")
    pytest.importorskip("hdbscan")
    from sklearn.datasets import make_blobs

    from metals.models.clustering import (
        assign_clusters,
        cluster_centroids,
        fit_clustering,
    )

    X_arr, _ = make_blobs(n_samples=200, n_features=8, centers=3, cluster_std=0.6, random_state=0)
    idx = pd.date_range("2020-01-01", periods=200, freq="D")
    X = pd.DataFrame(X_arr, index=idx, columns=[f"f{i}" for i in range(X_arr.shape[1])])
    cfg = ClusteringConfig(
        umap_n_components=2,
        umap_n_neighbors=15,
        hdbscan_min_cluster_size=10,
    )
    pipeline = fit_clustering(X, config=cfg, model_version="test_v1")
    assert pipeline.model_version == "test_v1"
    assert pipeline.feature_names == list(X.columns)

    assignments = assign_clusters(pipeline, X)
    assert len(assignments) == 200
    assert "cluster_id" in assignments.columns
    # Should find at least 2 non-noise clusters on 3-blob synthetic data.
    non_noise = assignments[assignments["cluster_id"] != -1]
    assert non_noise["cluster_id"].nunique() >= 2

    centroids = cluster_centroids(pipeline, X)
    assert "centroid" in centroids.columns
    assert centroids["n_members"].sum() == len(X)


def test_save_and_load_pipeline_round_trip(tmp_path, monkeypatch):
    """Persist a tiny pipeline shell (no real UMAP/HDBSCAN) and load it back."""
    import pickle

    from metals.models.clustering import (
        ClusterPipeline,
        MODEL_DIR,
        load_pipeline,
        save_pipeline,
    )

    # Use tmp_path for the model directory.
    monkeypatch.setattr("metals.models.clustering.MODEL_DIR", tmp_path)

    p = ClusterPipeline(
        config=ClusteringConfig(),
        umap_model=("fake_umap",),
        hdbscan_model=("fake_hdb",),
        feature_mean=np.zeros(4),
        feature_std=np.ones(4),
        feature_names=["a", "b", "c", "d"],
        model_version="t1",
        fit_at="2026-01-01T00:00:00Z",
    )
    path = save_pipeline(p)
    assert path.exists()
    sidecar = path.parent / "t1.json"
    assert sidecar.exists()
    loaded = load_pipeline("t1")
    assert loaded.model_version == "t1"
    assert loaded.feature_names == ["a", "b", "c", "d"]


def test_upsert_assignments_and_centroids():
    from metals.data.migrations.runner import apply_migrations
    from metals.models.clustering import upsert_assignments, upsert_centroids

    apply_migrations(verbose=False)
    assignments = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "cluster_id": [0, 1, 0],
            "confidence": [0.9, 0.4, 0.8],
        }
    )
    n = upsert_assignments(assignments, model_version="t_v1")
    assert n == 3

    centroids = pd.DataFrame(
        [
            {
                "cluster_id": 0,
                "n_members": 2,
                "centroid": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                "centroid_dim": 3,
            },
            {
                "cluster_id": 1,
                "n_members": 1,
                "centroid": np.array([0.5, 0.6, 0.7], dtype=np.float32),
                "centroid_dim": 3,
            },
        ]
    )
    nc = upsert_centroids(centroids, model_version="t_v1")
    assert nc == 2
