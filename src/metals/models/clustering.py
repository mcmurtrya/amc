"""UMAP + HDBSCAN scenario clustering for Phase 3 steps 3.10–3.11 and 3.14.

The contextual feature vector from ``metals.features.context.build_context``
gets reduced via UMAP and clustered with HDBSCAN. Each fitted pipeline is
persisted under a ``model_version`` label so cluster assignments can be
reproduced and compared across runs.

UMAP and HDBSCAN are imported lazily so the test suite can exercise the
pure-function pieces without them installed.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from metals.data.db import connection

MODEL_DIR = Path(__file__).resolve().parents[3] / "data" / "processed" / "clustering"


@dataclass(frozen=True)
class ClusteringConfig:
    """Hyperparameters for the scenario clustering pipeline."""

    umap_n_components: int = 7
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.0
    umap_metric: str = "euclidean"
    hdbscan_min_cluster_size: int = 20
    hdbscan_min_samples: int | None = None
    hdbscan_cluster_selection_method: str = "eom"
    random_state: int = 42


@dataclass
class ClusterPipeline:
    """Bundle of fitted UMAP, HDBSCAN, training mean/std, and metadata."""

    config: ClusteringConfig
    umap_model: object
    hdbscan_model: object
    feature_mean: np.ndarray
    feature_std: np.ndarray
    feature_names: list[str]
    model_version: str
    fit_at: str


def _standardize(
    X: pd.DataFrame, mean: np.ndarray | None = None, std: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score features. If ``mean``/``std`` are passed, apply that exact transform."""
    arr = X.to_numpy(dtype=np.float64, na_value=0.0)
    if mean is None:
        mean = arr.mean(axis=0)
    if std is None:
        std = arr.std(axis=0, ddof=0)
    std_safe = np.where(std == 0, 1.0, std)
    return (arr - mean) / std_safe, mean, std


def fit_clustering(
    X: pd.DataFrame,
    config: ClusteringConfig | None = None,
    model_version: str | None = None,
) -> ClusterPipeline:
    """Fit UMAP + HDBSCAN on the training rows of the contextual feature frame.

    The caller is responsible for restricting ``X`` to the training segment
    (typically every row before the Phase 6 hold-out). Any rows in ``X`` are
    *all* used to fit the pipeline.
    """
    import hdbscan
    import umap as umap_lib

    config = config or ClusteringConfig()
    if X.empty:
        raise ValueError("fit_clustering: empty feature frame.")

    X_std, mean, std = _standardize(X)

    umap_model = umap_lib.UMAP(
        n_components=config.umap_n_components,
        n_neighbors=config.umap_n_neighbors,
        min_dist=config.umap_min_dist,
        metric=config.umap_metric,
        random_state=config.random_state,
    )
    embedding = umap_model.fit_transform(X_std)

    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=config.hdbscan_min_cluster_size,
        min_samples=config.hdbscan_min_samples,
        cluster_selection_method=config.hdbscan_cluster_selection_method,
        prediction_data=True,
    )
    hdbscan_model.fit(embedding)

    version = model_version or f"phase3_{datetime.utcnow():%Y%m%d_%H%M}"
    return ClusterPipeline(
        config=config,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        feature_mean=mean,
        feature_std=std,
        feature_names=list(X.columns),
        model_version=version,
        fit_at=datetime.utcnow().isoformat() + "Z",
    )


def assign_clusters(
    pipeline: ClusterPipeline,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Project + cluster new rows. Returns a frame with
    columns ``timestamp_utc, cluster_id, confidence``.
    """
    import hdbscan

    if X.empty:
        return pd.DataFrame(columns=["timestamp_utc", "cluster_id", "confidence"])
    # Align columns to the trained ordering. Missing columns are filled with 0.
    aligned = X.reindex(columns=pipeline.feature_names, fill_value=0.0)
    X_std, _, _ = _standardize(aligned, mean=pipeline.feature_mean, std=pipeline.feature_std)
    embedding = pipeline.umap_model.transform(X_std)
    labels, strengths = hdbscan.approximate_predict(pipeline.hdbscan_model, embedding)
    return pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(X.index),
            "cluster_id": np.asarray(labels, dtype=int),
            "confidence": np.asarray(strengths, dtype=float),
        }
    )


def cluster_centroids(
    pipeline: ClusterPipeline,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Compute centroid coordinates in the UMAP space, per cluster.

    Returns ``cluster_id, n_members, centroid (np.ndarray), centroid_dim``.
    """
    if X.empty:
        return pd.DataFrame(columns=["cluster_id", "n_members", "centroid", "centroid_dim"])
    aligned = X.reindex(columns=pipeline.feature_names, fill_value=0.0)
    X_std, _, _ = _standardize(aligned, mean=pipeline.feature_mean, std=pipeline.feature_std)
    embedding = pipeline.umap_model.transform(X_std)
    labels = pipeline.hdbscan_model.labels_
    if len(labels) != len(embedding):
        # If new rows: project + re-cluster instead
        import hdbscan

        labels, _ = hdbscan.approximate_predict(pipeline.hdbscan_model, embedding)
    rows = []
    for cid in sorted(set(int(x) for x in labels)):
        mask = labels == cid
        if not mask.any():
            continue
        centroid = embedding[mask].mean(axis=0).astype(np.float32)
        rows.append(
            {
                "cluster_id": int(cid),
                "n_members": int(mask.sum()),
                "centroid": centroid,
                "centroid_dim": int(centroid.size),
            }
        )
    return pd.DataFrame(rows)


def save_pipeline(pipeline: ClusterPipeline) -> Path:
    """Persist fitted UMAP + HDBSCAN to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{pipeline.model_version}.pkl"
    with path.open("wb") as f:
        pickle.dump(pipeline, f)
    # Also write a JSON config sidecar for human inspection.
    sidecar = MODEL_DIR / f"{pipeline.model_version}.json"
    with sidecar.open("w") as f:
        json.dump(
            {
                "model_version": pipeline.model_version,
                "fit_at": pipeline.fit_at,
                "config": asdict(pipeline.config),
                "n_features": len(pipeline.feature_names),
            },
            f,
            indent=2,
        )
    return path


def load_pipeline(model_version: str) -> ClusterPipeline:
    path = MODEL_DIR / f"{model_version}.pkl"
    with path.open("rb") as f:
        return pickle.load(f)


def upsert_assignments(df: pd.DataFrame, model_version: str) -> int:
    """Persist a per-date cluster assignment frame."""
    if df.empty:
        return 0
    work = df.copy()
    work["model_version"] = model_version
    with connection() as conn:
        conn.register(
            "incoming_clusters",
            work[["timestamp_utc", "model_version", "cluster_id", "confidence"]],
        )
        conn.execute(
            """
            INSERT INTO cluster_assignments
                (timestamp_utc, model_version, cluster_id, confidence)
            SELECT timestamp_utc, model_version, cluster_id, confidence
            FROM incoming_clusters
            ON CONFLICT (timestamp_utc, model_version) DO UPDATE SET
                cluster_id = EXCLUDED.cluster_id,
                confidence = EXCLUDED.confidence
            """
        )
        conn.unregister("incoming_clusters")
    return len(work)


def upsert_centroids(df: pd.DataFrame, model_version: str) -> int:
    """Persist cluster centroids + (initially blank) labels."""
    if df.empty:
        return 0
    work = df.copy()
    work["model_version"] = model_version
    work["centroid"] = work["centroid"].apply(lambda a: np.asarray(a, dtype=np.float32).tobytes())
    work["label"] = None
    work["label_source"] = None
    work["description"] = None
    cols = [
        "model_version",
        "cluster_id",
        "n_members",
        "centroid",
        "centroid_dim",
        "label",
        "label_source",
        "description",
    ]
    with connection() as conn:
        conn.register("incoming_centroids", work[cols])
        conn.execute(
            """
            INSERT INTO cluster_centroids
                (model_version, cluster_id, n_members, centroid, centroid_dim,
                 label, label_source, description)
            SELECT model_version, cluster_id, n_members, centroid, centroid_dim,
                   label, label_source, description
            FROM incoming_centroids
            ON CONFLICT (model_version, cluster_id) DO UPDATE SET
                n_members    = EXCLUDED.n_members,
                centroid     = EXCLUDED.centroid,
                centroid_dim = EXCLUDED.centroid_dim
            """
        )
        conn.unregister("incoming_centroids")
    return len(work)
