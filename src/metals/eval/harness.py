"""Evaluation harness.

Every model run — baseline, transformer, ablation — registers itself and logs
its predictions here. Phase 6 validation reads from these tables to compute
lift tables without re-running anything.

Schema (created lazily):

    runs(run_id, name, model_type, target_type, config_json,
         created_at, git_hash, notes)

    run_predictions(run_id, timestamp_utc, ticker, horizon,
                    prediction, actual)

Idempotency: ``log_predictions`` upserts on the primary key
``(run_id, timestamp_utc, ticker, horizon)``.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from typing import Iterable

import numpy as np
import pandas as pd

from metals.data.db import connection

_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          VARCHAR PRIMARY KEY,
    name            VARCHAR NOT NULL,
    model_type      VARCHAR NOT NULL,
    target_type     VARCHAR NOT NULL,
    config_json     JSON,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    git_hash        VARCHAR,
    notes           VARCHAR
)
"""

_RUN_PREDICTIONS_DDL = """
CREATE TABLE IF NOT EXISTS run_predictions (
    run_id          VARCHAR     NOT NULL,
    timestamp_utc   TIMESTAMP   NOT NULL,
    ticker          VARCHAR     NOT NULL,
    horizon         INTEGER     NOT NULL,
    prediction      DOUBLE,
    actual          DOUBLE,
    PRIMARY KEY (run_id, timestamp_utc, ticker, horizon)
)
"""

_REQUIRED_PRED_COLS = ("timestamp_utc", "ticker", "horizon", "prediction", "actual")


def _git_hash() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def _ensure_schema(conn) -> None:
    conn.execute(_RUNS_DDL)
    conn.execute(_RUN_PREDICTIONS_DDL)


def register_run(
    name: str,
    model_type: str,
    target_type: str,
    config: dict | None = None,
    notes: str | None = None,
) -> str:
    """Create a new run and return its unique run_id."""
    run_id = str(uuid.uuid4())
    config_json = json.dumps(config or {}, default=str)
    with connection() as conn:
        _ensure_schema(conn)
        conn.execute(
            "INSERT INTO runs(run_id, name, model_type, target_type, config_json, "
            "git_hash, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [run_id, name, model_type, target_type, config_json, _git_hash(), notes],
        )
    return run_id


def log_predictions(run_id: str, df: pd.DataFrame) -> None:
    """Upsert predictions for a run."""
    missing = set(_REQUIRED_PRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"log_predictions: missing required columns {missing}")

    insert_df = df[list(_REQUIRED_PRED_COLS)].copy()
    insert_df.insert(0, "run_id", run_id)

    with connection() as conn:
        _ensure_schema(conn)
        conn.register("incoming_predictions", insert_df)
        conn.execute(
            """
            INSERT INTO run_predictions
                (run_id, timestamp_utc, ticker, horizon, prediction, actual)
            SELECT run_id, timestamp_utc, ticker, horizon, prediction, actual
            FROM incoming_predictions
            ON CONFLICT (run_id, timestamp_utc, ticker, horizon) DO UPDATE SET
                prediction = EXCLUDED.prediction,
                actual = EXCLUDED.actual
            """
        )
        conn.unregister("incoming_predictions")


def fetch_predictions(run_id: str) -> pd.DataFrame:
    """Return all predictions for a given run."""
    with connection() as conn:
        _ensure_schema(conn)
        return conn.execute(
            "SELECT timestamp_utc, ticker, horizon, prediction, actual "
            "FROM run_predictions WHERE run_id = ? "
            "ORDER BY timestamp_utc, ticker, horizon",
            [run_id],
        ).fetchdf()


def compute_metrics(run_id: str) -> pd.DataFrame:
    """Compute per-(ticker, horizon) summary metrics for a run."""
    cols = ["ticker", "horizon", "n", "rmse", "mae", "ic", "hit_rate", "mean_pred", "mean_actual"]
    df = fetch_predictions(run_id)
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = df.dropna(subset=["prediction", "actual"])
    if df.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for (ticker, horizon), g in df.groupby(["ticker", "horizon"]):
        pred = g["prediction"].to_numpy()
        act = g["actual"].to_numpy()
        n = len(g)
        rmse = float(np.sqrt(np.mean((pred - act) ** 2)))
        mae = float(np.mean(np.abs(pred - act)))
        if n >= 2 and np.std(pred) > 0 and np.std(act) > 0:
            ic = float(np.corrcoef(pred, act)[0, 1])
        else:
            ic = float("nan")
        hit_rate = float(np.mean(np.sign(pred) == np.sign(act)))
        rows.append(
            {
                "ticker": ticker,
                "horizon": int(horizon),
                "n": int(n),
                "rmse": rmse,
                "mae": mae,
                "ic": ic,
                "hit_rate": hit_rate,
                "mean_pred": float(np.mean(pred)),
                "mean_actual": float(np.mean(act)),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def compare_runs(run_ids: Iterable[str], metric: str = "rmse") -> pd.DataFrame:
    """Pivot a single metric across runs for side-by-side comparison."""
    run_ids = list(run_ids)
    if not run_ids:
        return pd.DataFrame()

    with connection() as conn:
        _ensure_schema(conn)
        placeholders = ",".join(["?"] * len(run_ids))
        runs = conn.execute(
            f"SELECT run_id, name FROM runs WHERE run_id IN ({placeholders})",
            run_ids,
        ).fetchdf()
    name_map = dict(zip(runs["run_id"], runs["name"]))

    frames = []
    for rid in run_ids:
        m = compute_metrics(rid)
        if m.empty:
            continue
        m = m.copy()
        m["run"] = name_map.get(rid, rid[:8])
        frames.append(m)

    if not frames:
        return pd.DataFrame()

    long = pd.concat(frames, ignore_index=True)
    if metric not in long.columns:
        raise ValueError(
            f"compare_runs: metric must be one of "
            f"{[c for c in long.columns if c not in ('ticker', 'horizon', 'run')]}"
        )
    return long.pivot_table(
        index=["ticker", "horizon"],
        columns="run",
        values=metric,
    )


def list_runs(limit: int = 50) -> pd.DataFrame:
    """Return the most recent runs."""
    with connection() as conn:
        _ensure_schema(conn)
        return conn.execute(
            "SELECT run_id, name, model_type, target_type, created_at, git_hash "
            "FROM runs ORDER BY created_at DESC LIMIT ?",
            [limit],
        ).fetchdf()


# ---------------------------------------------------------------------------
# Feature importances (Phase 1 cleanup)
# ---------------------------------------------------------------------------

_FEATURE_IMPORTANCES_DDL = """
CREATE TABLE IF NOT EXISTS run_feature_importances (
    run_id           VARCHAR     NOT NULL,
    split_id         INTEGER     NOT NULL,
    feature_name     VARCHAR     NOT NULL,
    importance       DOUBLE      NOT NULL,
    importance_type  VARCHAR     NOT NULL DEFAULT 'gain',
    PRIMARY KEY (run_id, split_id, feature_name, importance_type)
)
"""


def _ensure_importance_schema(conn) -> None:
    conn.execute(_FEATURE_IMPORTANCES_DDL)


def log_feature_importances(
    run_id: str,
    split_id: int,
    importances: dict[str, float],
    importance_type: str = "gain",
) -> None:
    """Upsert a dict of {feature_name: importance} for a single split.

    Parameters
    ----------
    run_id : str
        The harness run id (from register_run).
    split_id : int
        Walk-forward split index.
    importances : dict[str, float]
        Map from feature name to scalar importance.
    importance_type : str
        Free-form label, e.g. 'gain', 'split', 'shap_abs_mean'. Default 'gain'.
    """
    if not importances:
        return
    rows = pd.DataFrame(
        {
            "run_id": [run_id] * len(importances),
            "split_id": [int(split_id)] * len(importances),
            "feature_name": list(importances.keys()),
            "importance": [float(v) for v in importances.values()],
            "importance_type": [importance_type] * len(importances),
        }
    )
    with connection() as conn:
        _ensure_importance_schema(conn)
        conn.register("incoming_importances", rows)
        conn.execute(
            """
            INSERT INTO run_feature_importances
                (run_id, split_id, feature_name, importance, importance_type)
            SELECT run_id, split_id, feature_name, importance, importance_type
            FROM incoming_importances
            ON CONFLICT (run_id, split_id, feature_name, importance_type) DO UPDATE SET
                importance = EXCLUDED.importance
            """
        )
        conn.unregister("incoming_importances")


def fetch_feature_importances(
    run_id: str,
    importance_type: str | None = None,
) -> pd.DataFrame:
    """Return all logged importances for a run. Columns:
    run_id, split_id, feature_name, importance, importance_type."""
    with connection() as conn:
        _ensure_importance_schema(conn)
        if importance_type is None:
            return conn.execute(
                "SELECT * FROM run_feature_importances WHERE run_id = ? "
                "ORDER BY split_id, feature_name",
                [run_id],
            ).fetchdf()
        return conn.execute(
            "SELECT * FROM run_feature_importances "
            "WHERE run_id = ? AND importance_type = ? "
            "ORDER BY split_id, feature_name",
            [run_id, importance_type],
        ).fetchdf()


def aggregate_feature_importances(
    run_id: str,
    importance_type: str = "gain",
    normalize: bool = True,
) -> pd.DataFrame:
    """Average importances across splits, sorted high -> low.

    Columns: feature_name, mean_importance, std_importance, n_splits.
    If ``normalize=True`` each split's importances are divided by their sum
    before averaging, so cross-split comparisons aren't dominated by splits
    that happen to have larger raw gain values.
    """
    df = fetch_feature_importances(run_id, importance_type=importance_type)
    if df.empty:
        return pd.DataFrame(
            columns=["feature_name", "mean_importance", "std_importance", "n_splits"]
        )
    if normalize:
        sums = df.groupby("split_id")["importance"].transform("sum")
        df = df.assign(importance=df["importance"] / sums.replace(0, np.nan))
    g = df.groupby("feature_name")["importance"]
    out = pd.DataFrame(
        {
            "mean_importance": g.mean(),
            "std_importance": g.std(ddof=0).fillna(0.0),
            "n_splits": g.count(),
        }
    ).reset_index()
    return out.sort_values("mean_importance", ascending=False).reset_index(drop=True)
