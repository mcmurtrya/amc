"""Tests for the evaluation harness.

A temp DuckDB is used per test via the METALS_DB_PATH env var to isolate state.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def tmp_db(monkeypatch):
    """Redirect METALS_DB_PATH at a per-test temporary DuckDB file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.duckdb")
        monkeypatch.setenv("METALS_DB_PATH", db_path)
        yield db_path


def test_register_run_returns_uuid():
    from metals.eval.harness import register_run

    rid = register_run(
        name="test_run",
        model_type="lgbm_vol",
        target_type="realized_vol",
        config={"horizon": 5},
    )
    assert isinstance(rid, str)
    assert len(rid) >= 32


def test_log_and_fetch_predictions():
    from metals.eval.harness import fetch_predictions, log_predictions, register_run

    rid = register_run(
        name="test",
        model_type="lgbm_vol",
        target_type="realized_vol",
    )
    df = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2020-01-01", periods=5, freq="D"),
            "ticker": ["GC=F"] * 5,
            "horizon": [5] * 5,
            "prediction": [0.10, 0.12, 0.11, 0.09, 0.13],
            "actual": [0.11, 0.10, 0.12, 0.10, 0.14],
        }
    )
    log_predictions(rid, df)
    back = fetch_predictions(rid)
    assert len(back) == 5
    assert set(back["ticker"].unique()) == {"GC=F"}


def test_log_predictions_is_idempotent():
    from metals.eval.harness import fetch_predictions, log_predictions, register_run

    rid = register_run(name="test", model_type="lgbm_vol", target_type="realized_vol")
    df = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2020-01-01", periods=3, freq="D"),
            "ticker": ["GC=F"] * 3,
            "horizon": [5] * 3,
            "prediction": [0.1, 0.2, 0.3],
            "actual": [0.15, 0.25, 0.35],
        }
    )
    log_predictions(rid, df)
    log_predictions(rid, df)  # should not duplicate
    back = fetch_predictions(rid)
    assert len(back) == 3


def test_log_predictions_upserts_updated_values():
    from metals.eval.harness import fetch_predictions, log_predictions, register_run

    rid = register_run(name="test", model_type="lgbm_vol", target_type="realized_vol")
    df1 = pd.DataFrame(
        {
            "timestamp_utc": [pd.Timestamp("2020-01-01")],
            "ticker": ["GC=F"],
            "horizon": [5],
            "prediction": [0.1],
            "actual": [0.2],
        }
    )
    df2 = df1.copy()
    df2["prediction"] = [0.9]
    log_predictions(rid, df1)
    log_predictions(rid, df2)
    back = fetch_predictions(rid)
    assert len(back) == 1
    assert back.iloc[0]["prediction"] == pytest.approx(0.9)


def test_log_predictions_missing_column_raises():
    from metals.eval.harness import log_predictions, register_run

    rid = register_run(name="test", model_type="lgbm_vol", target_type="realized_vol")
    bad = pd.DataFrame({"timestamp_utc": [pd.Timestamp("2020-01-01")]})
    with pytest.raises(ValueError, match="missing required columns"):
        log_predictions(rid, bad)


def test_compute_metrics_perfect_predictions():
    from metals.eval.harness import compute_metrics, log_predictions, register_run

    rid = register_run(name="test", model_type="lgbm_vol", target_type="realized_vol")
    n = 50
    rng = np.random.default_rng(0)
    actual = rng.normal(size=n) * 0.02
    df = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2020-01-01", periods=n, freq="D"),
            "ticker": ["GC=F"] * n,
            "horizon": [5] * n,
            "prediction": actual,  # perfect
            "actual": actual,
        }
    )
    log_predictions(rid, df)
    m = compute_metrics(rid)
    row = m.iloc[0]
    assert row["n"] == n
    assert row["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert row["ic"] == pytest.approx(1.0, abs=1e-6)


def test_compute_metrics_empty_run():
    from metals.eval.harness import compute_metrics

    m = compute_metrics("00000000-0000-0000-0000-000000000000")
    assert m.empty


def test_compare_runs_pivot():
    from metals.eval.harness import compare_runs, log_predictions, register_run

    rid_a = register_run(name="a", model_type="lgbm", target_type="vol")
    rid_b = register_run(name="b", model_type="lgbm", target_type="vol")
    n = 10
    rng = np.random.default_rng(0)
    actual = rng.normal(size=n)
    df_a = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2020-01-01", periods=n, freq="D"),
            "ticker": ["GC=F"] * n,
            "horizon": [5] * n,
            "prediction": actual + rng.normal(scale=0.1, size=n),
            "actual": actual,
        }
    )
    df_b = df_a.copy()
    df_b["prediction"] = actual + rng.normal(scale=1.0, size=n)
    log_predictions(rid_a, df_a)
    log_predictions(rid_b, df_b)
    pivot = compare_runs([rid_a, rid_b], metric="rmse")
    assert "a" in pivot.columns
    assert "b" in pivot.columns
    # The lower-noise run a should have lower RMSE than b
    assert pivot["a"].iloc[0] < pivot["b"].iloc[0]


def test_list_runs_returns_recent():
    from metals.eval.harness import list_runs, register_run

    register_run(name="r1", model_type="lgbm", target_type="vol")
    register_run(name="r2", model_type="lgbm", target_type="vol")
    runs = list_runs(limit=5)
    assert len(runs) >= 2
    assert set(runs["name"]).issuperset({"r1", "r2"})
