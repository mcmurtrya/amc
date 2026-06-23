"""Tests for the feature-importance side of the eval harness."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def tmp_db(monkeypatch):
    """Per-test temp DuckDB so harness state doesn't bleed across tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.duckdb")
        monkeypatch.setenv("METALS_DB_PATH", db_path)
        yield db_path


def test_log_and_fetch_importances_round_trip():
    from metals.eval.harness import (
        fetch_feature_importances, log_feature_importances, register_run,
    )

    rid = register_run(name="test", model_type="lgbm_vol", target_type="realized_vol")
    log_feature_importances(rid, split_id=0,
                            importances={"feat_a": 10.0, "feat_b": 5.0})
    out = fetch_feature_importances(rid)
    assert len(out) == 2
    assert set(out["feature_name"]) == {"feat_a", "feat_b"}
    assert (out["importance_type"] == "gain").all()


def test_log_importances_is_idempotent_on_resubmit():
    from metals.eval.harness import (
        fetch_feature_importances, log_feature_importances, register_run,
    )

    rid = register_run(name="t", model_type="lgbm", target_type="vol")
    log_feature_importances(rid, 0, {"x": 1.0, "y": 2.0})
    log_feature_importances(rid, 0, {"x": 1.0, "y": 2.0})  # same again
    out = fetch_feature_importances(rid)
    assert len(out) == 2


def test_log_importances_upserts_updated_value():
    from metals.eval.harness import (
        fetch_feature_importances, log_feature_importances, register_run,
    )

    rid = register_run(name="t", model_type="lgbm", target_type="vol")
    log_feature_importances(rid, 0, {"x": 1.0})
    log_feature_importances(rid, 0, {"x": 9.0})  # update
    out = fetch_feature_importances(rid)
    assert len(out) == 1
    assert out.iloc[0]["importance"] == pytest.approx(9.0)


def test_log_importances_empty_dict_is_noop():
    from metals.eval.harness import (
        fetch_feature_importances, log_feature_importances, register_run,
    )

    rid = register_run(name="t", model_type="lgbm", target_type="vol")
    log_feature_importances(rid, 0, {})
    assert fetch_feature_importances(rid).empty


def test_fetch_by_importance_type_filters():
    from metals.eval.harness import (
        fetch_feature_importances, log_feature_importances, register_run,
    )

    rid = register_run(name="t", model_type="lgbm", target_type="vol")
    log_feature_importances(rid, 0, {"x": 1.0, "y": 2.0}, importance_type="gain")
    log_feature_importances(rid, 0, {"x": 3.0, "y": 4.0}, importance_type="split")
    only_gain = fetch_feature_importances(rid, importance_type="gain")
    assert len(only_gain) == 2
    assert (only_gain["importance_type"] == "gain").all()


def test_aggregate_normalizes_across_splits():
    from metals.eval.harness import (
        aggregate_feature_importances, log_feature_importances, register_run,
    )

    rid = register_run(name="t", model_type="lgbm", target_type="vol")
    # Two splits with very different raw gain scales but same proportions.
    # After normalization, mean importance per feature should be ~equal.
    log_feature_importances(rid, 0, {"x": 6.0, "y": 4.0})    # 60% / 40%
    log_feature_importances(rid, 1, {"x": 600.0, "y": 400.0})  # same split
    agg = aggregate_feature_importances(rid, normalize=True)
    assert len(agg) == 2
    x_row = agg[agg["feature_name"] == "x"].iloc[0]
    y_row = agg[agg["feature_name"] == "y"].iloc[0]
    assert x_row["mean_importance"] == pytest.approx(0.6)
    assert y_row["mean_importance"] == pytest.approx(0.4)
    # Sorted high -> low
    assert agg.iloc[0]["feature_name"] == "x"


def test_aggregate_unnormalized_uses_raw_values():
    from metals.eval.harness import (
        aggregate_feature_importances, log_feature_importances, register_run,
    )

    rid = register_run(name="t", model_type="lgbm", target_type="vol")
    log_feature_importances(rid, 0, {"x": 10.0})
    log_feature_importances(rid, 1, {"x": 20.0})
    agg = aggregate_feature_importances(rid, normalize=False)
    assert agg.iloc[0]["mean_importance"] == pytest.approx(15.0)
    assert agg.iloc[0]["n_splits"] == 2


def test_aggregate_empty_run_returns_empty_with_schema():
    from metals.eval.harness import aggregate_feature_importances

    out = aggregate_feature_importances("00000000-0000-0000-0000-000000000000")
    assert out.empty
    assert set(out.columns) == {"feature_name", "mean_importance",
                                "std_importance", "n_splits"}
