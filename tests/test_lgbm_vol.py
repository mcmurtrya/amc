"""Smoke test for the LightGBM baseline using synthetic data via monkeypatching.

We avoid hitting DuckDB or yfinance: we patch metals.models.lgbm_vol.load_prices
and load_macro to return synthetic data, then call run() to confirm the whole
pipeline executes and predictions land in the eval harness.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def tmp_db(monkeypatch):
    """Per-test temp DuckDB so the harness doesn't accumulate state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.duckdb")
        monkeypatch.setenv("METALS_DB_PATH", db_path)
        yield db_path


def _synthetic_prices(n: int = 1800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2014-01-01", periods=n, freq="B")
    cols = ["GC=F", "SI=F", "PL=F", "PA=F", "HG=F", "CL=F", "DX=F"]
    rets = rng.normal(0, 0.01, (n, len(cols)))
    levels = 1000 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(levels, index=idx, columns=cols)


def _synthetic_macro(idx: pd.DatetimeIndex, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(idx)
    return pd.DataFrame(
        {
            "DGS10": 3.0 + rng.normal(0, 0.05, n).cumsum() * 0.005,
            "DGS2": 2.0 + rng.normal(0, 0.05, n).cumsum() * 0.005,
            "T10YIE": 2.3 + rng.normal(0, 0.02, n).cumsum() * 0.003,
            "T5YIE": 2.5 + rng.normal(0, 0.02, n).cumsum() * 0.003,
            "DTWEXBGS": 100 + rng.normal(0, 0.3, n).cumsum() * 0.03,
            "VIXCLS": 18 + rng.normal(0, 1.5, n),
            "BAMLH0A0HYM2": 4.0 + rng.normal(0, 0.15, n),
            "GPR_DAILY": 100 + rng.normal(0, 25, n),
        },
        index=idx,
    )


def test_lgbm_baseline_runs_end_to_end(monkeypatch):
    pytest.importorskip("lightgbm")
    from metals.eval.harness import compute_metrics, fetch_predictions
    from metals.models import lgbm_vol

    prices = _synthetic_prices()
    macro = _synthetic_macro(prices.index)

    monkeypatch.setattr(lgbm_vol, "load_prices", lambda **kw: prices)
    monkeypatch.setattr(lgbm_vol, "load_macro", lambda **kw: macro)

    run_id = lgbm_vol.run(
        ticker="GC=F",
        target_kind="realized_vol",
        target_horizon=5,
        realized_vol_window=20,
        train_start="2014-06-01",
        val_days=120,
        test_days=120,
        step_days=120,
        min_train_days=3 * 365,
    )
    assert isinstance(run_id, str)

    preds = fetch_predictions(run_id)
    assert len(preds) > 0
    assert set(preds.columns) >= {"timestamp_utc", "ticker", "horizon", "prediction", "actual"}

    metrics = compute_metrics(run_id)
    assert not metrics.empty
    row = metrics.iloc[0]
    assert np.isfinite(row["rmse"])
