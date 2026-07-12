"""Tests for the BERTopic wrapper. Heavy deps are imported lazily, so the
purely-data tests run without BERTopic installed."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from metals.features.topics import (
    topic_prevalence_per_day,
)


@pytest.fixture(autouse=True)
def tmp_db(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.duckdb")
        monkeypatch.setenv("METALS_DB_PATH", db_path)
        yield db_path


def test_topic_prevalence_per_day_sums_to_one_per_day():
    ts = pd.to_datetime(
        [
            "2024-01-01 09:00",
            "2024-01-01 12:00",
            "2024-01-01 18:00",
            "2024-01-02 10:00",
            "2024-01-02 11:00",
        ]
    )
    topics = np.array([0, 0, 1, 2, 2])
    out = topic_prevalence_per_day(ts, topics, include_noise=False)
    sums = out.groupby("timestamp_utc")["prevalence"].sum()
    assert (sums - 1.0).abs().max() < 1e-9


def test_topic_prevalence_per_day_excludes_noise_by_default():
    ts = pd.to_datetime(["2024-01-01", "2024-01-01"])
    topics = np.array([-1, 0])
    out = topic_prevalence_per_day(ts, topics, include_noise=False)
    assert (out["topic_id"] != -1).all()
    assert len(out) == 1
    assert out.iloc[0]["prevalence"] == 1.0


def test_topic_prevalence_per_day_includes_noise_when_asked():
    ts = pd.to_datetime(["2024-01-01", "2024-01-01"])
    topics = np.array([-1, 0])
    out = topic_prevalence_per_day(ts, topics, include_noise=True)
    assert -1 in set(out["topic_id"])
    assert (out["prevalence"] == 0.5).all()


def test_topic_prevalence_empty_input():
    out = topic_prevalence_per_day(pd.Series([], dtype="datetime64[ns]"), np.array([], dtype=int))
    assert out.empty
    assert set(out.columns) == {"timestamp_utc", "topic_id", "prevalence"}


def test_upsert_and_load_topic_prevalence_round_trip():
    from metals.data.migrations.runner import apply_migrations
    from metals.features.topics import (
        load_topic_prevalence_wide,
        upsert_topic_prevalence,
    )

    apply_migrations(verbose=False)
    df = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
            "topic_id": [0, 1, 0],
            "prevalence": [0.7, 0.3, 1.0],
        }
    )
    n = upsert_topic_prevalence(df)
    assert n == 3
    wide = load_topic_prevalence_wide()
    assert not wide.empty
    assert "topic_0" in wide.columns and "topic_1" in wide.columns
    assert wide.loc[pd.Timestamp("2024-01-02"), "topic_0"] == pytest.approx(1.0)
    # Missing (date, topic) pairs become 0.0 on pivot
    assert wide.loc[pd.Timestamp("2024-01-02"), "topic_1"] == pytest.approx(0.0)


def test_fit_topic_model_requires_bertopic():
    """The function should import bertopic lazily; this just confirms the import
    path triggers cleanly when called. Skips if bertopic isn't installed."""
    pytest.importorskip("bertopic")
    from metals.features.topics import fit_topic_model

    # We don't actually fit — that requires sentence-transformers + a real
    # corpus. We assert the symbol is importable and callable.
    assert callable(fit_topic_model)
