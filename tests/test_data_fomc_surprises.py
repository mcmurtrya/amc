"""Tests for the Bauer-Swanson FOMC surprises cleaner (pure-function side)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metals.data.fomc_surprises import (
    REQUIRED_COLUMNS,
    SURPRISE_COLUMNS,
    clean_surprises_dataframe,
)


def _raw(rows: list[dict]) -> pd.DataFrame:
    cols = list(REQUIRED_COLUMNS)
    return pd.DataFrame([{c: r.get(c, np.nan) for c in cols} for r in rows], columns=cols)


def test_basic_rename_and_types():
    raw = _raw([{
        "Date": "2008-12-16",
        "Unscheduled": 0,
        "FF1": -0.42,
        "FF2": -0.36,
        "ED4": -0.18,
        "MPS": -1.05,
        "MPS_ORTH": -0.81,
    }])
    out = clean_surprises_dataframe(raw)
    assert len(out) == 1
    assert out.loc[0, "timestamp_utc"] == pd.Timestamp("2008-12-16")
    assert out.loc[0, "is_unscheduled"] is False or out.loc[0, "is_unscheduled"] == False  # noqa: E712
    assert out.loc[0, "mps"] == pytest.approx(-1.05)
    assert out.loc[0, "mps_orth"] == pytest.approx(-0.81)
    assert out.loc[0, "source"]   # source tag populated


def test_drops_rows_with_all_null_surprises():
    """A row with no surprise data at all must be dropped (1988 era)."""
    raw = _raw([
        {"Date": "1988-02-04", "Unscheduled": 1},  # all surprise cols NaN
        {"Date": "2010-11-03", "Unscheduled": 0, "MPS": -0.5, "MPS_ORTH": -0.4},
    ])
    out = clean_surprises_dataframe(raw)
    assert len(out) == 1
    assert out.loc[0, "timestamp_utc"] == pd.Timestamp("2010-11-03")


def test_keeps_rows_with_partial_surprises():
    """If even one surprise column is non-null the row is kept."""
    raw = _raw([
        {"Date": "1991-10-15", "Unscheduled": 0, "FF1": 0.12},  # only FF1
    ])
    out = clean_surprises_dataframe(raw)
    assert len(out) == 1
    assert out.loc[0, "ff1"] == pytest.approx(0.12)
    assert pd.isna(out.loc[0, "mps"])


def test_unscheduled_is_boolean():
    raw = _raw([
        {"Date": "2020-03-15", "Unscheduled": 1, "MPS": -2.0},
        {"Date": "2020-04-29", "Unscheduled": 0, "MPS":  0.1},
    ])
    out = clean_surprises_dataframe(raw)
    assert bool(out.loc[0, "is_unscheduled"]) is True
    assert bool(out.loc[1, "is_unscheduled"]) is False


def test_missing_required_column_raises():
    raw = pd.DataFrame({"Date": ["2010-01-01"], "MPS": [0.1]})  # missing many
    with pytest.raises(RuntimeError, match="missing expected columns"):
        clean_surprises_dataframe(raw)


def test_output_is_sorted_by_date():
    raw = _raw([
        {"Date": "2020-04-29", "Unscheduled": 0, "MPS": 0.1},
        {"Date": "2010-11-03", "Unscheduled": 0, "MPS": 0.0},
        {"Date": "2015-12-16", "Unscheduled": 0, "MPS": 0.2},
    ])
    out = clean_surprises_dataframe(raw)
    assert out["timestamp_utc"].is_monotonic_increasing
