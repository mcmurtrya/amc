"""Tests for the FRED ingestion utilities. Pure-function side only — no network."""

from __future__ import annotations

import pandas as pd

from metals.data.fred import EXPECTED_PER_YEAR, coverage_report


def _long_df(rows):
    return pd.DataFrame(rows, columns=["timestamp_utc", "series_id", "value"])


def test_coverage_report_empty_input_returns_empty_with_schema():
    out = coverage_report(_long_df([]), start="2010-01-01", end="2025-12-31")
    assert out.empty
    assert set(out.columns) >= {
        "series_id",
        "freq",
        "rows",
        "first_obs",
        "last_obs",
        "expected_rows",
        "coverage",
        "flagged",
    }


def test_coverage_report_flags_short_series():
    """A series with only ~2 obs in a 16-year window must be flagged."""
    df = _long_df(
        [
            (pd.Timestamp("2023-05-15"), "BAA10Y", 1.5),
            (pd.Timestamp("2024-01-01"), "BAA10Y", 1.6),
        ]
    )
    out = coverage_report(
        df,
        start="2010-01-01",
        end="2025-12-31",
        series_freq={"BAA10Y": "daily"},
    )
    row = out.iloc[0]
    assert row["series_id"] == "BAA10Y"
    assert bool(row["flagged"]) is True
    assert row["coverage"] < 0.01


def test_coverage_report_passes_full_series():
    """A series with ~252 obs/year over the window must not be flagged."""
    idx = pd.bdate_range("2020-01-01", "2024-12-31")
    df = _long_df([(t, "DGS10", 3.0) for t in idx])
    out = coverage_report(
        df,
        start="2020-01-01",
        end="2024-12-31",
        series_freq={"DGS10": "daily"},
    )
    row = out.iloc[0]
    assert bool(row["flagged"]) is False
    assert row["coverage"] > 0.9


def test_coverage_report_respects_frequency():
    """Weekly series with ~52 obs/year should pass at weekly expectation."""
    idx = pd.date_range("2020-01-01", "2024-12-31", freq="W-FRI")
    df = _long_df([(t, "WALCL", 1e12) for t in idx])
    out = coverage_report(
        df,
        start="2020-01-01",
        end="2024-12-31",
        series_freq={"WALCL": "weekly"},
    )
    row = out.iloc[0]
    assert row["freq"] == "weekly"
    assert bool(row["flagged"]) is False


def test_coverage_report_threshold_is_configurable():
    """Same data, stricter threshold should flip the flag."""
    idx = pd.bdate_range("2020-01-01", "2020-12-31")  # ~261 obs
    df = _long_df([(t, "DGS10", 3.0) for t in idx])
    relaxed = coverage_report(
        df,
        start="2020-01-01",
        end="2024-12-31",
        series_freq={"DGS10": "daily"},
        min_coverage=0.1,
    )
    strict = coverage_report(
        df,
        start="2020-01-01",
        end="2024-12-31",
        series_freq={"DGS10": "daily"},
        min_coverage=0.5,
    )
    assert bool(relaxed.iloc[0]["flagged"]) is False
    assert bool(strict.iloc[0]["flagged"]) is True


def test_expected_per_year_constants_present():
    """Smoke check the lookup table is sane."""
    assert EXPECTED_PER_YEAR["daily"] > EXPECTED_PER_YEAR["weekly"]
    assert EXPECTED_PER_YEAR["weekly"] > EXPECTED_PER_YEAR["monthly"]
