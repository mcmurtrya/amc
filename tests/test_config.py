"""Tests for YAML config loading."""

from __future__ import annotations

from metals.data.config import fred_series, load_yaml, universe


def test_load_yaml_finds_universe():
    cfg = load_yaml("universe")
    assert "metals" in cfg
    assert any(row["ticker"] == "GC=F" for row in cfg["metals"])


def test_universe_helper_returns_dict():
    cfg = universe()
    assert "metals" in cfg
    assert "etfs" in cfg
    assert "benchmarks" in cfg
    assert "date_range" in cfg


def test_fred_series_helper_returns_dict():
    cfg = fred_series()
    assert "series" in cfg
    series_ids = [row["id"] for row in cfg["series"]]
    assert "DGS10" in series_ids
    assert "T10YIE" in series_ids
