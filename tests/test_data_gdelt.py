"""Tests for the GDELT GKG parser (pure-function side; no network)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from metals.data.gdelt import build_query, load_themes, parse_gkg_rows


def test_load_themes_returns_nonempty_list():
    # NB: COMMODITIES_GOLD is in the planning doc but does NOT exist in the
    # real GDELT 2.0 vocabulary — see the comment block in configs/gdelt_themes.yaml.
    # We assert the canonical replacements: ECON_GOLDPRICE for gold,
    # ECON_INFLATION as a generic anchor.
    themes = load_themes()
    assert len(themes) > 5
    assert "ECON_GOLDPRICE" in themes
    assert "ECON_INFLATION" in themes


def test_build_query_includes_date_bounds_and_themes():
    sql = build_query("2024-01-01", "2024-01-31", ["COMMODITIES_GOLD", "ECON_INFLATION"])
    assert "20240101000000" in sql      # lo bound, exclusive of below
    assert "20240201000000" in sql      # hi bound = (end+1) * 1_000_000
    assert "COMMODITIES_GOLD" in sql
    assert "ECON_INFLATION" in sql
    assert "REGEXP_CONTAINS" in sql


def _raw_gkg_row(
    *,
    date_int: int,
    themes: str,
    tone: str = "0,0,0,0,0,0",
    source: str = "reuters.com",
    url: str = "https://www.reuters.com/article",
) -> dict:
    return {
        "date_int": date_int,
        "source_common_name": source,
        "document_identifier": url,
        "v2themes": themes,
        "v2tone": tone,
    }


def test_parse_extracts_themes_and_tone():
    themes = ["COMMODITIES_GOLD", "ECON_INFLATION", "CENTRAL_BANK"]
    raw = pd.DataFrame([_raw_gkg_row(
        date_int=20240115123000,
        themes="COMMODITIES_GOLD,42;ECON_INFLATION,150;NOT_A_THEME,200",
        tone="-2.5,1.0,3.5,4.5,12.3,4.1",
    )])
    out = parse_gkg_rows(raw, themes)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["timestamp_utc"] == pd.Timestamp("2024-01-15 12:30:00")
    parsed_themes = json.loads(row["themes"])
    assert "COMMODITIES_GOLD" in parsed_themes
    assert "ECON_INFLATION" in parsed_themes
    assert "NOT_A_THEME" not in parsed_themes
    assert row["tone_overall"] == pytest.approx(-2.5)
    assert row["tone_positive"] == pytest.approx(1.0)
    assert row["tone_negative"] == pytest.approx(3.5)
    assert row["tone_polarity"] == pytest.approx(4.5)
    assert row["tone_ard"] == pytest.approx(12.3)
    assert row["tone_sgrd"] == pytest.approx(4.1)


def test_parse_drops_rows_with_no_filtered_theme():
    themes = ["COMMODITIES_GOLD"]
    raw = pd.DataFrame([_raw_gkg_row(
        date_int=20240115000000,
        themes="ECON_INFLATION,5;CENTRAL_BANK,10",   # neither in filter
    )])
    out = parse_gkg_rows(raw, themes)
    assert out.empty


def test_parse_handles_missing_themes_or_tone():
    themes = ["COMMODITIES_GOLD"]
    raw = pd.DataFrame([
        _raw_gkg_row(date_int=20240115000000, themes="COMMODITIES_GOLD,5", tone=""),
        _raw_gkg_row(date_int=20240116000000, themes="COMMODITIES_GOLD,5", tone="x,y,z"),
    ])
    out = parse_gkg_rows(raw, themes)
    assert len(out) == 2
    assert pd.isna(out.iloc[0]["tone_overall"])
    # Row 1: tone fields are non-numeric, parser swallows quietly
    assert pd.isna(out.iloc[1]["tone_overall"])


def test_parse_empty_frame_returns_empty_with_schema():
    raw = pd.DataFrame(columns=[
        "date_int", "source_common_name", "document_identifier",
        "v2themes", "v2tone",
    ])
    out = parse_gkg_rows(raw, ["COMMODITIES_GOLD"])
    assert out.empty
    assert "timestamp_utc" in out.columns
    assert "tone_overall" in out.columns


def test_parse_offset_suffix_stripped_from_theme_codes():
    """Themes are stored as 'CODE,offset' in V2Themes - we keep only CODE."""
    raw = pd.DataFrame([_raw_gkg_row(
        date_int=20240115000000,
        themes="COMMODITIES_GOLD,100;COMMODITIES_GOLD,250",   # duplicate after offset strip
    )])
    out = parse_gkg_rows(raw, ["COMMODITIES_GOLD"])
    parsed = json.loads(out.iloc[0]["themes"])
    assert parsed == ["COMMODITIES_GOLD"]   # deduped
