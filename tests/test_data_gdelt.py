"""Tests for the GDELT GKG parser (pure-function side; no network)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from metals.data.gdelt import (
    PAGE_TITLE_MAX_CHARS,
    build_query,
    extract_page_title,
    extract_src_lang,
    load_themes,
    parse_gkg_rows,
)


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
    assert "20240101000000" in sql  # lo bound, exclusive of below
    assert "20240201000000" in sql  # hi bound = (end+1) * 1_000_000
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
    translation_info: str | None = "",
    extras: str | None = "<PAGE_TITLE>Gold steadies</PAGE_TITLE>",
) -> dict:
    return {
        "date_int": date_int,
        "source_common_name": source,
        "document_identifier": url,
        "v2themes": themes,
        "v2tone": tone,
        "translation_info": translation_info,
        "extras": extras,
    }


def test_parse_extracts_themes_and_tone():
    themes = ["COMMODITIES_GOLD", "ECON_INFLATION", "CENTRAL_BANK"]
    raw = pd.DataFrame(
        [
            _raw_gkg_row(
                date_int=20240115123000,
                themes="COMMODITIES_GOLD,42;ECON_INFLATION,150;NOT_A_THEME,200",
                tone="-2.5,1.0,3.5,4.5,12.3,4.1",
            )
        ]
    )
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
    raw = pd.DataFrame(
        [
            _raw_gkg_row(
                date_int=20240115000000,
                themes="ECON_INFLATION,5;CENTRAL_BANK,10",  # neither in filter
            )
        ]
    )
    out = parse_gkg_rows(raw, themes)
    assert out.empty


def test_parse_handles_missing_themes_or_tone():
    themes = ["COMMODITIES_GOLD"]
    raw = pd.DataFrame(
        [
            _raw_gkg_row(date_int=20240115000000, themes="COMMODITIES_GOLD,5", tone=""),
            _raw_gkg_row(date_int=20240116000000, themes="COMMODITIES_GOLD,5", tone="x,y,z"),
        ]
    )
    out = parse_gkg_rows(raw, themes)
    assert len(out) == 2
    assert pd.isna(out.iloc[0]["tone_overall"])
    # Row 1: tone fields are non-numeric, parser swallows quietly
    assert pd.isna(out.iloc[1]["tone_overall"])


def test_parse_empty_frame_returns_empty_with_schema():
    raw = pd.DataFrame(
        columns=[
            "date_int",
            "source_common_name",
            "document_identifier",
            "v2themes",
            "v2tone",
        ]
    )
    out = parse_gkg_rows(raw, ["COMMODITIES_GOLD"])
    assert out.empty
    assert "timestamp_utc" in out.columns
    assert "tone_overall" in out.columns
    assert "page_title" in out.columns
    assert "src_lang" in out.columns


def test_parse_offset_suffix_stripped_from_theme_codes():
    """Themes are stored as 'CODE,offset' in V2Themes - we keep only CODE."""
    raw = pd.DataFrame(
        [
            _raw_gkg_row(
                date_int=20240115000000,
                themes="COMMODITIES_GOLD,100;COMMODITIES_GOLD,250",  # duplicate after offset strip
            )
        ]
    )
    out = parse_gkg_rows(raw, ["COMMODITIES_GOLD"])
    parsed = json.loads(out.iloc[0]["themes"])
    assert parsed == ["COMMODITIES_GOLD"]  # deduped


# --- Wide-ingest enrichment: page_title / src_lang (migration 007) ---


def test_build_query_selects_enrichment_columns():
    sql = build_query("2024-01-01", "2024-01-31", ["ECON_INFLATION"])
    assert "TranslationInfo" in sql
    assert "Extras" in sql


def test_extract_page_title_unescapes_entities_and_collapses_whitespace():
    extras = "<PAGE_TITLE>Gold &amp; Silver &#8211; What&#x27;s\n  Next?</PAGE_TITLE>"
    assert extract_page_title(extras) == "Gold & Silver – What's Next?"


def test_extract_page_title_preserves_non_latin_scripts():
    extras = "<PAGE_TITLE>&#x642;&#x6CC;&#x645;&#x62A; &#x637;&#x644;&#x627;</PAGE_TITLE>"
    assert extract_page_title(extras) == "قیمت طلا"


def test_extract_page_title_absent_or_empty_is_none():
    assert extract_page_title(None) is None
    assert extract_page_title(float("nan")) is None
    assert extract_page_title("") is None
    assert extract_page_title("<SOME_OTHER_TAG>x</SOME_OTHER_TAG>") is None
    assert extract_page_title("<PAGE_TITLE>   </PAGE_TITLE>") is None


def test_extract_page_title_truncates_malformed_giants():
    extras = "<PAGE_TITLE>" + "x" * 10_000 + "</PAGE_TITLE>"
    title = extract_page_title(extras)
    assert title is not None
    assert len(title) == PAGE_TITLE_MAX_CHARS


def test_extract_src_lang_semantics():
    # Empty / NULL TranslationInfo (when the column WAS pulled) = English-original.
    assert extract_src_lang("") == "eng"
    assert extract_src_lang("   ") == "eng"
    assert extract_src_lang(None) == "eng"
    assert extract_src_lang(float("nan")) == "eng"
    # Machine-translated rows carry srclc.
    assert extract_src_lang("srclc:fra;eng:Moses 2.1.1") == "fra"
    assert extract_src_lang("SRCLC:ZH;eng:GT") == "zh"
    # Populated but unparseable = malformed, not English.
    assert extract_src_lang("eng:Moses 2.1.1") is None


def test_parse_derives_page_title_and_src_lang():
    raw = pd.DataFrame(
        [
            _raw_gkg_row(
                date_int=20240115123000,
                themes="ECON_INFLATION,5",
                translation_info="srclc:srp;eng:GT",
                extras="<PAGE_TITLE>Zlato &amp; srebro rastu</PAGE_TITLE>",
            ),
            _raw_gkg_row(
                date_int=20240115124500,
                themes="ECON_INFLATION,5",
                url="https://example.com/other",
                translation_info="",
                extras="<NO_TITLE_HERE/>",
            ),
        ]
    )
    out = parse_gkg_rows(raw, ["ECON_INFLATION"])
    assert out.iloc[0]["page_title"] == "Zlato & srebro rastu"
    assert out.iloc[0]["src_lang"] == "srp"
    assert pd.isna(out.iloc[1]["page_title"])
    assert out.iloc[1]["src_lang"] == "eng"


def test_parse_tolerates_narrow_frames_without_enrichment_columns():
    """Frames from the pre-007 five-column query must land as NULL/NULL —
    never 'eng' — so a later wide re-pull can still fill them via COALESCE."""
    row = _raw_gkg_row(date_int=20240115000000, themes="ECON_INFLATION,5")
    row.pop("translation_info")
    row.pop("extras")
    out = parse_gkg_rows(pd.DataFrame([row]), ["ECON_INFLATION"])
    assert len(out) == 1
    assert "page_title" in out.columns
    assert "src_lang" in out.columns
    assert pd.isna(out.iloc[0]["page_title"])
    assert pd.isna(out.iloc[0]["src_lang"])


# --- Upsert landing semantics (temp DB, real migrations) ---


@pytest.fixture()
def tmp_db(monkeypatch, tmp_path):
    db_file = tmp_path / "test_gdelt.duckdb"
    monkeypatch.setenv("METALS_DB_PATH", str(db_file))
    from metals.data.migrations.runner import apply_migrations

    apply_migrations(verbose=False)
    return db_file


def _fetch_headlines():
    from metals.data.db import connection

    with connection(read_only=True) as conn:
        return conn.execute(
            """
            SELECT headline_id, page_title, src_lang, tone_overall
            FROM headlines ORDER BY headline_id
            """
        ).fetchdf()


def test_upsert_wide_then_narrow_preserves_titles(tmp_db):
    from metals.data.gdelt import upsert_headlines

    themes = ["ECON_INFLATION"]
    wide_raw = pd.DataFrame(
        [
            _raw_gkg_row(
                date_int=20240115123000,
                themes="ECON_INFLATION,5",
                tone="-1.0,0,0,0,0,0",
                translation_info="srclc:fra;eng:Moses",
                extras="<PAGE_TITLE>L&#x27;or monte</PAGE_TITLE>",
            )
        ]
    )
    upsert_headlines(parse_gkg_rows(wide_raw, themes))
    landed = _fetch_headlines()
    assert len(landed) == 1
    assert landed.iloc[0]["page_title"] == "L'or monte"
    assert landed.iloc[0]["src_lang"] == "fra"

    # A narrow re-pull of the SAME row (pre-007 query shape: no enrichment
    # columns → NULLs) must update tone but keep the landed title/lang.
    narrow_row = _raw_gkg_row(
        date_int=20240115123000, themes="ECON_INFLATION,5", tone="2.5,0,0,0,0,0"
    )
    narrow_row.pop("translation_info")
    narrow_row.pop("extras")
    upsert_headlines(parse_gkg_rows(pd.DataFrame([narrow_row]), themes))

    after = _fetch_headlines()
    assert len(after) == 1  # same PK — updated, not duplicated
    assert after.iloc[0]["tone_overall"] == pytest.approx(2.5)
    assert after.iloc[0]["page_title"] == "L'or monte"  # COALESCE kept it
    assert after.iloc[0]["src_lang"] == "fra"

    # And a fresh wide pull CAN revise a title (EXCLUDED wins when non-NULL).
    wide_raw2 = wide_raw.copy()
    wide_raw2.loc[0, "extras"] = "<PAGE_TITLE>L&#x27;or monte encore</PAGE_TITLE>"
    upsert_headlines(parse_gkg_rows(wide_raw2, themes))
    final = _fetch_headlines()
    assert len(final) == 1
    assert final.iloc[0]["page_title"] == "L'or monte encore"
