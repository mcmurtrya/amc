"""Tests for the two-phase title backfill (pure, no BigQuery / no DuckDB)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from backfill_titles import build_fast_query, finish_title  # noqa: E402

from metals.data.gdelt import PAGE_TITLE_MAX_CHARS, extract_page_title  # noqa: E402


def test_finish_title_matches_extract_page_title():
    """finish_title(group) must equal extract_page_title on the whole Extras blob.

    The SQL pull extracts the regex group in BigQuery and only the post-regex
    normalisation runs in python — the two paths have to stay byte-for-byte
    equivalent (a 323K-row live comparison on 2020-01 had zero mismatches).
    """
    cases = [
        "Gold hits record high",
        "  Gold \n\t hits   record high  ",  # whitespace collapse incl. NBSP
        "Fed&#39;s Powell: &amp; then rates &quot;hold&quot;",  # entity decode, once
        "x" * (PAGE_TITLE_MAX_CHARS + 100),  # cap
        "",  # empty group -> None
        "官方回应黄金价格",  # non-latin passes through
    ]
    for raw in cases:
        assert finish_title(raw) == extract_page_title(f"<PAGE_TITLE>{raw}</PAGE_TITLE>"), raw


def test_finish_title_none_and_empty():
    assert finish_title(None) is None
    assert finish_title(float("nan")) is None
    assert finish_title("") is None
    assert finish_title("   ") is None  # collapses to empty -> None


def test_build_fast_query_swaps_select_keeps_where():
    themes = ["ECON_GOLD", "COMMODITIES_SILVER"]
    q = build_fast_query("2020-01-01", "2020-01-31", themes)
    # New projection: extraction in SQL, no fat/unused columns downloaded.
    assert "REGEXP_EXTRACT(Extras" in q
    assert "AS raw_title" in q
    assert "V2Tone" not in q
    assert "SourceCommonName" not in q
    # FROM/WHERE reused verbatim from build_query: partition pruning, DATE
    # bounds, and the theme filter must all survive the splice.
    assert q.count("SELECT") == 1
    assert '_PARTITIONTIME >= TIMESTAMP("2020-01-01")' in q
    assert "V2Themes IS NOT NULL" in q
    assert "ECON_GOLD|COMMODITIES_SILVER" in q
