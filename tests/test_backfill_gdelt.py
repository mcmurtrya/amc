"""Tests for the GDELT backfill planner (pure, no BigQuery / no network)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from backfill_gdelt import _chunks, _usd, gap_ranges  # noqa: E402


def test_gap_ranges_coalesces_consecutive_missing_months():
    present = {"2020-01", "2020-02", "2020-04"}  # March missing, then May+ missing
    gaps = gap_ranges("2020-01-01", "2020-05-31", present)
    assert ("2020-03-01", "2020-03-31") in gaps
    assert ("2020-05-01", "2020-05-31") in gaps
    # April is present -> never appears as a gap start/end
    assert all("2020-04" not in lo for lo, _ in gaps)


def test_gap_ranges_empty_when_fully_covered():
    present = {"2021-01", "2021-02", "2021-03"}
    assert gap_ranges("2021-01-01", "2021-03-31", present) == []


def test_gap_ranges_matches_real_corpus_shape():
    """The real corpus has 2020-01..2021-08 present; everything else 2015-2026 is a gap."""
    present = {f"2020-{m:02d}" for m in range(1, 13)} | {f"2021-{m:02d}" for m in range(1, 9)}
    gaps = gap_ranges("2015-02-18", "2026-06-30", present)
    # Two big gaps around the present block (pre-2020 and post-2021-08).
    assert any(lo.startswith("2015") for lo, _ in gaps)
    assert any(lo == "2021-09-01" for lo, _ in gaps)


def test_chunks_cover_range_without_overlap():
    chunks = _chunks("2022-01-01", "2022-03-15", chunk_days=30)
    assert chunks[0][0] == "2022-01-01"
    assert chunks[-1][1] == "2022-03-15"
    # contiguous: each chunk starts the day after the previous ends
    import pandas as pd
    for (_, e0), (s1, _) in zip(chunks, chunks[1:]):
        assert pd.Timestamp(s1) == pd.Timestamp(e0) + pd.Timedelta(days=1)


def test_usd_respects_free_tier():
    assert _usd(int(0.5 * 1024**4)) == 0.0        # within 1 TB/mo free tier
    assert abs(_usd(2 * 1024**4) - 6.25) < 1e-6   # (2-1) TB * $6.25/TB
