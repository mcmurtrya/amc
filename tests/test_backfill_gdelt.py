"""Tests for the GDELT backfill planner (pure, no BigQuery / no network)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from backfill_gdelt import _chunks, _usd, gap_ranges  # noqa: E402


def _days(start: str, end: str) -> set[str]:
    import pandas as pd

    return {d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="D")}


def test_gap_ranges_coalesces_consecutive_missing_days():
    # January, February, and April present; March missing, then May missing.
    present = _days("2020-01-01", "2020-02-29") | _days("2020-04-01", "2020-04-30")
    gaps = gap_ranges("2020-01-01", "2020-05-31", present)
    assert gaps == [("2020-03-01", "2020-03-31"), ("2020-05-01", "2020-05-31")]


def test_gap_ranges_empty_when_fully_covered():
    present = _days("2021-01-01", "2021-03-31")
    assert gap_ranges("2021-01-01", "2021-03-31", present) == []


def test_gap_ranges_resumes_after_mid_month_crash():
    """Regression for the 2026-07-02 overheat crash.

    The backfill died with 2016-11 populated only through the 21st. Month-level
    gap detection counted the month as present and would have silently skipped
    Nov 22-30; day-level detection must resume at exactly 2016-11-22.
    """
    present = _days("2015-02-18", "2016-11-21")
    gaps = gap_ranges("2015-02-18", "2019-12-31", present)
    assert gaps == [("2016-11-22", "2019-12-31")]


def test_gap_ranges_matches_real_corpus_shape():
    """2020-01..2021-08 present; everything else 2015-2026 is a gap."""
    present = _days("2020-01-01", "2021-08-31")
    gaps = gap_ranges("2015-02-18", "2026-06-30", present)
    assert gaps == [("2015-02-18", "2019-12-31"), ("2021-09-01", "2026-06-30")]


def test_chunks_cover_range_without_overlap():
    chunks = _chunks("2022-01-01", "2022-03-15", chunk_days=30)
    assert chunks[0][0] == "2022-01-01"
    assert chunks[-1][1] == "2022-03-15"
    # contiguous: each chunk starts the day after the previous ends
    import pandas as pd

    for (_, e0), (s1, _) in zip(chunks, chunks[1:], strict=False):
        assert pd.Timestamp(s1) == pd.Timestamp(e0) + pd.Timedelta(days=1)


def test_usd_respects_free_tier():
    assert _usd(int(0.5 * 1024**4)) == 0.0  # within 1 TB/mo free tier
    assert abs(_usd(2 * 1024**4) - 6.25) < 1e-6  # (2-1) TB * $6.25/TB
