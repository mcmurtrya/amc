"""Corpus coverage audit — find GDELT ingestion gaps in the ``headlines`` table.

The Stage-0 smoke turned up contiguous holes inside the title era (2024-01 has
only 2024-01-15; 2025-06 stops at 06-14) — bounded by fully-covered months, so
they are *our* ingestion gaps, not GDELT upstream holes (GKG is a continuous
15-min feed; the only known upstream empty day is 2017-08-29). This script maps
them so a targeted re-pull can be scoped before running it.

Reports: per-day row counts across the whole corpus, contiguous MISSING-day
ranges (0 rows), PARTIAL days (suspiciously low counts), and title-era
``page_title`` completeness per year.

    uv run python scripts/coverage_audit.py                 # full corpus
    uv run python scripts/coverage_audit.py --since 2019-09-22
    uv run python scripts/coverage_audit.py --list-missing   # every missing day
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta

import pandas as pd

from metals.data.db import connection

TITLE_ERA_START = "2019-09-22"
PARTIAL_FRAC = 0.10  # a day below this fraction of the median day is "partial"


def _day_counts(since: str | None) -> pd.DataFrame:
    where = "WHERE timestamp_utc >= ?" if since else ""
    params = [since] if since else []
    sql = (
        f"SELECT CAST(timestamp_utc AS DATE) AS d, count(*) AS n "
        f"FROM headlines {where} GROUP BY 1 ORDER BY 1"
    )
    with connection() as conn:
        df = conn.execute(sql, params).fetchdf()
    df["d"] = pd.to_datetime(df["d"]).dt.date
    return df


def _title_coverage_by_year(floor: str) -> pd.DataFrame:
    sql = (
        "SELECT strftime(timestamp_utc, '%Y') AS yr, count(*) AS n, "
        "count(page_title) AS pt FROM headlines WHERE timestamp_utc >= ? "
        "GROUP BY 1 ORDER BY 1"
    )
    with connection() as conn:
        return conn.execute(sql, [floor]).fetchdf()


def _contiguous(days: list[date]) -> list[tuple[date, date]]:
    ranges: list[tuple[date, date]] = []
    start = prev = None
    for d in sorted(days):
        if start is None:
            start = prev = d
        elif prev is not None and (d - prev).days == 1:
            prev = d
        else:
            assert start is not None and prev is not None
            ranges.append((start, prev))
            start = prev = d
    if start is not None and prev is not None:
        ranges.append((start, prev))
    return ranges


def _fmt_range(lo: date, hi: date) -> str:
    days = (hi - lo).days + 1
    return f"{lo} → {hi}  ({days} day{'s' if days > 1 else ''})"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--since", default=None, help="audit from this date (YYYY-MM-DD); default = full corpus"
    )
    ap.add_argument(
        "--list-missing", action="store_true", help="list every missing day, not just ranges"
    )
    args = ap.parse_args()

    dc = _day_counts(args.since)
    if dc.empty:
        print("No headline rows in range.")
        return

    present = set(dc["d"].tolist())
    lo, hi = dc["d"].min(), dc["d"].max()
    all_days = [lo + timedelta(days=i) for i in range((hi - lo).days + 1)]
    missing = [d for d in all_days if d not in present]
    median_day = int(dc["n"].median())
    # Per-YEAR median: GKG daily volume is non-stationary, so a single global
    # median false-flags low-volume early-era days and masks dips in busy years.
    dc = dc.assign(yr=pd.to_datetime(dc["d"]).dt.year)
    dc["yr_median"] = dc.groupby("yr")["n"].transform("median")
    partial = dc[dc["n"] < PARTIAL_FRAC * dc["yr_median"]]

    print(f"Corpus span: {lo} → {hi}  ({len(all_days):,} calendar days)")
    print(
        f"  present: {len(present):,}   MISSING: {len(missing):,}   "
        f"median rows/day: {median_day:,}   total rows: {int(dc['n'].sum()):,}"
    )
    print()

    gaps = _contiguous(missing)
    print(f"MISSING-day gaps (0 rows) — {len(gaps)} contiguous window(s):")
    if not gaps:
        print("  (none)")
    for lo_g, hi_g in gaps:
        print(f"  {_fmt_range(lo_g, hi_g)}")
    if args.list_missing and missing:
        print("  all missing days: " + ", ".join(str(d) for d in sorted(missing)))
    print()

    print(f"PARTIAL days (< {PARTIAL_FRAC:.0%} of that day's YEAR median):")
    if partial.empty:
        print("  (none)")
    for _, r in partial.iterrows():
        print(f"  {r['d']}: {int(r['n']):,} rows (year median {int(r['yr_median']):,})")
    print()

    title_floor = max(args.since, TITLE_ERA_START) if args.since else TITLE_ERA_START
    tc = _title_coverage_by_year(title_floor)
    print(f"Title-era page_title completeness by year (>= {title_floor}):")
    for _, r in tc.iterrows():
        pct = 100 * int(r["pt"]) / max(int(r["n"]), 1)
        print(f"  {r['yr']}: {int(r['pt']):>11,} / {int(r['n']):>11,}  ({pct:.2f}%)")

    print()
    total_missing_titleera = sum(1 for d in missing if str(d) >= TITLE_ERA_START)
    print(
        f"SUMMARY: {len(missing):,} missing days in {len(gaps)} window(s); "
        f"{total_missing_titleera:,} of them in the title era (re-pullable via "
        f"backfill_gdelt.py + the Extras/page_title pull, one process per month)."
    )


if __name__ == "__main__":
    main()
