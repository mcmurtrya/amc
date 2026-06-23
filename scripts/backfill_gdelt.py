"""Plan and execute a GDELT GKG backfill to fill the headlines coverage gaps.

The `headlines` corpus currently has a large hole (only 2020-01..2021-08 is
continuous, plus a 2024-01 fragment). This tool finds the missing months and,
optionally, fills them — safely.

Three modes, increasingly committal:

  --gaps      (default) Report which months are missing in the target range.
              Reads the local DuckDB only; no BigQuery, no credentials needed.
  --estimate  Add a BigQuery *dry run* per gap chunk to report bytes scanned and
              the $ cost to fill the gaps. Free (dry runs are not billed).
  --execute   Actually pull the missing chunks and upsert them, with a hard
              per-chunk `--max-gb` cap on bytes billed so a mistake cannot run up
              a bill. Idempotent: months already present are skipped.

--estimate / --execute require GOOGLE_APPLICATION_CREDENTIALS (BigQuery Data
Viewer + Job User + Read Session User), exactly like `metals.data.gdelt`.

Examples:
    uv run python scripts/backfill_gdelt.py                         # gaps, 2015-present
    uv run python scripts/backfill_gdelt.py --estimate              # + cost to fill
    uv run python scripts/backfill_gdelt.py --execute --max-gb 100  # fill, capped
    uv run python scripts/backfill_gdelt.py --start 2021-09-01 --end 2024-01-01 --execute
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from metals.data.db import connection  # noqa: E402
from metals.data.gdelt import (  # noqa: E402
    build_query,
    load_themes,
    parse_gkg_rows,
    upsert_headlines,
)

# GDELT GKG 2.0 (with V2Themes / V2Tone) begins 2015-02-18. Earlier history only
# exists in GKG 1.0, which has a different schema and is out of scope here.
GKG2_START = "2015-02-18"
PRICE_USD_PER_TB = 6.25  # BigQuery on-demand (US); first 1 TB/month is free.
FREE_TIER_TB = 1.0


def present_months(conn) -> set[str]:
    """Set of 'YYYY-MM' strings that already have >=1 headline row."""
    rows = conn.execute(
        "SELECT DISTINCT strftime(timestamp_utc, '%Y-%m') FROM headlines"
    ).fetchall()
    return {r[0] for r in rows if r[0]}


def gap_ranges(start: str, end: str, present: set[str]) -> list[tuple[str, str]]:
    """Contiguous runs of missing months in [start, end], as (first_day, last_day)."""
    months = pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M")
    ranges: list[tuple[str, str]] = []
    run: list[pd.Period] = []

    def flush() -> None:
        if run:
            lo, hi = run[0], run[-1]
            ranges.append((lo.start_time.date().isoformat(),
                           hi.end_time.date().isoformat()))
            run.clear()

    for m in months:
        if m.strftime("%Y-%m") in present:
            flush()
        else:
            run.append(m)
    flush()
    return ranges


def _chunks(start: str, end: str, chunk_days: int) -> list[tuple[str, str]]:
    out = []
    cur = pd.Timestamp(start).normalize()
    last = pd.Timestamp(end).normalize()
    while cur <= last:
        nxt = min(cur + pd.Timedelta(days=chunk_days - 1), last)
        out.append((cur.date().isoformat(), nxt.date().isoformat()))
        cur = nxt + pd.Timedelta(days=1)
    return out


def estimate_bytes(start: str, end: str, themes: list[str]) -> int:
    """BigQuery dry run: bytes the query *would* scan. Not billed."""
    from google.cloud import bigquery  # lazy: only needed for BQ modes

    client = bigquery.Client()
    cfg = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    job = client.query(build_query(start, end, themes), job_config=cfg)
    return int(job.total_bytes_processed)


def pull_chunk(start: str, end: str, themes: list[str], max_bytes: int) -> int:
    """Run the real query for one chunk (bytes-billed-capped) and upsert."""
    from google.cloud import bigquery

    client = bigquery.Client()
    cfg = bigquery.QueryJobConfig(maximum_bytes_billed=max_bytes)
    job = client.query(build_query(start, end, themes), job_config=cfg)
    raw = job.to_dataframe()
    return upsert_headlines(parse_gkg_rows(raw, themes))


def _usd(total_bytes: int) -> float:
    tb = total_bytes / 1024**4
    billable_tb = max(0.0, tb - FREE_TIER_TB)
    return billable_tb * PRICE_USD_PER_TB


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default=GKG2_START, help=f"YYYY-MM-DD (default {GKG2_START}).")
    ap.add_argument("--end", default=date.today().isoformat(), help="YYYY-MM-DD (default today).")
    ap.add_argument("--chunk-days", type=int, default=30, help="Days per BigQuery chunk.")
    ap.add_argument("--max-gb", type=float, default=100.0,
                    help="Hard cap on bytes billed per chunk (GB) under --execute.")
    ap.add_argument("--estimate", action="store_true", help="Dry-run cost to fill the gaps.")
    ap.add_argument("--execute", action="store_true", help="Actually pull the missing chunks.")
    args = ap.parse_args()

    with connection(read_only=True) as conn:
        present = present_months(conn)
    gaps = gap_ranges(args.start, args.end, present)

    print(f"Target range : {args.start} .. {args.end}")
    print(f"Months present: {len(present)}")
    if not gaps:
        print("No gaps — corpus is continuous over the target range.")
        return
    total_missing = sum(
        len(pd.period_range(lo, hi, freq="M")) for lo, hi in gaps
    )
    print(f"Missing months: {total_missing}, in {len(gaps)} gap range(s):")
    for lo, hi in gaps:
        print(f"  - {lo} .. {hi}")

    if not (args.estimate or args.execute):
        print("\nRe-run with --estimate for cost, or --execute --max-gb N to fill.")
        return

    themes = load_themes()
    max_bytes = int(args.max_gb * 1024**3)
    grand_bytes = 0
    grand_rows = 0
    for lo, hi in gaps:
        for cs, ce in _chunks(lo, hi, args.chunk_days):
            if args.execute:
                n = pull_chunk(cs, ce, themes, max_bytes)
                grand_rows += n
                print(f"  pulled {cs}..{ce}: {n:,} rows")
            else:
                b = estimate_bytes(cs, ce, themes)
                grand_bytes += b
                print(f"  {cs}..{ce}: scans {b/1024**3:,.1f} GB")

    if args.execute:
        print(f"\nDone. Rows upserted: {grand_rows:,}")
    else:
        print(f"\nTotal scan: {grand_bytes/1024**4:,.2f} TB"
              f"  ~= ${_usd(grand_bytes):,.2f} (after {FREE_TIER_TB:.0f} TB/mo free tier)")
        print("Dry runs are free. Re-run with --execute --max-gb N to fill.")


if __name__ == "__main__":
    main()
