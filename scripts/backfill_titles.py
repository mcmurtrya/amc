"""Two-phase page_title/src_lang backfill for *existing* headlines rows.

Filling migration-007 columns on rows that are already in the table is a
different workload from ingesting new rows, and the obvious path — the wide
``refresh()`` upsert — is pathologically slow at it: every row hits
``ON CONFLICT DO UPDATE``, and DuckDB's per-row conflict resolution through
the 140M-row ART primary-key index ran ~10 min per 10-day chunk when tried
live (2026-07-02), a 30-40 h job. The two phases below did the same 63.3M
rows in ~100 min of pulling + 31 s of updating:

  pull   BigQuery -> per-chunk parquet of (timestamp_utc, headline_id,
         page_title, src_lang). PAGE_TITLE is extracted *inside* BigQuery
         (REGEXP_EXTRACT on Extras), so the download is ~100-byte strings
         instead of multi-KB Extras blobs (~5x faster; scan cost identical —
         BigQuery bills scanned columns, not downloaded bytes). Files are
         written atomically and skipped when present, so a crashed pull
         resumes exactly. No DuckDB contact in this phase.
  apply  One bulk ``UPDATE ... FROM read_parquet(...)`` per year — a hash
         join, not 63M index conflicts (measured ~5 s per ~10M-row year).

The SQL title extraction is parity-tested against the python extractor
(``extract_page_title``): 323K-row live comparison had zero mismatches.
Run each phase per machine: the parquet directory is portable, so a second
machine (the A6000 server) only needs ``apply`` — no BigQuery re-scan.

Examples:
    uv run python scripts/backfill_titles.py pull --start 2020-01-01 \
        --end 2026-06-19 --outdir data/raw/title_backfill
    uv run python scripts/backfill_titles.py apply --dir data/raw/title_backfill

For multi-month pulls prefer one process per month (memory accumulates in the
BigQuery download path; see the 2026-07-02 journal entry):
    for m in 2020-01 2020-02 ...; do
        uv run python scripts/backfill_titles.py pull --start $m-01 ... ; done
"""

from __future__ import annotations

import argparse
import html
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from metals.data.db import connection  # noqa: E402
from metals.data.gdelt import (  # noqa: E402
    PAGE_TITLE_MAX_CHARS,
    build_query,
    extract_src_lang,
    load_themes,
)

SELECT_FAST = """
    SELECT
        DATE               AS date_int,
        DocumentIdentifier AS document_identifier,
        REGEXP_EXTRACT(Extras, r'(?is)<PAGE_TITLE>(.*?)</PAGE_TITLE>') AS raw_title,
        TranslationInfo    AS translation_info
""".strip()


def build_fast_query(start_date, end_date, themes) -> str:
    """The wide query's FROM/WHERE verbatim, with the SELECT swapped.

    Reusing ``build_query`` keeps the partition pruning, DATE bounds and the
    theme regex in one place; only the projected columns differ.
    """
    wide = build_query(start_date, end_date, themes)
    tail = "FROM" + wide.split("FROM", 1)[1]
    return f"{SELECT_FAST}\n    {tail}"


def finish_title(raw: object) -> str | None:
    """Post-regex steps of ``extract_page_title`` for the SQL-extracted group.

    Must stay byte-for-byte equivalent: unescape once, collapse whitespace,
    cap at PAGE_TITLE_MAX_CHARS, empty -> None.
    """
    if not isinstance(raw, str):
        return None
    title = " ".join(html.unescape(raw).split())
    return title[:PAGE_TITLE_MAX_CHARS] or None


def fetch_titles(start_date, end_date) -> pd.DataFrame:
    """One BigQuery pull, returning the 4-column update frame."""
    from google.cloud import bigquery

    themes = load_themes()
    client = bigquery.Client()
    raw = client.query(build_fast_query(start_date, end_date, themes)).to_dataframe()
    df = pd.DataFrame()
    df["timestamp_utc"] = pd.to_datetime(
        raw["date_int"].astype(str).str.zfill(14), format="%Y%m%d%H%M%S", errors="coerce"
    )
    url = raw["document_identifier"].astype("string")
    df["headline_id"] = (
        df["timestamp_utc"].dt.strftime("%Y%m%d%H%M%S").fillna("")
        + "_"
        + url.fillna("").str.slice(0, 200)
    )
    df["page_title"] = raw["raw_title"].map(finish_title).astype("string")
    df["src_lang"] = raw["translation_info"].map(extract_src_lang).astype("string")
    return df.dropna(subset=["timestamp_utc", "headline_id"]).reset_index(drop=True)


def run_pull(start: str, end: str, outdir: Path, chunk_days: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    cur = pd.Timestamp(start).normalize()
    ed = pd.Timestamp(end).normalize()
    while cur <= ed:
        nxt = min(cur + pd.Timedelta(days=chunk_days - 1), ed)
        f = outdir / f"titles_{cur.date()}_{nxt.date()}.parquet"
        if f.exists():
            print(f"  skip {f.name} (exists)", flush=True)
        else:
            df = fetch_titles(cur.date(), nxt.date())
            tmp = f.with_name(f.name + ".tmp")
            df.to_parquet(tmp, index=False)
            tmp.rename(f)
            print(f"  wrote {f.name}: {len(df):,} rows", flush=True)
        cur = nxt + pd.Timedelta(days=1)


def run_apply(parquet_dir: Path) -> int:
    """Yearly bulk UPDATE ... FROM the pulled parquet. Returns rows updated."""
    years = sorted({f.name[7:11] for f in parquet_dir.glob("titles_*.parquet")})
    if not years:
        raise SystemExit(f"No titles_*.parquet under {parquet_dir}")
    total = 0
    with connection() as conn:
        for year in years:
            glob = str(parquet_dir / f"titles_{year}-*.parquet")
            t0 = time.time()
            n = conn.execute(
                f"""
                UPDATE headlines
                SET page_title = src.page_title, src_lang = src.src_lang
                FROM (
                  SELECT * FROM read_parquet('{glob}')
                  QUALIFY row_number() OVER (PARTITION BY timestamp_utc, headline_id) = 1
                ) AS src
                WHERE headlines.timestamp_utc = src.timestamp_utc
                  AND headlines.headline_id = src.headline_id
                """
            ).fetchone()[0]
            total += n
            print(f"  {year}: {n:>12,} rows updated in {time.time() - t0:5.1f}s", flush=True)
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pull", help="BigQuery -> per-chunk parquet (no DuckDB contact).")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--chunk-days", type=int, default=31)

    a = sub.add_parser("apply", help="Yearly bulk UPDATE from the pulled parquet.")
    a.add_argument("--dir", type=Path, required=True)

    args = ap.parse_args()
    if args.cmd == "pull":
        run_pull(args.start, args.end, args.outdir, args.chunk_days)
    else:
        total = run_apply(args.dir)
        print(f"Done. Rows updated: {total:,}")


if __name__ == "__main__":
    main()
