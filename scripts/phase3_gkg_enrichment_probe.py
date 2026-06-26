"""Probe two GKG columns we don't currently ingest: Extras and TranslationInfo.

Phase 3 stores only DATE, SourceCommonName, DocumentIdentifier, V2Themes, V2Tone
(see ``metals.data.gdelt.build_query``). GKG carries ~22 more columns. Two of
them bear directly on the slug-recovery / language problems surfaced in the
text-quality audit:

  - ``Extras`` (V2EXTRASXML): an XML blob that *sometimes* contains
    ``<PAGE_TITLE>…</PAGE_TITLE>`` — the actual scraped article title. If
    coverage is high, this is a real headline source and slug recovery becomes
    unnecessary for those rows.
  - ``TranslationInfo``: populated only for documents GDELT translated from a
    non-English source (``srclc:<lang>;eng:<engine>``); empty for
    English-original articles. A near-free language label — the clean input for
    an English filter or for routing to a multilingual embedder.

This probe is theme-filtered with the SAME predicate as the real pipeline, so
the coverage numbers reflect the corpus actually in DuckDB — not all of GKG.

It ALWAYS dry-runs first and prints bytes scanned + estimated cost. Pulling the
large ``Extras`` XML column widens the scan well beyond the narrow-column
monthly pull, so confirm the dry-run number before ``--execute``. Default
window is deliberately short (3 days) to keep the probe cheap; widen with
``--start``/``--end`` once the dry-run cost looks acceptable.

Run:
    uv run python scripts/phase3_gkg_enrichment_probe.py                 # dry-run only
    uv run python scripts/phase3_gkg_enrichment_probe.py --execute       # download + stats
    uv run python scripts/phase3_gkg_enrichment_probe.py --start 2024-06-01 --end 2024-06-07 --execute
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter

import pandas as pd
from dotenv import load_dotenv

from metals.data.gdelt import TABLE, build_query, load_themes
from metals.data.text_prep import url_to_text

load_dotenv()

_PAGE_TITLE_RE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.IGNORECASE | re.DOTALL)
# TranslationInfo looks like "srclc:fra;eng:Moses 2.1.1" — pull the source lang.
_SRCLC_RE = re.compile(r"srclc:([a-z]{2,3})", re.IGNORECASE)

ONDEMAND_USD_PER_TB = 6.25  # US on-demand, per results/phase3_backfill_plan.md
FREE_TB_PER_MONTH = 1.0


def build_probe_query(start: str, end: str) -> str:
    """Same date + theme predicate as the pipeline, but selecting the two
    enrichment columns instead of the ingested ones."""
    base = build_query(start, end, load_themes())
    # Reuse the WHERE clause verbatim; swap the SELECT list.
    where = base[base.index("FROM"):]
    return (
        "SELECT\n"
        "    DocumentIdentifier AS url,\n"
        "    TranslationInfo    AS translation_info,\n"
        "    Extras             AS extras\n"
        f"{where}"
    )


def page_title(extras: object) -> str:
    if not isinstance(extras, str) or not extras:
        return ""
    m = _PAGE_TITLE_RE.search(extras)
    return m.group(1).strip() if m else ""


def src_lang(info: object) -> str:
    """Source language code; 'eng' when TranslationInfo is empty (English-original)."""
    if not isinstance(info, str) or not info.strip():
        return "eng"
    m = _SRCLC_RE.search(info)
    return m.group(1).lower() if m else "?"


def _bq_client():
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS not set — configure the GCP service "
            "account JSON in .env before probing GDELT."
        )
    from google.cloud import bigquery  # noqa: WPS433 — lazy import

    return bigquery, bigquery.Client()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2024-06-01")
    ap.add_argument("--end", default="2024-06-03", help="inclusive; short window keeps cost low")
    ap.add_argument("--execute", action="store_true",
                    help="download and compute coverage stats (else dry-run only)")
    ap.add_argument("--examples", type=int, default=12)
    args = ap.parse_args()

    bigquery, client = _bq_client()
    query = build_probe_query(args.start, args.end)
    print(f"Probe window {args.start} .. {args.end} (theme-filtered, same predicate as pipeline)\n")

    # --- Always dry-run first: report bytes + cost before touching the wire ---
    dry = client.query(
        query, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    )
    gb = dry.total_bytes_processed / 1e9
    tb = dry.total_bytes_processed / 1e12
    billed = max(0.0, tb - FREE_TB_PER_MONTH) * ONDEMAND_USD_PER_TB
    print(f"[dry-run] scans {gb:,.2f} GB  ({tb:.4f} TB)")
    print(f"[dry-run] cost: $0.00 if under the 1 TB/month free tier; "
          f"${tb * ONDEMAND_USD_PER_TB:,.2f} at full on-demand "
          f"(${billed:,.2f} once the free TB is used up)\n")

    if not args.execute:
        print("Dry-run only. Re-run with --execute to download and measure coverage.")
        return 0

    print("[execute] downloading ...")
    df = client.query(query).to_dataframe()
    n = len(df)
    if not n:
        print("No rows in window.")
        return 1

    titles = df["extras"].map(page_title)
    has_title = titles.str.len() > 0
    langs = df["translation_info"].map(src_lang)

    print(f"\nSampled rows: {n:,}\n" + "=" * 64)

    print("\n[A] PAGE_TITLE coverage in Extras (the real-headline question):")
    print(f"  rows with <PAGE_TITLE>: {has_title.sum():,}  ({100 * has_title.mean():.1f}%)")
    have = titles[has_title]
    if len(have):
        print(f"  median title length: {int(have.str.len().median())} chars; "
              f"mean words: {have.str.split().map(len).mean():.1f}")

    print("\n[B] Source language (TranslationInfo; 'eng' = English-original):")
    lc = Counter(langs)
    for lang, c in lc.most_common(12):
        print(f"  {lang:4s} {c:7,}  {100 * c / n:5.1f}%")
    print(f"  -> English-original share: {100 * lc.get('eng', 0) / n:.1f}%")

    print("\n[C] Real title vs slug recovery, where a PAGE_TITLE exists:")
    shown = 0
    for _, row in df[has_title].iterrows():
        if shown >= args.examples:
            break
        slug = url_to_text(row["url"])
        title = page_title(row["extras"])
        print(f"  TITLE: {title[:90]!r}")
        print(f"  SLUG : {slug[:90]!r}")
        print(f"  LANG : {src_lang(row['translation_info'])}\n")
        shown += 1

    # Cross-tab the headline question against the degeneracy finding: of the rows
    # where slug recovery is degenerate (<=2 tokens), how many does PAGE_TITLE rescue?
    slug_tokens = df["url"].map(lambda u: len(url_to_text(u).split()))
    degen = slug_tokens <= 2
    if degen.any():
        rescued = (degen & has_title).sum()
        print("[D] PAGE_TITLE rescue rate on degenerate-slug rows "
              "(the rows gating would otherwise drop from the embedding):")
        print(f"  degenerate slugs: {degen.sum():,}  "
              f"of which have a PAGE_TITLE: {rescued:,}  "
              f"({100 * rescued / max(1, degen.sum()):.1f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
