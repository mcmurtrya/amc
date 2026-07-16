"""GDELT 2.0 GKG ingestion via BigQuery.

Phase 3 steps 3.1–3.3 and 3.5.

The Global Knowledge Graph stores one row per article with extracted themes,
tone, persons/orgs/locations. We pull rows matching any of the themes in
``configs/gdelt_themes.yaml`` and write them to the ``headlines`` table.

GDELT GKG has no "headline" field per se — only the article URL, which we
store once in ``article_url``. (Earlier revisions also kept a ``headline``
column that was a byte-for-byte copy of ``article_url``; it was dropped in
migration 005 to reclaim storage.) The real article title *does* live in the
GKG ``Extras`` XML blob as ``<PAGE_TITLE>`` — but only from **2019-09-22**,
when GDELT switched the field on (0% before, ~99.2% of themed rows after;
measured 2026-07-02). We pull it into ``page_title`` (original language,
entity-decoded), along with the machine-translation source language from
``TranslationInfo`` into ``src_lang`` (empty upstream = English-original =
``'eng'``; populated for all dates back to 2015). Rows ingested before
migration 007 carry NULLs in both until re-pulled wide; rows before
2019-09-22 stay NULL ``page_title`` forever.

Run as:
    uv run python -m metals.data.gdelt --start 2024-01-01 --end 2024-01-31

Requires GOOGLE_APPLICATION_CREDENTIALS in .env pointing at a service-account
JSON with BigQuery Data Viewer + BigQuery Job User.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv

from metals.data.db import connection

load_dotenv()
SOURCE_TAG = "gdelt_gkg"
TABLE = "gdelt-bq.gdeltv2.gkg_partitioned"
THEMES_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "gdelt_themes.yaml"

# Extras is a flat XML blob; PAGE_TITLE is the scraped article <title>.
_PAGE_TITLE_RE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.IGNORECASE | re.DOTALL)
# TranslationInfo looks like "srclc:fra;eng:Moses 2.1.1" — pull the source lang.
_SRCLC_RE = re.compile(r"srclc:([a-z]{2,3})", re.IGNORECASE)
# Real titles are short; the cap only guards against malformed/unclosed XML.
PAGE_TITLE_MAX_CHARS = 512


def extract_page_title(extras: object) -> str | None:
    """Pull the article title out of the GKG ``Extras`` XML blob.

    Titles arrive HTML-entity-encoded (``&amp;#x27;`` etc.) in the article's
    original language; we unescape once and collapse internal whitespace.
    Returns ``None`` when Extras is missing, empty, or has no PAGE_TITLE tag —
    NULL in the table means "no title available", never an empty string.
    """
    if not isinstance(extras, str) or not extras:
        return None
    m = _PAGE_TITLE_RE.search(extras)
    if not m:
        return None
    title = " ".join(html.unescape(m.group(1)).split())
    return title[:PAGE_TITLE_MAX_CHARS] or None


def extract_src_lang(translation_info: object) -> str | None:
    """Source-language code from GKG ``TranslationInfo``.

    GDELT populates TranslationInfo only for machine-translated documents, so
    an *empty* value (when the column was pulled) means English-original and
    maps to ``'eng'``. A populated value without a parseable ``srclc:`` field
    is malformed → ``None``. Callers that never pulled the column must store
    NULL, not ``'eng'`` — absence of evidence is not English.
    """
    if not isinstance(translation_info, str) or not translation_info.strip():
        return "eng"
    m = _SRCLC_RE.search(translation_info)
    return m.group(1).lower() if m else None


def load_themes(path: Path | str = THEMES_CONFIG) -> list[str]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    themes = cfg.get("themes") or []
    if not themes:
        raise RuntimeError(f"No themes loaded from {path}")
    return list(themes)


def build_query(
    start_date: date | str,
    end_date: date | str,
    themes: Iterable[str],
) -> str:
    """Construct the GKG query for the date range and theme filter.

    Returns SQL targeting ``gdelt-bq.gdeltv2.gkg``. The result schema is
    ``date_int, source_common_name, document_identifier, v2themes, v2tone,
    translation_info, extras``. Extras/TranslationInfo widen the scan by
    ~20% over the old 5-column pull (measured 2026-07-02: 1.35 TB vs 1.13 TB
    for 2015-02-18..2019-12-31) and carry the article title + source language.
    """
    sd = pd.Timestamp(start_date)
    ed = pd.Timestamp(end_date)
    # GKG `DATE` is YYYYMMDDhhmmss as INT64. Exclusive upper bound = end + 1 day.
    date_lo = int(sd.strftime("%Y%m%d")) * 1_000_000
    date_hi = int((ed + pd.Timedelta(days=1)).strftime("%Y%m%d")) * 1_000_000
    # `_PARTITIONTIME` is the daily ingestion-time partition on `gkg_partitioned`.
    # Filtering it (not DATE) is what actually prunes; without this, BQ scans the
    # whole columns. Keep the DATE int filter for intra-day boundaries.
    pt_lo = sd.strftime("%Y-%m-%d")
    pt_hi = (ed + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # Build a REGEXP_CONTAINS predicate that matches any of our themes.
    # Each theme can appear with an offset suffix (e.g. "COMMODITIES_GOLD,123"),
    # so we anchor the match to either start-of-string or after a semicolon.
    pattern = "(^|;)(" + "|".join(themes) + ")(,|;|$)"

    return f"""
    SELECT
        DATE             AS date_int,
        SourceCommonName AS source_common_name,
        DocumentIdentifier AS document_identifier,
        V2Themes         AS v2themes,
        V2Tone           AS v2tone,
        TranslationInfo  AS translation_info,
        Extras           AS extras
    FROM `{TABLE}`
    WHERE _PARTITIONTIME >= TIMESTAMP("{pt_lo}")
      AND _PARTITIONTIME <  TIMESTAMP("{pt_hi}")
      AND DATE >= {date_lo}
      AND DATE <  {date_hi}
      AND V2Themes IS NOT NULL
      AND REGEXP_CONTAINS(V2Themes, r"{pattern}")
    """.strip()


def parse_gkg_rows(raw: pd.DataFrame, themes_filter: list[str]) -> pd.DataFrame:
    """Convert the BigQuery result frame to the headlines-table schema.

    Pure-function over a DataFrame so the parsing logic is testable without
    network access.
    """
    if raw.empty:
        return raw.assign(
            **{
                c: pd.Series(dtype="object")
                for c in (
                    "timestamp_utc",
                    "headline_id",
                    "source",
                    "themes",
                    "article_url",
                    "page_title",
                    "src_lang",
                    "tone_overall",
                    "tone_positive",
                    "tone_negative",
                    "tone_polarity",
                    "tone_ard",
                    "tone_sgrd",
                )
            }
        ).iloc[:0]

    df = raw.copy()
    df["timestamp_utc"] = pd.to_datetime(
        df["date_int"].astype(str).str.zfill(14),
        format="%Y%m%d%H%M%S",
        utc=False,
        errors="coerce",
    )

    theme_set = set(themes_filter)

    def _themes_to_list(s: object) -> list[str]:
        if not isinstance(s, str):
            return []
        seen: list[str] = []
        for token in s.split(";"):
            code = token.split(",")[0]  # strip offset suffix
            if code and code in theme_set and code not in seen:
                seen.append(code)
        return seen

    df["themes_list"] = df["v2themes"].map(_themes_to_list)
    df["themes"] = df["themes_list"].map(json.dumps)

    # V2Tone is "tone,positive_score,negative_score,polarity,ARD,SGRD".
    def _parse_tone(s: object) -> dict:
        if not isinstance(s, str) or not s:
            return {}
        parts = s.split(",")
        keys = ("overall", "positive", "negative", "polarity", "ard", "sgrd")
        out = {}
        for k, v in zip(keys, parts, strict=False):
            try:
                out[k] = float(v)
            except (ValueError, TypeError):
                pass
        return out

    tone = df["v2tone"].map(_parse_tone)
    df["tone_overall"] = tone.map(lambda d: d.get("overall"))
    df["tone_positive"] = tone.map(lambda d: d.get("positive"))
    df["tone_negative"] = tone.map(lambda d: d.get("negative"))
    df["tone_polarity"] = tone.map(lambda d: d.get("polarity"))
    df["tone_ard"] = tone.map(lambda d: d.get("ard"))
    df["tone_sgrd"] = tone.map(lambda d: d.get("sgrd"))

    df["article_url"] = df["document_identifier"].astype("string")
    df["source"] = df["source_common_name"].fillna("").astype("string")

    # Enrichment columns (migration 007). Tolerate narrow frames — pulls made
    # with the pre-wide query have no extras/translation_info, and their rows
    # must land as NULL (not 'eng'/'') so a later wide re-pull can fill them.
    if "extras" in df.columns:
        df["page_title"] = df["extras"].map(extract_page_title).astype("string")
    else:
        df["page_title"] = pd.Series(pd.NA, index=df.index, dtype="string")
    if "translation_info" in df.columns:
        df["src_lang"] = df["translation_info"].map(extract_src_lang).astype("string")
    else:
        df["src_lang"] = pd.Series(pd.NA, index=df.index, dtype="string")
    # Stable id = source + URL hash. We use the URL itself for traceability;
    # collisions in the same timestamp are extremely unlikely.
    df["headline_id"] = (
        df["timestamp_utc"].dt.strftime("%Y%m%d%H%M%S").fillna("")
        + "_"
        + df["article_url"].fillna("").str.slice(0, 200)
    )

    # Drop rows whose themes don't intersect the filter at all *before*
    # narrowing to the output columns.
    df = df[df["themes_list"].map(bool)]
    out_cols = [
        "timestamp_utc",
        "headline_id",
        "source",
        "themes",
        "article_url",
        "page_title",
        "src_lang",
        "tone_overall",
        "tone_positive",
        "tone_negative",
        "tone_polarity",
        "tone_ard",
        "tone_sgrd",
    ]
    out = df[out_cols].dropna(subset=["timestamp_utc", "headline_id"])
    return out.reset_index(drop=True)


def fetch_gkg(start_date: date | str, end_date: date | str) -> pd.DataFrame:
    """Run the BigQuery query and return the parsed-headlines DataFrame.

    Requires GOOGLE_APPLICATION_CREDENTIALS to be set.
    """
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS not set. Configure GCP service "
            "account JSON in .env before fetching GDELT."
        )
    from google.cloud import bigquery  # noqa: WPS433 — lazy import

    themes = load_themes()
    query = build_query(start_date, end_date, themes)
    client = bigquery.Client()
    job = client.query(query)
    # Use the BigQuery Storage API for the download. Requires the
    # `BigQuery Read Session User` role on the service account. The REST
    # fallback path fails with ConnectionResetError on multi-hundred-K-row
    # result sets — themed monthly GKG queries reliably exceed that.
    raw = job.to_dataframe()
    return parse_gkg_rows(raw, themes)


def upsert_headlines(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    with connection() as conn:
        conn.register("incoming_gkg", df)
        conn.execute(
            """
            INSERT INTO headlines
                (timestamp_utc, headline_id, source, themes,
                 article_url, page_title, src_lang,
                 tone_overall, tone_positive, tone_negative,
                 tone_polarity, tone_ard, tone_sgrd)
            SELECT
                timestamp_utc, headline_id, source, themes,
                article_url, page_title, src_lang,
                tone_overall, tone_positive, tone_negative,
                tone_polarity, tone_ard, tone_sgrd
            FROM incoming_gkg
            ON CONFLICT (timestamp_utc, headline_id) DO UPDATE SET
                source         = EXCLUDED.source,
                themes         = EXCLUDED.themes,
                article_url    = EXCLUDED.article_url,
                -- COALESCE: a narrow re-pull (NULL title/lang) must never
                -- clobber values that an earlier wide pull already landed.
                page_title     = COALESCE(EXCLUDED.page_title, headlines.page_title),
                src_lang       = COALESCE(EXCLUDED.src_lang, headlines.src_lang),
                tone_overall   = EXCLUDED.tone_overall,
                tone_positive  = EXCLUDED.tone_positive,
                tone_negative  = EXCLUDED.tone_negative,
                tone_polarity  = EXCLUDED.tone_polarity,
                tone_ard       = EXCLUDED.tone_ard,
                tone_sgrd      = EXCLUDED.tone_sgrd
            """
        )
        conn.unregister("incoming_gkg")
    return int(len(df))


def refresh(
    start_date: str,
    end_date: str,
    chunk_days: int = 7,
) -> dict:
    """Pull and upsert ``[start_date, end_date]`` in ``chunk_days``-sized chunks.

    Weekly chunks (~225K rows each) reliably complete inside the BigQuery
    result-download window even on the REST fallback; with the Storage API
    enabled they're trivially fast.
    """
    out_summary: dict[str, Any] = {"chunks": [], "rows_written": 0}
    sd = pd.Timestamp(start_date).normalize()
    ed = pd.Timestamp(end_date).normalize()
    cur = sd
    while cur <= ed:
        nxt = min(cur + pd.Timedelta(days=chunk_days - 1), ed)
        df = fetch_gkg(cur.date(), nxt.date())
        n = upsert_headlines(df)
        out_summary["chunks"].append(
            {"start": cur.date().isoformat(), "end": nxt.date().isoformat(), "rows": n}
        )
        out_summary["rows_written"] += n
        print(f"  {cur.date()} -> {nxt.date()}: {n} rows", flush=True)
        cur = nxt + pd.Timedelta(days=1)
    return out_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh GDELT GKG headlines.")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--chunk-days", type=int, default=7, help="Days per BigQuery chunk (default 7)."
    )
    args = parser.parse_args()
    s = refresh(args.start, args.end, chunk_days=args.chunk_days)
    print(f"\nTotal rows: {s['rows_written']}")


if __name__ == "__main__":
    main()
