"""Per-language precision mini-batch for the multilingual title gate.

Backlog F1's language bridge (schema v3.2 candidate) needs measured precision
before adoption: the `multilang.LANG_TERMS` lists are recall-first and written
non-natively, so some share of what they admit is "gold medal" noise in twenty
languages. This module samples admitted titles per language, has the annotator
model judge each title's relevance (the same judgment the production `relevant`
flag will make, so precision here = share of cap slots that would not be
wasted), and reports per-language precision with a false-positive taxonomy that
tells us WHICH stop-lists to write.

An English sample judged by the same instrument anchors the scale: eng
precision calibrates the judge's strictness, so per-language numbers are read
relative to it, not as absolutes.

Deliberately NOT date-blind: relevance judgment needs no date discipline (it is
a pre-filter measurement, not research annotation), and none is claimed.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pandas as pd

from metals.annotate import schema as sch
from metals.annotate.multilang import LANG_TERMS
from metals.annotate.pilot import PRICING, _approx_tokens, _client
from metals.annotate.titles import _STOP_RE, METAL_TITLE_RE
from metals.data.db import connection

# Bump when the judge prompt/schema/taxonomy changes.
JUDGE_VERSION = "v1.0"

PER_LANG_SAMPLE = 100
CHUNK_TITLES = 50
# A language whose measured precision (relative to the same-judge eng anchor)
# cannot plausibly clear ~60% even after stop-listing gets DROPPED from v3.2.
MIN_LANG_PRECISION = 0.60

# Why an admitted title is NOT about a precious metal as an asset/material.
FP_REASONS = [
    "sports_or_award",
    "place_name",
    "person_or_brand",
    "color_or_adjective",
    "money_currency_sense",
    "idiom_or_metaphor",
    "entertainment_media",
    "fashion_jewellery",
    "other_offtopic",
    "not_applicable",  # relevant == true
]

JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "titles": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "integer"},
                    "relevant": {"type": "boolean"},
                    "fp_reason": {"type": "string", "enum": FP_REASONS},
                },
                "required": ["id", "relevant", "fp_reason"],
            },
        }
    },
    "required": ["titles"],
}

JUDGE_SYSTEM_PROMPT = """\
You judge news titles for a precious-metals research corpus filter. Titles may be
in any language; judge each in its own language — do not translate first, do not
penalise a title for not being English.

For each numbered title decide `relevant`: is the title about a precious metal
(gold, silver, platinum, palladium, rhodium or other platinum-group metals) as a
financial asset, commodity, or physical material? Relevant includes: prices and
markets, investment and bullion/coins, mining and refining, recycling and scrap,
jewellery reported as metal demand or buying/selling (e.g. wedding-season gold
demand), central-bank reserves, and industrial PGM use such as catalytic
converters.

`relevant: false` when the metal word is used another way: sports medals and
awards; place names (Gold Coast); person, company, team, or brand names outside
the metals industry; colour or adjectival senses (Vietnamese "vàng" as yellow,
Turkish "altın" as golden); money or currency senses (Spanish "plata" as money,
the Polish złoty); idioms ("silver lining"); entertainment (platinum album, a
theatre named Palladium); and jewellery covered purely as fashion or celebrity
style with no metal-demand angle.

Return one record per title: {id, relevant, fp_reason}. `fp_reason` categorises
WHY an irrelevant title matched a metal keyword; use "not_applicable" when
relevant is true. If a title is ambiguous, judge from the most natural reading
of the headline alone.
"""


def judge_prompt_hash(model: str) -> str:
    """Fingerprint of (judge prompt, schema, version, model) — cache identity."""
    blob = json.dumps(
        {
            "system": JUDGE_SYSTEM_PROMPT,
            "schema": JUDGE_SCHEMA,
            "judge_version": JUDGE_VERSION,
            "model": model,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Sampling — one read-only scan, deterministic via hash ordering
# ---------------------------------------------------------------------------
def draw_sample(per_lang: int = PER_LANG_SAMPLE) -> pd.DataFrame:
    """Sample admitted titles per language: (lang, date, title).

    Non-English pools are the NEW admissions (multilingual term hit, not
    admitted by the current gate); the eng pool is the current gate itself (the
    judge-strictness anchor). Distinct titles only (case-folded), earliest date
    kept, ordered by a seeded hash so the draw is reproducible.
    """
    case_arms = "\n".join(
        f"WHEN src_lang = '{lang}' THEN regexp_matches(page_title, ?, 'i')" for lang in LANG_TERMS
    )
    params = [METAL_TITLE_RE.pattern, _STOP_RE.pattern] + [LANG_TERMS[lang] for lang in LANG_TERMS]
    sql = f"""
    WITH era AS (
        SELECT CAST(timestamp_utc AS DATE) AS d, src_lang, page_title, themes
        FROM headlines
        WHERE timestamp_utc >= '2019-09-22' AND page_title IS NOT NULL
    ),
    flagged AS (
        SELECT d, src_lang, page_title,
            (
                (regexp_matches(page_title, ?, 'i') OR contains(themes, 'ECON_GOLDPRICE'))
                AND NOT (regexp_matches(page_title, ?, 'i')
                         AND NOT contains(themes, 'ECON_GOLDPRICE'))
            ) AS cur,
            (CASE {case_arms} ELSE FALSE END) AS multi
        FROM era
    ),
    pool AS (
        SELECT src_lang, d, page_title
        FROM flagged
        WHERE (src_lang = 'eng' AND cur)
           OR (src_lang <> 'eng' AND multi AND NOT cur)
    ),
    dedup AS (
        SELECT src_lang,
               lower(page_title)      AS title_key,
               any_value(page_title)  AS title,
               min(d)                 AS d
        FROM pool
        GROUP BY 1, 2
    )
    SELECT src_lang AS lang, CAST(d AS VARCHAR) AS date, title
    FROM dedup
    QUALIFY row_number() OVER (
        PARTITION BY src_lang ORDER BY hash(title_key || 'precision-seed-42')
    ) <= {int(per_lang)}
    ORDER BY lang, date
    """
    with connection(read_only=True) as conn:
        return conn.execute(sql, params).fetchdf()


# ---------------------------------------------------------------------------
# Judging — chunked Batch requests, mirroring pilot.run_pilot
# ---------------------------------------------------------------------------
def _chunks(sample: pd.DataFrame) -> list[tuple[str, int, pd.DataFrame]]:
    """(lang, chunk_index, rows) in a stable order; ≤ CHUNK_TITLES rows each."""
    out: list[tuple[str, int, pd.DataFrame]] = []
    for lang, grp in sample.groupby("lang", sort=True):
        grp = grp.reset_index(drop=True)
        for i in range(0, len(grp), CHUNK_TITLES):
            out.append((str(lang), i // CHUNK_TITLES, grp.iloc[i : i + CHUNK_TITLES]))
    return out


def build_judge_params(titles: list[str], *, model: str) -> dict:
    """Message-create params for one chunk (a Batch request's ``params``)."""
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(titles))
    return {
        "model": model,
        "max_tokens": 8000,
        "system": [
            {
                "type": "text",
                "text": JUDGE_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": f"Judge the following titles.\n{numbered}"}],
        "output_config": {"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
    }


def estimate(sample: pd.DataFrame, *, model: str = sch.MODEL_DEFAULT) -> str:
    """Offline token/cost approximation for the judging batch."""
    chunks = _chunks(sample)
    system_tokens = _approx_tokens(JUDGE_SYSTEM_PROMPT)
    schema_tokens = _approx_tokens(json.dumps(JUDGE_SCHEMA))
    total_in = sum(
        system_tokens + schema_tokens + _approx_tokens("\n".join(grp["title"]))
        for _, _, grp in chunks
    )
    total_out = len(sample) * 22 + len(chunks) * 10  # ~22 tok/record + envelope
    lines = [
        f"Precision mini-batch estimate — {len(sample)} titles, "
        f"{sample['lang'].nunique()} languages, {len(chunks)} requests",
        f"  {'model':<20}{'in_tok':>9}{'out_tok':>9}{'batch$':>8}",
    ]
    for m, (in_per_m, out_per_m) in PRICING.items():
        usd = (total_in * in_per_m + total_out * out_per_m) / 1e6 * 0.5
        lines.append(f"  {m:<20}{total_in:>9,}{total_out:>9,}{usd:>8.2f}")
    lines.append(f"  (requested model: {model}; batch = 50% of standard)")
    return "\n".join(lines)


def run_judge(
    sample: pd.DataFrame,
    *,
    model: str = sch.MODEL_DEFAULT,
    out_path: str | Path,
    poll_seconds: int = 30,
    timeout_seconds: int = 6 * 3600,
) -> pd.DataFrame:
    """Submit the judging Batch, poll, parse, and write one row per title."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = _client()
    chunks = _chunks(sample)
    requests = [
        Request(
            custom_id=f"{lang}__{idx}",
            params=cast(
                MessageCreateParamsNonStreaming,
                build_judge_params(list(grp["title"]), model=model),
            ),
        )
        for lang, idx, grp in chunks
    ]
    by_key = {(lang, idx): grp for lang, idx, grp in chunks}

    batch = client.messages.batches.create(requests=requests)
    deadline = time.monotonic() + timeout_seconds
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        if time.monotonic() > deadline:
            raise TimeoutError(f"Batch {batch.id} did not finish within timeout.")
        time.sleep(poll_seconds)

    pulled_at = datetime.now(UTC).isoformat()
    phash = judge_prompt_hash(model)
    rows: list[dict] = []
    for result in client.messages.batches.results(batch.id):
        lang, idx_s = result.custom_id.split("__", 1)
        grp = by_key[(lang, int(idx_s))]
        base = {
            "lang": lang,
            "model": model,
            "judge_version": JUDGE_VERSION,
            "judge_prompt_hash": phash,
            "batch_id": batch.id,
            "pulled_at": pulled_at,
        }
        if result.result.type != "succeeded":
            for _, r in grp.iterrows():
                rows.append(
                    {
                        **base,
                        "date": r["date"],
                        "title": r["title"],
                        "ok": False,
                        "relevant": None,
                        "fp_reason": f"__error__:{result.result.type}",
                    }
                )
            continue
        msg = result.result.message
        text = next((blk.text for blk in msg.content if blk.type == "text"), "")
        try:
            records = {int(t["id"]): t for t in json.loads(text).get("titles", [])}
        except (json.JSONDecodeError, TypeError, KeyError, ValueError):
            records = {}
        for pos, (_, r) in enumerate(grp.iterrows(), start=1):
            rec = records.get(pos)
            rows.append(
                {
                    **base,
                    "date": r["date"],
                    "title": r["title"],
                    "ok": rec is not None,
                    "relevant": None if rec is None else bool(rec["relevant"]),
                    "fp_reason": None if rec is None else rec["fp_reason"],
                }
            )

    df = pd.DataFrame(rows).sort_values(["lang", "date"]).reset_index(drop=True)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def report(df: pd.DataFrame, *, examples_per_reason: int = 2) -> str:
    """Per-language precision table + FP taxonomy + keep/drop verdicts.

    Precision = judged-relevant / judged, over successfully judged titles. The
    eng row anchors the judge's strictness; verdicts apply to non-eng languages
    only (eng is the current gate, not a candidate).
    """
    ok = df[df["ok"] & df["relevant"].notna()]
    if ok.empty:
        return "No successfully judged titles — nothing to report."
    lines = [
        "Per-language precision (judge-anchored; eng = current gate baseline)",
        f"  {'lang':<6}{'judged':>7}{'relevant':>9}{'precision':>10}"
        "  verdict / top false-positive reasons",
    ]
    eng_prec = None
    eng_rows = ok[ok["lang"] == "eng"]
    if not eng_rows.empty:
        eng_prec = float(eng_rows["relevant"].mean())
    for lang, grp in ok.groupby("lang", sort=True):
        prec = float(grp["relevant"].mean())
        fps = grp.loc[~grp["relevant"].astype(bool), "fp_reason"].value_counts()
        top = ", ".join(f"{r}×{c}" for r, c in fps.head(3).items()) or "-"
        if lang == "eng":
            verdict = "ANCHOR"
        else:
            verdict = "KEEP" if prec >= MIN_LANG_PRECISION else "STOPLIST-OR-DROP"
        n_rel = int(grp["relevant"].sum())
        lines.append(f"  {lang:<6}{len(grp):>7}{n_rel:>9}{prec:>10.2f}  {verdict}  [{top}]")
    if eng_prec is not None:
        lines.append(
            f"\n  eng anchor precision {eng_prec:.2f} — read language rows relative to it; "
            f"bar for v3.2 inclusion: >= {MIN_LANG_PRECISION:.0%} after stop-listing."
        )
    lines.append("\nExample false positives (stop-list material):")
    for lang, grp in ok.groupby("lang", sort=True):
        fp = grp[~grp["relevant"].astype(bool)]
        if fp.empty or lang == "eng":
            continue
        lines.append(f"  [{lang}]")
        for reason, sub in fp.groupby("fp_reason"):
            for t in sub["title"].head(examples_per_reason):
                lines.append(f"    {reason}: {t[:110]}")
    n_err = int((~df["ok"]).sum())
    if n_err:
        lines.append(f"\n  WARNING: {n_err} titles unjudged (request errors / id mismatches).")
    return "\n".join(lines)
