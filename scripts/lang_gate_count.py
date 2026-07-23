"""Dry count: what would a multilingual keyword gate admit, per language?

Run: uv run python scripts/lang_gate_count.py  (read-only; ~90s scan of the
title-era corpus). First run 2026-07-23: +~520 unique titles/day vs the
current gate's ~576 — see journal.md.

Reproduces the pilot's CURRENT gate in SQL — (METAL_TITLE_RE keyword OR
ECON_GOLDPRICE theme) AND NOT (stop-phrase without the gold theme) — then
applies first-cut native-script metal terms per src_lang and counts rows the
multilingual terms admit that the current gate does not.

Recall-first term lists: precision is later work (per-language stop-phrases +
the LLM `relevant` flag); this measures the size of the composition trade.
"""

import re

from metals.annotate.multilang import LANG_TERMS
from metals.annotate.titles import _STOP_RE, METAL_TITLE_RE
from metals.data.db import connection

ENG = METAL_TITLE_RE.pattern
STOP = _STOP_RE.pattern

# Sanity: every pattern must compile under Python re (superset check for RE2).
for pat in LANG_TERMS.values():
    re.compile(pat)

case_arms = "\n".join(
    f"WHEN src_lang = '{lang}' THEN regexp_matches(page_title, ?, 'i')" for lang in LANG_TERMS
)
params = [ENG, STOP] + [LANG_TERMS[lang] for lang in LANG_TERMS]

SQL = f"""
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
)
SELECT src_lang,
       count(*)                                              AS n_rows,
       count(*) FILTER (cur)                                 AS n_cur,
       count(*) FILTER (multi AND NOT cur)                   AS n_new,
       count(DISTINCT (d, lower(page_title))) FILTER (cur)   AS uniq_cur,
       count(DISTINCT (d, lower(page_title))) FILTER (multi AND NOT cur) AS uniq_new,
       count(DISTINCT d)                                     AS n_days
FROM flagged
GROUP BY src_lang
ORDER BY n_new DESC
"""

with connection(read_only=True) as conn:
    df = conn.execute(SQL, params).fetchdf()
    total_days = conn.execute(
        "SELECT count(DISTINCT CAST(timestamp_utc AS DATE)) FROM headlines "
        "WHERE timestamp_utc >= '2019-09-22' AND page_title IS NOT NULL"
    ).fetchone()[0]

df["new_per_day"] = (df["uniq_new"] / total_days).round(1)
df["cur_per_day"] = (df["uniq_cur"] / total_days).round(1)

covered = df[df["src_lang"].isin(LANG_TERMS)].copy()
print(f"title-era days: {total_days}")
print()
cols = ["src_lang", "n_rows", "n_cur", "n_new", "uniq_new", "cur_per_day", "new_per_day"]
print(covered[cols].to_string(index=False))
print()
print(f"TOTAL current unique candidates/day (all langs): {df['cur_per_day'].sum():.0f}")
print(f"TOTAL new unique titles/day from multilingual terms: {covered['new_per_day'].sum():.0f}")
eng = df[df["src_lang"] == "eng"]
print(f"eng current/day: {float(eng['cur_per_day'].iloc[0]):.0f}")
