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

from metals.annotate.titles import _STOP_RE, METAL_TITLE_RE
from metals.data.db import connection

ENG = METAL_TITLE_RE.pattern
STOP = _STOP_RE.pattern

# First-cut native terms. Latin-script short words carry \b only where both
# edge characters are ASCII (RE2 \b is ASCII-only; a trailing \b after a
# diacritic inverts its meaning). CJK/Arabic/Cyrillic terms are plain
# substrings/stems.
LANG_TERMS: dict[str, str] = {
    "zho": "黄金|白银|贵金属|金价|银价|铂|钯|黃金|白銀|貴金屬|金價|銀價|鉑|鈀",
    "spa": r"\b(?:oro|plata|platino|paladio|rodio|lingotes?)\b|metales preciosos",
    "vie": "giá vàng|vàng|bạch kim|paladi|kim loại quý",
    "rus": "золот|серебр|платин|паллади|драгметалл|драгоценн",
    "fra": r"l'or\b|\bplatine|métaux précieux|\blingot|once d'or",
    "deu": "gold|silber|platin|palladium|edelmetall",
    "tur": r"\baltın\b|\baltin\b|gümüş|\bplatin\b|paladyum|külçe",
    "ben": "সোনা|স্বর্ণ|রুপা|রূপা|প্লাটিনাম|প্যালাডিয়াম",
    "ita": r"\b(?:oro|argento|platino|palladio|lingott\w*)\b|metalli preziosi",
    "ara": "الذهب|الفضة|بلاتين|بلاديوم|روديوم|المعادن الثمينة",
    "kor": "금값|금시세|금 시세|백금|팔라듐|귀금속|금괴|은값",
    "ind": r"\b(?:emas|perak|platina|paladium)\b|logam mulia|harga emas",
    "ell": "χρυσ|ασήμι|πλατίν|παλλάδι",
    "jpn": "金価格|金相場|金先物|プラチナ|パラジウム|貴金属|ゴールド|銀価格",
    "por": r"\b(?:ouro|prata|platina|paládio|paladio)\b|metais preciosos",
    "pol": "złoto|złota|srebr|platyn|pallad",
    "ukr": "золот|срібл|срібн|платин|палад",
    "tha": "ทองคำ|ราคาทอง|แพลตทินัม|แพลเลเดียม",
    "ron": r"\baur\b|aurul|argint|platin|paladiu",
    "hin": "सोना|सोने|चांदी|चाँदी|प्लैटिनम|पैलेडियम",
}

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
