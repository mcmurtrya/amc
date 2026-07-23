"""Multilingual metal-term vocabulary for the title gate (single source of truth).

First-cut, RECALL-FIRST native-script terms per GDELT ``src_lang`` code, written
2026-07-23 for the language-gap measurement (`scripts/lang_gate_count.py`:
+~520 unique titles/day vs the English gate's ~576 — see journal.md) and reused
by the precision mini-batch (`scripts/lang_precision_batch.py`).

Known ambiguities are deliberate — precision is the mini-batch's job, and the
production stop-lists get written from its measured false-positive patterns:
vàng is also "yellow", plata is money slang, złota collides with the złoty,
altın is also adjectival, French l'or aside "or" is a conjunction.

Regex notes: RE2's ``\\b`` is ASCII-only, so Latin-script short words carry
word boundaries only where both edge characters are ASCII (a trailing ``\\b``
after a diacritic inverts its meaning); CJK/Arabic/Cyrillic terms are plain
substrings/stems.
"""

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
