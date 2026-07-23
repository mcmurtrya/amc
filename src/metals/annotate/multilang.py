"""Multilingual metal-term vocabulary for the title gate (single source of truth).

Terms are RECALL-FIRST native-script patterns per GDELT ``src_lang`` code
(written 2026-07-23 for the language-gap measurement — `scripts/lang_gate_count.py`:
+~520 unique titles/day vs the English gate's ~576). ``LANG_STOP_TERMS`` holds
the per-language vetoes written from the precision mini-batch's MEASURED
false-positive patterns (terms v2, 2026-07-23): a title is admitted when its
language's terms hit AND its stops do not.

Terms v2 revisions, each answering a measured failure (journal 2026-07-23):

- **ron** — `AUR` the political party was 53/100 of false positives. Case is the
  only separator (party = uppercase acronym, metal = `aur`/`Aurul`), so the
  whole vocabulary moved from an external ``'i'`` regex flag to INLINE ``(?i)``
  prefixes, letting ron mix a case-folded stem group with a case-SENSITIVE
  ``\\b[Aa]ur\\b``. Callers must NOT pass a case-insensitivity flag anymore —
  an outer flag would re-fold ron and resurrect the party.
- **jpn** — ゴールド removed (70/100 FPs: fashion, brands, LEED-Gold, Goldman
  transliterated). The surviving terms are the precise price/metal compounds.
- **stops** — sports-medal vocabulary across European languages (medaglia
  d'oro ×40 in ita), La Plata place names (×49 in spa), Golden Dawn / Η Αυγή
  (ell), Sonali Bank (ben), the złoty currency (pol), l'or noir (fra).

Regex notes: RE2's ``\\b`` is ASCII-only — Latin-script short words carry word
boundaries only where both edge characters are ASCII; CJK/Arabic/Indic/Thai
terms are plain substrings/stems and take no ``(?i)`` (no case to fold).
"""

# Bump when terms/stops change; recorded into sample/results provenance.
TERMS_VERSION = "v2"

# The PRODUCTION bridge set (schema v3.2 freeze, 2026-07-23): the nine languages
# whose measured precision cleared the 0.60 bar / 0.58 eng anchor — six from the
# first mini-batch untouched, three promoted by the terms-v2 retest (jpn 0.68,
# ind 0.62, ron 0.61). ~265 admitted/day at measured precisions ≈ +195 relevant
# titles/day (~doubles the English gate's ~161). The other eleven languages'
# terms/stops remain below as the measurement record — they re-enter only via a
# smarter gate (LLM pre-gate decided at the pilot→full-run boundary), not via
# more stop words (journal 2026-07-23: residual noise is diffuse).
BRIDGE_LANGS: frozenset[str] = frozenset(
    {"zho", "vie", "ara", "tur", "tha", "kor", "ind", "ron", "jpn"}
)

LANG_TERMS: dict[str, str] = {
    "zho": "黄金|白银|贵金属|金价|银价|铂|钯|黃金|白銀|貴金屬|金價|銀價|鉑|鈀",
    "spa": r"(?i)\b(?:oro|plata|platino|paladio|rodio|lingotes?)\b|metales preciosos",
    "vie": "(?i)giá vàng|vàng|bạch kim|paladi|kim loại quý",
    "rus": "(?i)золот|серебр|платин|паллади|драгметалл|драгоценн",
    "fra": r"(?i)l'or\b|\bplatine|métaux précieux|\blingot|once d'or",
    "deu": "(?i)gold|silber|platin|palladium|edelmetall",
    "tur": r"(?i)\baltın\b|\baltin\b|gümüş|\bplatin\b|paladyum|külçe",
    "ben": "সোনা|স্বর্ণ|রুপা|রূপা|প্লাটিনাম|প্যালাডিয়াম",
    "ita": r"(?i)\b(?:oro|argento|platino|palladio|lingott\w*)\b|metalli preziosi",
    "ara": "الذهب|الفضة|بلاتين|بلاديوم|روديوم|المعادن الثمينة",
    "kor": "금값|금시세|금 시세|백금|팔라듐|귀금속|금괴|은값",
    "ind": r"(?i)\b(?:emas|perak|platina|paladium)\b|logam mulia|harga emas",
    "ell": "(?i)χρυσ|ασήμι|πλατίν|παλλάδι",
    "jpn": "金価格|金相場|金先物|プラチナ|パラジウム|貴金属|銀価格",
    "por": r"(?i)\b(?:ouro|prata|platina|paládio|paladio)\b|metais preciosos",
    "pol": "(?i)złoto|złota|srebr|platyn|pallad",
    "ukr": "(?i)золот|срібл|срібн|платин|палад",
    "tha": "ทองคำ|ราคาทอง|แพลตทินัม|แพลเลเดียม",
    # Case-folded stems, PLUS case-sensitive aur so the uppercase acronym
    # "AUR" (Alliance for the Union of Romanians) no longer matches.
    "ron": r"(?i:aurul|aurului|argint|platin|paladiu)|\b[Aa]ur\b",
    "hin": "सोना|सोने|चांदी|चाँदी|प्लैटिनम|पैलेडियम",
}

# Per-language vetoes, written from MEASURED false positives (2026-07-23).
# Applied as: admitted = terms hit AND NOT stops hit. Only languages whose
# mini-batch failures showed a concentrated, nameable pattern carry stops;
# the six languages that passed outright (zho/vie/ara/tur/tha/kor) are
# deliberately untouched so their measured numbers remain valid.
LANG_STOP_TERMS: dict[str, str] = {
    "spa": r"(?i)la plata\b|del plata\b|medall|olímpi|olimpi",
    "ita": "(?i)medagli|premi|olimpi|guida oro",
    "ron": "(?i)olimpi|medali",
    "rus": "(?i)медал|глобус|олимпи",
    "ukr": "(?i)медал|олімпі",
    "pol": "(?i)medal|olimpi|złotych|złotego",
    "deu": "(?i)medaille|olympi|goldene hochzeit",
    "fra": "(?i)l'or noir|l'or blanc|l'or vert|médaille|olympi",
    "hin": "पदक|ओलंपिक|ओलिंपिक",
    "ben": "সোনালী|সোনালি|পদক",
    "ind": "(?i)medali|olimpiade",
    "ell": "(?i)αυγή|μετάλλι|ολυμπι",
    "por": "(?i)ouro preto|ouro branco|ouro fino|medalh|olímpi|olimpi",
}


def multi_admit_case(column: str = "page_title") -> tuple[str, list[str]]:
    """SQL CASE arms + params: TRUE when the language's terms hit and stops don't.

    Patterns are self-contained (inline ``(?i)`` where wanted) — callers must
    call ``regexp_matches`` WITHOUT a flags argument, or ron's case-sensitive
    ``aur``/``AUR`` distinction is silently erased.
    """
    arms: list[str] = []
    params: list[str] = []
    for lang, pat in LANG_TERMS.items():
        stop = LANG_STOP_TERMS.get(lang)
        if stop:
            arms.append(
                f"WHEN src_lang = '{lang}' THEN regexp_matches({column}, ?) "
                f"AND NOT regexp_matches({column}, ?)"
            )
            params += [pat, stop]
        else:
            arms.append(f"WHEN src_lang = '{lang}' THEN regexp_matches({column}, ?)")
            params.append(pat)
    return "\n".join(arms), params
