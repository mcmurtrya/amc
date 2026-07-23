"""Per-day title loading: filter to genuinely metals titles, de-duplicate.

The corpus is ~29k headlines/day, but the GDELT theme codes in
``THEME_TO_METALS`` (ECON_CENTRALBANK, ECON_INFLATION, WB_1699 metal-ore-mining
which alone tags ~57% of the corpus, ...) match nearly all macro/financial news —
filtering on them leaves a ~30k/day firehose that is mostly *not* about metals. So
this narrows to titles that actually **name a metal** (or a PGM producer / bullion
coin) or carry the gold-price theme, vetoes obvious off-topic collocations,
collapses syndication, and caps per day. Real titles only (``page_title`` is
NULL / a URL slug before 2019-09-22), matching the titles-only primary run
(plans/phase_8_ssl_probing.md §5 trap 4, §8.1).

Known limitations (see plan §8.1 "filter review"):
- **English-centric.** The keyword gate is English (+ named producers/coins), so
  non-English metals titles the LLM could read are dropped at the pre-filter. Only
  gold has a non-English lifeline (the ECON_GOLDPRICE theme-OR); silver/PGM lose
  their entire non-English stratum. ~64% of gold-relevant news is non-English.
- **The cap is time-stratified** (reserves the US-afternoon session), fixing the
  earlier earliest-by-timestamp bias — but it still discards ~2.8x of titles/day on
  average, so a genuine metals story can still be dropped on a very high-volume day.
- **De-dup is exact-normalized text**, so paraphrase / cross-language copies of one
  wire story are not merged and inflate count features.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from metals.annotate.multilang import BRIDGE_LANGS, LANG_STOP_TERMS, LANG_TERMS
from metals.data.db import connection
from metals.features.text_daily import _parse_themes_field

# Cap distinct titles fed to the annotator per day, to bound token cost on the
# high-volume days. Dropped titles are reported, never silently hidden. Selection
# is TIME-STRATIFIED (see _select_capped) so the cap doesn't discard the
# US-afternoon session by clock.
MAX_TITLES_PER_DAY = 250

# --- Language bridge (schema v3.2 freeze, 2026-07-23) ------------------------
# Nine languages with measured precision >= the 0.58 English-gate anchor admit
# titles via native-script terms (metals.annotate.multilang). Because the cap
# binds, admission is a composition question: the base gate (English keywords /
# producers / gold theme — matched on EVERY row regardless of src_lang) keeps a
# floor of the budget, bridge languages share the rest proportionally with a
# per-language ceiling so zho (~119/day) cannot evict the long tail.
BASE_RESERVE_FRAC = 0.5  # >=50% of the cap for base-gate admissions
BRIDGE_LANG_CEILING_FRAC = 0.4  # no single bridge language takes >40% of the bridge share

# Compiled per-language (terms, stops|None). Patterns carry inline (?i) where
# intended — never wrap these with an external IGNORECASE, or ron's
# case-sensitive aur/AUR party fix is silently erased. Python re here vs RE2 in
# the DuckDB diagnostics: both engines are exercised by tests; the known edge
# (\b next to a non-ASCII letter) does not affect the frozen vocabulary.
_BRIDGE_RES: dict[str, tuple[re.Pattern[str], re.Pattern[str] | None]] = {
    lang: (
        re.compile(LANG_TERMS[lang]),
        re.compile(LANG_STOP_TERMS[lang]) if lang in LANG_STOP_TERMS else None,
    )
    for lang in sorted(BRIDGE_LANGS)
}

# The US trading session in UTC hours: ~COMEX open (13:20) / London PM fix (15:00)
# through FOMC (18-19), US equity close (20-21), and after-hours (21-22) — the
# most price-relevant window for a US dealer. When the cap binds, at least
# US_RESERVE_FRAC of the budget is reserved for this window before the rest is
# stratified across the remaining hours.
US_SESSION_LO, US_SESSION_HI = 13, 22
US_RESERVE_FRAC = 0.5

# GDELT began emitting the GKG Extras <PAGE_TITLE> tag on this date; before it,
# page_title is 0% (an upstream feature, not a gap we can backfill).
REAL_TITLE_START = "2019-09-22"

_WS = re.compile(r"\s+")

# Metal-naming pre-filter. The LLM's `relevant` flag does the fine-grained call;
# this only cuts the macro firehose to metals candidates. Covers the four majors
# + minor PGMs (Ir/Ru — converter scrap), tickers, major PGM producers/refiners
# (so supply stories naming the company, not the metal, survive), and anchored
# bullion-coin terms (AMC's coin-premium decisions). Bare "coin"/"sovereign"/
# "eagle"/"spot" are deliberately excluded (crypto / sovereign-debt / generic).
METAL_TITLE_RE = re.compile(
    r"\b(?:"
    r"gold|silver|platinum|palladium|rhodium|iridium|ruthenium|"
    r"bullion|precious metals?|platinum group metals?|pgms?|"
    r"gld|slv|xau|xag|xpt|xpd|pplt|pall|"
    r"nornickel|norilsk|sibanye|amplats|impala platinum|stillwater|northam|"
    r"zimplats|anglo american platinum|johnson matthey|heraeus|"
    r"catalytic converter|autocatalyst|comex|lbma|"
    r"krugerrand|gold eagle|silver eagle|maple leaf|proof coin|bullion coin|"
    r"numismatic|specie"
    r")\b",
    re.IGNORECASE,
)
_GOLD_THEME = "ECON_GOLDPRICE"

# Off-topic collocations that pass the keyword gate but are not the metal-as-asset
# (measured ~3-5% of keyword hits). Vetoed BEFORE the cap so they can't evict real
# stories; a gold-price theme rescues a match (rare true-positive protection).
_STOP_RE = re.compile(
    r"\b(?:"
    r"silver alert|"
    r"(?:gold|silver|platinum|palladium)\s+medal(?:s|ist)?|"
    r"gold\s+(?:coast|rush|standard|star|digger|finger|smith)|"
    r"silver\s+(?:lining|screen|spring|surfer|fox|state)|"
    r"platinum\s+(?:album|jubilee|record|blonde|hair)|"
    r"palladium\s+(?:theatre|theater)"
    r")\b",
    re.IGNORECASE,
)

# Recognized trailing outlet tags to strip before hashing, so "... - Reuters" and
# "... — Kitco" copies of one story de-dupe (a conservative, named list to avoid
# over-stripping legitimate dashed titles).
_OUTLET_SUFFIX = re.compile(
    r"\s*[-|–—:]\s*(?:reuters|bloomberg|kitco|cnbc|marketwatch|"
    r"yahoo(?:\s+finance)?|forbes|investing\.?com|fxstreet|barron'?s|wsj|"
    r"financial times|the economic times|business insider|seeking alpha|"
    r"benzinga|mining\.?com|money ?control|the hindu|ndtv)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DayTitles:
    """One day's annotator input plus provenance for joining results back."""

    date: str  # YYYY-MM-DD (UTC)
    headline_ids: list[str]  # aligned with ``titles`` by position (index 0 == title 1)
    titles: list[str]
    n_raw: int  # metal-relevant rows before de-duplication
    n_dropped_cap: int  # distinct titles dropped by MAX_TITLES_PER_DAY
    pre_title_era: bool = False  # True if date < 2019-09-22 (structurally titleless)
    # page_title-present rows that day BEFORE the metals filter. 0 with
    # pre_title_era False == a mid-era corpus gap (e.g. GDELT dropped ~all of
    # 2024-01), which must not be read as a quiet news day; >0 with n_raw 0 ==
    # a genuinely metals-quiet day.
    n_titled: int = 0
    # src_lang per kept title, aligned with ``titles`` (v3.2 — used by the
    # per-language offtopic split in checks.py). Empty for pre-era/gap days.
    langs: list[str] = field(default_factory=list)
    # Kept titles whose text carried a masked date/year (v3.3 blindness fix);
    # feeds the report-only `date_in_title_share` diagnostic.
    n_date_masked: int = 0


# --- Date masking (v3.3) -----------------------------------------------------
# Titles routinely embed the calendar date ("Gold Rate - March 30, 2020",
# "Giá vàng hôm nay 28/10/2019"), which defeats the date-blind design: the
# model reads the date from CONTENT, so withholding it as metadata measures
# nothing. Full dates -> [DATE] and standalone years -> [YEAR], applied to BOTH
# variants (the dated arm gets its date only via the explicit "Date:" header).
# Known residuals, accepted and measured (report-only `date_in_title_share`):
# day/month forms without a year ("28/10", "10月28日" handled; "hôm nay 28/10"
# masked only if a year appears), and unadorned prices in the 1900-2069 range
# that collide with years ("gold falls to 1950" masks as [YEAR] — comma forms
# "1,950" and currency-prefixed "$1950" are protected).
_MONTHS = (
    "Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    "Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)
_YR = r"(?:19|20)\d{2}"
_DATE_RES: tuple[re.Pattern[str], ...] = (
    re.compile(rf"\b{_YR}-\d{{1,2}}-\d{{1,2}}\b"),  # ISO
    re.compile(rf"\b\d{{1,2}}[./-]\d{{1,2}}[./-]{_YR}\b"),  # D/M/Y, M/D/Y
    re.compile(rf"\b{_YR}[./-]\d{{1,2}}[./-]\d{{1,2}}\b"),  # Y/M/D
    re.compile(rf"\b(?:{_MONTHS})\.? \d{{1,2}}(?:st|nd|rd|th)?,? ?{_YR}\b", re.I),
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)? (?:{_MONTHS})\.?,? ?{_YR}\b", re.I),
    re.compile(rf"{_YR}年(?:\d{{1,2}}月(?:\d{{1,2}}日)?)?"),  # CJK with year
    re.compile(r"\d{1,2}月\d{1,2}日"),  # CJK month-day
)
# Standalone years, protected from comma-grouped and currency-prefixed prices.
_YEAR_RE = re.compile(rf"(?<![\d,.$€£¥₹])\b{_YR}\b(?![\d,.]\d)")


def _mask_dates(title: str) -> str:
    """Replace calendar dates/years in title TEXT so blindness is real."""
    out = title
    for pat in _DATE_RES:
        out = pat.sub("[DATE]", out)
    return _YEAR_RE.sub("[YEAR]", out)


def _normalize(title: str) -> str:
    """De-dup key: HTML-decoded, NFKC-folded, outlet-suffix- and punctuation-
    stripped. Unicode-safe (``\\w`` keeps CJK/Arabic letters), so it collapses
    ``$2,000!``/``$2000``, curly vs straight quotes, and ``- Reuters`` outlet tags
    without erasing non-English titles."""
    t = html.unescape(title)
    t = unicodedata.normalize("NFKC", t).strip().lower()
    t = _OUTLET_SUFFIX.sub("", t)
    t = re.sub(r"[^\w\s]", "", t, flags=re.UNICODE)  # drop punctuation/symbols
    return _WS.sub(" ", t).strip()


def _systematic(sub: pd.DataFrame, k: int) -> pd.DataFrame:
    """Keep ``k`` rows evenly spaced across ``sub`` (assumed time-sorted).

    Deterministic even-stride sampling — spreads the kept rows across the block's
    time span instead of clustering at the earliest timestamps.
    """
    n = len(sub)
    if k <= 0 or n == 0:
        return sub.iloc[0:0]
    if n <= k:
        return sub
    idx = (np.arange(k) * n) // k  # k distinct, sorted indices spanning [0, n)
    return sub.iloc[idx]


def _admit(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(keep, via_bridge) masks for one day's titled rows.

    Base gate (unchanged from pre-v3.2): English keyword/producer/coin terms on
    every row regardless of language, OR the gold-price theme, minus stop-phrase
    collocations unless the theme rescues them. Bridge gate (v3.2): for rows
    whose ``src_lang`` is in ``BRIDGE_LANGS``, native terms hit and that
    language's stops do not. ``via_bridge`` marks rows admitted ONLY by the
    bridge — the unit the cap reserve allocates over.
    """
    themes = df["themes"].apply(_parse_themes_field)
    title = df["page_title"].astype(str)
    has_kw = title.str.contains(METAL_TITLE_RE, na=False)
    has_gold_theme = themes.apply(lambda t: _GOLD_THEME in t)
    is_stop = title.str.contains(_STOP_RE, na=False)
    base = (has_kw | has_gold_theme) & ~(is_stop & ~has_gold_theme)

    bridge = pd.Series(False, index=df.index)
    if "src_lang" in df.columns:
        langs = df["src_lang"].astype(str)
        for lang, (term_re, stop_re) in _BRIDGE_RES.items():
            in_lang = langs == lang
            if not in_lang.any():
                continue
            sub = title.loc[in_lang]
            hit = sub.str.contains(term_re, na=False)
            if stop_re is not None:
                hit &= ~sub.str.contains(stop_re, na=False)
            bridge.loc[in_lang] = hit
    keep = base | bridge
    return keep, bridge & ~base


def _allocate(counts: dict[str, int], budget: int, ceiling: int) -> dict[str, int]:
    """Largest-remainder proportional allocation, per-key ceiling, deterministic.

    Each key gets at most ``min(count, ceiling)``; the total never exceeds
    ``budget``. Ties break by key name so the draw is reproducible.
    """
    if budget <= 0 or not counts:
        return dict.fromkeys(counts, 0)
    total = sum(counts.values())
    if total == 0:
        return dict.fromkeys(counts, 0)
    caps = {k: min(v, ceiling) for k, v in counts.items()}
    ideal = {k: budget * counts[k] / total for k in counts}
    alloc = {k: min(int(ideal[k]), caps[k]) for k in counts}
    leftover = budget - sum(alloc.values())
    # Hand remaining slots to the largest fractional remainders with headroom.
    order = sorted(counts, key=lambda k: (-(ideal[k] - int(ideal[k])), k))
    while leftover > 0:
        progressed = False
        for k in order:
            if leftover == 0:
                break
            if alloc[k] < caps[k]:
                alloc[k] += 1
                leftover -= 1
                progressed = True
        if not progressed:  # every key at its ceiling — genuine slack remains
            break
    return alloc


def _select_language_stratified(df: pd.DataFrame, max_titles: int) -> pd.DataFrame:
    """v3.2 cap selection: base-gate floor + proportional bridge shares.

    Base admissions keep at least ``BASE_RESERVE_FRAC`` of the budget (selected
    with the existing US-session-aware ``_select_capped``); bridge languages
    share the rest proportionally to their day counts, each capped at
    ``BRIDGE_LANG_CEILING_FRAC`` of the bridge share and picked by even
    time-stride (bridge news cycles are not US-centred). Slack flows both ways
    so the budget fills whenever admissions allow.
    """
    if len(df) <= max_titles:
        return df
    base_df = df.loc[~df["_via_bridge"]]
    bridge_df = df.loc[df["_via_bridge"]]

    base_reserve = min(len(base_df), int(np.ceil(max_titles * BASE_RESERVE_FRAC)))
    bridge_budget = max_titles - base_reserve
    lang_counts = bridge_df.groupby("src_lang").size().to_dict()
    ceiling = max(1, round(bridge_budget * BRIDGE_LANG_CEILING_FRAC))
    alloc = _allocate(lang_counts, bridge_budget, ceiling)

    base_take = min(len(base_df), max_titles - sum(alloc.values()))
    leftover = max_titles - sum(alloc.values()) - base_take
    if leftover > 0:  # base exhausted — relax the ceiling and refill from bridge
        remaining = {k: lang_counts[k] - alloc[k] for k in lang_counts}
        extra = _allocate(remaining, leftover, ceiling=max(remaining.values(), default=0))
        alloc = {k: alloc[k] + extra.get(k, 0) for k in alloc}

    parts = [_select_capped(base_df, base_take)] if base_take else []
    for lang, k in sorted(alloc.items()):
        if k > 0:
            parts.append(_systematic(bridge_df.loc[bridge_df["src_lang"] == lang], k))
    kept = pd.concat(parts) if parts else df.iloc[0:0]
    return kept.sort_values("timestamp_utc")


def _select_capped(df: pd.DataFrame, max_titles: int) -> pd.DataFrame:
    """Select ``max_titles`` rows, reserving the US session, else stratified.

    Guarantees the US-afternoon window (``US_SESSION_LO..HI`` UTC) at least
    ``US_RESERVE_FRAC`` of the budget when it has enough titles, gives any unused
    slack back to whichever side has more, and evenly spreads each side's picks
    across time. Fills the budget exactly. ``df`` is assumed time-sorted.
    """
    if len(df) <= max_titles:
        return df
    hours = pd.to_datetime(df["timestamp_utc"]).dt.hour
    in_us = (hours >= US_SESSION_LO) & (hours <= US_SESSION_HI)
    us, other = df.loc[in_us], df.loc[~in_us]
    reserve = min(len(us), round(max_titles * US_RESERVE_FRAC))
    other_n = min(len(other), max_titles - reserve)
    us_n = min(len(us), max_titles - other_n)  # hand unused slack back to US
    kept = pd.concat([_systematic(us, us_n), _systematic(other, other_n)])
    return kept.sort_values("timestamp_utc")


def load_day_titles(date: str, *, max_titles: int = MAX_TITLES_PER_DAY) -> DayTitles:
    """Load, metals-filter, and de-duplicate one UTC day's titles."""
    pre_era = pd.Timestamp(date) < pd.Timestamp(REAL_TITLE_START)
    if pre_era:
        # Structurally titleless (GDELT emitted no PAGE_TITLE before 2019-09-22) —
        # flag it so a caller cannot mistake this for a genuinely newsless day.
        return DayTitles(date, [], [], 0, 0, pre_title_era=True)

    day = pd.Timestamp(date).normalize()
    nxt = day + pd.Timedelta(days=1)
    sql = (
        "SELECT timestamp_utc, headline_id, page_title, themes, src_lang "
        "FROM headlines "
        "WHERE timestamp_utc >= ? AND timestamp_utc < ? AND page_title IS NOT NULL "
        # headline_id tie-break: GDELT stamps arrive in 15-minute batches with
        # 500+ row tie groups, and DuckDB's parallel sort is unstable on ties —
        # without a total order the 250 selected titles differ per process.
        "ORDER BY timestamp_utc, headline_id"
    )
    with connection() as conn:
        df = conn.execute(sql, [str(day), str(nxt)]).fetchdf()

    n_titled = len(df)
    if n_titled:
        # Mask BEFORE admission/dedup so every downstream consumer (model,
        # auditor, dedupe key) sees the same blinded text.
        original = df["page_title"].astype(str)
        df["page_title"] = original.map(_mask_dates)
        df["_date_masked"] = df["page_title"] != original
    if df.empty:
        return DayTitles(date, [], [], 0, 0)  # corpus gap (n_titled 0)

    # Keep titles the base gate admits (metal/producer/coin keywords or the
    # gold-price theme, minus stop collocations) or the v3.2 language bridge
    # admits (nine measured languages; see _admit). Theme intersection remains a
    # proven no-op; residual off-topic hits are left for the LLM `relevant`
    # flag, reported per-language via the Stage-0 card.
    keep, via_bridge = _admit(df)
    df = df.assign(_via_bridge=via_bridge).loc[keep].reset_index(drop=True)
    n_raw = len(df)
    if n_raw == 0:
        return DayTitles(date, [], [], 0, 0, n_titled=n_titled)  # titled but metals-quiet

    # De-duplicate syndication on the normalized title key, keep earliest.
    df["_norm"] = df["page_title"].astype(str).map(_normalize)
    df = df.drop_duplicates(subset="_norm", keep="first").reset_index(drop=True)

    n_dropped_cap = max(0, len(df) - max_titles)
    df = _select_language_stratified(df, max_titles)

    return DayTitles(
        date=date,
        headline_ids=[str(x) for x in df["headline_id"].tolist()],
        titles=[str(x) for x in df["page_title"].tolist()],
        n_raw=n_raw,
        n_dropped_cap=n_dropped_cap,
        n_titled=n_titled,
        langs=[str(x) for x in df["src_lang"].tolist()] if "src_lang" in df.columns else [],
        n_date_masked=int(df["_date_masked"].sum()) if "_date_masked" in df.columns else 0,
    )
