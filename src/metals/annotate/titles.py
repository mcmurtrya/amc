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
from dataclasses import dataclass

import numpy as np
import pandas as pd

from metals.data.db import connection
from metals.features.text_daily import _parse_themes_field

# Cap distinct titles fed to the annotator per day, to bound token cost on the
# high-volume days. Dropped titles are reported, never silently hidden. Selection
# is TIME-STRATIFIED (see _select_capped) so the cap doesn't discard the
# US-afternoon session by clock.
MAX_TITLES_PER_DAY = 250

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
        "SELECT timestamp_utc, headline_id, page_title, themes "
        "FROM headlines "
        "WHERE timestamp_utc >= ? AND timestamp_utc < ? AND page_title IS NOT NULL "
        "ORDER BY timestamp_utc"
    )
    with connection() as conn:
        df = conn.execute(sql, [str(day), str(nxt)]).fetchdf()

    n_titled = len(df)
    if df.empty:
        return DayTitles(date, [], [], 0, 0)  # corpus gap (n_titled 0)

    # Keep titles that name a metal / producer / coin, or carry the gold-price
    # theme, minus obvious off-topic collocations (unless the gold theme rescues
    # them). A theme intersection does NOT help: metals_for_themes maps generic
    # macro themes (ECON_INFLATION, WB_1699 ~57% of corpus) so broadly it is true
    # for ~every keyword hit — GDELT themes cannot separate metals-financial news
    # from the macro firehose. Residual off-topic hits are left for the LLM
    # `relevant` flag, and their rate is reported as `corpus_offtopic_fraction`.
    themes = df["themes"].apply(_parse_themes_field)
    title = df["page_title"].astype(str)
    has_kw = title.str.contains(METAL_TITLE_RE, na=False)
    has_gold_theme = themes.apply(lambda t: _GOLD_THEME in t)
    is_stop = title.str.contains(_STOP_RE, na=False)
    keep = (has_kw | has_gold_theme) & ~(is_stop & ~has_gold_theme)
    df = df.loc[keep].reset_index(drop=True)
    n_raw = len(df)
    if n_raw == 0:
        return DayTitles(date, [], [], 0, 0, n_titled=n_titled)  # titled but metals-quiet

    # De-duplicate syndication on the normalized title key, keep earliest.
    df["_norm"] = df["page_title"].astype(str).map(_normalize)
    df = df.drop_duplicates(subset="_norm", keep="first").reset_index(drop=True)

    n_dropped_cap = max(0, len(df) - max_titles)
    df = _select_capped(df, max_titles)

    return DayTitles(
        date=date,
        headline_ids=[str(x) for x in df["headline_id"].tolist()],
        titles=[str(x) for x in df["page_title"].tolist()],
        n_raw=n_raw,
        n_dropped_cap=n_dropped_cap,
        n_titled=n_titled,
    )
