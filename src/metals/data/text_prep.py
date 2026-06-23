"""Text preparation for GDELT headlines (Phase 3 step 3.5).

GDELT GKG records carry no headline text — only the article URL
(``DocumentIdentifier``). Embedding the raw URL is weak signal, but most news
URLs embed a human-readable *slug* in the path
(``.../gold-hits-record-high-2024/12345``). This module turns a URL into the
cleanest readable text we can recover, so the embedding step (3.6) and the daily
aggregation (3.7) operate on words rather than on ``https://...``.

Pure functions only — no DB, no network — so they are cheap to unit-test and can
be applied on the fly when building embedding inputs (we deliberately do not
store the derived text as a 14M-row column).

Caveat: GDELT is global, so many slugs are non-English. This recovers the words;
it does not translate them. Language handling is a downstream concern.
"""

from __future__ import annotations

import re
from typing import Iterable
from urllib.parse import urlsplit

# File extensions that appear at the end of a path segment on news sites.
_EXT_RE = re.compile(
    r"\.(html?|s?html|ece|chn|cms|amp|aspx?|php|jsp|stm|asp)$", re.IGNORECASE
)
# Split camelCase ("currenciesNews" -> "currencies News").
_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")
_TOKEN_SPLIT_RE = re.compile(r"[-_\s]+")


def _strip_ext(segment: str) -> str:
    return _EXT_RE.sub("", segment)


def _segment_score(segment: str) -> tuple[int, int]:
    """Rank a path segment by how much readable text it carries.

    Returns (number of alphabetic words >= 3 chars, total length of those words).
    """
    words = [t for t in _TOKEN_SPLIT_RE.split(segment) if t.isalpha() and len(t) >= 3]
    return (len(words), sum(len(w) for w in words))


def slug_from_url(url: str) -> str:
    """Return the most readable path segment of ``url`` (extension stripped).

    Picks the path segment with the richest readable-word content, which skips
    over numeric id / section segments (``/news/265569381/the-real-slug``).
    """
    if not isinstance(url, str) or not url:
        return ""
    parts = urlsplit(url)
    if not parts.netloc:        # require scheme://host, else it is not a URL
        return ""
    segments = [_strip_ext(s) for s in parts.path.split("/") if s]
    if not segments:
        return ""
    best = max(segments, key=_segment_score)
    return best if _segment_score(best)[0] > 0 else ""


def clean_slug_text(slug: str) -> str:
    """Normalise a slug into lowercase space-separated words.

    Drops pure-number tokens (article ids, dates), id-like alphanumerics, and
    1-character noise; splits camelCase and hyphen/underscore boundaries.
    """
    if not slug:
        return ""
    slug = _strip_ext(slug)
    slug = _CAMEL_RE.sub(" ", slug)
    keep: list[str] = []
    for tok in _TOKEN_SPLIT_RE.split(slug):
        if not tok or len(tok) < 2:
            continue
        if tok.isdigit():
            continue
        n_digits = sum(c.isdigit() for c in tok)
        if n_digits and n_digits / len(tok) > 0.3:  # id-like (e.g. "BB14rrjS")
            continue
        keep.append(tok.lower())
    return " ".join(keep)


def url_to_text(url: str) -> str:
    """Recover readable text from a news URL. Empty string if nothing usable."""
    return clean_slug_text(slug_from_url(url))


def truncate_words(text: str, max_words: int = 256) -> str:
    """Truncate to ``max_words`` whitespace tokens (embedding efficiency)."""
    if not text:
        return ""
    words = text.split()
    return " ".join(words[:max_words])


def prepare_texts(urls: Iterable[str], max_words: int = 256) -> list[str]:
    """Convenience: map an iterable of URLs to cleaned, truncated embedding texts."""
    return [truncate_words(url_to_text(u), max_words) for u in urls]
