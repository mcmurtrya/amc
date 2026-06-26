"""Quantify URL-slug recovery quality across the whole GDELT corpus.

GDELT GKG stores only article URLs, so Phase 3 recovers readable text from the
URL slug (``metals.data.text_prep.url_to_text``) before embedding. A curated
look at clean English news domains shows near-perfect recovery, but that slice
is survivorship-biased. This script draws a *uniform* sample across all
~63M rows (no domain filter) and reports the distribution of recovery quality,
so the "how good is the slug as a headline proxy?" question gets a number
instead of a hand-picked table.

Read-only. Writes nothing to the DB.

Metrics reported:
  - recovery buckets: empty / weak (1-2 tok) / moderate (3-5) / rich (6+)
  - token-count percentiles
  - id-leak rate: fraction whose recovered text still contains a digit-bearing
    junk token (e.g. Reuters ``idINKBN20W0EA`` -> leaked ``inkbn20w0ea``)
  - trailing bare ``id`` rate (the common Reuters artifact)
  - English-likelihood proxy: fraction containing >=1 common English stopword
    (crude; flags the non-English share the recovery can't translate)
  - top TLDs in the sample, to show the domain mix behind the numbers

Run:
    uv run python scripts/phase3_slug_quality.py            # 5000-row sample
    uv run python scripts/phase3_slug_quality.py --n 20000 --examples 8
"""

from __future__ import annotations

import argparse
from collections import Counter
from urllib.parse import urlsplit

from metals.data.db import connection
from metals.data.text_prep import url_to_text

# A tiny, high-frequency English function-word set. Presence of any of these is
# a cheap proxy for "the recovered slug is English"; absence flags the global /
# non-English share that recovery captures as words but cannot translate.
_STOPWORDS = frozenset(
    "the to of in on for and is as at by with from up down over after "
    "new us out off into amid near high low cuts rises falls hits".split()
)


def _has_digit_token(text: str) -> bool:
    return any(any(c.isdigit() for c in tok) for tok in text.split())


def _is_english_ish(text: str) -> bool:
    return any(tok in _STOPWORDS for tok in text.split())


def _tld(url: str) -> str:
    host = urlsplit(url).netloc.lower()
    if not host:
        return "(none)"
    host = host.split(":")[0]
    parts = host.split(".")
    return parts[-1] if parts else "(none)"


def _percentile(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, int(q * len(sorted_vals)))
    return sorted_vals[idx]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=5000, help="sample size (rows)")
    ap.add_argument("--seed", type=int, default=42, help="repeatable-sample seed")
    ap.add_argument("--examples", type=int, default=5, help="examples per bucket")
    args = ap.parse_args()

    # SAMPLE does not accept bind params; args.n/seed are argparse ints, so safe.
    n_rows, seed = int(args.n), int(args.seed)
    with connection(read_only=True) as conn:
        rows = conn.execute(
            "SELECT article_url FROM headlines "
            f"USING SAMPLE reservoir({n_rows} ROWS) REPEATABLE({seed})"
        ).fetchall()

    urls = [r[0] for r in rows if r[0]]
    n = len(urls)
    if not n:
        print("No URLs sampled.")
        return 1

    token_counts: list[int] = []
    buckets: dict[str, list[tuple[str, str]]] = {
        "empty": [], "weak": [], "moderate": [], "rich": []
    }
    id_leak = trailing_id = english = 0
    tld_counter: Counter[str] = Counter()

    for url in urls:
        text = url_to_text(url)
        toks = text.split()
        nt = len(toks)
        token_counts.append(nt)
        tld_counter[_tld(url)] += 1

        if nt == 0:
            bucket = "empty"
        elif nt <= 2:
            bucket = "weak"
        elif nt <= 5:
            bucket = "moderate"
        else:
            bucket = "rich"
        if len(buckets[bucket]) < args.examples:
            buckets[bucket].append((url, text))

        if text:
            if _has_digit_token(text):
                id_leak += 1
            if toks[-1] == "id":
                trailing_id += 1
            if _is_english_ish(text):
                english += 1

    token_counts.sort()

    def pct(k: int) -> str:
        return f"{100 * k / n:5.1f}%"

    print(f"\nGDELT slug-recovery quality — uniform sample of {n} URLs "
          f"(seed {args.seed})\n" + "=" * 68)

    print("\nRecovery buckets (by recovered token count):")
    counts = Counter(
        "empty" if c == 0 else "weak" if c <= 2 else "moderate" if c <= 5 else "rich"
        for c in token_counts
    )
    for b in ("empty", "weak", "moderate", "rich"):
        label = {
            "empty": "empty   (0 tokens — no usable text)",
            "weak": "weak    (1-2 — section word / id only)",
            "moderate": "moderate(3-5 tokens)",
            "rich": "rich    (6+ tokens — headline-like)",
        }[b]
        print(f"  {label:42s} {counts[b]:6d}  {pct(counts[b])}")

    print("\nToken-count percentiles:")
    for q, lbl in [(0.10, "p10"), (0.25, "p25"), (0.50, "p50"),
                   (0.75, "p75"), (0.90, "p90"), (0.99, "p99")]:
        print(f"  {lbl}: {_percentile(token_counts, q):3d}")
    print(f"  mean: {sum(token_counts) / n:.2f}")

    nonempty = n - counts["empty"]
    print("\nNoise / language (of non-empty recoveries"
          f", n={nonempty}):")
    if nonempty:
        print(f"  digit-bearing junk token (id-leak): {id_leak:6d}  "
              f"{100 * id_leak / nonempty:5.1f}%")
        print(f"  trailing bare 'id' token:           {trailing_id:6d}  "
              f"{100 * trailing_id / nonempty:5.1f}%")
        print(f"  English-ish (>=1 stopword):         {english:6d}  "
              f"{100 * english / nonempty:5.1f}%")

    print("\nTop 12 TLDs in sample:")
    for tld, c in tld_counter.most_common(12):
        print(f"  .{tld:6s} {c:6d}  {pct(c)}")

    print("\nExamples per bucket:")
    for b in ("empty", "weak", "moderate", "rich"):
        print(f"\n  [{b}]")
        for url, text in buckets[b]:
            short = url if len(url) <= 88 else url[:85] + "..."
            print(f"    URL : {short}")
            print(f"    TEXT: {text!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
