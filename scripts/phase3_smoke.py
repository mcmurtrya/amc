"""Phase 3 pipeline smoke test.

Proves the (column-fixed) Phase 3 stage queries bind against the live
``headlines`` schema and that the real embed -> aggregate path runs on a
bounded slice. Read-only with respect to the DuckDB store (writes nothing to
any table); the only side effect is populating the on-disk embedding cache for
the small slice it processes.

Run:
    uv run python scripts/phase3_smoke.py
Exits non-zero if any check fails.
"""

from __future__ import annotations

import sys

import pandas as pd

from metals.data.db import connection
from metals.features.embeddings import embed_texts
from metals.features.text_daily import _parse_themes_field, aggregate_daily

SLICE = 2000

# The exact column projections each pipeline stage uses, post-fix. Bind-checking
# these against the live schema is what would have caught the document_identifier
# regression before a multi-hour run.
STAGE_QUERIES = {
    "embed":     "SELECT timestamp_utc, headline_id, article_url FROM headlines LIMIT 5",
    "aggregate": ("SELECT timestamp_utc, headline_id, source, themes, "
                  "article_url, tone_overall, tone_positive, tone_negative "
                  "FROM headlines LIMIT 5"),
    "topics":    "SELECT timestamp_utc, article_url FROM headlines LIMIT 5",
    "label":     ("SELECT timestamp_utc, article_url AS headline, article_url "
                  "FROM headlines LIMIT 5"),
}


def main() -> int:
    failures: list[str] = []

    def check(name: str, fn) -> None:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:  # noqa: BLE001 — smoke wants the message, not a trace
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
            failures.append(name)

    print("[1] stage queries bind against the live schema:")
    with connection(read_only=True) as conn:
        for name, q in STAGE_QUERIES.items():
            check(f"{name} query", lambda q=q: conn.execute(q).fetchdf())

    print(f"\n[2] embed -> aggregate on a {SLICE}-row slice (real GPU path, no DB writes):")
    with connection(read_only=True) as conn:
        hl = conn.execute(
            "SELECT timestamp_utc, headline_id, source, themes, article_url, "
            "tone_overall, tone_positive, tone_negative "
            "FROM headlines ORDER BY timestamp_utc DESC LIMIT ?",
            [SLICE],
        ).fetchdf()
    hl["themes_list"] = hl["themes"].apply(_parse_themes_field)
    hl["timestamp_utc"] = pd.to_datetime(hl["timestamp_utc"])

    def run_path() -> None:
        emb = embed_texts(hl["article_url"].astype(str).tolist())
        assert emb.shape[0] == len(hl), f"embedded {emb.shape[0]} != {len(hl)} rows"
        out = aggregate_daily(hl, embeddings=emb)
        assert not out.empty, "aggregate_daily returned no rows"
        print(f"        embedded {emb.shape[0]:,} urls -> dim {emb.shape[1]}")
        print(f"        daily feature rows: {len(out)}  metals: {sorted(out['metal'].unique())}")
        cols = ["timestamp_utc", "metal", "n_articles",
                "embedding_dispersion", "mean_tone_overall"]
        print(out[cols].head().to_string(index=False))

    check("embed->aggregate slice", run_path)

    print()
    if failures:
        print(f"SMOKE FAILED: {len(failures)} check(s) failed: {failures}")
        return 1
    print("SMOKE PASSED: stages bind and the embed->aggregate path runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
