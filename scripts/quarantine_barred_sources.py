"""Quarantine rows captured from data sources whose Terms of Use bar AMC's use.

Data classification, not schema: migration ``010_quarantine_flag.sql`` adds the
nullable ``quarantine_reason`` column to the affected tables; this script stamps
the rows that were captured before the 2026-07-16 ToU audit (journal.md) with a
per-source reason. A non-NULL ``quarantine_reason`` means the row was acquired
outside its source's licence and must be excluded from model training and from
any analysis shipped to AMC until a licence clears it; downstream loaders filter
on ``quarantine_reason IS NULL``.

Idempotent and reversible:
- Only rows with ``quarantine_reason IS NULL`` are stamped, so re-running is a
  no-op and legitimately-licensed rows added later (which land with NULL) are
  never touched.
- Clearing a quarantine once a licence lands is a one-liner:
  ``UPDATE <table> SET quarantine_reason = NULL WHERE ...``.

The four sources and why each is barred (all audited + adversarially verified
2026-07-16):
- coin_premiums  — APMEX / JM Bullion ToU: automated retrieval, commercial-use,
  and cached-dataset bars (JM Bullion also an anti-evasion clause).
- macro_consensus — Fair Economy (ForexFactory) FEED: copying "in part or in
  whole" prohibited without prior written consent.
- search_interest — Google Trends: the DATA is licensed, but these rows were
  acquired via the internal endpoint by defeating its non-browser 429 gate.
  Supersede with the sanctioned CSV export (see trends.py) before training.
- pgm_prices     — Johnson Matthey: "any use of the Prices without ... consent
  is prohibited", plus UK sui generis database right over a substantial extract.

Run as:
    uv run python scripts/quarantine_barred_sources.py --dry-run   # preview
    uv run python scripts/quarantine_barred_sources.py             # apply
"""

from __future__ import annotations

import argparse

from metals.data.db import connection

# One entry per barred table. The reason text is stored verbatim in the DB, so
# keep it short but specific enough to explain the exclusion without journal.md.
QUARANTINE_REASONS: dict[str, str] = {
    "coin_premiums": (
        "BARRED 2026-07-16: APMEX/JM Bullion ToU bar automated retrieval, "
        "commercial use, and cached datasets. No licence. See journal.md."
    ),
    "macro_consensus": (
        "BARRED 2026-07-16: Fair Economy (ForexFactory) FEED bars copying in "
        "part or whole without written consent. FEI consent pending. See journal.md."
    ),
    "search_interest": (
        "BARRED 2026-07-16: acquired via Google Trends internal endpoint past its "
        "non-browser 429 gate. Data is licensed; re-acquire via sanctioned CSV "
        "export before training. See journal.md."
    ),
    "pgm_prices": (
        "BARRED 2026-07-16: Johnson Matthey 'any use of the Prices' clause + UK "
        "database right over a substantial extract. JM consent pending. See journal.md."
    ),
}


def quarantine(dry_run: bool = False) -> dict[str, dict[str, int]]:
    """Stamp un-classified rows in each barred table. Returns per-table counts."""
    summary: dict[str, dict[str, int]] = {}
    with connection() as conn:
        for table, reason in QUARANTINE_REASONS.items():
            total = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            pending = conn.execute(
                f"SELECT count(*) FROM {table} WHERE quarantine_reason IS NULL"
            ).fetchone()[0]
            if not dry_run and pending:
                conn.execute(
                    f"UPDATE {table} SET quarantine_reason = ? WHERE quarantine_reason IS NULL",
                    [reason],
                )
            summary[table] = {"total": int(total), "stamped": int(pending)}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be stamped without writing.",
    )
    args = parser.parse_args()
    summary = quarantine(dry_run=args.dry_run)
    verb = "would stamp" if args.dry_run else "stamped"
    for table, counts in summary.items():
        print(f"{table:16} {counts['total']:>8,} rows total  {verb} {counts['stamped']:>8,}")
    total_stamped = sum(c["stamped"] for c in summary.values())
    print(
        f"\nTotal {verb}: {total_stamped:,} row(s)"
        + (" (dry run — nothing written)" if args.dry_run else "")
    )


if __name__ == "__main__":
    main()
