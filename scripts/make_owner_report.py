"""Generate the AMC owner's plain-language briefing as a PDF.

    uv run python scripts/make_owner_report.py
    uv run python scripts/make_owner_report.py --out results/owner_briefing.pdf

Live state (spread-floor figures, ledger status, data coverage) is read from
DuckDB at generation time, so re-running after new data lands produces an
updated document with no edits to the source.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from metals.report import owner_report

# Anchored to the repo root (not the cwd) so running the script from anywhere
# updates the canonical committed PDF instead of silently creating a new
# results/ directory wherever the shell happens to be.
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "results" / "amc_owner_briefing.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Destination PDF path (default: {DEFAULT_OUT}).",
    )
    args = parser.parse_args()

    path = owner_report.build(args.out)
    size_kb = path.stat().st_size / 1024
    print(f"Wrote {path} ({size_kb:,.0f} KB)")


if __name__ == "__main__":
    main()
