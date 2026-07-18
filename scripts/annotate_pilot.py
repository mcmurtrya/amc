"""Stage-0 LLM-annotator feasibility pilot (Phase 8 §8.1).

Cheap go/no-go test before annotating all ~1,678 title-era days. Stages:

    sample    draw the ~80-day stratified sample (deterministic)
    estimate  dry-run cost table across model tiers (no paid call by default)
    run       submit the Batch annotation (date-blind + date-visible A/B), cache
    check     compute the pre-registered pass/fail report card

Examples:
    uv run python scripts/annotate_pilot.py sample
    uv run python scripts/annotate_pilot.py estimate --model claude-opus-4-8
    uv run python scripts/annotate_pilot.py estimate --use-api-count      # exact tokens
    uv run python scripts/annotate_pilot.py run --out data/processed/annotate_pilot.parquet
    uv run python scripts/annotate_pilot.py check --results data/processed/annotate_pilot.parquet
"""

from __future__ import annotations

import argparse

import pandas as pd

from metals.annotate import pilot, schema
from metals.annotate.checks import report_card
from metals.annotate.sample import draw_sample

DEFAULT_OUT = "data/processed/annotate_pilot.parquet"


def _cmd_sample(args: argparse.Namespace) -> None:
    df = draw_sample(seed=args.seed)
    print(df.to_string(index=False))
    print(
        f"\n{len(df)} days: "
        + ", ".join(f"{k}={v}" for k, v in df["stratum"].value_counts().items())
    )
    if args.out:
        df.to_parquet(args.out, index=False)
        print(f"wrote {args.out}")


def _cmd_estimate(args: argparse.Namespace) -> None:
    df = draw_sample(seed=args.seed)
    n_variants = 2 if not args.no_ab else 1
    est = pilot.estimate_run(
        df,
        model=args.model,
        n_variants=n_variants,
        use_api_count=args.use_api_count,
    )
    print(pilot.format_estimate(est))


def _cmd_run(args: argparse.Namespace) -> None:
    df = draw_sample(seed=args.seed)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    print(
        f"Submitting Batch: {len(df)} days x {1 if args.no_ab else 2} variant(s), "
        f"model={args.model}, prompt_hash={schema.prompt_hash(args.model)}"
    )
    out = pilot.run_pilot(df, model=args.model, date_blind_ab=not args.no_ab, out_path=args.out)
    print(f"wrote {args.out} ({len(out)} rows, {int(out['ok'].sum())} ok)")
    print("\n" + report_card(out))


def _cmd_check(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.results)
    print(report_card(df))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seed", type=int, default=42)
    sub = p.add_subparsers(dest="stage", required=True)

    s = sub.add_parser("sample", help="draw the stratified day sample")
    s.add_argument("--out", default=None)
    s.set_defaults(func=_cmd_sample)

    e = sub.add_parser("estimate", help="dry-run cost estimate")
    e.add_argument("--model", default=schema.MODEL_DEFAULT)
    e.add_argument("--no-ab", action="store_true", help="single variant (no date-blind A/B)")
    e.add_argument("--use-api-count", action="store_true", help="exact tokens via count_tokens")
    e.set_defaults(func=_cmd_estimate)

    r = sub.add_parser("run", help="submit the Batch annotation")
    r.add_argument("--model", default=schema.MODEL_DEFAULT)
    r.add_argument("--no-ab", action="store_true")
    r.add_argument("--limit", type=int, default=None, help="cap days (smoke test)")
    r.add_argument("--out", default=DEFAULT_OUT)
    r.set_defaults(func=_cmd_run)

    c = sub.add_parser("check", help="compute the report card")
    c.add_argument("--results", default=DEFAULT_OUT)
    c.set_defaults(func=_cmd_check)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
