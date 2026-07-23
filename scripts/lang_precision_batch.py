"""Precision mini-batch for the multilingual title gate (backlog F1, v3.2 prep).

Measures per-language precision of `multilang.LANG_TERMS` before any schema-v3.2
adoption decision: samples admitted titles per language (plus an eng anchor),
has the annotator model judge relevance via the Batch API, and reports precision
+ a false-positive taxonomy per language. Stages:

    uv run python scripts/lang_precision_batch.py sample      # DB scan (~90 s)
    uv run python scripts/lang_precision_batch.py estimate    # offline cost table
    uv run python scripts/lang_precision_batch.py run         # PAID: submit batch + poll
    uv run python scripts/lang_precision_batch.py report      # precision table + FP examples
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from metals.annotate import precision, schema

SAMPLE_PATH = Path("data/processed/lang_precision_sample.parquet")
RESULTS_PATH = Path("data/processed/lang_precision_results.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sample = sub.add_parser("sample", help="draw the per-language title sample (read-only scan)")
    p_sample.add_argument("--per-lang", type=int, default=precision.PER_LANG_SAMPLE)

    p_est = sub.add_parser("estimate", help="offline token/cost estimate")
    p_est.add_argument("--model", default=schema.MODEL_DEFAULT)

    p_run = sub.add_parser("run", help="submit the judging Batch (PAID) and poll")
    p_run.add_argument("--model", default=schema.MODEL_DEFAULT)
    p_run.add_argument("--out", type=Path, default=RESULTS_PATH)

    p_rep = sub.add_parser("report", help="per-language precision report")
    p_rep.add_argument("--results", type=Path, default=RESULTS_PATH)

    args = parser.parse_args()

    if args.cmd == "sample":
        df = precision.draw_sample(per_lang=args.per_lang)
        SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(SAMPLE_PATH, index=False)
        counts = df.groupby("lang").size().to_dict()
        print(f"Wrote {SAMPLE_PATH} — {len(df)} titles across {df['lang'].nunique()} languages")
        print("  " + ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())))
    elif args.cmd == "estimate":
        df = pd.read_parquet(SAMPLE_PATH)
        print(precision.estimate(df, model=args.model))
    elif args.cmd == "run":
        df = pd.read_parquet(SAMPLE_PATH)
        res = precision.run_judge(df, model=args.model, out_path=args.out)
        n_ok = int(res["ok"].sum())
        print(f"Wrote {args.out} — {n_ok}/{len(res)} titles judged (model {args.model})")
    elif args.cmd == "report":
        res = pd.read_parquet(args.results)
        print(precision.report(res))


if __name__ == "__main__":
    main()
