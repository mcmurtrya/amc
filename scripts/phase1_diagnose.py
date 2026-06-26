"""Phase 1 cleanup: diagnose negative IC on silver, platinum, palladium.

Three questions to answer:
  1. Is the IC genuinely negative on Ag/Pt/Pd or is it within noise of zero?
  2. Which feature subset is responsible — price, spreads, or macro?
  3. Are feature importances stable across walk-forward splits?

Reads prices and macro from DuckDB (refresh first with
``uv run python -m metals.data.prices`` and ``... -m metals.data.fred``),
trains LightGBM under five feature configurations per metal, records every
configuration to the eval harness, and writes a markdown report to
``results/phase1_negative_ic_diagnosis.md``.

Run as:
    uv run python -m scripts.phase1_diagnose
or:
    uv run python scripts/phase1_diagnose.py
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from metals.eval.cv import walk_forward_splits
from metals.eval.harness import (
    aggregate_feature_importances,
    log_feature_importances,
    log_predictions,
    register_run,
)
from metals.features.assemble import build_feature_matrix
from metals.features.loaders import load_macro, load_prices
from metals.models.lgbm_vol import train_one_split

METALS = ["GC=F", "SI=F", "PL=F", "PA=F"]

# Feature-subset groupings, defined by substring on the column name.
SUBSETS: dict[str, tuple[str, ...]] = {
    "returns_and_vol": ("_ret_", "_rvol_", "_skew_", "_kurt_", "_maxdd_"),
    "spreads": ("Au_Ag", "Pt_Pd", "Au_Cu", "Au_Oil"),
    "macro": (
        "real_yield",
        "breakeven",
        "dxy_",
        "vix_",
        "yield_curve",
        "baa_spread",
        "gpr_",
    ),
}


def feature_subset(columns: list[str], substrs: tuple[str, ...]) -> list[str]:
    return [c for c in columns if any(s in c for s in substrs)]


def configurations(all_features: list[str]) -> dict[str, list[str]]:
    """Configurations to compare per metal.

    - ``all``: every feature
    - ``no_<group>``: ablation; drop just that group
    - ``only_<group>``: marginal; train on that group alone
    """
    cfg: dict[str, list[str]] = {"all": list(all_features)}
    for name, substrs in SUBSETS.items():
        group = feature_subset(all_features, substrs)
        if not group:
            continue
        cfg[f"no_{name}"] = [c for c in all_features if c not in group]
        cfg[f"only_{name}"] = group
    return cfg


def run_one_config(
    fm,
    feature_cols: list[str],
    label: str,
    splits,
    ticker: str,
) -> dict:
    """Train + log to the eval harness, return per-split IC vector."""
    sub_X = fm.X[feature_cols]
    sub_fm = type(fm)(
        X=sub_X,
        y=fm.y,
        target_name=fm.target_name,
        target_horizon=fm.target_horizon,
        feature_names=feature_cols,
    )
    name = f"diag_{ticker}_{label}_{datetime.now():%Y%m%d_%H%M}"
    rid = register_run(
        name=name,
        model_type="lgbm_vol_diag",
        target_type="realized_vol",
        config={"ticker": ticker, "subset": label, "n_features": len(feature_cols)},
        notes="phase1_negative_ic_diagnosis",
    )
    ics: list[float] = []
    for split in splits:
        preds, result, importances = train_one_split(sub_fm, split)
        log_predictions(rid, preds)
        for imp_type, imp_dict in importances.items():
            log_feature_importances(rid, split.split_id, imp_dict, importance_type=imp_type)
        ics.append(result.ic)
    return {
        "label": label,
        "run_id": rid,
        "n_features": len(feature_cols),
        "ics": ics,
        "ic_mean": float(np.nanmean(ics)),
        "ic_std": float(np.nanstd(ics)),
        "ic_pos_frac": float(np.mean([ic > 0 for ic in ics if not np.isnan(ic)])),
    }


def diagnose_metal(
    ticker: str,
    prices: pd.DataFrame,
    macro_wide: pd.DataFrame,
    horizon: int,
    vol_window: int,
    train_start: str,
    val_days: int,
    test_days: int,
    step_days: int,
    min_train_days: int,
) -> list[dict]:
    fm = build_feature_matrix(
        prices=prices,
        macro_wide=macro_wide,
        target_ticker=ticker,
        target_kind="realized_vol",
        target_horizon=horizon,
        realized_vol_window=vol_window,
    )
    splits = list(
        walk_forward_splits(
            timestamps=fm.X.index,
            train_start=train_start,
            val_days=val_days,
            test_days=test_days,
            step_days=step_days,
            min_train_days=min_train_days,
        )
    )
    if not splits:
        return [
            {
                "label": "(no splits)",
                "run_id": None,
                "n_features": len(fm.X.columns),
                "ics": [],
                "ic_mean": float("nan"),
                "ic_std": float("nan"),
                "ic_pos_frac": float("nan"),
            }
        ]
    out = []
    for label, cols in configurations(list(fm.X.columns)).items():
        out.append(run_one_config(fm, cols, label, splits, ticker))
    return out


def render_report(
    per_metal: dict[str, list[dict]],
    horizon: int,
    target_kind: str,
) -> str:
    lines: list[str] = []
    lines.append(f"# Phase 1 diagnosis: negative IC investigation\n")
    lines.append(f"_Generated {datetime.now():%Y-%m-%d %H:%M}_\n")
    lines.append(f"Target: `{target_kind}`, horizon: {horizon} trading days.\n")
    lines.append("## Per-metal subset comparison\n")
    for ticker, rows in per_metal.items():
        lines.append(f"### {ticker}\n")
        lines.append(
            "| subset | n_features | n_splits | mean IC | std IC | pos fraction | run_id |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for r in rows:
            n = len(r["ics"])
            run_id = (r["run_id"] or "")[:8]
            lines.append(
                f"| `{r['label']}` | {r['n_features']} | {n} | "
                f"{r['ic_mean']:+.3f} | {r['ic_std']:.3f} | "
                f"{r['ic_pos_frac']:.0%} | `{run_id}` |"
            )
        lines.append("")
    lines.append("## Reading guide\n")
    lines.append(
        "- **`all` row** is the production baseline.\n"
        "- **`no_<group>` ablation**: if mean IC *rises* when a group is dropped, "
        "that group is hurting predictions for this metal — a sign the feature "
        "is mismeasured or simply doesn't carry signal here.\n"
        "- **`only_<group>` marginal**: if a single group recovers most of the "
        "production IC alone, that group is doing the work; the others are noise.\n"
        "- **pos fraction** is the share of splits with IC > 0. A value near 0.5 "
        "indicates pure noise; near 1.0 indicates consistent signal.\n"
    )
    lines.append("## Next steps\n")
    lines.append(
        "Cross-reference the worst-performing metals against "
        "`aggregate_feature_importances(run_id)` to see which features the "
        "production model is *spending* its capacity on, then compare with "
        "subset-level IC above. Disagreement (high importance, low marginal "
        "lift) is the strongest signal that a feature is fitting noise.\n"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--train-start", default="2010-01-01")
    parser.add_argument("--val-days", type=int, default=180)
    parser.add_argument("--test-days", type=int, default=180)
    parser.add_argument("--step-days", type=int, default=180)
    parser.add_argument("--min-train-days", type=int, default=5 * 365)
    parser.add_argument("--metals", nargs="*", default=METALS)
    parser.add_argument(
        "--out",
        default="results/phase1_negative_ic_diagnosis.md",
        help="Output markdown path (relative to repo root).",
    )
    args = parser.parse_args()

    prices = load_prices(column="adj_close")
    macro_wide = load_macro()
    if prices.empty or macro_wide.empty:
        raise RuntimeError("Empty prices or macro DuckDB tables. Refresh ingestion before running.")

    per_metal: dict[str, list[dict]] = {}
    for ticker in args.metals:
        print(f"\n=== {ticker} ===")
        per_metal[ticker] = diagnose_metal(
            ticker=ticker,
            prices=prices,
            macro_wide=macro_wide,
            horizon=args.horizon,
            vol_window=args.vol_window,
            train_start=args.train_start,
            val_days=args.val_days,
            test_days=args.test_days,
            step_days=args.step_days,
            min_train_days=args.min_train_days,
        )
        for r in per_metal[ticker]:
            print(
                f"  {r['label']:>22s}  "
                f"n_feat={r['n_features']:>3d}  "
                f"mean_IC={r['ic_mean']:+.3f}  "
                f"pos={r['ic_pos_frac']:.0%}"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_report(per_metal, args.horizon, "realized_vol"))
    print(f"\nWrote diagnostic report to {out_path}")


if __name__ == "__main__":
    main()
