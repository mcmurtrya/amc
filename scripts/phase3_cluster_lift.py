"""Cluster -> forward-vol lift experiment, arms A/B/B_notext.

Executable form of the pre-registered design in
``results/phase3_cluster_lift_design.md`` (decision rules were fixed there
before any run; do not change them here without noting it in the journal).

Arms (all on the identical ``shared`` row set and identical folds):

  A         Phase-1 feature matrix only (``feature_set="full"``).
  B         A + per-fold regime features from the Option-C context
            (``build_regime_features``: one-hots + confidence + purged
            target encoding, refit per fold at ``boundary=split.train_end``).
  B_notext  B with the text-derived context columns removed before
            clustering. Run only if B beats A (attribution ablation).

Primary decision target: GC=F realized vol, h=5, w=20. SI=F and h=20 are
report-only secondaries. Arm C (title-era embeddings) is a separate
sub-experiment and is NOT run here — this script's readout gates it.

Usage:
  uv run python scripts/phase3_cluster_lift.py            # full pre-registered set
  uv run python scripts/phase3_cluster_lift.py --primary-only
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from metals.eval.cv import Split, walk_forward_splits
from metals.eval.harness import log_feature_importances, log_predictions, register_run
from metals.features.assemble import FeatureMatrix, build_feature_matrix
from metals.features.context import ContextConfig, build_context
from metals.features.loaders import load_macro, load_prices
from metals.features.regimes import RegimeFeatureConfig, build_regime_features, purge_days_for
from metals.features.text_daily import load_daily as load_text_daily
from metals.features.topics import load_topic_prevalence_wide
from metals.models.lgbm_vol import DEFAULT_LGBM_PARAMS, train_one_split

# Pre-registered pins (design doc "Determinism and bookkeeping").
PINNED_LGBM = {"seed": 42, "deterministic": True, "force_row_wise": True, "num_threads": 8}
FOLD_KW = dict(
    train_start="2015-02-19", val_days=180, test_days=180, step_days=180, min_train_days=5 * 365
)
REL_BAR_B_VS_A = -0.010
WIN_FRAC = 0.6
VOL_WINDOW = 20


def _is_text_col(col: str) -> bool:
    """Text-derived context columns per the design's B_notext definition."""
    return col == "n_articles" or col.startswith("mean_tone_") or col.startswith("topic_")


@dataclass
class ArmResult:
    run_id: str
    rmses: list[float]
    ics: list[float]

    @property
    def mean_rmse(self) -> float:
        return float(np.mean(self.rmses))


def run_arm(
    arm: str,
    fm: FeatureMatrix,
    splits: list[Split],
    ticker: str,
    horizon: int,
    context: pd.DataFrame | None = None,
) -> ArmResult:
    """Run one arm through every split, logging to the eval harness."""
    params = {**DEFAULT_LGBM_PARAMS, **PINNED_LGBM}
    purge = purge_days_for(horizon, VOL_WINDOW)
    run_id = register_run(
        name=f"lift_{arm}_{ticker}_h{horizon}_{datetime.now():%Y%m%d_%H%M}",
        model_type="lgbm_vol_lift",
        target_type="realized_vol",
        config={
            "design": "results/phase3_cluster_lift_design.md",
            "arm": arm,
            "ticker": ticker,
            "target_horizon": horizon,
            "realized_vol_window": VOL_WINDOW,
            "target_purge_days": purge,
            "lgbm_params": params,
            "folds": FOLD_KW,
            "n_base_features": len(fm.feature_names),
            "n_context_cols": None if context is None else int(context.shape[1]),
        },
        notes=f"cluster-lift pre-registered experiment, arm {arm}",
    )
    rmses: list[float] = []
    ics: list[float] = []
    for split in splits:
        if context is None:
            fm_fold = fm
        else:
            rf = build_regime_features(
                context,
                boundary=split.train_end,
                target=fm.y,
                config=RegimeFeatureConfig(target_purge_days=purge),
            )
            if not rf.index.equals(fm.X.index):
                raise RuntimeError("regime features not row-aligned with the shared matrix")
            X = pd.concat([fm.X, rf], axis=1)
            fm_fold = FeatureMatrix(
                X=X,
                y=fm.y,
                target_name=fm.target_name,
                target_horizon=fm.target_horizon,
                feature_names=list(X.columns),
            )
        pred_df, result, importances = train_one_split(fm_fold, split, params=params)
        log_predictions(run_id, pred_df)
        for imp_type, imp_dict in importances.items():
            log_feature_importances(run_id, split.split_id, imp_dict, importance_type=imp_type)
        rmses.append(result.rmse)
        ics.append(result.ic)
        print(
            f"  [{arm}] split {result.split_id:>2d} n={result.n_test:>4d} "
            f"rmse={result.rmse:.5f} ic={result.ic:+.3f}",
            flush=True,
        )
    print(f"  [{arm}] mean rmse={np.mean(rmses):.5f} mean ic={np.nanmean(ics):+.3f} ({run_id})")
    return ArmResult(run_id=run_id, rmses=rmses, ics=ics)


def decide(a: ArmResult, b: ArmResult, rel_bar: float) -> tuple[bool, float, int, int]:
    """Pre-registered rule: relative mean-RMSE bar AND per-split win count."""
    deltas = [rb - ra for ra, rb in zip(a.rmses, b.rmses, strict=True)]
    rel = (b.mean_rmse - a.mean_rmse) / a.mean_rmse
    wins = sum(d < 0 for d in deltas)
    need = math.ceil(WIN_FRAC * len(deltas))
    return (rel <= rel_bar and wins >= need), rel, wins, need


def shared_matrix(
    prices: pd.DataFrame, macro: pd.DataFrame, ticker: str, horizon: int, shared: pd.DatetimeIndex
) -> FeatureMatrix:
    fm = build_feature_matrix(
        prices=prices,
        macro_wide=macro,
        target_ticker=ticker,
        target_kind="realized_vol",
        target_horizon=horizon,
        realized_vol_window=VOL_WINDOW,
    )
    if not shared.isin(fm.X.index).all():
        raise RuntimeError(f"shared rows missing from the {ticker} h={horizon} feature matrix")
    X = fm.X.loc[shared]
    return FeatureMatrix(
        X=X,
        y=fm.y.loc[shared],
        target_name=fm.target_name,
        target_horizon=fm.target_horizon,
        feature_names=list(X.columns),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--primary-only", action="store_true", help="Skip SI=F / h=20 secondaries.")
    args = parser.parse_args()

    print("loading inputs...", flush=True)
    prices = load_prices(column="adj_close")
    macro = load_macro()
    text = load_text_daily()
    topics = load_topic_prevalence_wide()
    context, _ = build_context(
        prices=prices,
        macro_wide=macro,
        text_daily=text,
        topic_prevalence=topics,
        pca_fit_until=None,  # Option C context has no PCA; nothing to fold-fit
        config=ContextConfig(target_metal="gold", include_embeddings=False),
    )
    context = context.dropna()

    # Binding row policy: one shared row set for every arm.
    fm_primary_full = build_feature_matrix(
        prices=prices,
        macro_wide=macro,
        target_ticker="GC=F",
        target_kind="realized_vol",
        target_horizon=5,
        realized_vol_window=VOL_WINDOW,
    )
    shared = fm_primary_full.X.index.intersection(context.index)
    context = context.loc[shared]
    print(f"shared rows: {len(shared)} ({shared.min().date()} -> {shared.max().date()})")

    splits = list(walk_forward_splits(timestamps=shared, **FOLD_KW))
    print(f"folds: {len(splits)} (test {splits[0].val_end.date()} -> {splits[-1].test_end.date()})")

    rows: list[dict] = []

    def record(target: str, arm: str, res: ArmResult) -> None:
        for i, (rmse, ic) in enumerate(zip(res.rmses, res.ics, strict=True)):
            rows.append(
                {
                    "target": target,
                    "arm": arm,
                    "split_id": i,
                    "rmse": rmse,
                    "ic": ic,
                    "run_id": res.run_id,
                }
            )

    # ---- Primary: GC=F h=5 --------------------------------------------------
    print("\n=== primary: GC=F realized_vol h=5 ===")
    fm_p = shared_matrix(prices, macro, "GC=F", 5, shared)
    res_a = run_arm("A", fm_p, splits, "GC=F", 5)
    res_b = run_arm("B", fm_p, splits, "GC=F", 5, context=context)
    record("GC=F_h5", "A", res_a)
    record("GC=F_h5", "B", res_b)

    b_beats_a, rel, wins, need = decide(res_a, res_b, REL_BAR_B_VS_A)
    print(
        f"\nDECISION (pre-registered): B beats A = {b_beats_a} "
        f"(rel ΔRMSE {rel:+.4f} vs bar {REL_BAR_B_VS_A}; wins {wins}/{len(splits)}, need {need})"
    )

    # ---- Attribution ablation (gated) ---------------------------------------
    if b_beats_a:
        print("\n=== B_notext ablation (B beat A) ===")
        ctx_notext = context[[c for c in context.columns if not _is_text_col(c)]]
        res_bn = run_arm("B_notext", fm_p, splits, "GC=F", 5, context=ctx_notext)
        record("GC=F_h5", "B_notext", res_bn)
        gap_total = res_a.mean_rmse - res_b.mean_rmse
        gap_text = res_bn.mean_rmse - res_b.mean_rmse
        if gap_text <= 0:
            attribution = "macro clustering entirely"
        elif gap_text >= 0.5 * gap_total:
            attribution = "text"
        else:
            attribution = "mixed (below the 0.5 text bar -> macro-leaning)"
        print(f"attribution: {attribution} (gap_total={gap_total:.5f}, gap_text={gap_text:.5f})")

    # ---- Secondaries (report, never decide) ----------------------------------
    if not args.primary_only:
        for ticker, horizon in (("SI=F", 5), ("GC=F", 20)):
            print(f"\n=== secondary: {ticker} realized_vol h={horizon} (report-only) ===")
            fm_s = shared_matrix(prices, macro, ticker, horizon, shared)
            res_sa = run_arm("A", fm_s, splits, ticker, horizon)
            res_sb = run_arm("B", fm_s, splits, ticker, horizon, context=context)
            record(f"{ticker}_h{horizon}", "A", res_sa)
            record(f"{ticker}_h{horizon}", "B", res_sb)
            _, rel_s, wins_s, need_s = decide(res_sa, res_sb, REL_BAR_B_VS_A)
            print(
                f"secondary readout: rel ΔRMSE {rel_s:+.4f}, wins {wins_s}/{len(splits)} "
                f"(report-only; primary rules do not apply)"
            )

    out = pd.DataFrame(rows)
    out_path = "results/phase3_cluster_lift_readout.csv"
    out.to_csv(out_path, index=False)
    print(f"\nwrote {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
