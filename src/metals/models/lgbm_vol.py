"""LightGBM baseline for forecasting realized volatility.

This is the Phase 1 baseline that every later model must beat. It is
intentionally simple: a single LightGBM regressor per walk-forward split,
trained on the assembled feature matrix and evaluated on the next two
6-month windows (val + test).

Run as:
    uv run python -m metals.models.lgbm_vol \
        --ticker GC=F --target realized_vol --horizon 5

Predictions are logged to the evaluation harness under a unique run name
combining the ticker, target, and horizon.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from metals.eval.cv import Split, walk_forward_splits
from metals.eval.harness import (
    log_feature_importances,
    log_predictions,
    register_run,
)
from metals.features.assemble import FeatureMatrix, build_feature_matrix
from metals.features.loaders import load_macro, load_prices

DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 600,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "min_data_in_leaf": 50,
    "verbose": -1,
    "n_jobs": -1,
}

# Substrings identifying the price-history "returns and volatility" feature
# block (log returns, realized vol, skew/kurt, max drawdown) across every
# ticker in the panel. The Phase 1 diagnostic
# (results/phase1_negative_ic_diagnosis.md) found this 108-feature block is
# net-negative for IC on every metal, so the lean feature sets prune it.
RETURNS_VOL_SUBSTRINGS: tuple[str, ...] = (
    "_ret_",
    "_rvol_",
    "_skew_",
    "_kurt_",
    "_maxdd_",
)

FEATURE_SETS = ("full", "lean", "lean_own")


def feature_columns(all_cols: list[str], feature_set: str, ticker: str) -> list[str]:
    """Select feature columns for a named feature set.

    - ``full``: every feature (the original production baseline).
    - ``lean``: drop the entire returns-and-vol block, keeping only spreads +
      macro. Matches the diagnostic's ``no_returns_and_vol`` configuration.
    - ``lean_own``: drop other tickers' returns-and-vol but keep the *target*
      metal's own returns/vol (vol clustering is a legitimate own-series signal).
    """
    rv = {c for c in all_cols if any(s in c for s in RETURNS_VOL_SUBSTRINGS)}
    if feature_set == "full":
        return list(all_cols)
    if feature_set == "lean":
        drop = rv
    elif feature_set == "lean_own":
        own = {c for c in rv if c.startswith(f"{ticker}_")}
        drop = rv - own
    else:
        raise ValueError(f"Unknown feature_set {feature_set!r}; choose from {FEATURE_SETS}")
    return [c for c in all_cols if c not in drop]


@dataclass
class SplitResult:
    split_id: int
    rmse: float
    ic: float
    n_test: int


def _drop_unusable_rows(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Drop rows where target is NaN; LightGBM still tolerates NaNs in features."""
    mask = y.notna()
    return X.loc[mask], y.loc[mask]


def train_one_split(
    fm: FeatureMatrix,
    split: Split,
    params: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, SplitResult, dict[str, dict[str, float]]]:
    """Fit on split.train_idx, early-stop on val, predict on test.

    Returns
    -------
    predictions : DataFrame
        Per-row predictions and actuals, ready for the eval harness.
    result : SplitResult
        Summary metrics for the split.
    importances : dict[importance_type, dict[feature_name, value]]
        Booster importances by type. Keys: ``"gain"`` (sum of information gain)
        and ``"split"`` (number of times each feature is used to split).
    """
    import lightgbm as lgb

    params = {**DEFAULT_LGBM_PARAMS, **(params or {})}

    X_train, y_train = _drop_unusable_rows(fm.X.iloc[split.train_idx], fm.y.iloc[split.train_idx])
    X_val, y_val = _drop_unusable_rows(fm.X.iloc[split.val_idx], fm.y.iloc[split.val_idx])
    X_test, y_test = _drop_unusable_rows(fm.X.iloc[split.test_idx], fm.y.iloc[split.test_idx])

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    preds = model.predict(X_test)
    rmse = float(np.sqrt(np.mean((preds - y_test.values) ** 2)))
    if len(y_test) >= 2 and np.std(preds) > 0 and np.std(y_test) > 0:
        ic = float(np.corrcoef(preds, y_test.values)[0, 1])
    else:
        ic = float("nan")

    result = SplitResult(split_id=split.split_id, rmse=rmse, ic=ic, n_test=len(y_test))
    pred_df = pd.DataFrame(
        {
            "timestamp_utc": X_test.index,
            "ticker": fm.target_name.split("_")[0],  # original ticker prefix
            "horizon": fm.target_horizon,
            "prediction": preds,
            "actual": y_test.values,
        }
    )

    # Capture booster importances. feature_importances_ defaults to "split"
    # count; booster_.feature_importance("gain") is the more meaningful
    # information-gain measure. Map both onto our feature names.
    feat_names = list(X_train.columns)
    gain = model.booster_.feature_importance(importance_type="gain")
    split_cnt = model.booster_.feature_importance(importance_type="split")
    importances = {
        "gain": {f: float(v) for f, v in zip(feat_names, gain, strict=False)},
        "split": {f: float(v) for f, v in zip(feat_names, split_cnt, strict=False)},
    }
    return pred_df, result, importances


def run(
    ticker: str,
    target_kind: str = "realized_vol",
    target_horizon: int = 5,
    realized_vol_window: int = 20,
    train_start: str = "2010-01-01",
    val_days: int = 180,
    test_days: int = 180,
    step_days: int = 180,
    min_train_days: int = 5 * 365,
    feature_set: str = "full",
    notes: str | None = None,
) -> str:
    """End-to-end training run. Returns the eval-harness run_id."""
    prices = load_prices(column="adj_close")
    macro_wide = load_macro()

    if prices.empty:
        raise RuntimeError("No prices in DuckDB. Run `uv run python -m metals.data.prices` first.")
    if macro_wide.empty:
        raise RuntimeError(
            "No macro series in DuckDB. Run `uv run python -m metals.data.fred` first."
        )

    fm = build_feature_matrix(
        prices=prices,
        macro_wide=macro_wide,
        target_ticker=ticker,
        target_kind=target_kind,
        target_horizon=target_horizon,
        realized_vol_window=realized_vol_window,
    )

    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set {feature_set!r}; choose from {FEATURE_SETS}")
    if feature_set != "full":
        cols = feature_columns(list(fm.feature_names), feature_set, ticker)
        fm = FeatureMatrix(
            X=fm.X[cols],
            y=fm.y,
            target_name=fm.target_name,
            target_horizon=fm.target_horizon,
            feature_names=cols,
        )

    run_name = (
        f"lgbm_{ticker}_{target_kind}_h{target_horizon}_{feature_set}_{datetime.now():%Y%m%d_%H%M}"
    )
    run_id = register_run(
        name=run_name,
        model_type="lgbm_vol",
        target_type=target_kind,
        config={
            "ticker": ticker,
            "target_kind": target_kind,
            "target_horizon": target_horizon,
            "realized_vol_window": realized_vol_window,
            "train_start": train_start,
            "val_days": val_days,
            "test_days": test_days,
            "step_days": step_days,
            "min_train_days": min_train_days,
            "feature_set": feature_set,
            "lgbm_params": DEFAULT_LGBM_PARAMS,
            "n_features": len(fm.feature_names),
        },
        notes=notes,
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
        raise RuntimeError("No walk-forward splits produced — check date range and min_train_days.")

    summaries: list[SplitResult] = []
    for split in splits:
        pred_df, result, importances = train_one_split(fm, split)
        log_predictions(run_id, pred_df)
        for imp_type, imp_dict in importances.items():
            log_feature_importances(run_id, split.split_id, imp_dict, importance_type=imp_type)
        summaries.append(result)
        print(
            f"split {result.split_id:>2d}  "
            f"n={result.n_test:>4d}  rmse={result.rmse:.4f}  ic={result.ic:+.3f}"
        )

    mean_rmse = float(np.mean([s.rmse for s in summaries]))
    mean_ic = float(np.nanmean([s.ic for s in summaries]))
    print(f"\nMean across splits: rmse={mean_rmse:.4f}  ic={mean_ic:+.3f}")
    print(f"Run id: {run_id}")
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="LightGBM vol-forecasting baseline.")
    parser.add_argument("--ticker", default="GC=F")
    parser.add_argument("--target", default="realized_vol", choices=["realized_vol", "return"])
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--train-start", default="2010-01-01")
    parser.add_argument(
        "--feature-set",
        default="full",
        choices=list(FEATURE_SETS),
        help="full | lean (drop returns+vol block) | "
        "lean_own (keep only the target metal's own returns/vol)",
    )
    parser.add_argument("--notes", default=None)
    args = parser.parse_args()

    run(
        ticker=args.ticker,
        target_kind=args.target,
        target_horizon=args.horizon,
        realized_vol_window=args.vol_window,
        train_start=args.train_start,
        feature_set=args.feature_set,
        notes=args.notes,
    )


if __name__ == "__main__":
    main()
