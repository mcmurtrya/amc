"""Phase 6 hold-out scoring: models, benchmark suite, lift tables (6.2-6.4).

Hold-out and embargo per results/phase6_validation.md §6.1:
  boundary  2026-01-17  (last date any dev readout saw)
  embargo   boundary - 44 cal days = training rows end 2025-12-04
  scored    hold-out days > boundary with a computable target

Models (target: GC=F realized vol, h=5, w=20, annualized):
  lgbm_full        Phase-1 baseline, untuned `full` feature set, frozen at embargo
  lgbm_regime      + per-fold regime features (lift arm B construction)
  lgbm_sentiment   text features only (lagged tone/count/topics) — plan 6.3
  random_walk      pred = trailing 20d realized vol at t
  uncond_mean      pred = training-window target mean
  garch11          GARCH(1,1) on gold returns, params frozen at embargo,
                   forecast vol over [t+5, t+24]
  var2             VAR(2) on the four metals' trailing rvol, params frozen,
                   24-step-ahead gold rvol forecast (== the target window)

Metrics (6.4): RMSE, RMSE/RMSE_RW, IC, Diebold-Mariano vs lgbm_full
(Newey-West lag 24), moving-block bootstrap 95% CI on RMSE (block 24).
Every model logs predictions to the eval harness.

Usage: uv run python scripts/phase6_holdout.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metals.eval.harness import log_predictions, register_run
from metals.features.assemble import build_feature_matrix
from metals.features.context import ContextConfig, build_context
from metals.features.loaders import load_macro, load_prices
from metals.features.regimes import RegimeFeatureConfig, build_regime_features, purge_days_for
from metals.features.text_daily import load_daily as load_text_daily
from metals.features.topics import load_topic_prevalence_wide
from metals.models.lgbm_vol import DEFAULT_LGBM_PARAMS

BOUNDARY = pd.Timestamp("2026-01-17")
HORIZON, VOL_WINDOW = 5, 20
PURGE = purge_days_for(HORIZON, VOL_WINDOW)  # 44 calendar days
EMBARGO = BOUNDARY - pd.Timedelta(days=PURGE)
VAL_DAYS = 180
PINNED = {"seed": 42, "deterministic": True, "force_row_wise": True, "num_threads": 8}
DM_LAG = 24  # target windows overlap up to h + w - 1 trading days
SEED = 42


def fit_lgbm(x: pd.DataFrame, y: pd.Series, holdout_idx: pd.DatetimeIndex) -> pd.Series:
    import lightgbm as lgb

    train_mask = (x.index <= EMBARGO - pd.Timedelta(days=VAL_DAYS)) & y.notna()
    val_mask = (x.index > EMBARGO - pd.Timedelta(days=VAL_DAYS)) & (x.index <= EMBARGO) & y.notna()
    model = lgb.LGBMRegressor(**{**DEFAULT_LGBM_PARAMS, **PINNED})
    model.fit(
        x.loc[train_mask],
        y.loc[train_mask],
        eval_set=[(x.loc[val_mask], y.loc[val_mask])],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    return pd.Series(model.predict(x.loc[holdout_idx]), index=holdout_idx)


def garch_forecast(ret: pd.Series, holdout_idx: pd.DatetimeIndex) -> pd.Series:
    from arch import arch_model

    r = (ret.dropna() * 100).astype(float)
    am = arch_model(r, vol="GARCH", p=1, q=1, mean="Constant")
    res = am.fit(last_obs=EMBARGO, disp="off")
    fc = res.forecast(horizon=HORIZON + VOL_WINDOW - 1 + 1, start=holdout_idx[0], reindex=False)
    var = fc.variance  # rows: forecast origin; cols h.1 .. h.24
    window = var.iloc[:, HORIZON - 1 : HORIZON + VOL_WINDOW - 1]  # steps 5..24
    pred = np.sqrt(window.mean(axis=1)) * np.sqrt(252) / 100.0
    return pred.reindex(holdout_idx)


def var2_forecast(rvol_panel: pd.DataFrame, holdout_idx: pd.DatetimeIndex) -> pd.Series:
    from statsmodels.tsa.api import VAR

    panel = rvol_panel.dropna()
    fitted = VAR(panel.loc[:EMBARGO].to_numpy()).fit(2)
    steps = HORIZON + VOL_WINDOW - 1  # gold trailing-20 rvol at t+24 == target at t
    out = {}
    arr = panel.to_numpy()
    pos = {d: i for i, d in enumerate(panel.index)}
    for t in holdout_idx:
        if t not in pos or pos[t] < 2:
            out[t] = np.nan
            continue
        i = pos[t]
        fc = fitted.forecast(arr[i - 1 : i + 1], steps=steps)
        out[t] = fc[-1, 0]  # gold column
    return pd.Series(out).reindex(holdout_idx)


def main() -> None:
    prices = load_prices(column="adj_close")
    macro = load_macro()
    fm = build_feature_matrix(
        prices=prices,
        macro_wide=macro,
        target_ticker="GC=F",
        target_kind="realized_vol",
        target_horizon=HORIZON,
        realized_vol_window=VOL_WINDOW,
    )
    y = fm.y
    holdout_idx = fm.X.index[(fm.X.index > BOUNDARY) & y.notna()]
    print(
        f"embargo {EMBARGO.date()} | boundary {BOUNDARY.date()} | "
        f"scorable hold-out days: {len(holdout_idx)} "
        f"({holdout_idx.min().date()} -> {holdout_idx.max().date()})"
    )

    # Text/context inputs (Option C — reused for regime + sentiment models).
    context, _ = build_context(
        prices=prices,
        macro_wide=macro,
        text_daily=load_text_daily(),
        topic_prevalence=load_topic_prevalence_wide(),
        pca_fit_until=None,
        config=ContextConfig(target_metal="gold", include_embeddings=False),
    )

    preds: dict[str, pd.Series] = {}
    preds["lgbm_full"] = fit_lgbm(fm.X, y, holdout_idx)

    ctx_rows = context.dropna()
    rf = build_regime_features(
        ctx_rows,
        boundary=EMBARGO,
        target=y.reindex(ctx_rows.index),
        config=RegimeFeatureConfig(target_purge_days=PURGE),
    )
    xb = pd.concat([fm.X, rf.reindex(fm.X.index)], axis=1)
    preds["lgbm_regime"] = fit_lgbm(xb, y, holdout_idx)

    text_cols = [
        c
        for c in context.columns
        if c == "n_articles" or c.startswith("mean_tone_") or c.startswith("topic_")
    ]
    preds["lgbm_sentiment"] = fit_lgbm(context[text_cols].reindex(fm.X.index), y, holdout_idx)

    ret1d = np.log(prices["GC=F"].astype(float)).diff()
    trailing = ret1d.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std() * np.sqrt(252)
    preds["random_walk"] = trailing.reindex(holdout_idx)
    preds["uncond_mean"] = pd.Series(float(y[(y.index <= EMBARGO)].mean()), index=holdout_idx)
    preds["garch11"] = garch_forecast(ret1d, holdout_idx)

    rvol_panel = pd.DataFrame(
        {
            t: np.log(prices[t].astype(float)).diff().rolling(VOL_WINDOW).std() * np.sqrt(252)
            for t in ("GC=F", "SI=F", "PL=F", "PA=F")
        }
    )
    preds["var2"] = var2_forecast(rvol_panel, holdout_idx)

    # Common scorable days across all models.
    pred_df = pd.DataFrame(preds).loc[holdout_idx].dropna()
    actual = y.reindex(pred_df.index)
    n = len(pred_df)
    print(f"common scorable days: {n}")

    rng = np.random.default_rng(SEED)
    base_err = pred_df["lgbm_full"] - actual
    rw_rmse = float(np.sqrt(np.mean((pred_df["random_walk"] - actual) ** 2)))

    def dm_tstat(err: pd.Series) -> float:
        d = (err**2 - base_err**2).to_numpy()
        dbar = d.mean()
        # Newey-West long-run variance of mean(d)
        gamma0 = np.mean((d - dbar) ** 2)
        s = gamma0
        for lag in range(1, DM_LAG + 1):
            w = 1 - lag / (DM_LAG + 1)
            cov = np.mean((d[lag:] - dbar) * (d[:-lag] - dbar))
            s += 2 * w * cov
        return float(dbar / np.sqrt(s / len(d))) if s > 0 else float("nan")

    def block_boot_rmse_ci(err: np.ndarray, n_boot: int = 2000, block: int = 24):
        n_obs = len(err)
        n_blocks = int(np.ceil(n_obs / block))
        rmses = np.empty(n_boot)
        for b in range(n_boot):
            starts = rng.integers(0, max(n_obs - block, 1), size=n_blocks)
            sample = np.concatenate([err[s : s + block] for s in starts])[:n_obs]
            rmses[b] = np.sqrt(np.mean(sample**2))
        return float(np.quantile(rmses, 0.025)), float(np.quantile(rmses, 0.975))

    rows = []
    for name, p in pred_df.items():
        err = (p - actual).to_numpy()
        rmse = float(np.sqrt(np.mean(err**2)))
        ic = (
            float(np.corrcoef(p, actual)[0, 1])
            if np.std(p) > 0 and np.std(actual) > 0
            else float("nan")
        )
        lo, hi = block_boot_rmse_ci(err)
        rows.append(
            {
                "model": name,
                "rmse": rmse,
                "rmse_vs_rw": rmse / rw_rmse,
                "ic": ic,
                "dm_t_vs_lgbm": float("nan") if name == "lgbm_full" else dm_tstat(p - actual),
                "rmse_ci_low": lo,
                "rmse_ci_high": hi,
                "n": n,
            }
        )
        run_id = register_run(
            name=f"phase6_holdout_{name}",
            model_type="holdout_eval",
            target_type="realized_vol",
            config={
                "boundary": str(BOUNDARY.date()),
                "embargo": str(EMBARGO.date()),
                "model": name,
                "horizon": HORIZON,
                "vol_window": VOL_WINDOW,
                "n_scored": n,
            },
            notes="Phase 6.2/6.3 hold-out scoring",
        )
        log_predictions(
            run_id,
            pd.DataFrame(
                {
                    "timestamp_utc": pred_df.index,
                    "ticker": "GC=F",
                    "horizon": HORIZON,
                    "prediction": p.to_numpy(),
                    "actual": actual.to_numpy(),
                }
            ),
        )

    table = pd.DataFrame(rows).sort_values("rmse")
    table.to_csv("results/phase6_holdout_metrics.csv", index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("\nwrote results/phase6_holdout_metrics.csv")


if __name__ == "__main__":
    main()
