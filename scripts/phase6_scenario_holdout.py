"""Phase 6.5: scenario sign-validation on the hold-out.

For each macro scenario (GPR spike, DXY up/down — the FOMC scenarios cannot
fire: Bauer-Swanson ends 2023-12), thresholds are recomputed on PRE-hold-out
data only ([2010-01-01, 2026-01-17]) and applied forward to hold-out days —
fixing the §6.1 threshold-contamination finding. The comparison baseline
(the "training-period ATE") is re-estimated with the same end bound and the
same pre-hold-out thresholds.

Hold-out statistic: naive treated-minus-control difference in mean h=5
forward cumulative returns among hold-out days (a DoubleML fit on ~85 days
would be noise). Reported per the plan: n fires, sign agreement with the
training ATE, magnitude ratio. With months of OOS data the question is
"do the signs hold up?", not significance.

Usage: uv run python scripts/phase6_scenario_holdout.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metals.eval.harness import register_run
from metals.features.loaders import load_macro, load_prices
from metals.features.returns import compute_log_returns
from metals.features.scenarios import build_confounders
from metals.models.causal import DEFAULT_METALS, estimate_ate
from metals.models.lp import cumulative_log_returns

BOUNDARY = pd.Timestamp("2026-01-17")
TRAIN_START = pd.Timestamp("2010-01-01")
HORIZON = 5

# (id, macro series, transform periods, rule, param)
SCENARIOS = (
    ("gpr_spike", "GPR_DAILY", "diff", 1, "pct_high", 0.95),
    ("dxy_up_shock", "DTWEXBGS", "pct_change", 5, "sigma_high", 2.0),
    ("dxy_down_shock", "DTWEXBGS", "pct_change", 5, "sigma_low", 2.0),
)


def driver_series(macro_aligned: pd.DataFrame, series: str, transform: str, periods: int):
    s = macro_aligned[series].astype(float)
    return s.diff(periods) if transform == "diff" else s.pct_change(periods)


def treatment_from_pre_holdout_threshold(
    driver: pd.Series, rule: str, param: float
) -> tuple[pd.Series, float]:
    """Binary treatment over the FULL index with thresholds fit pre-boundary."""
    fit = driver.loc[TRAIN_START:BOUNDARY].dropna()
    if rule == "pct_high":
        thr = float(fit.quantile(param))
        t = (driver > thr).astype(float)
    elif rule == "sigma_high":
        thr = param * float(fit.std())
        t = (driver > thr).astype(float)
    elif rule == "sigma_low":
        thr = -param * float(fit.std())
        t = (driver < thr).astype(float)
    else:
        raise ValueError(rule)
    return t.where(driver.notna()), thr


def main() -> None:
    prices = load_prices(tickers=list(DEFAULT_METALS))
    macro = load_macro()
    returns_1d = compute_log_returns(prices, (1,))
    idx = returns_1d.index
    macro_aligned = macro.reindex(idx).ffill()

    rows = []
    for sid, series, transform, periods, rule, param in SCENARIOS:
        drv = driver_series(macro_aligned, series, transform, periods)
        treatment, thr = treatment_from_pre_holdout_threshold(drv, rule, param)
        for ticker in DEFAULT_METALS:
            outcome = cumulative_log_returns(returns_1d[f"{ticker}_ret_1d"], HORIZON)
            confounders = build_confounders(
                ticker,
                idx,
                prices=prices,
                macro=macro,
                exclude=("dxy_chg_5d",) if series == "DTWEXBGS" else ("gpr_chg_5d",),
            )
            pre = slice(TRAIN_START, BOUNDARY)
            train = estimate_ate(outcome.loc[pre], treatment.loc[pre], confounders.loc[pre])

            ho_mask = (idx > BOUNDARY) & outcome.notna() & treatment.notna()
            y_ho, t_ho = outcome[ho_mask], treatment[ho_mask]
            n_fire = int(t_ho.sum())
            if n_fire > 0 and n_fire < len(t_ho):
                diff = float(y_ho[t_ho == 1].mean() - y_ho[t_ho == 0].mean())
            else:
                diff = float("nan")
            sign_ok = (
                bool(np.sign(diff) == np.sign(train["ate"]))
                if np.isfinite(diff) and np.isfinite(train["ate"])
                else None
            )
            ratio = (
                diff / train["ate"]
                if np.isfinite(diff) and np.isfinite(train["ate"]) and train["ate"] != 0
                else float("nan")
            )
            rows.append(
                {
                    "scenario": sid,
                    "metal": ticker,
                    "threshold_pre_holdout": thr,
                    "train_ate": train["ate"],
                    "train_n_treated": train["n_treated"],
                    "holdout_n_fires": n_fire,
                    "holdout_n_days": int(len(t_ho)),
                    "holdout_diff": diff,
                    "sign_agree": sign_ok,
                    "magnitude_ratio": ratio,
                }
            )
            print(
                f"{sid:>15s} {ticker:>5s}  fires={n_fire:>3d}  "
                f"train_ate={train['ate']:+.4f}  holdout_diff={diff:+.4f}  "
                f"sign_agree={sign_ok}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    df.to_csv("results/phase6_scenario_holdout.csv", index=False)
    register_run(
        name="phase6_scenario_holdout",
        model_type="holdout_eval",
        target_type="cumret",
        config={
            "boundary": str(BOUNDARY.date()),
            "horizon": HORIZON,
            "scenarios": [s[0] for s in SCENARIOS],
            "note": "FOMC scenarios cannot fire (surprise data ends 2023-12)",
        },
        notes="Phase 6.5: scenario sign-validation, pre-holdout thresholds",
    )
    print("\nwrote results/phase6_scenario_holdout.csv")


if __name__ == "__main__":
    main()
