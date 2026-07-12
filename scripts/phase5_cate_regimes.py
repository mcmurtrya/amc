"""CATE of the hawkish-FOMC shock conditioned on Phase 3 regime labels (plan 5.4).

Asks: does the anchor causal finding (hawkish FOMC -> metals down, DML ATE
~-1.4% on gold at h=5) vary across the Phase 3 scenario regimes?

Design notes (read before trusting output):
- Effect modifiers are one-hot regimes from ``cluster_assignments``
  (`phase3_optC_tone_lag1_2024split`), **lagged one trading day**: the day-t
  assignment is built from day-t context (which includes day-t closes), and
  FOMC announcements land intraday — the same-day regime is partially
  post-treatment. The t-1 regime is strictly pre-treatment.
- Treatment thresholds are computed with the same window_start (2010) as the
  ATE run so the hawkish event set is identical; rows are then restricted to
  regime coverage (2015-02 onward), which shrinks the treated count — read
  per-regime patterns as sign-stability evidence, not point estimates.
- The regime clustering was trained through 2024 on the full context sample;
  this is a descriptive/in-sample conditioning variable, not a fold-local
  feature. Fine for effect-modification questions; not for forecasting.

Usage: uv run python scripts/phase5_cate_regimes.py
"""

from __future__ import annotations

import pandas as pd

from metals.data.db import connection
from metals.eval.harness import register_run
from metals.features.loaders import load_fomc_surprises, load_macro, load_prices
from metals.features.returns import compute_log_returns
from metals.features.scenarios import (
    build_confounders,
    build_treatment,
    confounder_exclusions,
    load_scenario_config,
)
from metals.models.causal import estimate_cate
from metals.models.lp import cumulative_log_returns

MODEL_VERSION = "phase3_optC_tone_lag1_2024split"
SCENARIO_ID = "hawkish_fomc"
HORIZON = 5
WINDOW_START = "2010-01-01"  # threshold window — must match the ATE run
TICKERS = ("GC=F", "SI=F")


def load_regimes() -> tuple[pd.Series, dict[int, str]]:
    with connection(read_only=True) as conn:
        asg = conn.execute(
            "SELECT timestamp_utc, cluster_id FROM cluster_assignments "
            "WHERE model_version = ? ORDER BY timestamp_utc",
            [MODEL_VERSION],
        ).fetchdf()
        lab = conn.execute(
            "SELECT cluster_id, label FROM cluster_centroids WHERE model_version = ?",
            [MODEL_VERSION],
        ).fetchdf()
    names = {int(r.cluster_id): (r.label or "noise") for r in lab.itertuples()}
    names.setdefault(-1, "noise")
    s = asg.set_index(pd.to_datetime(asg["timestamp_utc"]))["cluster_id"].astype(int)
    return s, names


def main() -> None:
    cfg = load_scenario_config()
    spec = next(s for s in cfg.scenarios if s.id == SCENARIO_ID)

    prices = load_prices(tickers=list(TICKERS))
    macro = load_macro()
    fomc = load_fomc_surprises()
    regimes, names = load_regimes()

    returns_1d = compute_log_returns(prices, (1,))
    trading_idx = returns_1d.index

    treatment = build_treatment(
        spec, trading_idx, fomc=fomc, macro=macro, window_start=WINDOW_START
    )
    # Strictly pre-treatment regime: previous trading day's assignment.
    regime_lag = regimes.reindex(trading_idx).shift(1)
    mods = pd.get_dummies(regime_lag.dropna().astype(int), prefix="regime").astype(float)
    mods.columns = [
        f"regime_{names.get(int(c.split('_')[1]), c)}".replace("-", "_") for c in mods.columns
    ]

    excl = confounder_exclusions(spec)
    print(f"regime coverage: {mods.index.min().date()} -> {mods.index.max().date()}")

    rows = []
    for ticker in TICKERS:
        own_ret = returns_1d[f"{ticker}_ret_1d"]
        outcome = cumulative_log_returns(own_ret, HORIZON)
        confounders = build_confounders(
            ticker, trading_idx, prices=prices, macro=macro, exclude=excl
        )
        idx = mods.index
        res = estimate_cate(
            outcome.reindex(idx),
            treatment.reindex(idx),
            confounders.reindex(idx),
            mods,
            seed=42,
        )
        cate = res["cate"]
        d = treatment.reindex(cate.index)
        reg = regime_lag.reindex(cate.index).astype(int).map(names)
        print(
            f"\n=== {ticker} h={HORIZON}: n={res['n_obs']}, treated={int(d.sum())}, "
            f"CATE mean={res['cate_mean']:+.4f} (ATE anchor ~-0.014 Au / -0.030 Ag), "
            f"std={res['cate_std']:.4f} ==="
        )
        summary = (
            pd.DataFrame({"cate": cate, "regime": reg, "treated": d})
            .groupby("regime")
            .agg(
                n=("cate", "size"),
                n_treated=("treated", "sum"),
                cate_mean=("cate", "mean"),
                cate_std=("cate", "std"),
                frac_negative=("cate", lambda s: float((s < 0).mean())),
            )
            .sort_values("cate_mean")
        )
        print(summary.to_string(float_format=lambda x: f"{x:+.4f}"))
        summary["ticker"] = ticker
        rows.append(summary.reset_index())

    out = pd.concat(rows, ignore_index=True)
    out_path = "results/phase5_cate_regimes.csv"
    out.to_csv(out_path, index=False)
    run_id = register_run(
        name=f"cate_regimes_{SCENARIO_ID}_h{HORIZON}",
        model_type="causal_cate",
        target_type="cumret",
        config={
            "method": "CausalForestDML",
            "scenario": SCENARIO_ID,
            "horizon": HORIZON,
            "model_version": MODEL_VERSION,
            "regime_lag_days": 1,
            "window_start": WINDOW_START,
            "tickers": list(TICKERS),
            "seed": 42,
        },
        notes="Plan 5.4: hawkish-FOMC CATE conditioned on lagged Phase 3 regime labels",
    )
    print(f"\nwrote {out_path}; harness run {run_id}")


if __name__ == "__main__":
    main()
