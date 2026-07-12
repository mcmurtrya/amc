"""Subsample stability of the scenario ATEs (plan 5.8).

Re-estimates every available scenario's h=5 ATE on three subsamples:
2010-2014, 2015-2019, 2020-2026 (the plan's 2007 left edge is data-
constrained to the 2010 modelling window). Treatment indicators are built
ONCE with the full-window thresholds (window_start=2010) so every subsample
uses the identical event set as the headline ATE run — this isolates
*effect* instability from *definition* drift; re-thresholding per subsample
would conflate the two.

Stability metric per (scenario, metal): the count of subsamples whose ATE
sign matches the full-window ATE sign (NaN subsamples don't count). No
placebos here — sign stability across windows is the 5.8 readout.

Usage: uv run python scripts/phase5_subsample_stability.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metals.eval.harness import register_run
from metals.features.loaders import load_fomc_surprises, load_macro, load_prices
from metals.features.returns import compute_log_returns
from metals.features.scenarios import (
    build_confounders,
    build_treatment,
    confounder_exclusions,
    load_scenario_config,
)
from metals.models.causal import DEFAULT_METALS, estimate_ate
from metals.models.lp import cumulative_log_returns

WINDOW_START = "2010-01-01"  # threshold window — matches the headline ATE run
HORIZON = 5
# Relaxed vs the production default (20): subsample FOMC cells have 10-13
# events, matching Phase 2's subsample LP precedent. Cells with n_treated < 20
# are flagged small_n and read as sign evidence only.
MIN_TREATED = 8
SUBSAMPLES: tuple[tuple[str, str, str], ...] = (
    ("2010_2014", "2010-01-01", "2014-12-31"),
    ("2015_2019", "2015-01-01", "2019-12-31"),
    ("2020_2026", "2020-01-01", "2026-12-31"),
)


def main() -> None:
    cfg = load_scenario_config()
    specs = [s for s in cfg.scenarios if s.available]

    prices = load_prices(tickers=list(DEFAULT_METALS))
    macro = load_macro()
    fomc = load_fomc_surprises()
    returns_1d = compute_log_returns(prices, (1,))
    trading_idx = returns_1d.index

    rows: list[dict] = []
    for spec in specs:
        treatment = build_treatment(
            spec, trading_idx, fomc=fomc, macro=macro, window_start=WINDOW_START
        )
        excl = confounder_exclusions(spec)
        for ticker in DEFAULT_METALS:
            own_ret = returns_1d[f"{ticker}_ret_1d"]
            outcome = cumulative_log_returns(own_ret, HORIZON)
            confounders = build_confounders(
                ticker, trading_idx, prices=prices, macro=macro, exclude=excl
            )
            windows = (("full", WINDOW_START, "2026-12-31"), *SUBSAMPLES)
            for name, start, end in windows:
                sl = slice(pd.Timestamp(start), pd.Timestamp(end))
                res = estimate_ate(
                    outcome.loc[sl],
                    treatment.loc[sl],
                    confounders.loc[sl],
                    min_treated=MIN_TREATED,
                )
                rows.append(
                    {
                        "scenario": spec.id,
                        "metal": ticker,
                        "window": name,
                        "ate": res["ate"],
                        "se": res["se"],
                        "n_treated": res["n_treated"],
                        "n_control": res["n_control"],
                        "small_n": bool(res["n_treated"] < 20),
                    }
                )
                print(
                    f"{spec.id:>15s} {ticker:>5s} {name:>9s}  "
                    f"ate={res['ate']:+.4f}  n_treated={res['n_treated']}",
                    flush=True,
                )

    df = pd.DataFrame(rows)

    # Stability score: subsample sign agreement with the full-window sign.
    def score(group: pd.DataFrame) -> pd.Series:
        full = group.loc[group["window"] == "full", "ate"]
        full_sign = np.sign(full.iloc[0]) if len(full) and np.isfinite(full.iloc[0]) else np.nan
        subs = group[group["window"] != "full"]
        valid = subs["ate"].apply(np.isfinite)
        agree = int((np.sign(subs.loc[valid, "ate"]) == full_sign).sum())
        return pd.Series(
            {
                "full_ate": full.iloc[0],
                "n_valid_subsamples": int(valid.sum()),
                "n_sign_agree": agree,
            }
        )

    stability = df.groupby(["scenario", "metal"]).apply(score, include_groups=False).reset_index()
    print("\n=== stability (h=5): sign agreement with full-window ATE ===")
    print(stability.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    df.to_csv("results/phase5_subsample_ates.csv", index=False)
    stability.to_csv("results/phase5_subsample_stability.csv", index=False)
    run_id = register_run(
        name="subsample_stability_h5",
        model_type="causal",
        target_type="cumret",
        config={
            "method": "DoubleML-IRM per subsample, fixed full-window treatment definitions",
            "horizon": HORIZON,
            "window_start": WINDOW_START,
            "subsamples": [list(s) for s in SUBSAMPLES],
            "scenarios": [s.id for s in specs],
        },
        notes="Plan 5.8: subsample sign-stability of scenario ATEs",
    )
    print(f"\nwrote results/phase5_subsample_{{ates,stability}}.csv; run {run_id}")


if __name__ == "__main__":
    main()
