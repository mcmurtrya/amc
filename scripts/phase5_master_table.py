"""Assemble the master scenario table (plan 5.9) — the phase's central output.

Consolidates, per scenario:
  - DoubleML ATEs + CIs for every metal x horizon (from double_ml_ates.parquet)
  - gold h=5 placebo p-value
  - cross-metal consistency score (plan 5.7, formalized here): share of the
    four metals whose h=5 ATE carries the modal sign
  - subsample stability score (plan 5.8): mean over metals of
    (sign-agreeing subsamples / valid subsamples), from the 5.8 run
  - triangulation agreement score: share of applicable estimators
    (LP / DML / SVAR) agreeing on the gold h=5 direction — per-method signs
    hand-coded below from the documented readouts, with sources
  - a hand-written economic interpretation

Deterministic assembly from committed inputs — not a model run, so it is not
harness-registered; provenance is this script + the input files' run ids.

Usage: uv run python scripts/phase5_master_table.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METAL_SHORT = {"GC=F": "au", "SI=F": "ag", "PL=F": "pt", "PA=F": "pd"}
HORIZONS = (1, 5, 20)

# Per-method gold h=5 direction, hand-coded from the documented readouts.
# LP: results/phase2_scenarios.md tables. DML: double_ml_ates.parquet.
# SVAR: results/phase5_svar_cate_readout.md §1 — event scenarios map to
# identified shocks (hawkish/dovish -> real-yield +/-; dxy up/down -> USD
# shock +/-; gpr -> risk-aversion). "0" = method reads the effect as null.
TRIANGULATION_SIGNS: dict[str, dict[str, str]] = {
    "hawkish_fomc": {"lp": "-", "dml": "-", "svar": "-"},
    "dovish_fomc": {"lp": "+", "dml": "+", "svar": "+"},  # all weak; sign-level only
    "gpr_spike": {"lp": "0", "dml": "-", "svar": "+"},  # methods disagree
    "dxy_up_shock": {"lp": "0", "dml": "+", "svar": "-"},  # event null vs shock -
    "dxy_down_shock": {"lp": "-", "dml": "0", "svar": "+"},  # event inversion vs shock +
}

INTERPRETATION: dict[str, str] = {
    "hawkish_fomc": (
        "Anchor finding, confirmed by LP, DML, and SVAR and sign-stable across "
        "eras: hawkish policy surprises depress all metals (Au -1.4% at h=5), "
        "hardest in rate-hike-expectation regimes; magnitude has decayed since "
        "the QE era; palladium unstable post-2020."
    ),
    "dovish_fomc": (
        "Weak positive mirror of the hawkish effect; never significant. The "
        "hawkish/dovish asymmetry itself is the finding (tail-risk-hedge "
        "reading), consistent across LP and DML."
    ),
    "gpr_spike": (
        "Null/fragile: LP null, DML borderline-negative, SVAR risk-aversion "
        "shock positive. The GPR index measures news intensity, not "
        "flight-to-safety; the safe-haven channel shows up only in the SVAR."
    ),
    "dxy_up_shock": (
        "Event study null (small unstable positives); the SVAR's pure USD "
        "shock is canonically negative for gold. Direction-consistent across "
        "subsamples but never significant."
    ),
    "dxy_down_shock": (
        "The event definition is contaminated: gold is textbook-positive in "
        "2020-26 while the apparent inversion concentrates in PGMs "
        "(COVID-era liquidation/industrial episodes). USD-gold channel fine "
        "per the SVAR."
    ),
}


def triangulation_score(signs: dict[str, str]) -> float:
    votes = [s for s in signs.values()]
    best = max(votes.count(s) for s in set(votes))
    return best / len(votes)


def main() -> None:
    ate = pd.read_parquet("data/processed/double_ml_ates.parquet")
    stab = pd.read_csv("results/phase5_subsample_stability.csv")

    rows = []
    for scenario, g in ate.groupby("scenario"):
        row: dict[str, object] = {
            "scenario_name": scenario,
            "definition_type": "event",
            "definition_yaml_id": scenario,
        }
        for _, r in g.iterrows():
            m, h = METAL_SHORT[r["metal"]], int(r["horizon"])
            row[f"ate_{m}_h{h}"] = r["ate"]
            row[f"ci_{m}_h{h}_low"] = r["ci_low"]
            row[f"ci_{m}_h{h}_high"] = r["ci_high"]
        gold5 = g[(g["metal"] == "GC=F") & (g["horizon"] == 5)].iloc[0]
        row["placebo_pvalue"] = gold5["placebo_pvalue"]
        row["n_treated"] = int(gold5["n_treated"])
        row["n_control"] = int(gold5["n_control"])

        h5 = g[g["horizon"] == 5]
        signs = np.sign(h5["ate"].to_numpy())
        modal = max((signs == 1).sum(), (signs == -1).sum())
        row["cross_metal_consistency_score"] = float(modal / len(signs))

        s = stab[stab["scenario"] == scenario]
        valid = s[s["n_valid_subsamples"] > 0]
        row["subsample_stability_score"] = float(
            (valid["n_sign_agree"] / valid["n_valid_subsamples"]).mean()
        )
        row["triangulation_agreement_score"] = triangulation_score(TRIANGULATION_SIGNS[scenario])
        row["triangulation_signs"] = ",".join(
            f"{k}:{v}" for k, v in TRIANGULATION_SIGNS[scenario].items()
        )
        row["economic_interpretation"] = INTERPRETATION[scenario]
        rows.append(row)

    master = pd.DataFrame(rows).sort_values(
        ["triangulation_agreement_score", "subsample_stability_score"],
        ascending=False,
    )
    master.to_parquet("data/processed/scenario_master.parquet", index=False)
    master.to_csv("results/phase5_scenario_master.csv", index=False)

    view_cols = [
        "scenario_name",
        "ate_au_h5",
        "placebo_pvalue",
        "cross_metal_consistency_score",
        "subsample_stability_score",
        "triangulation_agreement_score",
        "n_treated",
    ]
    print(master[view_cols].to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print(
        f"\nwrote data/processed/scenario_master.parquet + "
        f"results/phase5_scenario_master.csv ({len(master)} scenarios, "
        f"{master.shape[1]} columns)"
    )


if __name__ == "__main__":
    main()
