"""Double/debiased ML treatment-effect estimation for Phase 5 (steps 5.2-5.4).

For each scenario (a binary treatment ``T_t`` from ``metals.features.scenarios``)
we estimate the average treatment effect on the cumulative forward log return
``Y_{t+h}`` of a metal, controlling for a confounder set ``X_t``:

    ATE_h = E[ Y_{t+h}(T=1) - Y_{t+h}(T=0) ]

via DoubleML's *interactive regression model* (IRM): a flexible outcome
regression ``g(X) = E[Y | X, T]`` and a propensity score ``m(X) = P(T=1 | X)``,
both LightGBM, cross-fitted over K folds and combined in the doubly-robust
(efficient-influence-function) score. This is Neyman-orthogonal, so first-stage
ML bias does not contaminate the ATE, and the standard error is valid.

Three pieces:

* ``estimate_ate``     — one debiased ATE with a 95% CI (the core).
* ``placebo_pvalue``   — re-estimate with the treatment shifted by random
  offsets (plan 5.3); a real effect should sit in the tail.
* ``estimate_cate``    — heterogeneity via ``econml.CausalForestDML`` (plan 5.4).

The outcome reuses ``lp.cumulative_log_returns`` so the DoubleML ATEs and the
Phase 2 local projections share an identical outcome definition — the basis for
the Phase 5 triangulation. DoubleML assumes no unmeasured confounders; the
placebo, sign, and cross-method-agreement checks are the discipline around that
(it cannot be conjured from observational data alone).

Run as:
    uv run python -m metals.models.causal --metals GC=F SI=F --horizons 1 5 20
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_METALS: tuple[str, ...] = ("GC=F", "SI=F", "PL=F", "PA=F")

# Deliberately modest LightGBM nuisance learners: enough flexibility to debias,
# shallow enough to cross-fit fast and not overfit the propensity score.
DEFAULT_LEARNER_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "n_jobs": 1,
    "random_state": 0,
    "verbose": -1,
}

_ATE_COLUMNS: tuple[str, ...] = (
    "scenario",
    "metal",
    "horizon",
    "ate",
    "se",
    "ci_low",
    "ci_high",
    "n_treated",
    "n_control",
    "placebo_pvalue",
)


@dataclass(frozen=True)
class CausalResult:
    """DoubleML ATE estimates across (scenario, metal, horizon).

    ``ate`` columns: scenario, metal, horizon, ate, se, ci_low, ci_high,
    n_treated, n_control, placebo_pvalue.
    """

    ate: pd.DataFrame
    scenario_ids: list[str]
    metals: list[str]
    horizons: list[int]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_output_path() -> Path:
    return _repo_root() / "data" / "processed" / "double_ml_ates.parquet"


def estimate_ate(
    outcome: pd.Series,
    treatment: pd.Series,
    confounders: pd.DataFrame,
    *,
    n_folds: int = 5,
    ci_alpha: float = 0.05,
    learner_params: dict[str, Any] | None = None,
    min_obs: int = 200,
    min_treated: int = 20,
    seed: int = 0,
) -> dict[str, float]:
    """Debiased ATE of binary ``treatment`` on ``outcome`` given ``confounders``.

    Parameters
    ----------
    outcome
        Forward outcome ``Y_{t+h}`` (e.g. ``lp.cumulative_log_returns``), indexed
        by date. The trailing ``h`` NaN rows are dropped during alignment.
    treatment
        Binary 0/1 indicator at time ``t``; must share ``outcome``'s index.
    confounders
        Control matrix ``X_t``; same index. Rows with any NaN are dropped.
    n_folds
        Cross-fitting folds (K). Default 5.
    ci_alpha
        CI level (0.05 -> 95%).
    learner_params
        Overrides merged onto ``DEFAULT_LEARNER_PARAMS`` for both nuisances.
    min_obs, min_treated
        Guards: below these (total rows / treated / control) the estimate is
        returned as NaN rather than fitting an unstable model.
    seed
        Seeds the cross-fitting split (DoubleML draws it off the numpy global
        RNG) and the learners, for reproducibility.

    Returns
    -------
    dict with keys ate, se, ci_low, ci_high, n_treated, n_control, n_obs.
    """
    import doubleml as dml
    from lightgbm import LGBMClassifier, LGBMRegressor

    if not outcome.index.equals(treatment.index):
        raise ValueError("outcome and treatment must share an index.")
    if not outcome.index.equals(confounders.index):
        raise ValueError("confounders must share the same index as outcome.")

    x_cols = list(confounders.columns)
    df = pd.concat(
        [outcome.rename("y"), treatment.rename("d").astype("float64"), confounders],
        axis=1,
    ).dropna()
    n_obs = int(len(df))
    n_treated = int((df["d"] == 1).sum())
    n_control = int((df["d"] == 0).sum())
    nan_result = {
        "ate": float("nan"),
        "se": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "n_treated": n_treated,
        "n_control": n_control,
        "n_obs": n_obs,
    }
    if n_obs < min_obs or n_treated < min_treated or n_control < min_treated:
        return nan_result

    params = {**DEFAULT_LEARNER_PARAMS, **(learner_params or {})}
    params["random_state"] = seed
    np.random.seed(seed)
    data = dml.DoubleMLData.from_arrays(
        df[x_cols].to_numpy(dtype="float64"),
        df["y"].to_numpy(dtype="float64"),
        df["d"].to_numpy(dtype="float64"),
    )
    irm = dml.DoubleMLIRM(
        data,
        ml_g=LGBMRegressor(**params),
        ml_m=LGBMClassifier(**params),
        n_folds=n_folds,
        score="ATE",
    )
    irm.fit()
    ci = irm.confint(level=1 - ci_alpha)
    return {
        "ate": float(irm.coef[0]),
        "se": float(irm.se[0]),
        "ci_low": float(ci.iloc[0, 0]),
        "ci_high": float(ci.iloc[0, 1]),
        "n_treated": n_treated,
        "n_control": n_control,
        "n_obs": n_obs,
    }


def placebo_pvalue(
    outcome: pd.Series,
    treatment: pd.Series,
    confounders: pd.DataFrame,
    real_ate: float,
    *,
    n_trials: int = 100,
    min_offset: int = 5,
    max_offset: int = 60,
    seed: int = 2026,
    **ate_kwargs: Any,
) -> dict[str, float]:
    """Placebo test (plan 5.3): fraction of shifted-treatment ATEs as large as the real one.

    Each trial shifts the treatment indicator by a random offset of
    ``+/- [min_offset, max_offset]`` trading days and re-estimates. The p-value
    is ``mean(|ATE_placebo| >= |real_ate|)`` over valid trials. A genuine effect
    should sit in the tail; ``placebo_pvalue > 0.10`` is suspect.

    Returns ``{"placebo_pvalue", "n_valid"}``.
    """
    if np.isnan(real_ate) or n_trials <= 0:
        return {"placebo_pvalue": float("nan"), "n_valid": 0.0}
    rng = np.random.default_rng(seed)
    exceed = 0
    valid = 0
    for _ in range(n_trials):
        mag = int(rng.integers(min_offset, max_offset + 1))
        k = mag if rng.random() < 0.5 else -mag
        shifted = treatment.shift(k).fillna(0).astype("int8")
        if shifted.nunique() < 2:
            continue
        res = estimate_ate(outcome, shifted, confounders, **ate_kwargs)
        if np.isnan(res["ate"]):
            continue
        valid += 1
        if abs(res["ate"]) >= abs(real_ate):
            exceed += 1
    if valid == 0:
        return {"placebo_pvalue": float("nan"), "n_valid": 0.0}
    return {"placebo_pvalue": exceed / valid, "n_valid": float(valid)}


def estimate_cate(
    outcome: pd.Series,
    treatment: pd.Series,
    confounders: pd.DataFrame,
    effect_modifiers: pd.DataFrame,
    *,
    learner_params: dict[str, Any] | None = None,
    n_estimators: int = 500,
    min_obs: int = 300,
    seed: int = 0,
) -> dict[str, Any]:
    """Conditional ATE via ``econml.CausalForestDML`` (plan 5.4).

    Estimates the treatment effect as a function of ``effect_modifiers`` (X)
    while controlling for ``confounders`` (W). Use to ask e.g. "is the FOMC
    effect on gold larger when real yields are negative?".

    Heterogeneity from daily data is noisy — read sign-stable patterns, not
    point differences. Returns ``{cate (per-date Series), cate_mean, cate_std,
    frac_positive, n_obs}``; degenerate inputs give a NaN summary.
    """
    from econml.dml import CausalForestDML
    from lightgbm import LGBMClassifier, LGBMRegressor

    for name, frame in (("effect_modifiers", effect_modifiers), ("confounders", confounders)):
        if not outcome.index.equals(frame.index):
            raise ValueError(f"{name} must share the same index as outcome.")
    if not outcome.index.equals(treatment.index):
        raise ValueError("treatment must share the same index as outcome.")

    mod_cols = list(effect_modifiers.columns)
    conf_cols = list(confounders.columns)
    df = pd.concat(
        [
            outcome.rename("y"),
            treatment.rename("d").astype("float64"),
            effect_modifiers,
            confounders,
        ],
        axis=1,
    ).dropna()
    nan_summary: dict[str, Any] = {
        "cate": pd.Series(dtype="float64"),
        "cate_mean": float("nan"),
        "cate_std": float("nan"),
        "frac_positive": float("nan"),
        "n_obs": int(len(df)),
    }
    if len(df) < min_obs or df["d"].nunique() < 2:
        return nan_summary

    params = {**DEFAULT_LEARNER_PARAMS, **(learner_params or {})}
    params["random_state"] = seed
    est = CausalForestDML(
        model_y=LGBMRegressor(**params),
        model_t=LGBMClassifier(**params),
        discrete_treatment=True,
        n_estimators=n_estimators,
        random_state=seed,
    )
    est.fit(
        df["y"].to_numpy(dtype="float64"),
        df["d"].to_numpy(dtype="float64"),
        X=df[mod_cols].to_numpy(dtype="float64"),
        W=df[conf_cols].to_numpy(dtype="float64"),
    )
    cate = np.asarray(est.effect(df[mod_cols].to_numpy(dtype="float64")), dtype="float64")
    return {
        "cate": pd.Series(cate, index=df.index, name="cate"),
        "cate_mean": float(np.mean(cate)),
        "cate_std": float(np.std(cate)),
        "frac_positive": float(np.mean(cate > 0)),
        "n_obs": int(len(df)),
    }


def estimate_scenarios(
    specs: Sequence[Any],
    *,
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    fomc: pd.DataFrame | None = None,
    metals: Sequence[str] = DEFAULT_METALS,
    horizons: Sequence[int] = (1, 5, 20),
    window_start: str = "2010-01-01",
    n_folds: int = 5,
    placebo_trials: int = 100,
    placebo_horizon: int = 5,
    learner_params: dict[str, Any] | None = None,
) -> CausalResult:
    """Estimate ATEs (and the h=``placebo_horizon`` placebo p-value) for every
    (scenario, metal, horizon). Pure given loaded frames — no DB or harness IO.

    ``specs`` is a sequence of ``ScenarioSpec`` (see ``metals.features.scenarios``).
    ``prices`` is wide adj-close; ``macro`` is wide FRED; ``fomc`` is the FOMC
    surprises frame (required only if a scenario sources from it).
    """
    from metals.features.returns import compute_log_returns
    from metals.features.scenarios import (
        build_confounders,
        build_treatment,
        confounder_exclusions,
    )
    from metals.models.lp import cumulative_log_returns

    returns_1d = compute_log_returns(prices, (1,))
    trading_idx = returns_1d.index
    hs = tuple(int(h) for h in horizons)
    win = pd.Timestamp(window_start)

    rows: list[dict[str, Any]] = []
    for spec in specs:
        # Estimate only within the modelling window (Phase 2's `loc[WINDOW:]`),
        # where the treatment thresholds are defined.
        treatment = build_treatment(
            spec, trading_idx, fomc=fomc, macro=macro, window_start=window_start
        ).loc[win:]
        excl = confounder_exclusions(spec)
        for ticker in metals:
            ret_col = f"{ticker}_ret_1d"
            if ret_col not in returns_1d.columns:
                continue
            own_ret = returns_1d[ret_col]
            confounders = build_confounders(
                ticker, trading_idx, prices=prices, macro=macro, exclude=excl
            ).loc[win:]
            for h in hs:
                outcome = cumulative_log_returns(own_ret, h).loc[win:]
                res = estimate_ate(
                    outcome,
                    treatment,
                    confounders,
                    n_folds=n_folds,
                    learner_params=learner_params,
                )
                placebo = {"placebo_pvalue": float("nan")}
                if h == placebo_horizon and not np.isnan(res["ate"]):
                    placebo = placebo_pvalue(
                        outcome,
                        treatment,
                        confounders,
                        res["ate"],
                        n_trials=placebo_trials,
                        n_folds=n_folds,
                        learner_params=learner_params,
                    )
                rows.append(
                    {
                        "scenario": spec.id,
                        "metal": ticker,
                        "horizon": int(h),
                        "ate": res["ate"],
                        "se": res["se"],
                        "ci_low": res["ci_low"],
                        "ci_high": res["ci_high"],
                        "n_treated": res["n_treated"],
                        "n_control": res["n_control"],
                        "placebo_pvalue": placebo["placebo_pvalue"],
                    }
                )

    ate_df = pd.DataFrame(rows, columns=list(_ATE_COLUMNS))
    return CausalResult(
        ate=ate_df,
        scenario_ids=[s.id for s in specs],
        metals=list(metals),
        horizons=list(hs),
    )


def run(
    *,
    metals: Sequence[str] = DEFAULT_METALS,
    horizons: Sequence[int] | None = None,
    scenario_ids: Sequence[str] | None = None,
    n_folds: int = 5,
    placebo_trials: int = 100,
    placebo_horizon: int = 5,
    window_start: str | None = None,
    notes: str | None = None,
    output_path: str | None = None,
) -> str:
    """End-to-end DoubleML ATE estimation against the canonical DuckDB.

    Loads prices/macro/FOMC surprises, builds treatments + confounders for every
    available scenario, estimates ATEs and the h=``placebo_horizon`` placebo
    p-value, writes ``data/processed/double_ml_ates.parquet``, and registers the
    run with the eval harness (for provenance — ATEs do not fit the per-row
    prediction schema, so nothing is logged via ``log_predictions``). Returns the
    harness ``run_id``.
    """
    from metals.eval.harness import register_run
    from metals.features.loaders import load_fomc_surprises, load_macro, load_prices
    from metals.features.scenarios import load_scenario_config

    cfg = load_scenario_config()
    win = window_start or cfg.window_start
    hs = tuple(int(h) for h in (horizons if horizons else cfg.horizons))
    specs = [s for s in cfg.scenarios if s.available]
    if scenario_ids:
        wanted = set(scenario_ids)
        specs = [s for s in specs if s.id in wanted]
    if not specs:
        raise RuntimeError(
            "No available scenarios to estimate; check configs/scenarios.yaml "
            "or the --scenarios filter."
        )

    prices = load_prices(tickers=list(metals))
    macro = load_macro()
    fomc = load_fomc_surprises()
    if prices.empty or macro.empty:
        raise RuntimeError(
            "prices/macro are empty — build the DuckDB first: "
            "`uv run python -m metals.data.migrations.runner` then ingest "
            "prices/FRED/FOMC surprises."
        )

    result = estimate_scenarios(
        specs,
        prices=prices,
        macro=macro,
        fomc=fomc,
        metals=metals,
        horizons=hs,
        window_start=win,
        n_folds=n_folds,
        placebo_trials=placebo_trials,
        placebo_horizon=placebo_horizon,
    )

    out_path = Path(output_path) if output_path else _default_output_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.ate.to_parquet(out_path)

    run_name = f"causal_dml_{datetime.now():%Y%m%d_%H%M}"
    return register_run(
        name=run_name,
        model_type="causal",
        target_type="cumret",
        config={
            "method": "DoubleML-IRM",
            "metals": list(metals),
            "horizons": list(hs),
            "scenarios": [s.id for s in specs],
            "n_folds": n_folds,
            "placebo_trials": placebo_trials,
            "placebo_horizon": placebo_horizon,
            "window_start": win,
            "output_path": str(out_path),
        },
        notes=notes,
    )


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Phase 5 DoubleML ATE estimation.")
    p.add_argument("--metals", nargs="+", default=list(DEFAULT_METALS))
    p.add_argument("--horizons", nargs="+", type=int, default=None)
    p.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="scenario ids to restrict to (default: all available).",
    )
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--placebo-trials", type=int, default=100)
    p.add_argument("--notes", default=None)
    args = p.parse_args()
    run_id = run(
        metals=args.metals,
        horizons=args.horizons,
        scenario_ids=args.scenarios,
        n_folds=args.n_folds,
        placebo_trials=args.placebo_trials,
        notes=args.notes,
    )
    print(run_id)


if __name__ == "__main__":
    main()
