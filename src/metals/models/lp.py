"""Jordà local projections for event-driven impulse responses.

The Phase 2 workhorse. For each horizon h, we fit

    sum_{k=1..h} r_{t+k} = alpha_h + beta_h * D_t + gamma_h' * X_t + e_{t,h}

by OLS, with Newey–West (HAC) standard errors of bandwidth h to account for
the moving-average structure of the cumulative return on the LHS.

``beta_h`` is the impulse response at horizon h: the cumulative effect on
the metal's log return of a unit move in the treatment ``D``. Plot beta vs.
h with bands ±1.96 * se to get the standard IRF figure.

Notes
-----
* The cumulative-return convention is the one used by Jordà (2005) and most
  macro/finance LP papers. It tells a "level" story: "h days after a hawkish
  FOMC surprise, gold is down ~1.5%."
* The treatment must be measured *at* time t, not after, otherwise the
  estimated beta absorbs future information and the regression is no longer
  causal. Same applies to controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LPResult:
    """Estimated impulse response across horizons.

    ``irf`` columns: horizon, beta, se, t_stat, ci_low, ci_high, n_obs.
    """

    irf: pd.DataFrame
    treatment_name: str
    control_names: list[str]


def cumulative_log_returns(
    returns_1d: pd.Series,
    horizon: int,
) -> pd.Series:
    """For each t, the sum of daily log returns at indices t+1 .. t+h.

    The last ``horizon`` rows are NaN by construction. This is the dependent
    variable in the per-horizon LP regression.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    return (
        returns_1d.rolling(horizon, min_periods=horizon).sum().shift(-horizon)
    )


def local_projection(
    returns_1d: pd.Series,
    treatment: pd.Series,
    controls: pd.DataFrame | None = None,
    horizons: Sequence[int] = (1, 3, 5, 10, 20, 60),
    ci_alpha: float = 0.05,
) -> LPResult:
    """Estimate the impulse response of ``returns_1d`` to ``treatment``.

    Parameters
    ----------
    returns_1d
        Daily log returns of the outcome asset, indexed by date.
    treatment
        The shock at time t. Can be a binary event indicator or a continuous
        magnitude (e.g. a standardised surprise). Must share an index with
        ``returns_1d``.
    controls
        Optional contemporaneous controls; same index. Each column enters
        the regression linearly at time t (no lags — caller's responsibility
        to lag any feature that needs it).
    horizons
        Horizons (in trading days) at which to estimate the impulse response.
    ci_alpha
        Confidence-interval significance level. Default 0.05 → 95% CI.

    Returns
    -------
    LPResult with one row per requested horizon.
    """
    import statsmodels.api as sm
    from scipy import stats

    if not returns_1d.index.equals(treatment.index):
        raise ValueError("returns_1d and treatment must share an index.")
    if controls is not None and not returns_1d.index.equals(controls.index):
        raise ValueError("controls must share the same index as returns_1d.")

    z = stats.norm.ppf(1 - ci_alpha / 2)
    treatment_name = treatment.name or "treatment"
    control_names = list(controls.columns) if controls is not None else []

    rows: list[dict] = []
    for h in sorted(horizons):
        y = cumulative_log_returns(returns_1d, h).rename("y")
        X_parts = [treatment.rename(treatment_name)]
        if controls is not None:
            X_parts.extend(controls[c] for c in control_names)
        X = pd.concat(X_parts, axis=1)
        X = sm.add_constant(X, has_constant="add")
        df = pd.concat([y, X], axis=1).dropna()
        if len(df) < 30:
            rows.append({
                "horizon": h, "beta": np.nan, "se": np.nan, "t_stat": np.nan,
                "ci_low": np.nan, "ci_high": np.nan, "n_obs": len(df),
            })
            continue

        res = sm.OLS(df["y"], df.drop(columns="y")).fit(
            cov_type="HAC", cov_kwds={"maxlags": h}
        )
        beta = float(res.params[treatment_name])
        se = float(res.bse[treatment_name])
        rows.append({
            "horizon": h,
            "beta": beta,
            "se": se,
            "t_stat": beta / se if se > 0 else float("nan"),
            "ci_low": beta - z * se,
            "ci_high": beta + z * se,
            "n_obs": int(len(df)),
        })

    return LPResult(
        irf=pd.DataFrame(rows),
        treatment_name=treatment_name,
        control_names=control_names,
    )
