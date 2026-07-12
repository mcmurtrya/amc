"""Sign-restricted structural VAR (plan step 5.5).

4-variable daily VAR over [real-yield change, DXY log return, S&P 500 log
return, gold log return], identified by impact sign restrictions via the
Rubio-Ramirez rejection algorithm. Per the 2026-06-23 build decision the
bands are Bayesian: reduced-form parameters are drawn from a noninformative
Normal-inverse-Wishart posterior, each draw gets one random orthogonal
rotation, and draws violating the impact restrictions (or stationarity) are
rejected — so the 16/84 bands carry estimation *and* identification
uncertainty.

Variables enter stationarized (yield change + log returns): a "+" impact on
``real_yield`` means the yield *rises* on impact; gold's IRF is a return,
cumulated for level responses.

Baseline restrictions (impact, h=0; "?" = unrestricted):

    shock          real_yield  dxy  spx        gold
    real_yield     +           +    "- or 0"   -
    risk_aversion  -           ?    -          +
    usd            ?           +    ?          -

``ALT_SIGN_RESTRICTIONS`` is the pre-declared robustness set (plan "common
pitfalls"): the real-yield shock leaves S&P unrestricted, and risk-aversion
adds the flight-to-USD "+" on DXY.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Any

import numpy as np
import pandas as pd

from metals.features.leakage import assert_chronological

VARIABLES: tuple[str, ...] = ("real_yield", "dxy", "spx", "gold")

# Sign codes: "+" strictly positive, "-" strictly negative, "-0" nonpositive.
SIGN_RESTRICTIONS: dict[str, dict[str, str]] = {
    "real_yield": {"real_yield": "+", "dxy": "+", "spx": "-0", "gold": "-"},
    "risk_aversion": {"real_yield": "-", "spx": "-", "gold": "+"},
    "usd": {"dxy": "+", "gold": "-"},
}
ALT_SIGN_RESTRICTIONS: dict[str, dict[str, str]] = {
    "real_yield": {"real_yield": "+", "dxy": "+", "gold": "-"},
    "risk_aversion": {"real_yield": "-", "dxy": "+", "spx": "-", "gold": "+"},
    "usd": {"dxy": "+", "gold": "-"},
}


@dataclass
class SvarResult:
    """Accepted-draw IRFs and bookkeeping for one restriction set."""

    lag: int
    n_accepted: int
    n_draws: int
    horizons: int
    # irfs[shock] has shape (n_accepted, horizons+1, n_vars) — per-period
    # responses of the stationarized variables to a one-sd shock.
    irfs: dict[str, np.ndarray] = field(default_factory=dict)

    def quantiles(self, shock: str, var: str, qs=(0.16, 0.5, 0.84)) -> pd.DataFrame:
        """Percentile bands of the *cumulative* IRF for one shock/variable."""
        v = VARIABLES.index(var)
        cum = np.cumsum(self.irfs[shock][:, :, v], axis=1)
        out = {f"q{int(q * 100)}": np.quantile(cum, q, axis=0) for q in qs}
        return pd.DataFrame(out, index=pd.RangeIndex(cum.shape[1], name="h"))


def build_svar_data(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Assemble the stationarized 4-variable frame on gold trading days."""
    for col in ("GC=F", "^GSPC"):
        if col not in prices.columns:
            raise ValueError(f"{col!r} missing from prices.")
    for col in ("DGS10", "T10YIE", "DTWEXBGS"):
        if col not in macro.columns:
            raise ValueError(f"{col!r} missing from macro.")
    m = macro.reindex(prices.index).ffill()
    real_yield = (m["DGS10"] - m["T10YIE"]).astype(float)
    out = pd.DataFrame(
        {
            "real_yield": real_yield.diff(),
            "dxy": np.log(m["DTWEXBGS"].astype(float)).diff(),
            "spx": np.log(prices["^GSPC"].astype(float)).diff(),
            "gold": np.log(prices["GC=F"].astype(float)).diff(),
        },
        index=prices.index,
    ).dropna()
    assert_chronological(out)
    return out


def _lagged_design(y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, Y) for OLS: X = [1, y_{t-1}, ..., y_{t-p}]."""
    t_total, n = y.shape
    rows = t_total - p
    x = np.ones((rows, 1 + n * p))
    for i in range(1, p + 1):
        x[:, 1 + (i - 1) * n : 1 + i * n] = y[p - i : t_total - i]
    return x, y[p:]


def fit_var_ols(y: np.ndarray, p: int) -> dict[str, np.ndarray | float]:
    """OLS reduced-form VAR(p). Returns B (k x n), residuals U, Sigma, XtX_inv."""
    x, yy = _lagged_design(y, p)
    xtx = x.T @ x
    xtx_inv = np.linalg.inv(xtx)
    b = xtx_inv @ x.T @ yy
    u = yy - x @ b
    dof = max(yy.shape[0] - x.shape[1], 1)
    sigma = (u.T @ u) / dof
    return {"B": b, "U": u, "Sigma": sigma, "XtX_inv": xtx_inv, "T": float(yy.shape[0])}


def select_lag_bic(y: np.ndarray, max_lag: int = 5, min_lag: int = 1) -> int:
    """BIC lag selection on a common estimation sample (rows after max_lag)."""
    best_p, best_bic = min_lag, np.inf
    for p in range(min_lag, max_lag + 1):
        x, yy = _lagged_design(y[max_lag - p :], p)  # same effective sample for all p
        b = np.linalg.lstsq(x, yy, rcond=None)[0]
        u = yy - x @ b
        t_eff = yy.shape[0]
        sigma_ml = (u.T @ u) / t_eff
        sign, logdet = np.linalg.slogdet(sigma_ml)
        if sign <= 0:
            continue
        k_params = x.shape[1] * yy.shape[1]
        bic = t_eff * logdet + k_params * np.log(t_eff)
        if bic < best_bic:
            best_p, best_bic = p, bic
    return best_p


def _companion(b: np.ndarray, n: int, p: int) -> np.ndarray:
    """Companion matrix from stacked coefficient matrix B (k x n), k = 1 + n*p."""
    f = np.zeros((n * p, n * p))
    f[:n, :] = b[1:, :].T  # drop intercept row; A_i blocks are B[1+n(i-1):1+ni].T
    if p > 1:
        f[n:, :-n] = np.eye(n * (p - 1))
    return f


def irf_from_var(b: np.ndarray, impact: np.ndarray, horizons: int, n: int, p: int) -> np.ndarray:
    """Per-period IRFs: (horizons+1, n, n_shocks) given an impact matrix."""
    f = _companion(b, n, p)
    out = np.empty((horizons + 1, n, impact.shape[1]))
    psi = np.eye(n * p)
    out[0] = impact
    for h in range(1, horizons + 1):
        psi = f @ psi
        out[h] = psi[:n, :n] @ impact
    return out


def _check_cell(value: float, code: str) -> bool:
    if code == "+":
        return value > 0.0
    if code == "-":
        return value < 0.0
    if code == "-0":
        return value <= 0.0
    raise ValueError(f"Unknown sign code {code!r}")


def match_shocks(
    impact: np.ndarray, restrictions: dict[str, dict[str, str]]
) -> dict[str, tuple[int, int]] | None:
    """Assign distinct impact columns (with sign flips) to each restricted shock.

    Returns ``{shock: (column, flip)}`` with ``flip`` in {+1, -1}, or None if no
    assignment satisfies every restriction. Columns are searched in permutation
    order; the first full assignment wins (draw-level acceptance is what
    matters for the Rubio-Ramirez algorithm, not which valid assignment).
    """
    shocks = list(restrictions)
    n_cols = impact.shape[1]
    ok: dict[str, list[tuple[int, int]]] = {}
    for shock in shocks:
        cells = restrictions[shock]
        candidates = []
        for col in range(n_cols):
            for flip in (1, -1):
                vec = impact[:, col] * flip
                if all(_check_cell(vec[VARIABLES.index(v)], c) for v, c in cells.items()):
                    candidates.append((col, flip))
        if not candidates:
            return None
        ok[shock] = candidates
    for combo in permutations(range(n_cols), len(shocks)):
        assign: dict[str, tuple[int, int]] = {}
        for shock, col in zip(shocks, combo, strict=True):
            hit = next(((c, f) for c, f in ok[shock] if c == col), None)
            if hit is None:
                break
            assign[shock] = hit
        else:
            return assign
    return None


def estimate_svar(
    data: pd.DataFrame,
    *,
    lag: int | None = None,
    horizons: int = 60,
    n_target: int = 500,
    max_draws: int = 50_000,
    restrictions: dict[str, dict[str, str]] | None = None,
    seed: int = 42,
) -> SvarResult:
    """NIW-posterior + random-rotation sign-restricted SVAR (pure, no IO)."""
    from scipy.stats import invwishart

    restr = restrictions or SIGN_RESTRICTIONS
    y = data[list(VARIABLES)].to_numpy(dtype="float64")
    n = y.shape[1]
    p = lag or select_lag_bic(y)
    fit = fit_var_ols(y, p)
    b_hat = np.asarray(fit["B"])
    u = np.asarray(fit["U"])
    xtx_inv = np.asarray(fit["XtX_inv"])
    scale = u.T @ u
    dof = int(fit["T"]) - b_hat.shape[0]
    rng = np.random.default_rng(seed)
    chol_xtx_inv = np.linalg.cholesky(xtx_inv)

    irfs: dict[str, list[np.ndarray]] = {s: [] for s in restr}
    n_draws = 0
    while n_draws < max_draws and len(next(iter(irfs.values()))) < n_target:
        n_draws += 1
        sigma_d = invwishart.rvs(df=dof, scale=scale, random_state=rng)
        chol_sigma = np.linalg.cholesky(sigma_d)
        z = rng.standard_normal(b_hat.shape)
        b_d = b_hat + chol_xtx_inv @ z @ chol_sigma.T
        # Stationarity: explosive draws produce divergent IRFs — reject.
        if np.max(np.abs(np.linalg.eigvals(_companion(b_d, n, p)))) >= 1.0:
            continue
        q, r = np.linalg.qr(rng.standard_normal((n, n)))
        q = q @ np.diag(np.sign(np.diag(r)))  # Haar-uniform rotation
        impact = chol_sigma @ q
        assign = match_shocks(impact, restr)
        if assign is None:
            continue
        for shock, (col, flip) in assign.items():
            shock_impact = (impact[:, [col]] * flip).reshape(n, 1)
            irf = irf_from_var(b_d, shock_impact, horizons, n, p)
            irfs[shock].append(irf[:, :, 0])

    accepted = len(next(iter(irfs.values())))
    return SvarResult(
        lag=p,
        n_accepted=accepted,
        n_draws=n_draws,
        horizons=horizons,
        irfs={s: np.stack(v) if v else np.empty((0, horizons + 1, n)) for s, v in irfs.items()},
    )


def run(
    *,
    horizons: int = 60,
    n_target: int = 500,
    seed: int = 42,
    notes: str | None = None,
    output_path: str = "results/phase5_svar_irfs.csv",
) -> str:
    """End-to-end SVAR against the canonical DuckDB; both restriction sets.

    Writes cumulative-IRF quantile bands per (restriction set, shock, variable)
    and registers the run with the eval harness. Returns the run_id.
    """
    from metals.eval.harness import register_run
    from metals.features.loaders import load_macro, load_prices

    prices = load_prices(column="adj_close")
    macro = load_macro()
    data = build_svar_data(prices, macro).loc["2010-01-01":]

    frames = []
    meta: dict[str, Any] = {}
    for set_name, restr in (("baseline", SIGN_RESTRICTIONS), ("alt", ALT_SIGN_RESTRICTIONS)):
        res = estimate_svar(
            data, horizons=horizons, n_target=n_target, restrictions=restr, seed=seed
        )
        meta[set_name] = {
            "lag": res.lag,
            "n_accepted": res.n_accepted,
            "n_draws": res.n_draws,
            "acceptance": res.n_accepted / max(res.n_draws, 1),
        }
        print(f"[{set_name}] lag={res.lag} accepted {res.n_accepted}/{res.n_draws} draws")
        for shock in restr:
            for var in VARIABLES:
                q = res.quantiles(shock, var).reset_index()
                q.insert(0, "variable", var)
                q.insert(0, "shock", shock)
                q.insert(0, "restriction_set", set_name)
                frames.append(q)

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(output_path, index=False)
    print(f"wrote {output_path} ({len(out)} rows)")

    return register_run(
        name="svar_sign_restricted",
        model_type="svar",
        target_type="irf",
        config={
            "variables": list(VARIABLES),
            "window_start": "2010-01-01",
            "n_obs": int(len(data)),
            "horizons": horizons,
            "n_target": n_target,
            "seed": seed,
            "restrictions": {"baseline": SIGN_RESTRICTIONS, "alt": ALT_SIGN_RESTRICTIONS},
            **meta,
        },
        notes=notes,
    )


if __name__ == "__main__":
    run(notes="Plan 5.5: first real-DB SVAR run (baseline + alt sign sets)")
