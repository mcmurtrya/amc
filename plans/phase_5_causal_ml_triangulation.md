# Phase 5 — Causal ML and Method Triangulation

## Goal
Estimate causal effects of each scenario type using double/debiased ML, validate with sign-restricted structural VAR, and reconcile against the event-driven (Phase 2) and clustering (Phases 3–4) methods. This phase produces the project's central quantitative deliverable: the scenario master table.

## Prerequisites
- Phases 2, 3, 4 complete
- Scenarios defined: from events (Phase 2) and from clusters (Phase 3/4)

## Steps

### 5.1 Compile the master scenario list
Pull together every scenario candidate from prior phases:

Event-defined (Phase 2):
- Hawkish FOMC surprise (top tercile of policy-rate surprise)
- Dovish FOMC surprise (bottom tercile)
- CPI upside surprise
- CPI downside surprise
- NFP upside / downside surprise
- GPR daily change > 95th percentile
- DXY 5-day change > 2σ (both directions)

Cluster-defined (Phase 3 and Phase 4):
- Each interpretable cluster as a binary indicator

For each: precise definition of treatment indicator `T_t`, outcome `Y_{t+h}`, confounder set `X_t` (lagged returns, macro state, recent positioning). Store as a YAML file in `configs/scenarios.yaml` for reproducibility.

### 5.2 DoubleML ATE estimation
`src/metals/models/causal.py`:
- Use the `DoubleML` package
- Outcome model: LightGBM regressor
- Propensity model: LightGBM classifier
- Cross-fitting K=5
- For each `(scenario, metal, horizon)` triple, estimate ATE with 95% CI
- Horizons: 1, 5, 20 days

Output: a long-format DataFrame `(scenario, metal, horizon, ate, se, ci_low, ci_high, n_treated, n_control)` written to `data/processed/double_ml_ates.parquet`.

### 5.3 Placebo treatments
For each real scenario, repeat estimation with treatment dates shifted by random offsets of ±5 to ±60 trading days. Run 100 placebo trials per scenario.

For each scenario compute a placebo p-value: fraction of placebo ATEs at h=5 that exceed |real ATE| in magnitude. Scenarios with placebo p > 0.10 are suspect.

### 5.4 Conditional treatment effects (CATE)
Use Causal Forests (via `econml.dml.CausalForestDML`):
- Treatment effect as a function of macro state (TIPS level, DXY z-score, VIX regime)
- Specifically test: "Is the FOMC effect on gold larger when real yields are negative?"
- Output for each scenario: a per-date CATE estimate plus aggregate heterogeneity statistics

Don't over-claim heterogeneity. With ~250 trading days/year, CATE estimates are noisy. Look for sign-stable patterns, not point-estimate differences.

### 5.5 Sign-restricted structural VAR
`src/metals/models/svar.py`:

4-variable VAR over daily data: real yield (DGS10 − T10YIE), DXY return, S&P 500 return, gold return.

Lag length: select via BIC, typically 2–5 lags.

Sign restrictions for identification (each restriction on impact, h=0):

| Shock | real yield | DXY | S&P | gold |
|-------|-----------|-----|-----|------|
| Real-yield | + | + | − or 0 | − |
| Risk-aversion (flight to safety) | − | ? | − | + |
| USD shock | ? | + | ? | − |

Use the `pyflux` or `arviz`-compatible setup, or hand-roll with Rubio-Ramirez algorithm. Compute IRFs over 60 days with 16/84 percentile bands from posterior draws.

### 5.6 Triangulation table
For each scenario, three estimates of the gold impact at h=5 days:
- β from local projection (Phase 2)
- ATE from DoubleML (5.2)
- Implied effect from SVAR if the scenario maps to a structural shock (5.5)

Build a single comparison table with columns: `scenario, lp_beta_h5, lp_ci, dml_ate_h5, dml_ci, svar_implied_h5, svar_band, agreement_score`.

Agreement score: count how many of the three intervals overlap with each other. 3 = strong agreement, 2 = partial, 1 = weak, 0 = full disagreement (interesting and worth investigating).

### 5.7 Cross-metal consistency
For each scenario, check that the cross-metal pattern matches economic theory:
- Monetary/macro scenarios (FOMC, CPI, real yields): Au and Ag should react similarly in sign; Pt/Pd weaker or noisier
- Industrial/cyclical scenarios: Pt/Pd react more, Au less
- Supply scenarios (mining strikes, sanctions): effect concentrated in the directly affected metal

Compute a consistency score per scenario and add to the master table. Failures of cross-metal consistency are red flags.

### 5.8 Subsample stability
Re-estimate each scenario's ATE on:
- 2007–2014 (pre-Taper-Tantrum normalization)
- 2015–2019 (low-vol "normal")
- 2020–2026 (COVID, inflation, geopolitical regime)

A scenario with stable sign across all three subsamples is robust. Sign flips signal regime dependence — still interesting but less generalizable.

### 5.9 Master scenario table
Final output `data/processed/scenario_master.parquet`:
- `scenario_name`
- `definition_type` (event / cluster)
- `definition_yaml_id`
- `ate_{metal}_{horizon}` and `ci_{metal}_{horizon}` for each metal and h ∈ {1, 5, 20}
- `placebo_pvalue`
- `triangulation_agreement_score`
- `cross_metal_consistency_score`
- `subsample_stability_score`
- `n_treated`, `n_control`
- `economic_interpretation` (hand-written)

This is the central scientific output of the project.

### 5.10 Write-up
`results/phase5_triangulation.md`:
- Method-by-scenario comparison plots
- Highlight high-agreement scenarios as robust findings
- Highlight disagreements with explicit hypotheses about why
- Note which scenarios fail robustness
- Explicit list of "things I would have liked to test but couldn't" — this matters more than the headline findings

## Deliverables
- `metals.models.causal` module with DoubleML estimation and placebo tests
- `metals.models.svar` module with sign-restricted IRFs
- `scenario_master.parquet` — the central deliverable
- `results/phase5_triangulation.md`

## Common pitfalls
- Cherry-picking the method that gave the "right" answer. Pre-register your specifications (write them down in the journal *before* estimating).
- Treating CATE heterogeneity as causal interaction without enough power. Daily financial data has fewer effective observations than it appears.
- Forgetting that DoubleML assumes no unmeasured confounders. You cannot conjure causal identification from observational data alone — sign restrictions, placebo tests, and cross-method agreement are your discipline.
- Sign-restricted VAR identification depends critically on the chosen restrictions. Document them explicitly and try an alternative set as a robustness check.
- Reporting only point estimates. The CI matters more than the point estimate, especially with ~12 scenarios × 4 metals × 3 horizons = 144 tests where multiple-comparisons corrections start to bite.
