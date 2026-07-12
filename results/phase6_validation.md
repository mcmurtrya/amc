# Phase 6 — Validation (in progress)

## 6.1 Hold-out freeze and contamination audit (2026-07-11)

**Designated hold-out: 2026-01-18 → 2026-05-22** (the price-data right edge;
~85 trading days). The plan's 12-month ideal is not attainable and we do not
pretend otherwise: development was walk-forward from the start, so every
earlier window has appeared in some test readout. The controlling fact is
that the **last model-selection readout** (cluster-lift experiment, fold 10)
had its test window end **2026-01-17**; no metric computed on any later date
has ever been read during development.

Audit trail, pipeline by pipeline:

| Pipeline | Hold-out status |
|---|---|
| Cluster-lift folds (Phase 3/5 model selection) | Last test window ends 2026-01-17 → hold-out untouched. |
| Phase 3 clustering fit | `train_until` 2024; hold-out rows only ever received out-of-sample `approximate_predict` assignments. Fit clean. |
| Phase 3 LLM labels | Read hold-out-era headlines (descriptive only; no parameter or threshold derives from them). Disclosed, accepted. |
| Phase 5 ATE/SVAR estimation windows | **Contaminated** (estimated through 2026-05). For 6.5, training-period ATEs are re-estimated with an end bound of 2026-01-17. |
| Scenario thresholds (GPR pct, DXY sigma) | **Contaminated** (computed in-window through 2026-05). For 6.5, thresholds are recomputed on pre-hold-out data only and applied forward. |
| Phase 1 feature-set diagnosis | Walk-forward readouts in June 2026 plausibly included early-2026 test windows. Mitigation: hold-out scoring uses the **untuned default** feature set (`full`), not the diagnosis-selected lean variants. Residual design-level contamination disclosed. |
| FOMC treatments | Bauer–Swanson ends 2023-12 → no hold-out firings possible; FOMC scenarios cannot be sign-validated on this hold-out. |

**Training embargo:** the target (h=5, w=20 realized vol) reads returns
through t+24 trading days, so training rows within `purge_days_for(5,20)=44`
calendar days of the boundary have targets that peek into the hold-out.
Models are trained on rows ≤ **2025-12-04** (boundary − 44d), with the last
180 pre-embargo days as the early-stopping validation slice. Scorable
hold-out predictions run through ~2026-04-16 (last date whose target window
completes by the data edge).

*(Sections 6.2–6.5 appended by the runner scripts; 6.7–6.9 drafted after.)*

## 6.2–6.4 Hold-out scoring, benchmarks, lift table

Runner `scripts/phase6_holdout.py`; per-model predictions in the harness;
metrics `phase6_holdout_metrics.csv`. 63 common scorable days
(2026-01-20 → 2026-04-20). **Effective sample is tiny** — with 24-trading-day
overlapping target windows, 63 days ≈ 2–3 independent observations; the DM
statistics (NW lag 24) and block-bootstrap CIs account for overlap but cannot
manufacture power.

| Model | RMSE | vs RW | IC | DM t (vs lgbm_full) | RMSE 95% CI |
|---|---|---|---|---|---|
| var2 | 0.127 | 0.69 | −0.00 | −1.18 | [0.090, 0.141] |
| garch11 | 0.131 | 0.70 | −0.29 | −0.46 | [0.071, 0.137] |
| lgbm_full | 0.140 | 0.76 | **+0.21** | — | [0.093, 0.161] |
| lgbm_regime | 0.149 | 0.80 | −0.00 | **+3.43** | [0.106, 0.172] |
| lgbm_sentiment | 0.163 | 0.88 | +0.20 | **+2.90** | [0.116, 0.192] |
| uncond_mean | 0.178 | 0.96 | 0.00 | +3.64 | [0.132, 0.208] |
| random_walk | 0.186 | 1.00 | −0.14 | +2.12 | [0.119, 0.224] |

Readings:

1. **Classical vol models beat the ML stack on the hold-out** — VAR(2) and
   GARCH(1,1) post the lowest RMSE, though neither significantly beats
   LightGBM (|DM t| ≤ 1.2). The roadmap's prior ("a tuned LightGBM beats
   most transformers in honest evaluations") extends one rung down:
   on this window, mean-reverting classical baselines beat LightGBM too.
2. **Regime and sentiment features hurt out-of-sample, significantly**
   (DM t +3.4 / +2.9 vs lgbm_full) — the Phase 3/5 forecasting null
   re-confirms on virgin data, now as an actual OOS penalty.
3. **Random walk is the worst model** — trailing vol was a bad level
   anchor in a window where vol shifted; everything mean-reverting won.
   LightGBM's IC (+0.21) is the only meaningfully positive rank signal.

## 6.5 Scenario sign-validation on the hold-out

Runner `scripts/phase6_scenario_holdout.py` (pre-hold-out thresholds,
re-estimated pre-hold-out ATEs); table `phase6_scenario_holdout.csv`.
FOMC scenarios structurally untestable (surprise data ends 2023-12).

| Scenario | Fires | Sign agreement | Reading |
|---|---|---|---|
| gpr_spike | 12 | 2/4 metals | Coin-flip, as its master-table scores predicted. |
| dxy_up_shock | 0 | — | No +2σ dollar rallies in the window. |
| dxy_down_shock | 5 | **4/4** | Signs agree but magnitudes are 5–90× the training ATE (Ag −17.9% raw diff): the hold-out fires cluster in one risk-off episode — the *contamination mechanism itself* replicating out-of-sample, not a validated causal effect. |

## 6.6 Component ablation — adapted

The plan's ablation targets the Phase 4 transformer, which was deliberately
descoped. The spirit of 6.6 is answered by design elsewhere: the lift
experiment (regime features: no lift, now an OOS penalty), the sentiment-only
model above (worse than the full set), and Phase 1's feature-set diagnosis.
A transformer ablation becomes relevant only if the deferred Phase 4 runs.

## 6.7 Limitations

- **Data quality.** Text carries a 2019-09 title-era break and no per-metal
  signal; COT is ingested but effectively unused; the HY-OAS license
  truncation was caught but similar silent truncations remain possible.
  Timestamp discipline (UTC, one-day text lag, futures-settle-before-FOMC)
  is tested but rests on documented conventions, not tick data.
- **Identification.** DML assumes no unmeasured confounders — untestable;
  placebos and triangulation discipline but do not prove. SVAR conclusions
  are conditional on the sign restrictions (the alternative set agreed —
  comfort, not proof). Macro-event treatments (2σ moves) are demonstrably
  endogenous; only the FOMC treatments approach exogeneity.
- **Regime instability.** The anchor effect's magnitude decayed ~2.3× from
  the QE era to 2020-26; palladium's responses are regime-dependent in every
  method. Post-2023 monetary policy is causally untested (no surprise data).
- **Modeling choices.** No hyperparameter search anywhere (pinned defaults
  by design — a robustness virtue but an efficiency unknown); architectures
  not tried: state-space/mixed-frequency models, transformers (deferred),
  headline-level text models (gated off by the lift null).
- **External validity.** The pipeline (walk-forward harness, scenario
  registry, triangulation) transfers to any macro asset; the findings are
  metals-specific — gold's real-yield elasticity and the PGM supply regime
  do not generalize.
- **Statistical power.** The binding constraint everywhere: 35 hawkish
  events, 26 in the regime-covered window, 63 hold-out days ≈ 2-3
  independent vol observations, 5 hold-out DXY fires. Multiplicity across
  the 60-cell ATE grid is disciplined by placebos and pre-registration but
  not formally corrected.

## Status of remaining plan items

- **6.8/6.9 (full methodology + findings write-ups)**: **done** —
  `results/phase6_methodology.md` (9-section standalone methodology) and
  `results/phase6_findings.md` (scenario-first findings). Both drafted from
  committed source material; no analyses re-run.
- **6.10 (reproducibility package)**: partially standing (uv-pinned deps,
  harness records, master table, scripts); missing: single-command refresh/
  retrain entry points and a model-weights checkpoint.
- **6.11 (cleanup + v1.0 tag)**: pending the write-ups.
