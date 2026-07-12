# Methodology — Drivers of Precious-Metals Prices

**Phase 6.8 deliverable. 2026-07-11.** This is the consolidated methodology
record for the full research program (Phases 0–6). It is written to stand on
its own: a reader who has seen none of the per-phase notes should be able to
reconstruct what was built, why each choice was made, and where the results
can and cannot be trusted. Companion documents: `phase6_findings.md` (what we
learned), `phase6_validation.md` (the hold-out evaluation), and the per-phase
write-ups it cites throughout.

The organizing claim of the whole project is narrow and defensible: **hawkish
monetary-policy surprises depress precious-metals prices (gold ≈ −1.4% over
five trading days), and that effect survives three independent estimators,
three eras, and a virgin hold-out — while almost every other candidate driver
we tested does not survive honest evaluation.** The methodology below exists to
earn the right to make that distinction between what held up and what didn't.

---

## 1. Research question and approach

### 1.1 The question

What actually moves the prices of the four exchange-traded precious metals —
gold (`GC=F`), silver (`SI=F`), platinum (`PL=F`), and palladium (`PA=F`) — and
how confidently can each candidate driver be established? "Drivers" is meant in
two distinct senses that the project keeps deliberately separate:

- **Predictive** — does a feature or regime label improve an honest
  out-of-sample forecast of near-term realized volatility?
- **Causal** — does a well-defined economic shock (a hawkish FOMC surprise, a
  geopolitical-risk spike, a dollar shock) move forward returns, in a sense
  that would survive placebo and cross-estimator scrutiny?

These are different questions with different bars, and a recurring theme of the
results is that a variable can matter for one and not the other. The Phase 3
news-regime taxonomy is the sharpest example: it is a useful *effect modifier*
in the causal analysis and a *net-negative* feature in the predictive one.

### 1.2 The approach, and why it is shaped the way it is

The program is deliberately **evaluation-first**. This is a research codebase,
not a deployed service; nothing depends on latency or API ergonomics, and
everything depends on not fooling ourselves. Three commitments follow from that
and recur in every phase:

1. **Leakage is prevented by construction, not caught after the fact.** Every
   feature pipeline passes a mandatory look-ahead guard (`features/leakage.py`)
   before any model sees it; a day's text strictly precedes the forward returns
   it is used to predict; the volatility target's forward window is enforced to
   be strictly future. The single most important number in Phase 1 — a mean IC
   of +0.64 on the first run — was a leakage artifact, and the guard was
   strengthened in direct response (§4.1).

2. **Walk-forward evaluation only, never a random split.** Financial series are
   non-stationary; a random split leaks the future into the past through the
   fold structure itself. Every forecasting metric in this project comes from
   an expanding-window walk-forward harness.

3. **Triangulation over any single method.** For the causal questions, three
   estimators with *different bias profiles* are pointed at identical treatment
   definitions (Phase 5). Agreement across methods that fail in different ways
   is stronger evidence than a tighter confidence interval from any one of
   them. Disagreement is treated as a finding about measurement, not swept
   aside.

The program runs in seven sequential phases (`plans/00_roadmap.md`): scoping
(0), a LightGBM volatility baseline (1), event-study local projections (2),
text/clustering (3), a multimodal transformer (4, deliberately descoped — see
§4.5), causal triangulation (5), and validation + write-up (6). Three
scenario-discovery methods come online at Phases 2, 3, and 5 and are reconciled
in Phase 5. Gold is the designated primary target throughout — deepest data,
clearest macro drivers, "the right metal to fail on first."

---

## 2. Data sources and processing

All data lands in a single local **DuckDB** file
(`data/processed/metals.duckdb`), the source of truth for the entire project;
all DB I/O goes through `data/db.py`. Every timestamp is stored in **UTC**, and
every row carries an as-of timestamp so leakage can be prevented by
construction rather than audited later. One module per source under `data/`.

### 2.1 Prices

Daily OHLCV for the four metals futures and their ETF analogues (`GLD`, `SLV`,
`PPLT`, `PALL`) via `yfinance` (`data/prices.py`). Futures are modelled
directly; ETFs are carried for cross-checking. The futures/ETF close-to-close
correlations are 0.886–0.918, below the 0.95 bar we set at scoping — almost
certainly the COMEX ~1:30 PM settle vs the 4:00 PM ETF close, not a data
error. That timing mismatch is not a nuisance to be smoothed away: it is
load-bearing for the FOMC event-study convention (§6.1). Platinum and palladium
have thin-trading gaps in 2007–2009, sidestepped by starting every model at
`2010-01-01`.

### 2.2 Macro (FRED)

Via `data/fred.py`: 10-year TIPS (`DFII10`), nominal 10Y and 2Y (`DGS10`,
`DGS2`), the broad dollar index (`DTWEXBGS`), VIX (`VIXCLS`), 10Y breakeven
inflation (`T10YIE`), WTI (`DCOILWTICO`), copper futures, and a credit spread.
The real yield used throughout is `DGS10 − T10YIE`.

**A documented data-quality save.** The credit-spread feature was originally
`BAMLH0A0HYM2` (ICE BofA HY OAS), which silently arrived truncated to only
2023-05-23 onward — a licensed third-party series that FRED can clip to a short
license window without notice. This was caught in the Phase 1 data audit
(feature input to `compute_macro_features`, missing for >85% of the training
window) and replaced with the FRBSTL-calculated `BAA10Y`. The lesson —
**prefer Fed-calculated series over licensed indices, and audit coverage on
ingest** — is now enforced by a FRED coverage check and is the first entry in
the journal's lessons-learned.

### 2.3 GPR (geopolitical risk)

The Caldara–Iacoviello daily GPR index (`data/gpr.py`), a news-text-based
measure of the *intensity* of geopolitical media coverage. That "intensity, not
crisis" character turns out to be the whole story of why the GPR event scenario
fails (§6, and findings §2).

### 2.4 FOMC surprises (Bauer–Swanson)

`data/fomc_surprises.py` ingests the Bauer–Swanson monetary-policy-surprise
series; the treatment uses `MPS_ORTH` (the component orthogonalized to the
prior news cycle, to reduce the "Fed information effect" confound). **This
series ends 2023-12-13.** That right edge is the single most consequential data
constraint in the project: it caps the FOMC event count at 35, and it means the
2024–2026 cutting cycle is entirely untested — including, fatally for Phase 6,
that **no FOMC scenario can fire in the hold-out window** (§7.3).

### 2.5 COT positioning

CFTC Commitments-of-Traders (`data/cot.py`), lagged correctly: Tuesday
positioning is released Friday afternoon, so using it as a Tuesday feature is
the classic leak in this field and is avoided. COT is ingested and lag-correct
but ends up **effectively unused** in the final models — a disclosed limitation
(§8), not a silent omission.

### 2.6 Text (GDELT GKG)

The largest data object in the project: **139.9M GDELT Global Knowledge Graph
headlines**, 2015-02-18 → 2026-06-19, pulled from BigQuery
(`data/gdelt.py` → the `headlines` table). Full provenance and the hard
constraints are in `phase3_gdelt_data_assessment.md`; the three facts that
shaped everything downstream:

1. **There is no per-metal news signal.** Metal-specific filtering leaves too
   little volume to be usable, so all text features are a *single shared
   "market" row per day*. This alone bounds what text can ever contribute to a
   metal-specific forecast.
2. **`PAGE_TITLE` exists only from 2019-09-22** (0% before). Pre-2019 rows can
   never get titles from GKG — only URL slugs. Titles were backfilled to
   ~99.5% within their valid era via a two-phase BigQuery pull (2026-07-02;
   portable parquets in `data/raw/title_backfill/`, 7.6 GB).
3. **Tone (`V2Tone`) is the trustworthy full-history signal** — 100% coverage,
   no era break. The final Phase 3 pipeline ("Option C") leans on tone and
   avoids the title-era discontinuity entirely.

One upstream hole (2017-08-29 empty in GDELT itself) and a title-era break at
2019-09 are the known text-data scars, both disclosed in the limitations.

### 2.7 Timestamp discipline

Everything is UTC; text is joined to trading days **lagged one trading day** (a
same-day join was found and fixed in adversarial review, journal 2026-07-02);
the FOMC event-study excludes the announcement-day close-to-close move because
COMEX settles before the 2 PM announcement. These conventions are tested, but
they rest on documented market structure rather than on tick data — an honest
limit (§8).

---

## 3. Feature engineering

Features are built in `features/` (`returns.py`, `macro.py`, `spreads.py`),
assembled into wide ML matrices by `assemble.py` + `loaders.py`, and every
matrix passes `features/leakage.py` before training.

### 3.1 Numeric features

- **Price-derived:** multi-horizon log returns; realized volatility at 5/20/60
  days; rolling skew/kurtosis; max drawdown; four log spreads (Au/Ag, Pt/Pd,
  Au/Cu, and one more) with their z-scores.
- **Macro:** real yield, breakevens, DXY level/change/percentile, VIX
  level/change/percentile, curve slope, GPR level/change. Macro is aligned to
  the price index and forward-filled **at feature-build time only** (never
  across the target boundary).

The full matrix is 142 features. Two lean variants exist because of the Phase 1
diagnosis (§4.1): `lean` (34 features — spreads + macro, dropping the entire
returns-and-vol block) and `lean_own` (43 — `lean` plus only the target metal's
own returns/vol).

### 3.2 The volatility target

For metal *m* and date *t*, the target is the annualized realized volatility of
one-day log returns over the **strictly forward window** `[t+h, t+h+w−1]` with
`h = 5`, `w = 20`. The last `h + w − 1 = 24` rows are NaN by construction and
that is enforced (`assert_target_strictly_future(min_nan_tail=24)`). The choice
to forecast *volatility* rather than direction is deliberate: vol is far more
forecastable than direction and gives a baseline later phases must beat. This
24-day forward window is also what forces the 44-calendar-day training embargo
in Phase 6 (§7.2).

### 3.3 Text features

From the shared daily "market" row: `V2Tone` tone means (the trusted
full-history signal), article counts, and BERTopic theme prevalences — all
joined to trading days lagged one trading day. These feed both the Phase 3
context vector and the `lgbm_sentiment` benchmark.

### 3.4 The daily context vector and regime labels

The Phase 3 scenario-discovery vector combines macro state + news tone/themes +
recent returns into a daily context vector, UMAP-reduced to 7 dimensions and
clustered with HDBSCAN (`features/context.py`, `embeddings.py`, `topics.py`).
The resulting cluster IDs, one-hot encoded and **lagged one trading day**, are
the regime features used both as a forecasting input (where they fail) and as a
causal effect modifier (where they are informative). Crucially, when regime
labels are used as a *forecasting feature* the clustering is **refit per
walk-forward fold** at each fold's training boundary (`features/regimes.py`,
leakage-tested); the full-sample clustering is used only for *descriptive*
causal conditioning, never as a forward feature.

---

## 4. Modeling architecture

Layered `data → features → models → eval`. Every model run — no matter how toy
— logs predictions and metrics to the evaluation harness (`eval/harness.py`);
walk-forward splits come from `eval/cv.py`.

### 4.1 Phase 1 — LightGBM volatility baseline

One LightGBM regressor per metal (`models/lgbm_vol.py`), retrained on every
walk-forward split (600 trees, lr 0.03, `num_leaves` 31, feature/bagging
fraction 0.85, `min_data_in_leaf` 50, early stopping at 50 rounds on a 6-month
validation slice). Expanding train ≥ 5 years, 6-month validation / test / step,
starting 2010-01-01.

Two findings from this phase shaped the rest of the project:

- **The leakage tripwire.** The first run posted mean IC ≈ +0.64 (per-split up
  to +0.94). The scoping sanity band for a 5-day vol forecast is IC ∈
  [0.05, 0.20], with "IC > 0.30 → look hard for leakage." The cause was a
  trailing-window target (`shift_target` left 15 of 20 target days observable at
  *t*). The forward-window fix (§3.2) produced honest numbers, and the leakage
  guard was strengthened to require `min_nan_tail = h + w − 1` for
  window-valued targets. **A guard that only checks the NaN tail is too weak.**
- **The returns-and-vol block is net-negative for IC.** The negative-IC
  diagnosis (`phase1_negative_ic_diagnosis.md`) showed that dropping the
  108-feature cross-ticker returns/vol block *raises* IC on gold, silver, and
  platinum — flipping silver and platinum from negative to positive. The `lean`
  set (macro + spreads) is the production baseline for Au/Ag/Pt; palladium
  stays on `full` (it is noise-of-zero on every set — it trades on its supply
  squeeze, not the macro/monetary channel this feature set captures). Honest
  gold IC lands at ≈ +0.07–0.10; the other three sit near zero. This modest
  baseline is the correct anchor: it is what everything later must beat.

### 4.2 Phase 2 — Jordà local projections (LP)

`models/lp.py`: per-horizon OLS of the cumulative h-step-ahead log return on a
treatment indicator + controls (lagged own return, lagged own vol, DXY 5-day
change, VIX level, real yield), with Newey–West HAC errors at bandwidth `h`
(Jordà's recommendation). This is the **first scenario method** and the first
estimator in the eventual triangulation. LP is not a forecasting run and does
not log to the harness.

### 4.3 Phase 3 — UMAP + HDBSCAN clustering

`models/clustering.py`: the daily context vector → UMAP(7) → HDBSCAN, trained
through 2024 and assigned forward via `approximate_predict`, producing a
7-cluster + noise taxonomy (the **second scenario method**). The pipeline is 8
ordered, resumable, month-chunked stages
(`gdelt → embed → aggregate → topics → context → cluster → analyze → label`,
`scripts/phase3_pipeline.py`). Cluster labels are LLM-assigned (Opus 4.8, a
cause-based vocabulary with honest-confidence grades). See §5.2.

### 4.4 Phase 5 — causal estimators

The **third scenario method** and the two additional triangulation legs:

- **DoubleML-IRM** (`models/causal.py`): interactive-regression-model ATE with
  LightGBM nuisance functions, 5-fold cross-fitting, 100-trial
  shuffled-treatment placebo p-values.
- **CausalForestDML** effect modification, conditioning the hawkish-FOMC
  treatment on the (lagged) Phase 3 regime one-hots.
- **Sign-restricted SVAR** (`models/svar.py`): a 4-variable daily VAR
  (Δreal-yield, DXY return, S&P return, gold return; 2010+; BIC chose lag 1),
  identified by NIW-posterior + Haar-rotation draws under sign restrictions,
  with baseline and alternative restriction sets (500 accepted draws each,
  47–63% acceptance).

### 4.5 Phase 4 — the transformer, and why it was descoped

The roadmap's flashiest phase (iTransformer/PatchTST + cross-attention text
stream) was **deliberately re-scoped to an optional numeric-only experiment and
deferred.** The reasons are principled, not logistical: (a) Phase 3
demonstrated a *text-lift null* at the primary target, so the multimodal
premise starts from a demonstrated absence of the signal it was meant to
exploit; (b) the roadmap itself flags that "a tuned LightGBM beats most
transformers in honest evaluations," and financial-ML transformer gains are
routinely look-ahead artifacts; (c) Phase 6 later confirmed that even LightGBM
loses to classical vol models on the hold-out (§7). Building an expensive
cross-attention model to chase a signal three cheaper methods say is absent
would have been motivated reasoning. The descoping is itself a methodological
result, and it means the Phase 6.6 component-ablation plan (which targeted the
transformer) is answered by design elsewhere rather than by a transformer run
(§7.5).

### 4.6 Benchmarks (Phase 6)

For honest hold-out comparison: random walk (`pred = value_t`), unconditional
historical mean, **GARCH(1,1)** (`arch` package), **VAR(2)** on the numeric
features, and a **sentiment-only** LightGBM (Phase 3 text features only). These
are the yardsticks the ML stack is measured against in §7.

---

## 5. Scenario identification — three methods, and why three

The project discovers "scenarios" three different ways and reconciles them.
Three methods, not one, because each has a different failure mode, and a
scenario that looks real under all three despite those different failure modes
is far more credible than one that survives a single estimator.

### 5.1 Event-driven (Phase 2 LP)

Five pre-registered event scenarios with explicit treatment definitions:
hawkish FOMC (top-tercile `MPS_ORTH`), dovish FOMC (bottom tercile), GPR spike
(top-5% daily change), DXY +2σ up-shock, DXY −2σ down-shock. Each is estimated
as an IRF with HAC bands and a cross-metal sign verdict. **What this method
adds:** clean, interpretable, economically-named treatments with a transparent
identifying assumption (the event is as-good-as-random conditional on
controls). **Its failure mode:** the treatment definition can be *contaminated*
— a "2σ dollar-down" event is not exogenous; it clusters in risk-off episodes
(this is exactly what happens to the DXY scenarios).

### 5.2 Unsupervised (Phase 3 clustering)

Daily market context → UMAP+HDBSCAN → a 7-cluster taxonomy, LLM-labelled with
honest-confidence grades:

| id | days | label | conf |
|---|---|---|---|
| 0 | 208 | fed-rate-hike-expectations | low |
| 1 | 445 | diffuse-macro-noise-baseline | low |
| 2 | 505 | unclear | low |
| 3 | 379 | mixed-newsflow-crude-uptrend | low |
| 4 | 103 | covid-recovery-stimulus-rebound | medium |
| 5 | 67 | trade-war-dovish-fed-tailwind | high |
| 6 | 414 | unclear | low |

**What this method adds:** it discovers regimes without being told what to look
for — the labeller independently found the 2019 trade-war/dovish-Fed gold rally
and the 2020 stimulus rebound from headlines alone, blind to any outcome data.
9 of 14 labels across both model versions were graded *low* confidence, which is
the honest-confidence prompt working as intended: the small distinctive
clusters are real and crisply labelable; the large clusters are regime mixtures
the labeller correctly marked "unclear." **Its failure mode:** GDELT world news
describes the macro *backdrop*, not metals-specific catalysts (the
no-per-metal-signal constraint), so the regimes are diffuse — which is exactly
why they fail the forecasting gate (§5.3) yet still work as causal effect
modifiers.

### 5.3 The pre-registered lift gate (Phase 3)

Before the taxonomy can be called a *predictive* scenario method, it has to
earn it. The design (`phase3_cluster_lift_design.md`, registered 2026-07-03,
**before any run**) is a paired walk-forward comparison: Arm A = Phase-1
features; Arm B = A + per-fold regime features (clustering refit per fold,
one-hots + confidence + purged target encoding). Decision rule, fixed in
advance: B beats A iff mean-RMSE improves ≥ 1.0% **and** B wins ≥ 60% of splits.

**Readout (11 folds, primary target GC=F h=5 rvol):** rel ΔRMSE −0.37% (vs the
−1.0% bar), 4/11 split wins (need 7). **B does not beat A.** Because the
decision rule was fixed before the run, this is a genuine null, not a
garden-of-forking-paths artifact. The consequence was concrete: the
corpus-scale PAGE_TITLE embedding run and its GPU provisioning were *not*
justified, and the gate was closed. (One report-only lead: the h=20 secondary
came in −2.12% / 7-of-11, which would have passed a primary-style bar — but
chasing it honestly requires a fresh pre-registered design at that horizon,
which was not run.)

### 5.4 Causal-ML (Phase 5 DoubleML) and reconciliation

The same five event treatments are re-estimated by DoubleML, and where a
scenario maps to a structural shock, checked against the SVAR. This closes the
triangle: an event scenario (LP), an ML treatment-effect estimate (DoubleML),
and a structural shock (SVAR) all pointed at the same economic mechanism. The
reconciliation logic and its master table are §6.

---

## 6. Causal estimation

### 6.1 Treatment definitions and the timing convention

Treatments are fixed in a scenario registry that predates the estimation runs
(pre-registration by construction). The FOMC event-day return convention is the
subtle piece: the cumulative LHS sums returns from `t+1` to `t+h`, **excluding**
the announcement-day close-to-close move, because COMEX futures settle (~1:30
PM ET) *before* the 2 PM FOMC statement, so the reaction lands in `r_{t+1}`. The
~0.89 futures/ETF close-to-close correlation corroborates the early-settle
timing. Threshold cutoffs (terciles, percentiles, σ bands) are computed
in-window in Phases 2/5 — a modest in-sample data-snooping cost that is
explicitly re-thresholded on pre-hold-out data in Phase 6 (§7.4).

### 6.2 The three estimators and their bias profiles

The point of triangulation is that the three legs fail differently:

- **Jordà LP** — linear controls, HAC errors. Fails if the confounding is
  nonlinear or if the treatment definition is contaminated.
- **DoubleML-IRM** — LightGBM nuisances, cross-fitting, placebo p-values. Buys
  robustness to nonlinear confounding and some efficiency, but assumes **no
  unmeasured confounders** (untestable) and inherits any contamination in the
  treatment definition.
- **Sign-restricted SVAR** — identifies structural shocks (real-yield,
  risk-aversion, USD) from comovement sign restrictions rather than from an
  event label. Fails if the sign restrictions are wrong, but is *immune to
  event-definition contamination* because it never uses the event label — which
  is exactly why it is the decisive tie-breaker on the GPR and DXY scenarios.

### 6.3 The master table and its scores

The central causal deliverable is `data/processed/scenario_master.parquet` (CSV
mirror `phase5_scenario_master.csv`): per-scenario ATEs and CIs for all four
metals at h = 1/5/20, placebo p-values, treated/control counts, and three
0–1 agreement scores — **cross-metal consistency**, **subsample stability**,
and **triangulation agreement** — plus a compact `lp/dml/svar` sign string and
an economic-interpretation note. The scores are what let the findings document
rank scenarios by robustness rather than by point-estimate size.

### 6.4 Robustness passes

Every scenario is put through: **placebo** (100 shuffled-treatment trials at
h=5), **subsample stability** (2010–14 / 2015–19 / 2020–26 with fixed treatment
definitions), **cross-metal consistency** (does silver react as predicted?),
and **effect modification** (CausalForestDML on the regime labels). The
hawkish-FOMC scenario is the only one of the five that passes the full pass;
the other four earn their place as documented measurement lessons rather than
as standalone findings (findings §2).

---

## 7. Validation and ablation (Phase 6)

The purpose of Phase 6 is to stress-test every model and scenario claim against
data that was never touched during development, and to record honestly where
the claims break.

### 7.1 Freezing the hold-out (and why it isn't 12 months)

The plan asks for a 12-month untouched hold-out. **That ideal is not
attainable here and we do not pretend it is.** Development was walk-forward from
the start, so every earlier window has appeared in some test readout. The
controlling fact is the date of the *last* model-selection readout: the
cluster-lift experiment's final fold ended **2026-01-17**, and no metric on any
later date was ever read during development. The hold-out is therefore
**2026-01-18 → 2026-05-22** (the price-data right edge, ~85 trading days).

A pipeline-by-pipeline contamination audit (in `phase6_validation.md`)
classifies each component as clean or contaminated and states the mitigation.
The honest ledger:

- **Clean:** cluster-lift folds (last test ends 2026-01-17); Phase 3 clustering
  fit (trained through 2024, hold-out rows only ever got out-of-sample
  assignments).
- **Contaminated, and re-done for Phase 6:** the Phase 5 ATE/SVAR estimation
  windows and the scenario thresholds (both were computed in-window through
  2026-05) — for 6.5 these are **re-estimated with an end bound of 2026-01-17
  and thresholds recomputed on pre-hold-out data only** (§7.4).
- **Disclosed and accepted:** Phase 3 LLM labels read hold-out-era headlines
  (descriptive only, no parameter derives from them); Phase 1's feature-set
  diagnosis plausibly saw early-2026 test windows, mitigated by scoring the
  hold-out with the **untuned default** feature set rather than the
  diagnosis-selected lean variants.
- **Structurally untestable:** FOMC treatments — Bauer–Swanson ends 2023-12, so
  no hawkish/dovish event can fire in the hold-out at all. **The anchor finding
  cannot be sign-validated OOS.** This is stated plainly rather than hidden.

### 7.2 The training embargo

The target reads returns through `t+24` trading days, so a naive train/hold-out
boundary would let training rows peek across it. Training is therefore capped at
rows ≤ **2025-12-04** (`purge_days_for(5,20) = 44` calendar days before the
boundary), with the last 180 pre-embargo days as the early-stopping validation
slice. Scorable hold-out predictions run through ~2026-04-16 (the last date
whose 24-day target window completes before the data edge). This is the same
purge logic the walk-forward CV uses, applied to the hold-out boundary.

### 7.3 Model + benchmark scoring

Runner `scripts/phase6_holdout.py`; predictions logged to the harness; metrics
in `phase6_holdout_metrics.csv`. **63 common scorable days**
(2026-01-20 → 2026-04-20). The effective sample is *tiny*: with 24-trading-day
overlapping target windows, 63 days is roughly 2–3 independent observations.
Diebold–Mariano statistics use Newey–West lag 24 and CIs come from a block
bootstrap, both of which account for the overlap — but neither can manufacture
power. Metrics (lower RMSE better; RMSE-vs-RW < 1 beats the random walk; DM t
vs `lgbm_full`, positive means the named model is *worse*):

| Model | RMSE | vs RW | IC | DM t (vs lgbm_full) | RMSE 95% CI |
|---|---|---|---|---|---|
| var2 | 0.127 | 0.69 | −0.00 | −1.18 | [0.090, 0.141] |
| garch11 | 0.131 | 0.70 | −0.29 | −0.46 | [0.071, 0.137] |
| lgbm_full | 0.140 | 0.76 | **+0.21** | — | [0.093, 0.161] |
| lgbm_regime | 0.149 | 0.80 | −0.00 | **+3.43** | [0.106, 0.172] |
| lgbm_sentiment | 0.163 | 0.88 | +0.20 | **+2.90** | [0.116, 0.192] |
| uncond_mean | 0.178 | 0.96 | 0.00 | +3.64 | [0.132, 0.208] |
| random_walk | 0.186 | 1.00 | −0.14 | +2.12 | [0.119, 0.224] |

Three readings (full discussion in findings §5): classical vol models (VAR(2),
GARCH) post the lowest RMSE, though neither *significantly* beats LightGBM
(|DM t| ≤ 1.2); regime and sentiment features **significantly hurt** OOS
(DM t +3.4 / +2.9), re-confirming the Phase 3 lift null now as an actual OOS
penalty on virgin data; and the random walk is the *worst* model, so
LightGBM's +0.21 IC is the only meaningfully positive rank signal in the table.

### 7.4 Scenario sign-validation

Runner `scripts/phase6_scenario_holdout.py`, using **pre-hold-out thresholds
and re-estimated pre-hold-out ATEs** (the §7.1 mitigation). The right question
with one year of OOS data is "do the signs hold?", not "are they significant?"

| Scenario | Fires | Sign agreement | Reading |
|---|---|---|---|
| gpr_spike | 12 | 2/4 metals | Coin-flip, exactly as its master-table scores predicted. |
| dxy_up_shock | 0 | — | No +2σ dollar rallies in the window; untestable. |
| dxy_down_shock | 5 | **4/4** | Signs agree but magnitudes are 5–90× the training ATE (silver raw diff −17.9%). The 5 fires cluster in one risk-off episode — the *contamination mechanism itself replicating OOS*, not a validated causal effect. |

The `dxy_down_shock` row is the most instructive result in the whole hold-out:
a naïve reader sees "4/4 signs agree" and calls it validated; the honest read
is that the same event-definition contamination diagnosed in Phases 2 and 5
reproduced out-of-sample. FOMC scenarios do not appear because they are
structurally untestable (§7.1).

### 7.5 Ablation — adapted

Plan step 6.6 targets the Phase 4 transformer, which was descoped (§4.5). The
*spirit* of the ablation — "what actually mattered?" — is answered by design
elsewhere and more cheaply: the cluster-lift experiment (regime features: no
lift, now an OOS penalty), the `lgbm_sentiment` benchmark (text-only: worse
than the full set), and the Phase 1 feature-set diagnosis (the returns/vol
block is net-negative). A transformer ablation becomes relevant only if the
deferred Phase 4 is ever built.

---

## 8. Limitations

The section that separates research from PR. Each failure mode gets an honest
paragraph; none is a throwaway.

**Data quality.** Text carries a 2019-09 title-era break and, more fundamental,
*no per-metal signal* — every text feature is one shared market row. COT
positioning is ingested and lag-correct but effectively unused. The HY-OAS
license truncation was caught, but similar silent truncations remain possible
in any licensed series. Timestamp discipline (UTC, one-day text lag,
futures-settle-before-FOMC) is unit-tested but rests on documented market
conventions, not on tick data — the FOMC event convention in particular would
be settled by intraday data we do not have.

**Identification.** DoubleML assumes no unmeasured confounders — untestable;
placebos and triangulation *discipline* the claims but do not *prove* them. The
SVAR conclusions are conditional on the sign restrictions (the alternative
restriction set agreed — comfort, not proof). The macro-event treatments (2σ
moves) are demonstrably endogenous — the DXY-down scenario is the worked
example — and only the FOMC treatments approach exogeneity, and only through
the orthogonalized `MPS_ORTH` surprise.

**Regime instability.** The anchor effect's *magnitude* is not stable even
though its sign is: it decayed ~2.3× from the QE era (−2.11% in 2010–14) to
−0.91% in 2020–26. Palladium's responses are regime-dependent under every
method. And post-2023 monetary policy is causally *untested* — the entire
2024–26 cutting cycle sits beyond the Bauer–Swanson right edge. A hawkish-FOMC
claim quoted without its era qualifier would be overstated.

**Modeling choices.** There is **no hyperparameter search anywhere** — every
model uses pinned defaults by design. This is a robustness virtue (nothing is
tuned to the test set) but an efficiency unknown (a tuned LightGBM might well
beat the classical baselines it currently loses to). Architectures not tried:
state-space / mixed-frequency models, the deferred transformers, and
headline-level text models (gated off by the lift null rather than shown
useless at the token level).

**External validity.** The *pipeline* — walk-forward harness, pre-registered
scenario registry, three-method triangulation, contamination-audited hold-out —
transfers to any macro asset (oil, copper, equities). The *findings* do not:
gold's real-yield elasticity and the PGM supply regime are metals-specific.
Nothing here says how equities respond to a hawkish surprise.

**Statistical power.** The binding constraint everywhere. 35 hawkish FOMC
events total (26 in the regime-covered window); three Phase 3 regimes contain
*zero* treated events, so their CATEs are pure forest extrapolation; the
top-amplitude regime cell rests on 3 events; the hold-out is ≈ 2–3 independent
vol observations and 5 DXY fires. Multiplicity across the 60-cell ATE grid is
disciplined by placebos and pre-registration but **not formally corrected** — a
family-wise correction is deferred.

---

## 9. Future work

Ordered by value-per-effort, drawn from the "things I would have liked to test"
lists in Phases 5 and 6:

1. **Extend the FOMC surprise series past 2023-12.** The single
   highest-value data extension. An updated Bauer–Swanson pull or a
   futures-implied reconstruction would make the 2024–26 cutting cycle testable
   and — critically — let the anchor finding be sign-validated on a real
   hold-out, which it currently cannot be.
2. **Macro-release surprise scenarios (CPI, NFP).** The most obvious missing
   treatment family; blocked only by consensus-forecast ingestion
   (`available: false` in the registry today).
3. **A sharper GPR treatment** (top-1% or continuous magnitude) to test the
   index-dilution hypothesis directly — the SVAR says the safe-haven *channel*
   is real, so the fault is the instrument.
4. **A powered CATE.** 26 treated events across 7 regimes cannot support
   per-regime point estimates; doubling the event count (item 1) or coarsening
   to 2–3 super-regimes would move the regime-heterogeneity result from
   suggestive to confirmatory.
5. **Metal-specific SVAR blocks**, especially for palladium and its supply
   regime, to test whether the PGM anomalies are identifiable shocks or noise.
6. **Intraday identification** for the FOMC event to separate the announcement
   from same-day confounds — daily bars cannot, and the settle-timing note only
   mitigates.
7. **Formal multiplicity control** across the 60-cell ATE grid.
8. **The optional numeric-only transformer** (Phase 4) — worth building for the
   methodological completeness of an honest transformer-vs-LightGBM comparison,
   not because any prior evidence predicts it will win.

---

*Companion: `phase6_findings.md` (results, scenario-first). Central artifacts:
`data/processed/scenario_master.parquet`, `phase6_holdout_metrics.csv`,
`phase6_scenario_holdout.csv`, and the eval harness run records. Per-phase
detail: `phase1_baseline.md`, `phase2_scenarios.md`, `phase3_writeup.md`,
`phase5_triangulation.md`, `phase6_validation.md`.*
