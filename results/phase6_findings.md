# Findings — Drivers of Precious-Metals Prices

**Phase 6.9 deliverable. 2026-07-11.** This is the substance of the project:
what actually moves gold, silver, platinum, and palladium, stated at the
confidence level the evidence supports and no higher. It leads with the
scenarios, because the scenarios are the research and the models are only the
instruments used to find them. Method detail lives in `phase6_methodology.md`;
the hold-out evaluation is `phase6_validation.md`; the causal master table is
`data/processed/scenario_master.parquet` (mirror `phase5_scenario_master.csv`).

**The one-line result.** *Hawkish monetary-policy surprises depress precious
metals — gold about −1.4% over five trading days — and that is the only one of
five candidate scenarios that survives three independent estimators, three
eras, and out-of-sample scrutiny. Everything else we tested either fails
robustness for a diagnosable measurement reason, or adds no forecastable
information at all.*

---

## 1. What reliably moves which metals

### 1.1 Hawkish FOMC surprises → all metals down (the anchor finding)

A top-tercile hawkish monetary-policy surprise (Bauer–Swanson `MPS_ORTH`)
depresses every precious metal over the following week. Point estimates for
cumulative log return at h=5, with the two event-study estimators side by side:

| Metal | LP (Phase 2) | DoubleML (Phase 5) | DML placebo p |
|---|---|---|---|
| Gold | −1.50% [−2.40, −0.61] | **−1.43%** [−2.23, −0.64] | < 0.01 |
| Silver | −2.97% [−4.92, −1.02] | **−2.95%** [−4.67, −1.24] | < 0.01 |
| Platinum | −1.74% [−3.07, −0.42] | **−1.68%** [−2.88, −0.48] | < 0.01 |
| Palladium | −1.75% [−3.85, +0.35] | −1.61% [−3.58, +0.35] | 0.09 |

The two estimators — a linear local projection with HAC errors and a
LightGBM-nuisance DoubleML treatment effect — agree to within ~0.1 pp on every
metal, with an identical metal ordering. The **third** estimator, a
sign-restricted SVAR that never uses the FOMC event label, corroborates the
mechanism structurally: a 1-sd real-yield innovation moves gold −0.55%
[−0.84, −0.28], and a top-tercile hawkish surprise is a 2–3 sd real-yield
event, so the implied −1.1% to −1.7% brackets both event-study point estimates.
Sign, ordering, and rough magnitude line up across three methods that fail in
different ways. This is what "triangulated" means here, and it is why this is
the only finding the project puts full weight behind.

The economics are textbook and both channels point the same way: a hawkish
surprise raises the real yield (the discount rate for zero-yielding metals) and
typically strengthens the dollar (raising the price for non-USD buyers). Master
table scores: triangulation 1.00, cross-metal 1.00, subsample stability 0.875.

### 1.2 The hawkish/dovish asymmetry

Dovish surprises do **not** mirror hawkish ones. They produce small, never
significant positives under both LP and DML (gold h=5 +0.57% by both methods,
placebo p = 0.15). The asymmetry — a −1.43% response to hawkish shocks against a
+0.57% response to dovish ones — is itself a finding, and a regime-level one:
precious metals behave as tail-risk hedges that react strongly to shocks
*threatening* the hedge thesis and only mildly to shocks that *marginally
support* an already-priced thesis. (One isolated wrinkle: DML finds palladium
dovish h=5 at +1.91% with placebo p ≈ 0.00 — but one significant cell in a
family of twelve dovish tests is not a finding, and it is flagged as such.)

### 1.3 Cross-metal ordering: Ag > Pt > Pd ≈ Au

Under the hawkish treatment the magnitude ordering is stable everywhere: silver
reacts hardest (most leveraged, highest beta), platinum next, gold the cleanest
and smallest, palladium the noisiest and least stable. This ordering is
preserved across both event-study estimators and all three subsample eras — a
cross-metal consistency that is itself evidence the effect is real rather than a
gold-specific artifact.

### 1.4 Regime heterogeneity (suggestive, not confirmatory)

Conditioning the hawkish effect on the Phase 3 news-regime labels
(CausalForestDML, regimes lagged one day), the gold h=5 effect is **negative in
every regime** but its amplitude varies ~10×:

| Regime (t−1) | treated n | CATE |
|---|---|---|
| fed-rate-hike-expectations | 3 | **−2.76%** |
| noise | 0 | −1.73% |
| covid-recovery-stimulus-rebound | 0 | −1.71% |
| unclear | 13 | −1.18% |
| diffuse-macro-noise-baseline | 7 | −0.87% |
| mixed-newsflow-crude-uptrend | 3 | −0.26% |

The effect is largest precisely in the regime the LLM independently labelled
`fed-rate-hike-expectations` — a label assigned from headlines alone, blind to
any outcome data. When the news-state already says "rate-hike expectations," an
actual hawkish surprise hits hardest, which is economically sensible
(tightening regimes are when policy surprises matter most for real yields). The
sign-universality is solid; the ordering is **suggestive only** — 26 treated
events, three regimes hold zero treated events (their CATEs are pure forest
extrapolation), and the top cell rests on 3 events.

---

## 2. How the three methods agreed and disagreed

Triangulation was designed so that agreement across differently-biased
estimators counts for more than any single interval, and so that *disagreement*
is treated as information. It delivered both.

**Where all three agreed — the anchor.** Hawkish→down is the clean case: LP,
DML, and SVAR agree on sign, ordering, and (scaled) magnitude. Nothing else in
the project reaches this bar.

**Where disagreement was itself the finding — GPR and DXY.** Two scenarios
split the estimators, and in both the split *diagnosed a measurement problem*:

- **GPR spike:** LP null (+0.05%), DML borderline-negative (−0.89%, CI
  straddling zero), SVAR risk-aversion shock *positive* (+0.46%, band excluding
  zero). The resolution: the Caldara–Iacoviello index measures news *intensity*,
  not flight-to-safety, and its top-5% days pool background news cycles with
  true crises. The SVAR — which identifies risk-aversion from comovements
  (yields ↓, equities ↓) rather than from the index — recovers the textbook
  safe-haven response. So the *channel is real and the instrument is the
  problem.*
- **DXY down-shock:** event studies null-to-sign-inverted (LP negative on PGMs;
  DML reproduces the inversion, so it is a *sample feature, not an LP artifact*),
  while the SVAR's pure USD shock hits gold canonically (−0.40% per sd, band
  excluding zero). The subsample pass decomposed it cleanly: gold is
  textbook-positive in 2020–26 (+1.05%) while the apparent inversion is
  PGM-concentrated (platinum −3.85%, palladium −2.80% in 2020–26). The "wrong
  sign" is COVID-era liquidation/industrial episodes contaminating the event
  definition — *the USD→gold channel is fine; the 2σ-move treatment is not
  exogenous.*

The methodological payoff: in both cases the SVAR, immune to event-definition
contamination because it never uses the event label, was the decisive
tie-breaker. Three methods were not redundant.

---

## 3. Scenarios that failed robustness (and why that is a result)

Per the master table, three of the five scenarios fail and should never be
cited as standalone findings:

| Scenario | Triangulation | Stability | Cross-metal | Verdict |
|---|---|---|---|---|
| gpr_spike | 0.33 | 0.50 | 0.50 | Null/fragile — index mismeasures flight-to-safety |
| dxy_down_shock | 0.33 | 0.50 | 0.50 | Sign-inverted — endogenous event definition |
| dxy_up_shock | 0.33 | 0.92 | 0.75 | Directionally stable, never significant |

These earn their keep as **documented measurement lessons**, not failures to
bury. Each one taught the project something transferable: that a news-intensity
index is not a crisis index; that a "2σ move" is not an exogenous shock when the
moves cluster in a single regime; and that direction-consistency without
significance is not evidence. The hold-out then confirmed all three
independently (§5).

The predictive analogue of a failed scenario is the **Phase 3 text/regime
lift null**: an unsupervised 7-cluster news-regime taxonomy that is genuinely
interpretable (it found the 2019 trade-war rally and 2020 stimulus rebound from
headlines alone) but adds **no forecastable information** over conventional
price/macro features at the primary target. Pre-registered decision rule
(≥1.0% RMSE improvement and ≥60% split wins); actual −0.37% and 4/11. Because
the rule was fixed before the run, it is a real null, not a forking-paths
artifact — and it closed the corpus-scale embedding gate. A regime taxonomy can
be real, interpretable, and causally informative (§1.4) while being useless for
forecasting. Holding those apart is one of the project's central results.

---

## 4. Cross-metal patterns — confirmed and surprising

**Confirmed.** The prior that monetary-policy scenarios hit the macro-sensitive
metals (Au, Ag) cleanly and the industrial metals (Pt, Pd) more noisily held up:
palladium is the unstable metal under *every* method (the only hawkish placebo
p > 0.05; the only metal whose hawkish sign flips post-2020; regime-dependent
everywhere). It trades on its supply squeeze and industrial demand, not the
monetary channel this feature set captures — which is also why it is the one
metal kept on the `full` feature set in the baseline while the others use
`lean`.

**Surprising.** Gold is the *only* metal with a positive predictive IC in the
Phase 1 volatility baseline; silver and platinum sit at or below zero until the
returns-and-vol feature block is *dropped* (the `lean` set flips them positive).
That a cross-ticker returns/vol block would be net-*negative* for forecast IC —
that adding own-series vol clustering back in (`lean_own`) takes silver and
platinum negative again — was counterintuitive and is the reason the production
baseline is deliberately lean. More signal came from *removing* features than
from adding them.

---

## 5. What the hold-out changed

The 63-day virgin hold-out (2026-01-20 → 2026-04-20; ≈ 2–3 independent
observations given 24-day overlapping windows) delivered three results that
sharpen rather than contradict the story:

1. **Classical vol models beat the ML stack.** VAR(2) (RMSE 0.127) and
   GARCH(1,1) (0.131) posted the lowest errors, below `lgbm_full` (0.140) —
   though neither *significantly* (|DM t| ≤ 1.2). The roadmap's prior ("a tuned
   LightGBM beats most transformers") extends one rung further down on this
   window: mean-reverting classical baselines beat LightGBM too. This vindicates
   the decision to descope the transformer.
2. **Regime and sentiment features hurt OOS, significantly.** `lgbm_regime`
   (DM t +3.4) and `lgbm_sentiment` (DM t +2.9) are significantly *worse* than
   the plain feature set. The Phase 3 forecasting null re-confirms on virgin
   data — now as an actual out-of-sample penalty, not merely an absence of lift.
3. **The random walk is the worst model.** Trailing vol was a bad level anchor
   in a window where volatility shifted, so everything mean-reverting won;
   LightGBM's +0.21 IC is the only meaningfully positive rank signal in the
   table.

And the sharpest single lesson: **`dxy_down_shock` fired 5 times and all 4
metals agreed in sign — but at 5–90× the training magnitude** (silver raw diff
−17.9%), because the fires clustered in one risk-off episode. That is the
contamination mechanism from §2 *replicating out-of-sample* — the strongest
possible confirmation that the scenario measures a regime, not a causal effect.
A reader who stopped at "4/4 signs agree" would draw exactly the wrong
conclusion. The anchor FOMC finding, by contrast, **could not be tested at
all** on this hold-out (Bauer–Swanson ends 2023-12) — an honest gap, stated
plainly, not a passed test.

---

## 6. The three finding-level paragraphs

**Most counterintuitive.** More news data did nothing — and structured
market-regime knowledge actively hurt the forecast. A 139.9M-headline GDELT
corpus, embedded, topic-modelled, and clustered into an interpretable regime
taxonomy, produced *zero* predictive lift over plain price/macro features at the
primary target, and when those regime labels were added as features they made
the out-of-sample forecast **significantly worse** (DM t +3.4). The intuition
that "more and richer data must help" is simply false here: the news signal is
diffuse (no per-metal content), whatever regime information exists is already
spanned by the macro features the clusters are partly built from, and the extra
parameters bought only overfitting. The same taxonomy is genuinely useful as a
causal effect modifier — so this is not "text is worthless," it is "predictive
value and causal/descriptive value are different things," which is the more
uncomfortable and more important lesson.

**Most robust.** Hawkish monetary surprises depress precious metals. Three
estimators with different bias profiles (local projection, DoubleML, sign-
restricted SVAR) agree on sign, cross-metal ordering, and scaled magnitude
(gold ≈ −1.4% at h=5, placebo p < 0.01); the sign is stable across all three
eras 2010–2026 on gold, silver, and platinum; and the effect is negative in
every one of seven news regimes. It is grounded in two reinforcing textbook
channels (real-yield discounting and USD appreciation). Nothing else in the
project comes close to this weight of corroboration — and its one honest
weakness is disclosed, not hidden: it could not be re-validated on the hold-out
because the surprise series ends in 2023.

**Most fragile-seeming.** The regime-heterogeneity result — that the hawkish
effect is largest in the LLM-labelled `fed-rate-hike-expectations` regime — is
the most seductive and the least powered finding in the project. The story is
elegant (the news-state that says "hikes are coming" is where an actual hike
surprise bites hardest) and the sign-universality across regimes is solid. But
the amplitude *ordering* rests on 26 treated events spread across seven regimes,
three of which contain zero treated events and get their CATEs by pure forest
extrapolation, with the headline cell carrying only 3 events. It is reported as
*suggestive* on purpose; a family-wise correction is not applied; and it would
take doubling the FOMC event count (which requires extending the surprise series
past 2023) or coarsening to 2–3 super-regimes to make it confirmatory. Quoted
without those caveats it would be the project's most overstated claim.

---

## 7. Open questions

1. **Does the anchor finding hold in the 2024–26 cutting cycle?** Untested —
   the surprise series ends 2023-12. Extending it (an updated Bauer–Swanson pull
   or a futures-implied reconstruction) is the highest-value next step and the
   only way to give the anchor finding a real out-of-sample test.
2. **Do macro-release surprises (CPI, NFP) move metals like FOMC surprises do?**
   The most obvious missing treatment family; blocked only on consensus-forecast
   ingestion.
3. **Is the safe-haven channel recoverable with a sharper GPR treatment?** The
   SVAR says the channel is real; a top-1% or continuous-magnitude GPR treatment
   would test whether the index-dilution hypothesis is the whole explanation.
4. **Is palladium's instability identifiable or just noise?** A metal-specific
   SVAR block (especially for the PGM supply regime) would tell us whether the
   palladium anomalies are structural shocks or measurement noise.
5. **Would a tuned model or a state-space model beat the classical baselines?**
   No hyperparameter search was run anywhere by design; the hold-out says
   classical vol models currently win, but "currently" is doing work — a tuned
   LightGBM or a mixed-frequency model was never tried.

---

*Central artifacts: `data/processed/scenario_master.parquet`,
`phase6_holdout_metrics.csv`, `phase6_scenario_holdout.csv`. Method and full
limitations: `phase6_methodology.md`. Per-phase detail: `phase2_scenarios.md`,
`phase3_writeup.md`, `phase5_triangulation.md`, `phase6_validation.md`.*
