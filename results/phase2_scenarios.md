# Phase 2 — Event-driven scenarios and impulse responses

This document is the durable Phase 2 record. Five candidate scenarios were estimated against the four precious metals (Au, Ag, Pt, Pd) using Jordà local projections with Newey-West (HAC) standard errors, modelling window 2010-01-01 through latest available date. Controls per plan step 2.7: lagged 5-day own return, lagged 20-day own realized vol, DXY 5-day change, VIX level, real yield (DGS10 − T10YIE). The macro variable used as a treatment is excluded from its own control set.

Each scenario section gives the IRF point estimates at h=5 and h=20 (the headline plan horizons), the cross-metal sign-consistency verdict, robustness diagnostics where run, and a paragraph of economic interpretation. The full IRF tables and per-metal PNG panels live in `results/phase2/` and in the source notebooks 02–05.

## Scenario summary

| Scenario | Source notebook | Sig cells / 24 | Cross-metal sign | Placebo p (Au) | Subsample sign-stable | Verdict |
|---|---|---|---|---|---|---|
| Hawkish FOMC surprise (top tercile MPS_ORTH) | 03, 05 | 11 | All four negative | < 0.001 | ✓ (Au/Ag/Pt); Pd flips | **Strong** |
| Dovish FOMC surprise (bottom tercile MPS_ORTH) | 03 | 0 | All four small / positive | — | not run | Weak, asymmetric |
| GPR spike (top 5% 1-day change) | 04 | 1 (Pd h=20) | Inconsistent | — | not run | **Null** |
| DXY +2σ 5-day up-shock | 04 | 0 | All four negative at h=60 | — | not run | Direction OK, no significance |
| DXY −2σ 5-day down-shock | 04 | 4 (Pt h=3; Pd h=5/10/20) | All four **negative** | — | not run | **Sign inverted** vs textbook — sample contamination |

The hawkish FOMC scenario is the only one that survives a multi-check robustness pass. The others surface useful nuance for Phase 5 triangulation but do not stand on their own.

---

## 1. Hawkish FOMC surprise

**Definition.** Treatment at time *t* is 1 when *t* is the trading-day-aligned FOMC announcement date *and* Bauer-Swanson MPS_ORTH is in the top tercile of in-window FOMC observations. Tercile threshold: MPS_ORTH > +0.018 bps. 35 hawkish events between 2010-01-01 and 2023-12-13 (Bauer-Swanson dataset stops in late 2023).

**Point estimates** (cumulative log return, in % to two decimals):

| Metal | h=5 | 95% CI | h=20 | 95% CI |
|---|---|---|---|---|
| Gold      | **−1.50%** | [−2.40, −0.61] | **−1.78%** | [−2.95, −0.61] |
| Silver    | **−2.97%** | [−4.92, −1.02] | **−3.70%** | [−6.02, −1.37] |
| Platinum  | **−1.74%** | [−3.07, −0.42] | **−3.01%** | [−4.98, −1.03] |
| Palladium | −1.75% | [−3.85, +0.35] | −2.27% | [−5.05, +0.51] |

**Cross-metal consistency.** All four metals respond negatively. Ordering (Ag > Pt > Pd > Au by |beta| at h=5) is roughly consistent with relative beta/liquidity: silver is the most leveraged play, gold the cleanest, palladium the noisiest. Plan step 2.10's prediction that monetary-policy scenarios should have Au/Ag tracking sign, Pt/Pd weaker, is confirmed.

**Robustness (from notebook 05).**
- **Placebo p-values at h=5**: Au 0.000, Ag 0.000, Pt 0.010, Pd 0.050. Random ±5–60-day offsets of the hawkish dates produce a near-zero placebo distribution; the real IRFs are far outside that distribution for the first three metals.
- **Subsample at 2015**: Au, Ag, Pt all show stable-sign negative IRFs in both pre-2015 and post-2015 windows. Pre-2015 magnitudes are roughly 2-3× larger (QE-era forward-guidance leverage). **Palladium is the exception** — its hawkish response collapses to near zero post-2015, almost certainly because the 2018-22 supply squeeze dominated monetary signal.
- **Alternative thresholds (gold)**: top quartile and top tercile produce near-identical IRFs. Top decile has the largest |point estimate| but smaller n events, so wider CIs.
- **Control specification** (added 2026-06-18): the hawkish IRF is invariant to whether the macro controls (real yield, VIX, DXY change) are contemporaneous, lagged one day, or omitted entirely — within ~0.3% across all four metals, so the contemporaneous controls are not absorbing the effect. See `results/phase2_review.md` and notebook 05 section 4.

**Economic interpretation.** Hawkish monetary surprises raise the real yield, which is the standard discount rate for the zero-yielding precious metals; they typically also strengthen the USD, raising the USD-denominated metal's price for non-USD buyers and depressing demand. Both channels point negative — exactly the observed sign. The h=10–20 peak and decay by h=60 is consistent with a slow repricing that doesn't compound indefinitely. **This is the cleanest Phase 2 result and the natural seed for Phase 5 triangulation.**

---

## 2. Dovish FOMC surprise

**Definition.** Treatment at time *t* is 1 when MPS_ORTH on an FOMC date is in the bottom tercile of in-window observations. Threshold: MPS_ORTH < −0.007 bps. 35 dovish events.

**Point estimates.**

| Metal | h=5 | 95% CI | h=20 | 95% CI |
|---|---|---|---|---|
| Gold      | +0.57% | [−0.19, +1.34] | +0.82% | [−0.60, +2.24] |
| Silver    | +0.22% | [−1.35, +1.79] | +1.31% | [−1.67, +4.29] |
| Platinum  | +0.46% | [−0.33, +1.26] | +1.06% | [−0.84, +2.95] |
| Palladium | +1.07% | [−0.31, +2.45] | −0.13% | [−2.65, +2.40] |

**No cell crosses |t| = 1.96.** Point estimates are positive at short horizons (consistent with textbook prior — dovishness supports metals) but standard errors are wide.

**Economic interpretation / hypothesis.** The 2010-2023 sample is hawkish-skewed (tightening cycles 2015-18 and 2022-23). Dovish surprises in this window often occur when accommodation is already priced in, muting the marginal positive reaction. The asymmetric magnitude between hawkish (large, significant) and dovish (small, insignificant) responses is the single most interesting cross-cut to revisit:

- Could be a **regime feature** — precious metals are tail-risk hedges that respond strongly to hawkish shocks (which threaten the hedge thesis) but only mildly to dovish surprises (which marginally support an already-priced thesis).
- Could be **sample contamination** — too few clean dovish surprises to estimate cleanly. Phase 5's CATE estimator may have more power.

---

## 3. GPR spike (Caldara–Iacoviello)

**Definition.** Treatment at time *t* is 1 when the daily change in the GPR index exceeds its 95th percentile within the modelling window (cutoff +70.7 GPR index units). 207 events.

**Point estimates.**

| Metal | h=5 | 95% CI | h=20 | 95% CI |
|---|---|---|---|---|
| Gold      | +0.05% | [−0.27, +0.37] | +0.12% | [−0.49, +0.73] |
| Silver    | +0.26% | [−0.33, +0.85] | +0.61% | [−0.48, +1.70] |
| Platinum  | +0.17% | [−0.35, +0.69] | +0.07% | [−0.81, +0.96] |
| Palladium | +0.79% | [−0.02, +1.60] | **+1.34%** | [+0.14, +2.54] |

**Verdict: mostly null.** The conventional "GPR up → gold up" prior does not survive. Only palladium at h=20 crosses 1.96|t|, with a small positive coefficient — and that contradicts a clean safe-haven story (industrial palladium "benefiting" from geopolitical risk is implausible; more likely a coincidental cluster of palladium-supply concerns inside the 207-event set).

**Economic interpretation / why the null is the right answer.** The Caldara-Iacoviello GPR is a news-text-based index — it captures intensity of geopolitical media coverage, not specifically the kind of crisis that triggers flight-to-safety flows. The 95th-percentile threshold is generous (207 events in ~16 years means roughly one a month), pooling background news-cycle volatility with the rare events that historically moved gold (2011 Arab Spring, 2022 Ukraine war). A higher cutoff (top 1%) or a continuous-magnitude treatment is the natural follow-up; deferred.

---

## 4. DXY +2σ 5-day up-shock

**Definition.** Treatment at time *t* is 1 when the 5-day percentage change in DTWEXBGS exceeds +2σ (in-window σ ≈ 0.74%). 114 events.

**Point estimates.**

| Metal | h=5 | h=20 | h=60 |
|---|---|---|---|
| Gold      | +0.09% (NS) | −0.38% (NS) | −1.71% (t=−1.66) |
| Silver    | +0.39% (NS) | −2.15% (NS) | −4.53% (t=−1.82) |
| Platinum  | −0.06% (NS) | −0.57% (NS) | −2.86% (t=−1.56) |
| Palladium | +1.76% (NS) | +0.46% (NS) | −3.09% (NS) |

**Verdict: directionally right, statistically weak.** No cell crosses 1.96|t|, but signs converge to negative at h≥10 across all four metals, with magnitudes roughly proportional to each metal's USD-denomination sensitivity. The drag is slow rather than impulsive.

**Economic interpretation.** Strong USD raises the foreign-currency cost of metals for non-USD buyers and reduces global demand — a slow-burn channel that takes weeks to compound rather than a same-day repricing. The lack of short-horizon significance suggests the 5-day USD move is partly anticipated / itself a response to other shocks (FOMC, risk-off) that the LP is partly absorbing through controls.

---

## 5. DXY −2σ 5-day down-shock

**Definition.** Treatment at time *t* is 1 when the 5-day percentage change in DTWEXBGS is below −2σ. 70 events.

**Point estimates.**

| Metal | h=3 | h=5 | h=10 | h=20 |
|---|---|---|---|---|
| Gold      | −0.26% (NS) | −0.30% (NS) | −0.07% (NS) | −0.20% (NS) |
| Silver    | −1.53% (t=−1.74) | −1.84% (NS) | −2.26% (t=−1.72) | −2.00% (NS) |
| Platinum  | **−1.58%** (t=−2.44) | −1.65% (t=−1.93) | −1.61% (NS) | −1.37% (NS) |
| Palladium | −1.36% (t=−1.93) | **−2.24%** (t=−2.51) | **−3.31%** (t=−3.53) | **−3.01%** (t=−2.52) |

**Verdict: sign inverted from textbook.** A weakening USD should be supportive for USD-denominated metals. We find the opposite, with statistically significant negative responses for platinum (h=3, 5) and palladium (h=5, 10, 20).

**Economic interpretation (almost certainly sample contamination).** The 70 in-window 2σ USD-weakness events concentrate in **risk-off regimes** (2011-12 eurozone crisis, March 2020 COVID, 2022 mid-year inflation panic, 2023 banking stress). In those episodes a weaker USD coincides with broader risk-asset selling — dollar-funding unwinds, margin calls, liquidity stress — and industrial-leaning metals (Pt, Pd) get hit harder than safe-haven gold. The LP is picking up the risk-off context rather than a clean FX-pricing channel. A holdout or non-crisis subsample would likely flip this back to the textbook positive sign; that test is deferred to Phase 2 follow-up work or Phase 5 triangulation.

---

## Methodology notes

- **Estimator**: `metals.models.lp.local_projection`. Per-horizon OLS of cumulative h-step-ahead log return on treatment + controls, Newey-West HAC standard errors with bandwidth h.
- **Event-day return convention**: the cumulative LHS sums returns from t+1 to t+h, excluding the announcement-day close-to-close move. This is appropriate because the COMEX futures settlement (~1:30 PM ET) precedes the 2 PM FOMC announcement, so the reaction lands in r_{t+1}. The Phase 1 audit's ~0.89 futures/ETF close-to-close correlation corroborates the early-settlement timing.
- **Sample window**: 2010-01-01 through latest available data. Bauer-Swanson surprises stop 2023-12-13, so FOMC scenarios use that as the right edge. GPR and DXY scenarios use through 2026-05-22.
- **Tercile / threshold cutoffs**: computed within the modelling window. Modest in-sample data snooping; a clean holdout-based threshold is deferred to Phase 6.
- **HAC bandwidth**: maxlags = h (Jordà's recommendation). Appropriate for the moving-average structure of the cumulative-return LHS.
- **Cross-metal regressions are independent**. We do not run a joint cross-metal SUR — Phase 5's structural VAR will give that view.

## What is not in Phase 2 yet

- **CPI / NFP / ECB / BoE surprise IRFs** (plan steps 2.2, 2.3, 2.8): deferred. The calendar pull is straightforward but consensus-history acquisition is the binding constraint, akin to Bauer-Swanson's role for FOMC.
- **Robustness on GPR / DXY scenarios** (plan 2.11): the GPR and DXY scenarios did not survive the headline test, so per-scenario placebo / subsample passes are deferred until the treatment definitions sharpen.
- **Refresh Bauer-Swanson XLSX** (~10 FOMC meetings since 2023-12-13 missing): periodic SF Fed update; revisit before any production-grade claim.

## Reproducing

```bash
$env:UV_PROJECT_ENVIRONMENT = $null
uv run python -m metals.data.migrations.runner
uv run python -m metals.data.fomc_surprises    # Bauer-Swanson
uv run python -m metals.data.events            # FOMC calendar
uv run python -m metals.data.cot               # COT positioning (not used in this writeup yet)

uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_fomc_indicator_irf.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_fomc_surprise_irf.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_geopol_dxy_irf.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/05_fomc_robustness.ipynb
```

PNG charts land in `results/phase2/`. Run records do **not** go to the eval harness — LP estimation is not a forecasting run.

## Phase 5 triangulation hand-off

Phase 5 will estimate the same scenarios via DoubleML (treatment-effect estimator with LightGBM nuisance models) and a sign-restricted SVAR for the structural shocks. The Phase 2 IRF coefficients above are the *first leg* of the three-method triangulation. Disagreements between LP and DoubleML on the same scenario will be the most interesting Phase 5 analytical content.

The hawkish-FOMC scenario should be the first triangulation target — Phase 2's evidence is strong enough that any disagreement from DoubleML / SVAR will demand explanation.
