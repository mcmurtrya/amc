# Phase 5 second pass: SVAR (5.5) + regime-conditioned CATE (5.4)

**2026-07-11.** Follows `phase5_dml_vs_lp_first_pass.md`. Two new runs:
the sign-restricted SVAR (third estimator for the triangulation) and the
hawkish-FOMC CATE conditioned on the Phase 3 regime labels.

## 1. Sign-restricted SVAR — the third estimator agrees

`metals.models.svar`: 4-variable daily VAR (Δreal-yield, DXY ret, S&P ret,
gold ret, 2010+), BIC chose lag 1, NIW-posterior + Haar-rotation
identification, 500 accepted draws per restriction set (acceptance 47–63%).
Cumulative gold IRF, median [16/84], both restriction sets nearly identical
(robustness pass — baseline shown):

| Shock (1 sd) | h=5 gold response | Band |
|---|---|---|
| Real-yield ↑ | **−0.55%** | [−0.84, −0.28] |
| Risk-aversion (flight to safety) | **+0.46%** | [+0.17, +0.77] |
| USD ↑ | **−0.40%** | [−0.69, −0.10] |

Three readings:

1. **Monetary channel corroborated a third way.** A 1-sd daily real-yield
   innovation depresses gold ~0.55%. A top-tercile hawkish FOMC surprise is
   a multiple-sd real-yield event; at ~2–3 sd the implied response (−1.1% to
   −1.7%) brackets the DML ATE (−1.43%) and LP IRF (−1.50%). Sign, ordering
   (bands exclude zero), and rough magnitude all line up. **LP, DML, and
   SVAR now agree on the anchor finding.**
2. **The safe-haven channel exists after all.** The GPR event study was null,
   but the SVAR's risk-aversion shock (identified from comovements: yields ↓,
   equities ↓) moves gold +0.46% with a band excluding zero — supporting
   Phase 2's diagnosis that the GPR *index* mismeasures flight-to-safety
   rather than the channel being absent.
3. **A pure USD shock hurts gold, per textbook.** The identified USD shock
   is canonically negative for gold — reinforcing that the DXY *event
   study's* sign inversion is contamination of the event definition (mixed
   shock origins), not a broken USD channel.

IRFs are essentially flat after h≈5 (lag-1 daily VAR: impact-dominated);
h=20/60 add little. Run registered in the harness; bands in
`phase5_svar_irfs.csv`.

## 2. Regime-conditioned CATE — sign-universal, amplitude varies ~10×

`scripts/phase5_cate_regimes.py`: CausalForestDML, hawkish-FOMC treatment
(same 2010-window thresholds as the ATE run), moderators = Phase 3 regime
one-hots **lagged one trading day** (strictly pre-treatment), sample =
regime coverage 2015-02 → 2026-05 (26 of the 35 hawkish events). Run
`51dd25cb-6405-4a32-8c84-f0a43b874872`; table `phase5_cate_regimes.csv`.

Gold h=5 CATE by regime (piecewise-constant by construction; "unclear"
aggregates clusters 2 and 6, hence its nonzero within-group std):

| Regime (t−1) | n | treated | CATE |
|---|---|---|---|
| fed-rate-hike-expectations | 208 | 3 | **−2.76%** |
| trade-war-dovish-fed-tailwind | 74 | 0 | −2.76% |
| noise | 125 | 0 | −1.73% |
| covid-recovery-stimulus-rebound | 117 | 0 | −1.71% |
| unclear | 1,047 | 13 | −1.18% |
| diffuse-macro-noise-baseline | 679 | 7 | −0.87% |
| mixed-newsflow-crude-uptrend | 462 | 3 | −0.26% |

Overall CATE mean −1.16% (vs −1.43% ATE on the full 2010+ window — same
ballpark on the restricted sample). Silver orders the same way at the top
(fed-rate-hike-expectations −2.68%) with two weak-cell sign flips.

**The headline read:** the hawkish effect on gold is **negative in every
regime** (100% of days), but its amplitude varies ~10×, and it is largest
in the regime the LLM labelled `fed-rate-hike-expectations` — a label
assigned from headlines alone, blind to any outcome data. The taxonomy that
failed the *forecasting* gate is informative as an *effect modifier*: when
the news-state already says "rate-hike expectations," an actual hawkish
surprise hits hardest. Economically sensible (tightening regimes are when
policy surprises are most consequential for real yields) and exactly the
kind of heterogeneity Phase 5 exists to find.

**Caveats (read before quoting):** 26 treated events total; three regimes
contain **zero** treated events, so their CATEs are pure forest
extrapolation; the top cell has only 3 events. Per the module docstring,
the defensible claim is sign-stability plus a *suggestive* amplification
ordering — not per-regime point estimates. The regime clustering is also
full-sample-trained (descriptive conditioning, fine here; never a
forecasting feature).

## Updated triangulation picture

| Channel | LP (Phase 2) | DML (5.2–5.3) | SVAR (5.5) | Verdict |
|---|---|---|---|---|
| Hawkish/monetary → metals ↓ | Strong | Strong | Consistent (sign + scaled magnitude) | **Confirmed, three methods** |
| Flight-to-safety → gold ↑ | Null (GPR events) | Fragile/sign-flip | **Present** (+0.46, band > 0) | Channel real; GPR index mismeasures it |
| USD ↑ → gold ↓ | Null / sign-inverted events | Null | **Present** (−0.40, band < 0) | Channel real; event definition contaminated |
| Regime heterogeneity | — | — | — | Sign-universal; amplified in rate-hike-expectations regime (suggestive) |

Remaining for the phase: 5.7 cross-metal consistency (formalize), 5.8
subsample stability (the DXY puzzle and the CATE ordering are the targets),
5.9 master table, 5.10 write-up.
