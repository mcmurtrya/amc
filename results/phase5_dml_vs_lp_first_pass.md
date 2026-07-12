# Phase 5 first pass: DoubleML ATEs vs Phase 2 local projections

**2026-07-11.** First real-DB run of the Phase 5 causal scaffolding
(`metals.models.causal.run()`, harness run
`6b80f2b3-ad08-4acd-a676-73ac9a44319b`; table
`data/processed/double_ml_ates.parquet`). Same treatment definitions, window
(2010+), and horizons as Phase 2's local projections — the two methods differ
in estimator (DoubleML-IRM with LightGBM nuisances + 5-fold cross-fitting vs
Jordà LP with HAC errors) and confounder handling (ML adjustment vs linear
controls). Placebos: 100 shuffled-treatment trials at h=5.

## 1. Hawkish FOMC — corroborated, almost exactly

Cumulative log return at h=5, % (LP CI is 95%; DML CI is ±1.96·SE):

| Metal | LP (Phase 2) | DML (Phase 5) | DML placebo p |
|---|---|---|---|
| Gold | −1.50 [−2.40, −0.61] | **−1.43** [−2.23, −0.64] | 0.00 |
| Silver | −2.97 [−4.92, −1.02] | **−2.95** [−4.67, −1.24] | 0.00 |
| Platinum | −1.74 [−3.07, −0.42] | **−1.68** [−2.88, −0.48] | 0.00 |
| Palladium | −1.75 [−3.85, +0.35] | −1.61 [−3.58, +0.35] | 0.09 |

Point estimates agree to within ~0.1 pp on every metal; the metal ordering
(Ag > Pt > Pd ≈ Au by |effect|) is identical; palladium is the weak link in
both (its Phase 2 subsample instability recurs here as the only placebo
p > 0.05). h=20 agrees the same way (Au: LP −1.78 vs DML −1.62; Ag −3.70 vs
−3.27; Pt −3.01 vs −2.90). DML intervals are modestly tighter than LP's HAC
intervals throughout — the ML confounder adjustment buys some efficiency.

**This is the triangulation the phase is named for**: two estimators with
different bias profiles, same answer. Hawkish-FOMC-hurts-metals is now the
project's most robust causal finding.

## 2. Dovish FOMC — asymmetry confirmed, one new wrinkle

DML matches LP's picture: small positive, mostly insignificant effects
(gold h=5 +0.57 both methods; placebo p = 0.15). The hawkish/dovish
asymmetry (−1.43 vs +0.57 on gold) survives the method change — supporting
Phase 2's tail-risk-hedge interpretation. New wrinkle: DML finds palladium
dovish h=5 **+1.91 [+0.32, +3.49], placebo p = 0.00**, where LP had an
insignificant +1.07. One significant cell in a family of 12 dovish tests is
not a finding yet; flag for the 5.7 cross-metal consistency pass.

## 3. GPR spike — methods disagree; verdict stays null

LP said null-to-positive (gold +0.05 n.s.); DML says gold h=5 **−0.89**
[−1.82, +0.04], placebo p = 0.03 — borderline *negative*, CI straddling
zero, and no other metal moves. When two reasonable estimators pull opposite
signs on a weak effect, the triangulation verdict is *fragile/null*, matching
Phase 2's diagnosis that the GPR index measures news intensity, not
flight-to-safety crises. The top-1%-cutoff follow-up remains the right next
probe if this scenario is ever revisited.

## 4. DXY shocks — the "wrong-sign" puzzle is method-invariant

DXY-up: null in both methods. DXY-down: DML reproduces LP's textbook-
violating *negative* metals response (Pd h=5 −2.37 [−4.13, −0.60], placebo
p = 0.07; Pd h=20 −2.27). Since the inversion survives a completely
different estimator, it is a **feature of the sample, not an LP artifact** —
strengthening Phase 2's contamination hypothesis (2010–23 dollar-down
episodes cluster with risk-off/liquidation events). Worth a dedicated
subsample look in step 5.8.

## Bottom line for the triangulation table (5.6)

| Scenario | Phase 2 LP | Phase 5 DML | Triangulated verdict |
|---|---|---|---|
| Hawkish FOMC | Strong | Strong (3/4 placebo p ≤ 0.00) | **Confirmed — anchor finding** |
| Dovish FOMC | Weak, asymmetric | Weak, asymmetric | Asymmetry confirmed |
| GPR spike | Null | Sign-flipped, borderline | Null/fragile |
| DXY up | Null | Null | Null |
| DXY down | Sign-inverted | Sign-inverted | Sample feature, not artifact |

Remaining for the full 5.6–5.9 pass: the sign-restricted SVAR (5.5) as the
third estimator, CATE/effect-modification (5.4 machinery, now interesting to
condition on the Phase 3 regime labels), subsample stability (5.8), and the
master table (5.9).
