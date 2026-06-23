# Phase 2 review — events + local projections

A code/methodology review of the Phase 2 work (Jordà local projections for
event-driven scenarios). Conducted 2026-06-18 against the current data
(post-`BAA10Y` refresh, compacted DB). Companion to `results/phase2_scenarios.md`,
which remains the primary results record.

## Verdict

Phase 2 is the most solid part of the project. The estimator is correct and
well-tested, the headline hawkish-FOMC result reproduces exactly from raw data,
and the two weaknesses probed in this review both held up under direct testing.
It is a sound first leg for the Phase 5 triangulation.

## 1. Estimator (`src/metals/models/lp.py`) — correct

- `cumulative_log_returns(r, h)` builds the strictly-forward window
  `r_{t+1} + ... + r_{t+h}` (rolling-sum then `shift(-h)`); the last `h` rows are
  NaN by construction. No contemporaneous return leaks into the LHS.
- Per-horizon OLS of the cumulative return on `treatment + controls`, Newey–West
  HAC SEs with `maxlags = h`. This is textbook Jordà (2005).
- Treatment and controls enter at `t`; outcome is strictly future — no look-ahead
  in the estimator itself.
- Tests (`tests/test_models_lp.py`) include a known-IRF recovery test, a
  zero-effect CI test, a control-sign test, and index-mismatch guards.

## 2. Independent reproduction of the headline result

Re-running the gold hawkish-FOMC IRF from raw data (MPS_ORTH top tercile,
35 events, 2010–2023, controls as in notebook 03) reproduces the
`phase2_scenarios.md` table to the basis point:

| Horizon | beta (this review) | beta (writeup) | t-stat |
|---|---:|---:|---:|
| h=1  | -0.71% | —      | -2.62 |
| h=5  | -1.50% | -1.50% | -3.28 |
| h=20 | -1.78% | -1.78% | -2.97 |

The pipeline is faithful and the result is real.

## 3. New result — robustness to control specification

The own-return / own-vol controls are correctly lagged (`shift(1)`), but the
three macro controls (real yield, VIX, DXY 5-day change) are measured *on the
FOMC day*. For an FOMC treatment those are transmission channels, so a natural
worry is that they absorb part of the effect (mediator / "bad control" bias). We
tested gold three ways:

| Spec | h=1 | h=5 | h=20 |
|---|---:|---:|---:|
| (a) contemporaneous macro controls (as-is) | -0.71% | -1.50% | -1.78% |
| (c) macro controls lagged 1 day            | -0.71% | -1.51% | -1.77% |
| (b) no controls at all                     | -0.70% | -1.45% | -1.57% |

The estimate is essentially invariant to the control specification (≤0.2% at any
horizon). The contemporaneous controls are *not* materially attenuating the
hawkish effect, which **strengthens** the result. No code change needed; worth a
one-line note in the writeup.

## 4. Concerns probed and dismissed

- **Unscheduled-meeting contamination.** Suspected that intermeeting emergency
  actions (e.g. March 2020) might pollute the dovish tercile and explain the
  dovish/hawkish asymmetry. Checked the data: Bauer–Swanson has no clean surprise
  for March 2020, and there is exactly **one** unscheduled event in-window
  (2019-10-11, mild hawkish). The asymmetry is therefore genuine, not an artifact.
  (Hygiene: still worth filtering `is_unscheduled` explicitly; impact is one
  event out of 35.)
- **Control mediation.** See §3 — not material.

## 5. Open items and recommendations

- **COT release offset — FIXED in this pass.** `cot.py` previously used a fixed
  `+3 day` Tuesday→Friday map, which is wrong on federal-holiday weeks (the CFTC
  delays publication). Replaced with a holiday-aware `release_date()` that pushes
  the nominal Friday one business day per in-week federal holiday and snaps to the
  next business day (Thanksgiving / July-4 weeks now release the following
  Monday). Erring later is leakage-safe. Two tests added. NOTE: COT is ingested
  but not yet consumed in the LP analysis, so this is preventive.
- **In-sample tercile thresholds** (data snooping). Already flagged in the
  writeup; thresholds are computed in-window. Deferred to a holdout-based cut in
  Phase 6. Low impact on sign, possible mild impact on magnitude.
- **Bauer–Swanson stops 2023-12.** ~10 recent FOMC meetings are missing. Refresh
  the SF Fed XLSX before any production-grade claim.
- **Event-day-return convention — supported by your own audit.** The IRF
  cumulates from `t+1`, excluding the FOMC-day close-to-close move. That is
  correct only if the GC=F daily close precedes the 2 PM announcement. The Phase 1
  data audit found futures/ETF close-to-close correlations of ~0.89 (not ~0.97) —
  the signature of a COMEX ~1:30 PM settlement vs ETF 4 PM close — so the 2 PM
  FOMC lands in `r_{t+1}` as assumed. Consistent; worth a one-line note since the
  IRF magnitude hinges on it.
- **HAC bandwidth.** `maxlags = h` is Jordà's own choice and defensible; the
  Lazarus–Lewis–Stock rule of thumb would nudge it to ~1.3h. SEs may be a hair
  optimistic at long horizons. Not worth changing unless a result is borderline.
- **Multiple testing.** 5 scenarios × 4 metals × 6 horizons with no formal
  correction; the informal "sig cells / 24" counting is fine given the placebo
  test backstops the one scenario (hawkish) that is actually claimed. Good hygiene
  overall — the weak/null scenarios are honestly labeled.

## 6. Phase 5 readiness

The hawkish-FOMC scenario is the right first triangulation target: it survived
independent reproduction plus a control-robustness check it had not previously
been put through. Any disagreement from DoubleML / SVAR in Phase 5 should be
treated as informative rather than as a reason to doubt the LP leg.
