# Phase 9: Real-yield surprises and a broadened scheduled-event set

Added 2026-07-18. **Status: scoped, not started.** A design briefing for review — no
code written yet, now wired into `00_roadmap.md` as Phase 9 (2026-07-18).

This phase re-specifies the Phase-5 anchor finding rather than adding features. The
hawkish-FOMC result (`results/phase5_triangulation.md`: gold −1.4% at h=5, triangulated
LP/DML/SVAR) had two honest weaknesses that are **specification failures, not noise**:
(1) it could not be validated on the Phase-6 hold-out because `MPS_ORTH` ends 2023-12-13
(≈35 usable events, no 2024–26 test), and (2) the SVAR already showed the effect runs
*through real yields* (a 1-sd real-yield innovation moves gold −0.55%), so the FOMC
surprise is a noisy proxy for the real thing. The fix is to make **the real-yield
surprise itself the treatment**, identified on scheduled-release windows, across a much
larger event set. Gold is fundamentally a zero-coupon real-rate duration asset; targeting
that mechanism directly is a cleaner and higher-powered version of the finding we already
have.

It inherits Phase 5/7 discipline verbatim: **baseline-first, causal-not-predictive,
placebo + triangulation + pre-registration**, walk-forward CV only, the
`features/leakage.py` guard before any fit, harness logging, UTC, a `journal.md` entry
per session. This is an *identification/insight* phase, not a bid to beat the Phase-1 vol
baseline.

---

## 0. The recommendation in one paragraph

Build three treatments in order of identification cleanliness and test each with the
existing triangulation stack (`models/lp.py`, `models/causal.py`, `models/svar.py`):
**(A)** decompose the FOMC surprise already in `fomc_surprises` (`ff1/ff2/ed4`) into a
Gürkaynak–Sack–Swanson–style *target* vs *path/forward-guidance* factor pair and test
which one gold loads on — pure specification improvement, no new data; **(B)** a
**same-evening Δreal-yield surprise** (`DFII2`/`DFII10`), the exact analog of the existing
Hanson–Stein `fomc_yield_surprises` ΔDGS2 table (migration 012), extended from FOMC to the
**broadened scheduled-event set** (FOMC + CPI + Employment, whose dates already live in
`configs/fomc_calendar.csv` and `configs/bls_calendar.csv`); **(C)** an
**inflation-surprise decomposition** at CPI/PCE releases that splits gold's response into
its breakeven (inflation-hedge) and real-yield (opportunity-cost) components — the two
channels offset, and which dominates is the finding. The payoff is a real-yield IRF for
gold that (i) has hundreds of event-days instead of 35, so it can be validated on
2024–26, and (ii) maps to a sharper FOMC/CPI hedge-timing rule for AMC's float.

---

## 1. Honest framing — what "better-specified" buys, and its ceiling

- **The mechanism is real yields.** Gold has no yield; its opportunity cost is the real
  rate. Empirically it trades like negative duration on 10y TIPS, and the Phase-5 SVAR
  confirmed the sign and magnitude. The FOMC event study recovers this *plus* noise from
  the target/QE components of the surprise. Targeting real yields directly removes that
  noise.
- **Power is the binding win.** FOMC ≈ 8/yr, CPI ≈ 12/yr, Employment ≈ 12/yr, 2010–2026 ⇒
  **~450–550 scheduled event-days**, versus the 35 clean `MPS_ORTH` events. That is what
  lets the finding clear a 2024–26 hold-out — the exact gap that sank the anchor finding's
  validation.
- **The ceiling is intraday.** We are daily-only (`plans/phase_8_ssl_probing.md` FACT 1).
  Clean high-frequency monetary identification (Kuttner; Gürkaynak–Sack–Swanson;
  Nakamura–Steinsson) uses a 30-minute window; we approximate with a **same-evening /
  same-day** change and must be honest that other same-day news contaminates it. Databento
  1-minute `GC/SI` + `ZQ` around the 2:00pm ET FOMC print (owned licence, per
  `results/amc_paid_data_review.md`) can tighten the *price* side for FOMC, but **intraday
  TIPS do not exist for us**, so the real-yield treatment stays daily. State this as a
  limitation, not a solved problem.
- **Honest modal outcome.** A cleaner, better-powered real-yield IRF that *validates* on
  2024–26 is the ambitious win; a defensible null ("the real-yield channel is present but
  too noisy at daily resolution to beat a random-walk hedge rule") is the acceptable,
  shippable one, mirroring the Phase-3 pre-registered null.

---

## 2. The three treatments

### 2.1 Treatment A — factor-decomposed FOMC surprise (cleanest identification, no new data)

`fomc_surprises` already stores the raw futures-based components `ff1, ff2, ed4` plus
`mps, mps_orth`. Extract a two-factor rotation (target-rate factor vs path/forward-guidance
factor) in the Gürkaynak–Sack–Swanson / Swanson (2021) style: PCA of the standardized
`(ff1, ff2, ed4)` matrix on the **train prefix only**, rotate so factor 1 loads on the
front contract (target) and factor 2 is orthogonal (path). Test which factor gold's
forward return loads on. Hypothesis: gold responds to the **path/real-rate factor**, not
the current-meeting target — a sharper, more defensible claim than "hawkish surprise."
Identification here is unambiguous (the surprise is exogenous monetary news measured in a
tight futures window); this is the anchor the noisier daily treatments are validated
against.

### 2.2 Treatment B — same-evening Δreal-yield surprise, broadened event set

Mirror the existing `fomc_yield_surprises` construction (migration 012, `delta_dgs2_bp =
(release − prev) × 100`, `pulled_at` vintage pinned) but on **real** yields: a
`realyield_surprises` table carrying `delta_dfii2_bp` and `delta_dfii10_bp` (FRED `DFII5`
is the shortest TIPS; use `DFII10` as primary, `DFII5` as the short leg — there is no
2y TIPS, so name the proxy honestly). Populate it not only on FOMC dates but on every
`release_type ∈ {CPI, EMPSIT}` date in `configs/bls_calendar.csv` and every scheduled
`fomc_calendar.csv` meeting. The treatment on event day *t* is the same-day (or
release-to-next-close) real-yield change; the outcome is gold's cumulative forward log
return. Identifying assumption: on a scheduled release day the real-yield move is
dominated by the news surprise, not by gold. Weaker than Treatment A (daily window), so it
is always reported *beside* A, never instead of it.

### 2.3 Treatment C — inflation-surprise decomposition (high power, novel split)

At CPI/PCE releases, the inflation surprise (`actual − point-in-time consensus`) moves gold
through two offsetting channels: breakevens up (inflation hedge → gold up) and expected
real yields up (opportunity cost → gold down). Decompose the release-day response of gold
into its `ΔT10YIE` (breakeven) and `ΔDFII10` (real-yield) projections and estimate which
dominates. **Consensus must be point-in-time** — the FXMacroData refutation
(`results/amc_paid_data_review.md`: retro-generated "consensus" carries a 2026 generation
stamp) is the cautionary tale; use `data/consensus.py`/`macro_consensus` first-print actuals
plus a genuine as-of consensus, and if a clean consensus is unavailable, fall back to the
model-free release-day yield decomposition (no consensus needed) and say so.

---

## 3. Estimation — reuse the triangulation stack

Per treatment, per metal (gold primary; Ag/Pt/Pd for the cross-metal consistency check):

- **Local projections** (`models/lp.py::local_projection`, `cumulative_log_returns`,
  `LPResult`): regress the h-step cumulative log return (h ∈ {1,5,20,60}) on the treatment
  + Phase-2 controls, HAC/Newey–West errors at bandwidth h. This is the Phase-2 machinery
  applied to a continuous real-yield treatment instead of an event dummy.
- **DoubleML** (`models/causal.py`): treatment = the real-yield / factor surprise,
  nuisance = the price/macro panel, 5-fold cross-fitting, 100 shuffled-treatment placebos —
  the Phase-5.2/5.3 protocol.
- **SVAR cross-check** (`models/svar.py`): the existing 4-variable sign-restricted VAR
  (Δreal-yield, DXY return, S&P return, gold return) already estimates the real-yield shock
  structurally and never uses an event label — it is the third leg and the mechanism
  anchor.
- **Triangulation + robustness** exactly as Phase 5: placebo (random event dates → zero),
  subsample stability (pre/post-2015 and a held-out **2024–26** era — the whole point),
  cross-metal sign consistency, lookback sensitivity. Log every run through
  `eval/harness.py` (`register_run`/`log_predictions`/`compare_runs`); the load-bearing
  comparison for any hedge claim is against **lagged realized vol / random walk**, not
  against zero.

---

## 4. Leakage & specification traps (each becomes an assertion/test)

1. **Real-yield vintage.** Pin the DFII vintage with `pulled_at` exactly as
   `fomc_yield_surprises` does; a backfill from the current FRED vintage is
   `is_realtime=false` and must be flagged — later-revised real yields leak.
2. **Consensus retro-generation** (Treatment C). Any "historical consensus" must prove its
   generation timestamp precedes the release (the FXMacroData trap). Prefer the
   consensus-free yield-decomposition variant when in doubt.
3. **COT Friday-close lag.** Unchanged from Phase 2 — positioning is release-dated, never
   re-lagged (`data/cot.py`).
4. **Window-target tail.** For any realized-vol secondary target, `min_nan_tail = h+w−1`
   (`features/leakage.py::assert_target_strictly_future`).
5. **Train-prefix-only factor rotation** (Treatment A) and **train-prefix-only
   standardizers** — the GSS PCA and any scaler are fitted objects; fit on `split.train_idx`,
   apply forward (`eval/cv.py::walk_forward_splits`, `check_no_leakage`).
6. **Same-day contamination.** A daily real-yield change on a release day includes non-event
   news; report the FOMC subset (tightest identification) separately from the pooled event
   set, and treat the pooled estimate as a lower bound on identification quality.
7. **Multiple testing.** Pre-declare the grid (3 treatments × 4 horizons × 4 metals ×
   {breakeven,real-yield} splits) and control FDR (BH/BY), raw and adjusted p — the Phase-5
   discipline.

---

## 5. Success / null

- **Success (ambitious):** a real-yield / path-factor IRF on gold that is sign- and
  magnitude-stable across eras **including a 2024–26 hold-out**, survives placebo and
  DML at FDR, and whose Treatment-A (clean) and Treatment-B (daily) estimates agree — plus
  a demonstrated decomposition showing gold loads on the path/real-rate factor and (at CPI)
  on the net real-yield channel. This directly upgrades the Phase-7 **FOMC/CPI hedge-timing**
  rule: a pre-FOMC/CPI real-yield-surprise nowcast tells AMC when to lighten or hedge the
  float, now with an OOS-validated effect size and a block-bootstrap CI.
- **Null (acceptable, shippable):** the real-yield channel is present but daily-resolution
  noise makes it too weak to beat a random-walk hedge rule OOS. Business call: keep the
  hedge discretionary around scheduled events, and do **not** buy intraday TIPS-adjacent data
  to chase it (a direct input to the paid-data review). A clean, pre-registered no.

---

## 6. Execution order

1. Treatment A first (no new data): factor-rotate `ff1/ff2/ed4`, LP + DML + SVAR, validate
   on 2024–26. If A doesn't survive, stop — B and C are noisier by construction.
2. Build `realyield_surprises` (Treatment B) with the vintage-pinning of migration 012;
   broaden to CPI/EMPSIT dates; re-run the stack.
3. Treatment C only if a point-in-time consensus (or the consensus-free decomposition)
   is clean.
4. Pre-register the frozen grid, primary metric per horizon, block length, and decision
   rule in `journal.md` before touching the 2024–26 hold-out. Run `ruff`/`ruff format`/
   `mypy`/`pytest` before any change is done; append a session entry.

**Grounding files:** `models/lp.py`, `models/causal.py`, `models/svar.py`,
`data/fomc_surprises.py` (`ff1/ff2/ed4/mps_orth`), `fomc_yield_surprises` (migration 012),
`configs/fomc_calendar.csv`, `configs/bls_calendar.csv`, `data/consensus.py`,
`features/leakage.py`, `eval/cv.py`, `eval/harness.py`, `results/phase5_triangulation.md`
(the finding being re-specified), `results/phase6_validation.md` (the hold-out that must
now pass).
