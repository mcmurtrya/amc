# Research backlog — candidate paths for AMC

Standing, curated list of research directions that are **not yet phases**. Created
2026-07-21 from three brainstorming passes (ML-with-new-data, business/operations,
ML round two). Companion to `00_roadmap.md`: the roadmap records what is *committed*,
this records what is *considered*. Items graduate into a phase plan, or get killed
with a reason recorded here.

**Nothing in this file has been run.** Effect sizes and payoffs described below are
hypotheses about where value might be, not findings.

## How to read the status column

| Status | Meaning |
|---|---|
| **Ready** | Buildable today with data in hand |
| **Ledger** | Blocked on AMC's transaction export (see `plans/phase_7_amc_program.md` §7.1) |
| **Purchase** | Blocked on a paid data acquisition |
| **Licence** | Blocked on a ToU/licence clearance (2026-07-16 audit) |
| **Prereq** | Blocked on internal work, named inline |

## Standing rule for anything that re-opens a null

Items B1, G1 and G3 re-test conclusions Phase 6 already reached, on legitimately
different tasks (distributional vs point, pooled vs per-metal, decision loss vs RMSE).
That is defensible research, and it is also exactly what motivated reasoning looks like
from outside. **Each must have its pass mark written down before it runs**, the way the
Phase 3 lift gate was (`results/phase3_writeup.md` §pre-registration). The programme's
most credible asset is that its nulls were pre-registered; do not spend it.

---

## A. Making the anchor finding usable

The hawkish-FOMC result (gold ≈ −1.4%/wk, triangulated) is the one durable finding.
Its weaknesses are all about *coverage and currency*, not validity.

**A1. Extend the treatment through the cutting cycle — Ready.**
Bauer–Swanson `mps_orth` ends 2023-12-13 (354 events); the ΔDGS2 proxy in
`fomc_yield_surprises` runs to 2026-04-29 (172 events). They **overlap on 136 events**,
with **20 events after Bauer–Swanson ends**. Learn the mapping on the overlap, impute
forward, and the 2024–26 cycle becomes testable for the first time — closing the gap the
write-ups call the single highest-value one to close.
*Method:* small-N supervised regression (Gaussian process / Bayesian ridge — 136 points is
not deep-learning territory). The methodological substance is downstream: the imputed
treatment is an **estimated regressor**, so its uncertainty must propagate into the local
projection as errors-in-variables or the confidence bands will be spuriously tight.
*Honest prior:* live either way. The effect may have decayed to nothing, which is itself
the answer AMC needs. Overlaps Phase 9 treatment (B).

**A2. Intraday event identification — Purchase (~$1/mo + one-time backfill).**
COMEX settles ~1:30 PM ET; the FOMC statement lands at 2:00 PM. **The daily data
structurally cannot see the announcement reaction** — Phase 2 correctly pushes it into
`r_{t+1}`, bundled with the overnight session and everything else
(`results/phase2_scenarios.md:128`). Databento intraday would isolate the 2:00–2:30 PM
window and sharpen the estimate on *the same 35 events* — the only way to buy precision
when you cannot buy more events.

**A3. Change-point detection on the effect's decay — Ready.**
The effect decayed ~2.3× since the QE era, currently handled with fixed era splits. A
Bayesian change-point or time-varying-parameter model would learn *where* it actually
broke and whether it is still moving — which is what determines whether the finding is
still usable today rather than merely true historically.

**A4. Pre-FOMC intake policy — Ready.**
Eight scheduled meetings a year, known dates, a measured adverse move. Nobody has costed
the obvious response: widen spreads or pause intake in the window. Straight cost-benefit —
deals lost against losses avoided. Needs no new data and is the most direct translation of
the flagship finding into something executable at the counter.

## B. The spread floor and tail risk

**B1. Conformal prediction for the floor — Ready. (Pre-register.)**
The floor's job is "breach no more than q% of the time." Conformal prediction gives
**distribution-free finite-sample coverage** — a guarantee rather than an estimate — and
block/weighted variants handle the exchangeability violation from autocorrelation. This
fits the business requirement more exactly than either quantile regression or an EVT tail
fit, both of which remain reasonable comparison arms. Replaces the flagged
`tail=normal_approx` placeholder (`k=1.645`).
*Why ML can win here where it lost before:* Phase 6's null was about conditional **means**.
Distributional calibration is a different task.

**B2. Simulation-based stress testing — Ready.**
Bounded at ~40–50 effective regimes, manufacturing power may beat further modelling. A
regime-switching / block-bootstrap simulator lets breach rates be tested against scenarios
history never produced.

**B3. Does float correlate with volatility? — Ledger.**
The cushion assumes holding period is independent of price moves. If metal sits *longer*
when prices fall — refiners backed up, buyers scarce, everyone waiting for a bounce — then
float and vol rise together and **the current floor understates risk in the dangerous
direction**. Cheap to test once the ledger lands; would change the formula's structure, not
just its parameters.

**B4. Price the hedge — Ready.**
Never actually costed. The crux is granularity: a 100 oz COMEX gold contract is enormous
relative to a small dealer's float, so the real question is whether micro contracts make
hedging feasible at all, and what basis risk sits between the hedge instrument and the
actual refiner payable. "Don't hedge, widen the spread instead" would be a valuable answer
to have in writing.

## C. Operations and inventory

**C1. Optimal refining batch timing — Ledger.**
Float enters the floor as √(float), so it is the dominant term — and it is a **decision**,
not a constant. Refiner fixed fees and lot minimums pull toward batching; price risk pulls
toward shipping now. Shortening float tightens the floor, which lets AMC bid more
aggressively. *Method:* classical threshold policy first; least-squares Monte Carlo
(Longstaff–Schwartz) for the continuation value if state dependence proves rich.

**C2. Liquidity-at-risk — Ready (sharpens with ledger).**
Scrap inflow rises with prices, so cash needed to buy inventory rises with spot: working
capital is squeezed hardest exactly when the opportunity is best. A distinct risk from price
risk, never modelled, and for a small dealer it may bind first.

**C3. Learned payable curve — Ledger / refiner settlements.**
Regress dollars-out on fine-content-in by metal, lot-size band and refiner, with lot-size
nonlinearity. Replaces the fixed-haircut exit placeholder (Au 2 / Ag 5 / Pt 5 / Pd 8%).

**C4. Float survival model — Ledger.**
Kaplan–Meier before anything neural. Open lots are **censored, not missing**; the current
naive mean excludes them and therefore reads float too short and the floor too tight — the
unsafe direction.

## D. Risk channels nobody has modelled

**D1. Assay and fineness risk — Ledger.**
AMC pays on *assayed* fine content. For small lots the dollar variance from assay error and
karat misgrading may exceed the variance from price movement over a two-week float — in
which case the programme has been optimising the wrong risk. Nothing in the codebase has
ever looked at this.
*Method:* predict fine ounces from gross weight, karat class and item type; flag outliers
with **conformal** intervals so the flags carry an actual error rate. Doubles as data-entry
QC and as a detector for systematic under/over-assay.

**D2. Is coin premium a hedge or an amplifier? — Ledger (premium panel is Licence-blocked).**
Premium over melt is its own price with its own dynamics. If premiums spike when spot
crashes (retail panic buying), coin inventory is genuinely *less* risky than melt-equivalent
scrap and deserves a different spread schedule. If they fall together, risk compounds.
Determines whether coin and scrap can share a risk model at all.

## E. Demand side and pricing

**E1. A real pricing experiment — Ledger + deliberate design.**
The **only** idea in this backlog that produces *experimental* rather than observational
evidence, and AMC controls the treatment. Till data already records offers made vs accepted;
varying the spread by small amounts and measuring acceptance gives a causal estimate of
AMC's own demand curve — the strongest evidence class available anywhere in this programme.
*ML form:* contextual bandit mapping state (spot, vol, float, weekday, FOMC proximity) to a
spread offer, reward = acceptance × margin. **Critical:** without deliberate randomisation
in the logged decisions, offline policy evaluation is confounded by whatever set the quote.
The experiment design *is* the ML project, not a precursor to it.

**E2. Scrap-inflow nowcasting — Ledger.**
Count/Poisson model over calendar, price level and momentum, targeting walk-ins and lots.
Note the natural leading indicator (search interest) is currently **Licence**-blocked.

**E3. Seasonality of scrap flow and coin demand — Ledger.**
Tax season, post-holiday liquidation, gift-giving cycles. Unglamorous, cheap, directly
useful for staffing and cash planning.

## F. Text, and creating information the corpus lacks

**F1. Run the LLM-annotator pilot — Ready. Highest readiness-to-value in the backlog.**
`src/metals/annotate/` is **built, tested and never run** (no results file exists): frozen
date-blind prompt, stratified 80-day sampler, Batch-API runner, and a **pre-registered
pass/fail report card**, driven by `scripts/annotate_pilot.py`. Cost $32.66 (Opus batch,
80-day pilot ×2 variants); full 1,678-day run $68.51–$342.54 depending on tier
(schema-v3 re-estimate, 2026-07-21 — supersedes the ~$30/$63–314 v2 figures this entry
originally quoted).
*Why it matters beyond its price:* `results/amc_paid_data_review.md` concluded that **no
purchase at any price** yields per-metal news signal, and that the only lever which adds
per-metal information is a *method* — this one. It is also the input to Phase 10's PGM
supply-event ledger. Expected result is itself a null; that is fine and pre-registered.

**F2. Text → market-identified crisis instrument — Ready.**
GPR was diagnosed as measuring news *intensity*, not flight-to-safety, while the SVAR
recovers the real channel from market comovement. Learn a mapping from each day's headlines
to the SVAR's risk-aversion shock — distilling a structurally identified latent onto text.
**This is not the experiment that already failed:** Phase 3/6 tested text → *forward*
returns/vol; this is text → *contemporaneous identified shock*, a measurement task, and
Phase 6 explicitly separated predictive from causal/descriptive value. Then test whether the
learned index produces a triangulated safe-haven IRF where GPR gave a coin flip.
*Risk:* high null risk, and the target is itself estimated — same errors-in-variables care
as A1.

**F3. Phase 8 Stage-A walk-forward — Prereq: `daily_text_features.mean_embedding` is
all-NULL and must be rematerialised.** Library layer is built (`features/ssl_views.py`,
`models/factor_ssl.py`, `eval/probes.py`); the driver, harness wiring and pre-registration
are not.

## G. Methods for the small-N problem

**G1. Pooled panel volatility model — Ready. (Pre-register.)**
Phase 6 tested *per-metal* ML against classical baselines. **Pooled ML against per-metal
GARCH is untested**, and pooling across four metals plus copper and oil roughly quadruples
the data behind shared dynamics. Metal embeddings or fixed effects carry the differences.
Cheap, and a legitimate re-opening of the ML-vs-classical question on different grounds.

**G2. Hierarchical Bayesian pooling across metals × horizons × eras — Ready.
(Pre-register.)**
Estimating 35 events separately per metal wastes information. Partial pooling — a multi-task
GP over the event-response surface — shrinks the noisy palladium estimate toward the complex
while still letting it differ. Palladium is currently the weakest link in the anchor finding;
this is the right tool for that regime.

**G3. Decision-loss retraining — Ready. (Pre-register.)**
Retrain existing vol models against the spread floor's asymmetric business loss (cost of a
breach vs margin given up) instead of RMSE. Phase 6 may have shown ML loses on symmetric
error while winning under the loss AMC actually faces. Very cheap — reuses the harness and
the existing models.

## H. Blocked on licence

**H1. PGM and rhodium tail calibration + supply-shock study (Phase 10) — Licence.**
169,920 rows (1992→2026, including rhodium) sit quarantined. Rhodium has **no exchange
price**, making these the only possible tail calibration for AMC's fattest-tail exposure —
and palladium is precisely where the monetary channel broke. Clearing this licence is likely
the highest-leverage non-research action available.

---

## Suggested order

1. **F1** — built, cheap, pre-registered, and the only item that *adds information* rather
   than reprocessing what exists.
2. **A1** — converts the flagship finding from untestable to tested.
3. **B1** — a coverage guarantee is what the counter decision actually needs.
4. **A4** — near-zero cost, uses the finding already trusted.
5. **G1 / G3** — cheap, honest re-tests.

**The moment the ledger lands:** B3 and C4 first (both concern whether the floor is wrong in
the *unsafe* direction), then C1, D1, E1.
