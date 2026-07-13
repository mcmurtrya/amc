# Phase 7: The AMC program — dealer decision support

Added 2026-07-12. Phases 0–6 answered *what moves metals prices*; Phase 7 turns
those answers into operating decisions for **AMC Company**, the business the
research serves: a small dealer that buys scrap Au/Ag/Pt/Pd (assaying fine
content) and buys/sells gold coin and specie — structurally long physical metal
over a days-to-weeks inventory float.

Three source documents govern this phase:

- `results/phase5_amc_business_implications.pdf` — the Phase 5 findings translated
  into AMC's hedging, spread, and inventory decisions (adversarially verified).
- `results/amc_data_acquisition_program.md` (+ `.pdf`) — the start-now
  data-acquisition program (step 7.1 below, expanded).
- `results/amc_paid_data_review.md` (+ `.pdf`) — the verified paid-data buy/skip
  review (2026-07-12): Databento CME backfill (~$0–125, unblocks 7.2/7.6 labels
  now) and Greysheet ($299/yr, wholesale coin benchmark) are the only buys; JM
  PGM prices (incl. rhodium, free) flagged as candidate collector 6; sentiment
  feeds and enterprise assessments ruled out with evidence.

The portfolio below came out of a structured multi-agent brainstorm
(2026-07-12): 25 raw ideas → 10 distinct projects → adversarial feasibility
review. Its governing lesson matched the roadmap's own warning: in 8 of 10
projects a transformer was decoration. **Every project here is baseline-first;
any transformer runs as a gated bake-off behind a pre-registered kill
criterion.** All existing conventions hold (walk-forward CV only, leakage guard,
harness logging, UTC, journal entries).

**Amended 2026-07-12 (evening):** the paid-data review's consequences folded in
after a second adversarial pass (6 proposal lenses, per-item verification):
collectors 6–7, the funded 7.2 intraday upgrade, implied-vol benchmark arms in
7.3/7.4, the rhodium axis in 7.5/7.6, new §7.8 (macro-release movers), a
purchased-history provenance gate in 7.7, and a budgeted Ordering.

## 7.1 Data acquisition — the seven collectors (start immediately)

The binding constraint on the whole portfolio is data that cannot be
backfilled. Full specification, rationale, and sequencing in
`results/amc_data_acquisition_program.md`. Summary:

1. **AMC ledger ingest** — schema + validating importer for AMC's own books
   (`amc_scrap_lots`, `amc_coin_trades`, optional `amc_till_daily`). Local-only.
   Converts per-unit price risk into dollar inventory VaR; float-duration
   histogram; realized premiums as ground truth.
2. **Retail coin-premium panel** — daily polite scrape of a fixed product
   basket from two large online dealers (ask, buyback bid, spot, timestamp).
   Wayback snapshots are validation-only (selection-biased), never training.
3. **Google Trends archiver** — weekly as-pulled snapshots of a fixed sell-side
   term set, stored verbatim; Trends renormalizes per request, so only a
   real-time archive is honest. Setup-time history kept but flagged
   non-real-time.
4. **CME daily volume/open-interest collector** — forward capture of the
   public daily figures (Yahoo Pt/Pd volume is ~40% zeros). History is no
   longer a wait: the funded Databento backfill (below) supplies official
   settlement/volume/OI 2010-06+, and this collector becomes the
   forward-continuity leg of a spliced series. Splice gate before any model
   consumes the merge: numeric agreement on the live overlap AND
   preliminary-vs-final OI semantics classified (a model trained on official
   finals but fed scraped preliminaries live overstates real-time
   performance).
5. **Event calendars + surprise upkeep** — (a) FOMC 2024–26+ calendar into
   `events` **with per-meeting statement release times** (~2:15pm ET pre-2011,
   12:30pm on presser meetings 2011–12, 2:00pm from 2013; presser start times
   separately) — 7.2's intraday windows are defined off these, never
   hardcoded; (b) BLS release calendars for CPI and the Employment Situation;
   (c) keep the extended surprise series (7.2) refreshed each meeting evening;
   (d) **macro consensus capture** — scrape CPI/payrolls consensus before each
   8:30 release (real-time by construction, `pulled_at`-stamped) into a
   `macro_consensus` table. Descoped 2026-07-12 from "two independent sites"
   to the one free machine-readable feed verified to exist (ForexFactory
   JSON — Trading Economics guest API discontinued, FXStreet auth-gated,
   myfxbook trailing-only): the table keys on `consensus_source`, so a second
   feed slots in whenever one appears — re-check yearly. Feed never publishes
   actuals — first prints come from ALFRED in 7.8;
   historical reconstruction admitted only where a pre-release Wayback capture
   proves provenance (expect ~30–50 of ~193 prints; the rest stay second-class
   under the 7.7 purchased-history gate).
6. **Johnson Matthey PGM base prices (Rh/Ir/Ru + JM Pt/Pd cross-check)** —
   one-time decades-deep historical CSV pull + weekly forward capture.
   Rhodium is typically the dominant value in catalytic-converter scrap and
   has no exchange price; nothing else in the stack prices it. Stale-quote
   run-length flag (Rh quotes plateau); JM Pt/Pd cross-checked against CME
   settles to characterize quote timing. The one backfillable collector —
   rides along because it is free and feeds the 7.5/7.6 rhodium axis.
7. **Premium-side forward capture** — (a) daily Greysheet wholesale bid/ask
   snapshots for the benchmark products via the basic API (urgency drops only
   if the pending sales inquiry confirms a deep downloadable bid archive —
   send that email first); (b) periodic Terapeak realized-price snapshots
   (the rolling 3-year window ages out permanently); (c) conditional: log a
   wholesale maker feed (FizTrade-class) daily if AMC holds or opens a
   trading account. Goldhub India/China premia ride along as a standard
   backfillable ingest with publication-lag logging (`pulled_at` vs stated
   date — the lag the 7.4 analyses must assume).

**The paid sprint (same first week):** open the Databento account, verify
pre-2017 completeness (the CME FIX flat-file reconstruction era) inside the
$125 free credits, then pull statistics + definition schemas for GC/SI/PL/PA
futures AND options, plus 1-minute bars for event windows — include ZQ, ZT,
and GE/SR3 (target *and* path instruments for 7.2) in the same pull. ~$0–425
all-in, portal-estimated before purchase; credits expire ~6 months after
signup — the one hard clock in this plan. Subscribe to Greysheet ($299/yr).

Engineering rules for all collectors: DuckDB migrations numbered from `008`,
append-only, UTC, `source`/`pulled_at` provenance columns, real-time flags on
any retro-captured rows, fail-loudly alerting on schema drift or missed days.
The laptop DB is the sole corpus copy — the deferred off-site backup lands in
the same sprint as the first collector. Sequencing: collectors 1–5 and the
backup own the critical path; 6–7 must not extend it.

## 7.2 FOMC hedge playbook, live again (~3 days; first analysis job)

Extend the monetary-surprise series past its 2023-12 end with two measures
built in the same pass: (a) the FOMC-day ΔDGS2 (Hanson–Stein convention) as
the same-evening live stand-in, and (b) the funded intraday upgrade — a
GSS-style **target+path** surprise from Databento 1-minute bars (Kuttner-scaled
ZQ for target; ZT/GE/SR3 for path — ZQ alone is structurally near-zero across
the 2010–15 ZLB, where the identifying variation was guidance) in 30-minute
windows bracketed off the per-meeting release times in `events` (never
hardcoded; statement times moved twice before 2013). Validate both against
MPS/MPS_ORTH on the overlap — ZQ against MPS's target component, the composite
against full MPS — so a ZLB-era mismatch cannot masquerade as instrument
failure. Re-run LP/DoubleML/CATE on 2015–2026 (raises treated events toward
the count that makes CATE confirmatory). Ship the per-meeting hedge playbook:
hedge notional per $100k of Au/Ag/Pt float, Pd excluded (supply-driven),
dovish tails not hedged, expanding-window thresholds only. **Gate:** the
Stage-2 text stance-encoder is built only if the intraday composite (not
ΔDGS2) correlates < 0.7 with MPS on the overlap. **IV-cycle appendix:** event
study of ATM implied vol and 25-delta skew around statements (2010+, option
settles) plus a hedge-cost ledger — puts bought t−5 vs t−1 vs futures/collar
alternatives — so the playbook prescribes instrument and timing alongside
notional. Settlement clock: options settle ~1:30pm ET, *before* the
statement, so the t=0 settle is a pre-statement read and the IV crush lands
at t+1; stratify by press-conference regime.

## 7.3 Float-window tail engine (spread floors; ~1.5 weeks + gated experiment)

Forecast conditional quantiles (q05/q10/q25/q50) of the running minimum and
terminal cumulative return over 1–20-day floats, all four metals jointly →
daily per-metal **spread-floor sheet** (minimum discount below spot for a
chosen underwater probability). Stage 1 (ships the artifact): numeric-only from
2007, GARCH-t/EGARCH-t path simulation + filtered historical simulation +
per-target AND pooled LightGBM pinball regression; add a max-horizon (20-day)
embargo to `cv.py`; `min_nan_tail = h` leakage gate; coverage tests pooled
across folds. Stage 2 (the sanctioned Phase 4 numeric experiment): fixed
pre-registered iTransformer config; **kill criterion: must beat GARCH-t on
conditional coverage** (pooled-LightGBM control mandatory so a win is not
misattributed to attention). **Implied-quantile benchmark arm (mandatory):**
option-implied quantiles from self-computed ATM IV + 25-delta skew (Black-76
on daily option settles, 2010+; GVZ cross-check) in two variants — raw
risk-neutral (diagnostic only; the variance risk premium makes it over-cover)
and expanding-window VRP-adjusted, which is the must-beat gate on pooled
conditional coverage and pinball loss before the sheet ships modeled
quantiles. IV level/term/skew also enter the LightGBM arm as features. Once
7.1's ledger lands, weight the horizon grid by AMC's real float durations to
convert per-unit VaR into dollar VaR.

## 7.4 Fast artifacts (~1 week each, baseline-first)

- **Event-window vol risk card** — Yang-Zhang RV from ETFs; HAR-RV +
  event-position dummies and direct multi-horizon LightGBM (days-to-event per
  forecast day, pre-event state, regime cluster as effect modifier per Phase
  5); event-stratified QLIKE on purged splits. Benchmark arm: pre-event
  implied vol (GVZ / self-computed ATM IV at the prior close, horizon-matched
  to the shortest expiry covering the event window) in raw and
  expanding-window-debiased variants — the ship/no-ship gate and any fallback
  use the **debiased** arm, and a shipped IV number is labeled as such.
  TFT/PatchTST only as a gated bake-off.
- **Physical-tightness index** — futures-vs-ETF basis z-scores (all four
  pairs), futures/ETF volume ratio (all four pairs from 2010 via the
  backfill), commercial-vs-managed-money divergence; validated as a *nowcast*
  against real premium series: the WorthPoint one-time realized-premium
  backfill (~2006+, ~$30–60, episode-window extraction, retro-flagged
  validation-only) plus Goldhub India/China premia, over the documented Au/Ag
  blowouts AND 2–3 PGM tightness episodes (2019–20 Pd squeeze, Mar-2020
  dislocation), with a permutation base-rate control; feeds buy-spread
  decisions and the premium work once 7.1/collector-2 data accrues.
- **Premium-dynamics studies (later tier, once premium series exist)** —
  (a) Goldhub India/China lead-lag: mechanically endogenous as a mover test
  (spot sits inside the premium's definition; sticky local prices plus mean
  reversion fake "premium leads spot") — only the India import-duty
  quasi-experiment leg is clean; publication-lag gate mandatory; a null is
  acceptable. (b) Blowout decomposition: asymmetric pass-through
  (expanding-window estimation ONLY) splits the mechanical sticky-retail
  premium from residual demand; realized (WorthPoint) and posted-ask series
  are never pooled in one regression. Deliverable either way is the coin-desk
  premium playbook (expected premium path conditional on spot-gap direction),
  not a mover claim.

## 7.5 Text/NLP projects (the corpus earns its keep; 2–6 weeks)

- **PGM supply-shock event ledger** — entity-anchored retrieval (producers,
  mines, smelters, Eskom, Nornickel, sanctioning bodies; never metal names) over
  the title era (2019-09-22+); recall-validate against 5 known events AND the
  systematic set of de-plateaued top-1% rhodium move-day clusters (collector
  6, title era) — unmatched clusters audited from sources *outside* the corpus
  (library Factiva), miss bins split "absent from corpus" vs "retrieval
  missed", permutation base-rate control on the ±3-day matching; four-way
  classifier bake-off (rules → frozen MiniLM+logistic → zero-shot LLM →
  fine-tuned encoder, escalation-gated). Fills the missing supply axis for the
  metal Phase 5 showed is not Fed-hedgeable.
- **Supply-shock causal pass (candidate mover: the Pd axis)** — tiered
  hand-curated event list 2010–2026 (accidents / strikes / sanctions; honest
  count ~15–22, of which ~7–9 cleanly exogenous accidents) under a
  **price-blind inclusion rule**: an event enters iff a contemporaneous
  physical-impact disclosure exists (force majeure, production-guidance cut,
  strike filing — from JM annual reports, company releases, Factiva), never
  "days Pd moved"; list and tiers frozen in the scenario registry before
  estimation (the Phase 5 pattern). Jordà LP h=1–10 on Pd/Pt/Rh with gold as
  placebo outcome and FOMC as placebo treatment, permutation p-values; DML
  gated on ≥25 clean events after ledger expansion; no CATE at any realistic
  count — sign-test tier, stated as such. Payoff: the Pd counterpart to the
  FOMC rule (dated supply shock → widen converter/Pd spreads, cap PGM float).
- **Flight-to-safety index (GPR replacement)** — price-observable FTS labels
  (gold↑ ∧ S&P↓ ∧ VIX↑, 2007+); ladder G0 (LightGBM on tone/theme aggregates)
  → G1 (zero-shot crisis-prototype cosine pooling over prefiltered headlines)
  → G2 (learned attention pooler, admitted only if it beats G1 out-of-fold).
  Trainable index scoped to the title era; pre-2019 slug-scored and
  second-class. Payoff: sharper SVAR instrument + coin-desk crisis cue.

## 7.6 Positioning projects (LightGBM-first; ~2 weeks each)

- **Shadow COT + crowding tail** — migration: add `report_date` to
  `positioning`; nowcast intra-week managed-money flow from the price/volume
  path since last report Tuesday (backfilled CME volume/OI 2010+ as primary
  activity input; ETF volume retained as cross-check); tail heads
  P(5d < −1.5σ) + 5%/10% quantiles.
- **PGM liquidation hazard alarm** — pooled Pt+Pd discrete-time hazard,
  regularized logistic + LightGBM, strict false-alarm budget, ≥10-day
  purge/embargo, pre-onset vs in-drawdown alarm split. Labels available now
  (backfilled OI): the onset definition is **pre-registered before anyone
  inspects the backfill** and computed on roll-neutral aggregate OI (summed
  across contract months — per-contract OI contracts mechanically at every
  quarterly roll); the onset count is recomputed from the registered
  definition, not anchored on "~38". Rhodium features (lagged return,
  momentum, Rh/Pd + Rh/Pt ratio z-scores; previous-completed-quote alignment
  in the leakage guard) are kept only if precision at the alarm budget is
  non-inferior in a with/without ablation. Pre-registered secondary readout,
  descriptive tier: realized episodes split Rh-confirmed vs Rh-silent
  (fundamental repricing vs positioning washout) using a past-only
  confirmation window — hindsight labels are not live discriminators (the
  Phase 3 lesson). Companion estimand, causal language banned: pooled
  conditional-predictive profiles (drawdown-depth and time-to-trough quantiles
  given onset, episode-clustered bootstrap) — the severity leg the de-risking
  pace needs. Sample-size honesty: precision at the alarm budget, no
  transformer question is decidable here.
- **Crowding-conditioned FOMC response (feeds 7.2 hedge sizing)** —
  pre-registered interaction LP (surprise × pre-meeting managed-money
  net-long percentile) plus crowding in the CATE modifier set. Conditioning
  uses the last **released** COT report (release-date aligned per
  `positioning.timestamp_utc` — the as-of Tuesday publishes Friday, so
  conditioning on it is the canonical leakage bug); ~18–22 treated events per
  binary cell → monotonicity/sign evidence only, shipped as a hedge-notional
  multiplier, not a new mover.

## 7.7 Standing gates

- Every model logs to the harness; every claim that reaches AMC gets the same
  adversarial-review treatment as `phase5_amc_business_implications`.
- Transformers run only as pre-registered, kill-criterioned bake-offs
  (7.3 Stage 2, 7.4, 7.5 G2) — never as the default architecture.
- Anything touching AMC's ledger stays on the local machine.
- Parked pending fresh pre-registration: the h=20 gold-vol confirmation (the
  −2.12% lead traces to regime features, not text sequences).
- Purchased or third-party "historical / point-in-time" series enter training
  only after provenance verification. Hindsight-sensitive series (consensus,
  forecasts, sentiment, revised-methodology indices, back-adjusted continuous
  contracts) require positive evidence that generation timestamps precede the
  events they describe — probe the vendor's own API/files (the FXMacroData
  refutation is the template); unverifiable rows are flagged non-real-time
  and excluded from training. Mechanical series (exchange settles, OI)
  require a license/reproducibility check instead (the Norgate
  delete-on-lapse lesson).

## 7.8 Macro-release movers — CPI + payrolls (gated on 7.1 item 5d)

Candidate movers #2/#3, run as two arms of one design (shared code, consensus
infrastructure, and multiplicity accounting): ~193 monthly prints each
2010-06+ — roughly 5× the hawkish-FOMC event count, with non-overlapping h≤10
windows. Treatment: co-primary surprise measures — the market-based 8:30→8:35
intraday jump (ZT/ZQ from the Databento event-day bars; the identification
workhorse, no provenance problem) and the consensus-based first-print surprise
(ALFRED actual minus provenance-audited consensus; the interpretable data-unit
cross-check, second-class wherever reconstruction is unverified). Design:
Jordà LP h=0–10 with HAC errors + DoubleML with placebos, expanding-window
standardization only, cross-release dummies (FOMC/NFP days inside outcome
windows). Sign pre-registration is **mechanism-conditional**: metals load
negatively on the release-day *yield response* (era-robust); the reduced-form
sign is pre-registered era-conditionally (2020–26: hot CPI → Au/Ag/Pt down,
Pd null; 2010–19: weak/no prior — the repo's own Phase 2 note documents gold
*rising* on pre-2022 upside CPI surprises), with the Phase 5-style subsample
pass as core analysis, not robustness. Payoff: extends the event playbook to
CPI mornings and payrolls Fridays, and roughly triples treated events for the
CATE that Phase 5 could only call suggestive.

## Ordering

Off-site backup and 7.1 collectors 1–5 first; the paid sprint (Databento
backfill incl. options + event-day minute bars; Greysheet) lands in the same
first week — Databento's free credits expire ~6 months after signup, the one
hard clock. Collectors 6–7 follow without extending the critical path. 7.2's
ΔDGS2 pass proceeds in parallel and does not wait for the backfill; the
intraday composite and any 7.6 labeling do. Then 7.4 (fast artifacts), 7.3
Stage 1, 7.8, and 7.5/7.6 as capacity allows — the hazard alarm is no longer
accrual-blocked and becomes eligible right after 7.4; shadow-COT stays
capacity-scheduled; 7.3 Stage 1 is not displaced. Gated experiments last.
Phase 6 close-out (6.10 repro entry points, 6.11 cleanup + v1.0 tag) proceeds
independently and should finish before heavy 7.3+ modelling.

**Budget & accounts:** year-one cash ≤ ~$725 (Databento ~$0–425 all-in with
options; Greysheet $299/yr; WorthPoint ~$30–60 when the premium study runs).
Vendor accounts are AMC-owned; credentials live in `.env`, never in the repo.
