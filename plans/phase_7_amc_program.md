# Phase 7: The AMC program — dealer decision support

Added 2026-07-12. Phases 0–6 answered *what moves metals prices*; Phase 7 turns
those answers into operating decisions for **AMC Company**, the business the
research serves: a small dealer that buys scrap Au/Ag/Pt/Pd (assaying fine
content) and buys/sells gold coin and specie — structurally long physical metal
over a days-to-weeks inventory float.

Two source documents govern this phase:

- `results/phase5_amc_business_implications.pdf` — the Phase 5 findings translated
  into AMC's hedging, spread, and inventory decisions (adversarially verified).
- `results/amc_data_acquisition_program.md` (+ `.pdf`) — the start-now
  data-acquisition program (step 7.1 below, expanded).

The portfolio below came out of a structured multi-agent brainstorm
(2026-07-12): 25 raw ideas → 10 distinct projects → adversarial feasibility
review. Its governing lesson matched the roadmap's own warning: in 8 of 10
projects a transformer was decoration. **Every project here is baseline-first;
any transformer runs as a gated bake-off behind a pre-registered kill
criterion.** All existing conventions hold (walk-forward CV only, leakage guard,
harness logging, UTC, journal entries).

## 7.1 Data acquisition — the five collectors (start immediately)

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
4. **CME daily volume/open-interest collector** — forward-only capture of the
   public daily figures (historical daily OI is paywalled; Yahoo Pt/Pd volume
   is ~40% zeros). Enables shadow-COT and liquidation-alarm labels.
5. **Event calendars + surprise upkeep** — FOMC 2024–26+ and BLS CPI release
   calendars into `events`; then keep the extended surprise series (7.2)
   refreshed each meeting evening.

Engineering rules for all five: DuckDB migrations numbered from `008`,
append-only, UTC, `source`/`pulled_at` provenance columns, real-time flags on
any retro-captured rows, fail-loudly alerting on schema drift or missed days.
The laptop DB is the sole corpus copy — the deferred off-site backup lands in
the same sprint as the first collector.

## 7.2 FOMC hedge playbook, live again (~3 days; first analysis job)

Extend the monetary-surprise series past its 2023-12 end using the FOMC-day
ΔDGS2 (Hanson–Stein convention; optional Kuttner futures-implied variant needs
a small new ingest), validated against MPS/MPS_ORTH on the 77-meeting GDELT-era
overlap (report correlation + sign agreement). Re-run LP/DoubleML/CATE on
2015–2026 (raises treated events toward the count that makes CATE
confirmatory). Ship the per-meeting hedge playbook: hedge notional per $100k of
Au/Ag/Pt float, Pd excluded (supply-driven), dovish tails not hedged,
expanding-window thresholds only. **Gate:** a Stage-2 text stance-encoder
(pre/post-statement decomposition off 15-minute headline timestamps) is built
only if the ΔDGS2 series correlates < 0.7 with MPS on the overlap.

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
misattributed to attention). Once 7.1's ledger lands, weight the horizon grid
by AMC's real float durations to convert per-unit VaR into dollar VaR.

## 7.4 Fast artifacts (~1 week each, baseline-first)

- **Event-window vol risk card** — Yang-Zhang RV from ETFs; HAR-RV +
  event-position dummies and direct multi-horizon LightGBM (days-to-event per
  forecast day, pre-event state, regime cluster as effect modifier per Phase
  5); event-stratified QLIKE on purged splits. TFT/PatchTST only as a gated
  bake-off.
- **Physical-tightness index** — futures-vs-ETF basis z-scores (all four
  pairs), futures/ETF volume ratio (Au/Ag only until CME volume lands),
  commercial-vs-managed-money divergence; validated as a *nowcast* by event
  study against known premium episodes; feeds buy-spread decisions and the
  premium work once 7.1/collector-2 data accrues.

## 7.5 Text/NLP projects (the corpus earns its keep; 2–6 weeks)

- **PGM supply-shock event ledger** — entity-anchored retrieval (producers,
  mines, smelters, Eskom, Nornickel, sanctioning bodies; never metal names) over
  the title era (2019-09-22+); recall-validate against 5 known events; four-way
  classifier bake-off (rules → frozen MiniLM+logistic → zero-shot LLM →
  fine-tuned encoder, escalation-gated). Fills the missing supply axis for the
  metal Phase 5 showed is not Fed-hedgeable.
- **Flight-to-safety index (GPR replacement)** — price-observable FTS labels
  (gold↑ ∧ S&P↓ ∧ VIX↑, 2007+); ladder G0 (LightGBM on tone/theme aggregates)
  → G1 (zero-shot crisis-prototype cosine pooling over prefiltered headlines)
  → G2 (learned attention pooler, admitted only if it beats G1 out-of-fold).
  Trainable index scoped to the title era; pre-2019 slug-scored and
  second-class. Payoff: sharper SVAR instrument + coin-desk crisis cue.

## 7.6 Positioning projects (LightGBM-first; ~2 weeks each)

- **Shadow COT + crowding tail** — migration: add `report_date` to
  `positioning`; nowcast intra-week managed-money flow from the price/volume
  path since last report Tuesday (ETF volume as activity proxy until CME
  collector accrues); tail heads P(5d < −1.5σ) + 5%/10% quantiles.
- **PGM liquidation hazard alarm** — pooled Pt+Pd discrete-time hazard
  (~38 onsets 2010+), regularized logistic + LightGBM, strict false-alarm
  budget, ≥10-day purge/embargo, pre-onset vs in-drawdown alarm split. Sample
  size honesty: precision at the alarm budget, no transformer question is
  decidable here.

## 7.7 Standing gates

- Every model logs to the harness; every claim that reaches AMC gets the same
  adversarial-review treatment as `phase5_amc_business_implications`.
- Transformers run only as pre-registered, kill-criterioned bake-offs
  (7.3 Stage 2, 7.4, 7.5 G2) — never as the default architecture.
- Anything touching AMC's ledger stays on the local machine.
- Parked pending fresh pre-registration: the h=20 gold-vol confirmation (the
  −2.12% lead traces to regime features, not text sequences).

## Ordering

7.1 collectors (+ off-site backup) and 7.2 first — they are cheap and the clock
runs only forward. Then 7.4 (fast artifacts), 7.3 Stage 1, and 7.5/7.6 as
capacity allows; gated experiments last. Phase 6 close-out (6.10 repro entry
points, 6.11 cleanup + v1.0 tag) proceeds independently and should finish
before heavy 7.3+ modelling.
