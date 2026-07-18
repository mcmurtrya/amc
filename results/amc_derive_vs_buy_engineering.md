# Derive vs. Buy: A Data-Engineering Program for AMC

**Prepared 2026-07-18.** Third companion to the AMC data notes:
`results/amc_data_acquisition_program.md` (the free, non-backfillable *collectors* to
start now) and `results/amc_paid_data_review.md` (the small-business *buy/skip* review).
Where the paid review asked "what should AMC pay for," this note asks the sharper
question underneath it: **of the datasets worth having, which can AMC actually acquire —
and which of them is it cheaper to *build from data it already owns* than to buy?**
Audience: deep business experience, above-average but non-expert statistics. Acronyms are
spelled out on first use.

---

## Bottom line

Most of the value AMC is chasing **is not for sale as a product.** The single
highest-return activity is *deriving* constructs the vendors do not publish as a field —
implied-volatility surfaces, lease-rate and squeeze alarms, retail-vs-wholesale premium
wedges, a tail-loss library — out of data AMC **already owns or can compute**, and then
**joining those onto AMC's own transaction ledger** (dollar risk on the actual book,
realized premium versus the national benchmark, per-converter exit value). That work costs
essentially nothing in licensing and clears the Terms-of-Use (ToU) gate cleanly, because
AMC owns the inputs.

Of the genuinely *net-new purchases*, exactly one is a clean buy (deep rhodium and
platinum-group-metal price history, a few hundred dollars); two are cheap but conditional
(Norgate deep futures history; the Eco Cat catalytic-converter database); one needs a
dealer trading account AMC may already qualify for (a live wholesale two-way feed); and
one — the differentiated dealer-specific retail-ask and stock-out panel — is **barred by
default** and opens only with written consent or by capturing AMC's own data forward.
Everything sold as "enterprise alternative data" (satellite smelter monitoring, SFA Oxford
analytics, a multi-dealer consortium, a Bloomberg Data License) is either
institution-building, budget-indefensible, or already ruled out by AMC's own research
priors.

The rest of this note makes that concrete: how each candidate is actually acquired
(Part A), what data engineering can extract from the owned and cheap-to-own sources
(Part B), the leakage discipline every derivation must obey, and the honest list of
clever-looking builds that do not survive (Part C).

## The four hard research priors (the filter behind every verdict)

These are load-bearing empirical findings from Phases 3–6, not opinions. A signal that is
merely *computable* does not clear them.

- **P1 — sentiment and regime features hurt out-of-sample.** Phase 6 validation found
  classical volatility baselines beat machine learning, and news-sentiment / regime-state
  features added zero forecasting lift and *degraded* out-of-sample accuracy (the sentiment
  model's Diebold–Mariano statistic was +2.90 *worse*). Any derived feature whose only use
  is a sentiment or regime signal starts from "no" and must clear an incremental-information
  and block-permutation null it has little reason to pass.
- **P2 — the independent-sample wall.** The daily record has roughly 40–50 *effective
  independent* observations (about 2,800 rows at ~99% autocorrelation). More history mostly
  buys duplicative regimes. The exceptions that escape the wall are (a) genuine tail
  *exceedances* — crisis episodes — and (b) *operational reference* constructs used at the
  counter, which are not fitted forecasters at all.
- **P3 — leakage is easy to inject in a derivation.** A back-adjusted/stitched futures
  series fabricates backwardation at every contract roll; vendor "point-in-time" history is
  routinely retro-generated; a dated news title lets a model know what happened next. Every
  derived series needs genuine as-of provenance.
- **P4 — the float is days-to-weeks.** Intraday and tick-level derivations are decoration
  for AMC's decisions, with the single exception of the funded Federal Open Market Committee
  (FOMC) announcement window.
- **P5 — the ToU gate is commercial + model-training + cached-local.** "Free" is not
  "cleared." Scraping public market pages for AMC's use has failed this gate repeatedly
  (the Chicago Mercantile Exchange and World Gold Council Goldhub cases). Government open
  data (the Federal Reserve Economic Data service "FRED," the U.S. Mint, customs portals)
  is clean; a paid commercial licence can convert *barred → licensed*.

---

## Part A — Feasibility: how each move is actually acquired

Four acquisition modes, in rough order of ease:

- **Buy off-the-shelf** — a vendor sells it, and (subject to a ToU read) it clears
  commercial use, local caching, and model training.
- **Derive-and-compute** — build it from data AMC already owns or can get cheaply; zero
  ToU exposure because AMC owns the inputs.
- **Scrape-within-ToU** — pull from public pages. For AMC's use this is **barred by
  default** (P5); it clears only where a site's terms genuinely permit commercial
  cache-and-train, or where a licence converts it.
- **Commission-only** — no vendor exists; AMC must create the primary data (mystery-shops,
  a dealer consortium, a bespoke geospatial contract). This is business development, not a
  data buy.

### The five flagship "unlimited-resources" moves, bucketed

| # | Move | How it is actually acquired | Verdict |
|---|---|---|---|
| 1 | Deep rhodium + PGM price history | Johnson Matthey / Anglo free series + CPM Group Yearbook (~$170 one-time) | **Buy / derive — the one clean, cheap, on-mission purchase** |
| 2 | Catalytic-converter content database (Eco Cat) | Off-the-shelf subscription (~$20–115/mo) | **Buy** — but operational counter tool, owner's call, not a research buy |
| 3 | Norgate deep US futures 1975–2009 | Vendor subscription (~$270/yr) | **Buy, but defer** until tail work proves pre-2010 crises help |
| 4 | Live wholesale two-way feed (Dillon Gage FizTrade, A-Mark) | Log AMC's own feed — needs an approved dealer account | **Buy, account-gated** (plausible AMC qualifies) |
| 5 | Dealer-specific retail ask + **stock-out flags** | Per-dealer written consent, or capture forward | **Barred to buy/scrape** — this is the retired Collector 2 |

The honest headline: **only rhodium/PGM history is an unambiguous buy.** And the single
most differentiated crisis signal — the retailer stock-out / ship-delay flag that gapped
coin premiums *ahead* of the melt price in 2020–21 — is precisely the one AMC cannot
purchase or legally scrape. It lives only in live pages, it is non-backfillable, and its
whole nature is that it requires consent. If AMC wants it, that is a
forward-capture-under-consent project, not a purchase.

### Full triage, sorted by bucket

**Buy off-the-shelf (clean, licensed, cheap-to-moderate).**
- *Deep rhodium/PGM history* — Johnson Matthey / Anglo producer series plus the CPM Group
  PGM Yearbook ($170). Producer-published price data is the cleanest non-government source;
  still run the ToU gate on Johnson Matthey's file terms before caching into the training
  database (free is not cleared — the Goldhub lesson).
- *Norgate deep futures* (~$270/yr). The licence requires **deleting all data and
  derivatives on lapse**, so "buy six months and cancel" is non-compliant; compliant use is
  the recurring fee. Rhodium is **not** covered (it has no exchange price).
- *Eco Cat converter database* (~$20–115/mo). Sold to dealers/recyclers for exactly this
  per-piece counter-pricing use, so it clears cleanly for lookup; a bulk export or a
  resale-model training use would exceed the licence.
- *CPM Group PGM Yearbook* (~$170 one-time) — unambiguously ownable; fills pre-1990 rhodium
  detail.

**Derive-and-compute from owned-or-cheap data ($0 licensing, no ToU exposure) — the big
bucket.**
- *Everything off the Chicago Mercantile Exchange (CME) statistics feed* (recommended buy,
  via Databento; official daily settlement, open interest, volume per contract and per
  option strike, plus one-minute bars): the implied-volatility surface, the
  calendar-spread-implied lease/forward rate and squeeze flag, open-interest term
  structure, the self-computed FOMC surprise. All from **raw per-contract settlements**,
  never stitched.
- *Premium wedges* (retail benchmark minus wholesale bid) from the Greysheet Coin Dealer
  Network (CDN) application programming interface (API) — a single benchmark per side is
  enough for a two-series wedge; no multi-dealer panel is required.
- *Macro constructs* off already-ingested FRED and Commodity Futures Trading Commission
  (CFTC) positioning data: the Cboe Gold Volatility Index (GVZ) tail-vol floor (GVZ is
  **confirmed not yet ingested** — a genuine free add), real-yield and dollar factors,
  managed-money crowding.
- *AMC's own payable curve* (fine-content-in versus dollars-out) from refining-settlement
  invoices, and *AMC's own realized-premium curve* from the ledger — both owned, both
  local, both high-value.

**Scrape — mostly barred, stated plainly.**
- Dealer-specific asks / buyback / stock-out flags (APMEX, JM Bullion): **barred** (the
  retired, quarantined Collector 2). Written per-dealer consent is the only clearance;
  non-backfillable, so capture forward.
- eBay Terapeak / WorthPoint realized premiums: treat as **barred by default** for
  train-and-cache until the terms are read; the clean route is the eBay Marketplace
  Insights API licence, not scraping. Terapeak's ~3-year window ages out permanently — if
  licensed, snapshot forward now.
- Local competitor pages and trade-press premiums: barred by default, and mostly there is
  nothing posted to scrape anyway. The clean exception is **AMC's own mystery-shop
  quotes** — those are primary observations, not scraping.

**Commission-only (build an institution — all low feasibility).**
- Multi-dealer ledger/premium consortium, multi-refiner payable panel, local-competition
  panel, satellite smelter monitoring. No vendor sells these; each is a multi-year
  business-development play. In every case the *owned leg* — AMC's ledger, its own payable
  curve, its own counter observations — is the clean 80% of the value.
- *Bloomberg / London Stock Exchange Group (LSEG) Data License* is the one clean *vehicle*
  that converts barred → licensed for the exchange-for-physical basis, PGM lease rates, and
  point-in-time consensus — but it is five-figures-plus per year, its value is ToU
  clearance rather than new signal, and most of what it clears already has a free or
  derived substitute. Rational only once AMC is materially larger.

---

## Part B — What data engineering uncovers from paid data

### The meta-point (read this first)

The vendors sell **fields**; the money is in what AMC can build *on top of* them that they
do not sell. Two categories:

**(i) Derived constructs no vendor publishes as a field.** An implied-volatility surface
for silver and the platinum-group metals — there is *no* published PGM volatility index
anywhere, so AMC manufactures it from option settlements. The lease-rate / backwardation
squeeze alarm, which rebuilds the retired London gold-forward (GOFO) lease series from raw
settlements. Retail-minus-wholesale premium wedges, quoted nowhere. A deep-tail loss
library. A dated rhodium supply-disruption event ledger mined from news titles — the only
real-time supply signal for a metal with no exchange price.

**(ii) Joins onto AMC's proprietary ledger.** This is AMC's unfair advantage and no
competitor can replicate it: dollar Value-at-Risk (VaR) on the *actual* book (the tail-loss
distribution scaled by AMC's real float-size distribution), realized premium versus the
national benchmark (an execution scorecard by product and channel), per-piece converter
exit value (converter content multiplied by live PGM/rhodium prices), and an FOMC hedge
notional sized to AMC's *actual* float.

Every construct below is an **operational or reference object, not a learned regime
feature** — which is exactly why it escapes P1 and P2. Tail-exceedance and counter-reference
uses are the explicit carve-outs from the independent-sample wall.

### Decision 1 — buy-spread-floors (how wide to quote)

The governing relationship is
`max_buy = wholesale_exit_floor − k · tail_vol · sqrt(float_days) − float_carry`.
Every build here feeds a term in it.

**Tier-1 builds:**
- **Calendar-spread-implied forward/lease rate + backwardation squeeze flag** (CME feed).
  The curve carry *is* AMC's literal cost of holding the float; a near-over-far inversion
  is physical tightness that widens the exit AMC hedges into. A direct physical observable
  that never touches the regime wall. *(Also serves PGM alarms.)*
- **Option-implied volatility surface: at-the-money implied vol + 25-delta risk reversal**
  (CME feed). Forward volatility the vendor never indexes for silver or the PGMs. Feeds the
  spread-width term. *(Also PGM alarms.)*
- **Conservative wholesale exit-floor anchor** (Greysheet). The exact Greysheet →
  decision bridge: the low-side envelope built only from *realized past captures* over a
  window at least as long as the float duration.
- **90% "junk" silver bag premium-over-melt** (Greysheet). The closest listed benchmark to
  what AMC actually pays for melt-grade silver; the premium *sign flip* (a discount in a
  scrap glut, a spike in a retail squeeze) is a physical-market read. *(Also
  coin-premium-intel.)*
- **GVZ + variance-risk-premium tail-vol floor** (FRED). The only forward-looking
  volatility input to decision 1; GVZ is confirmed not yet ingested — a genuine free add.
- **Extreme-value tail calibration + float-horizon maximum-adverse-excursion (MAE)
  distribution** (deep history). Recalibrates the worst-case terminal-loss and
  worst-point-during-hold quantiles the shipped sheet emits, from essentially one crisis
  (2008) to several independent ones. The maximum adverse excursion — the worst point
  *during* the hold, not just at the end — is the more faithful forced-sale risk object.
- **Effective-sample-size / declustering audit** (deep history). The honesty governor that
  quantifies how many *independent* tail draws AMC actually has and refuses to let the tail
  engine overstate itself. This is the precondition that makes the deep-history buy
  defensible.

**Tier-2:** wholesale bid/ask spread stress gauge (Greysheet — verify the ask moves
independently of the bid before shipping); bullion premium term structure (sovereign →
round → bar slope); conditional recovery-time distribution (the *duration* half of the
floor — how long AMC carries underwater inventory).

### Decision 2 — event-hedging (FOMC and macro)

**Tier-1:**
- **Self-computed monetary-policy surprise + per-metal co-jump betas** (one-minute bars in
  the FOMC window). This is the superior intraday measure the daily 2-year-yield proxy
  stood in for, and it adds platinum/palladium announcement betas the proxy does not carry.
  It sits in the one funded, embargo-clean intraday slice P4 explicitly exempts from
  "decoration."
- **Hawkish-surprise hedge-sizing table** (FRED/CFTC): surprise basis points → dollar move
  per metal per horizon, anchored on the one scenario that survived Phase 5 triangulation
  (hawkish gold −1.4% / silver −2.9% / platinum −1.7% at the one-week horizon). Packaged as
  a frozen counter lookup — the project's purpose translated into a decision tool.

**Conditional buy:** point-in-time survey consensus for the Consumer Price Index (CPI) and
payrolls (Bloomberg Data License) roughly *triples* the surprise-event count with genuinely
independent tail events — but gate the purchase behind confirming the free path (first
prints from the St. Louis Fed's ALFRED archive plus published surprise series) is truly
inadequate.

### Decision 3 — PGM alarms (the most under-served decision)

**Tier-1:**
- **Rhodium converter-basket ratio history + rhodium tail range** (Johnson Matthey / Anglo
  deep fixings). The strongest single build in the whole program. Rhodium dominates
  converter-scrap value, has **no exchange price anywhere**, so there is zero
  market-implied rhodium volatility to read — the producer fixing is the *only* multi-decade
  calibration of the rhodium tail (the 2008 ~10× round-trip). It directly sets the
  converter-scrap discount AMC quotes.
- **PGM supply-disruption dated event ledger** (news titles → events table, from the owned
  GDELT corpus). Not sentiment — a dated events *catalog*. For a metal with no price, a
  first-seen-dated supply catalog is the only real-time supply signal obtainable.
- **OTC PGM lease rate as a listed-futures lead** (Bloomberg, if licensed). The one genuine
  *unlock* rather than upgrade: the gold lease self-derives from owned settlements, but that
  self-derive *breaks precisely for illiquid platinum/palladium* — it fails exactly at AMC's
  most differentiated decision.

**Tier-2:** PGM thin-book execution-cost gauge (on thin days AMC's own hedge moves the
market); managed-money crowding alarm from positioning (a tail exceedance P2 exempts);
cross-metal lower-tail co-exceedance counts (kept as declustered empirical counts with wide
confidence intervals — *not* a 5×5 copula VaR claim, which is under-identified on ~4–5
crises).

### Decision 4 — coin-premium-intel

**Tier-1:**
- **Retail-minus-wholesale premium wedge and its stress blow-out** (Greysheet). The literal
  2020–21 squeeze signature and the licensed replacement for the retired retail-premium
  collector. Scope caveat (not leakage): the retail benchmark is CDN's *published*
  benchmark, not AMC's realizable retail, so any "gross margin AMC captures" framing is
  overstated; the surviving value is the stress detector and buyback-spread posture.
- **Crisis-episode realized-premium elasticity curves** (realized-premium data, via licence
  — *not* scraping). In 2008 and March 2020 dealers went "call for price" and posted panels
  did not exist, so realized premiums are the only surviving observation. A one-time
  labelled lookup table that escapes P1/P4.

**Tier-2:** cross-sovereign brand basis (Eagle / Maple / Krugerrand — isolates
US-specific stress); premium-over-melt licensable history (build only if a deep as-*published*
bid archive is truly purchasable *and* each bid carries an original publication date);
a labelled crisis-regime episode library (a frozen reference registry, banned as a
supervised target).

### The ledger joins — AMC's proprietary edge (do these regardless of any purchase)

- **Dollar-VaR on the actual book** = the extreme-value / MAE tail library × AMC's real
  float-size distribution. Turns a generic per-ounce quantile into the dollar drawdown on
  *AMC's* inventory. **This is the highest-value build in the program and is spec'd in full
  in the companion `results/amc_spread_floor_engine_spec.md`.**
- **Ledger-vs-market realized-premium reconciliation** — an execution scorecard flagging
  systematic over-paying or under-charging by product and channel. The ledger leg is owned
  and clean; point it at the owned Greysheet benchmark first.
- **Per-piece converter exit value** = Eco Cat serial/photo lookup × live PGM/rhodium
  prices — the sharp end of PGM alarms at the counter.
- **FOMC hedge notional** = surprise basis points → dollar move × AMC's *actual* float
  notional and duration.
- **AMC's own payable curve** from refining-settlement invoices — sets the exit level in
  the spread floor directly.

---

## Leakage discipline (load-bearing — a derivation is worthless if it peeks)

Every construct above must obey the same look-ahead rules the codebase already enforces via
`src/metals/features/leakage.py` (`assert_chronological`,
`assert_target_strictly_future`, `assert_features_have_history`). The derivation-specific
traps:

- **Point-in-time roll, never stitched.** Every futures-derived series (lease rate, term
  structure, exchange-for-physical basis, tail library) must be built from the raw
  per-contract (near, far) pair *live on each date*. A back-adjusted/continuous front
  fabricates backwardation at every roll and injects roll look-ahead. The tail library must
  never span a roll, and limit-locked settlements (e.g. 1980 silver) must be flagged
  non-executable — an untradeable settle is not a realizable exit.
- **As-of stamping and open-interest timing.** Volume is knowable the same evening; open
  interest is final only the next morning (T+1). Align each statistic to its own receive
  timestamp. Positioning data is Tuesday-positioning / Friday-release — honor the release
  date, never the "as-of" date, when a model reads it.
- **Trailing-only envelopes.** The Greysheet exit-floor anchor and every stress baseline
  must be built from strictly-prior (≤ t−1) captures over a window at least as long as the
  float duration. A contemporaneous or forward bid overstates a mark AMC can only realize on
  exit.
- **Date-blind text.** For the supply-event ledger, the as-of date is the first-seen
  capture timestamp, never a date parsed out of the title text; drop recap/outlook titles.
- **Provenance against retro-generation (the FXMacroData trap).** Any purchased
  "point-in-time" history — survey consensus, premium archives, producer annual prices —
  must carry an *original publication date*, be stamped not-real-time permanently, and be
  paired to spot at the publication vintage. Vendor "history" is routinely as-revised.
- **Frozen thresholds.** Tercile boundaries, exceedance thresholds, and extreme-value
  parameters are frozen from the training vintage before the hold-out — the Phase 6.5 lesson
  that in-window shock thresholds replicated the mechanism 5–90× too strongly.

---

## Part C — Engineering that looks clever but does not survive

Included so the program is not a naive catalog. Each item is computable; none survives.

**Sentiment / regime derivations — start from "no" under P1.** A shadow-positioning nowcast
(daily open-interest bridging weekly positioning) is exactly the modeled-positioning class
Phase 6 falsified and is duplicative of owned positioning data. An options max-pain /
put-call fear index is a positioning-regime feature, weakest in the thin PGM books it
targets. LLM tone scores, theme-prevalence aggregates, and news-volume exceedance are the
news-attention class Phases 3 and 6 killed (cluster-lift null of −0.37% against a −1.0%
bar). A sold-through-rate demand nowcast is the cleanest kill — an explicit demand-regime
signal with survivorship bias on a handful of crisis onsets. The *only* legitimate spend in
this whole family is a one-time ~$30–150 LLM run to *close* the "should AMC buy a
$20–40k/yr news feed" question empirically — value is falsification, not a feature.

**Intraday microstructure — decoration for a days-to-weeks float (P4).** A FOMC-window
settlement-vs-volume-weighted-price dislocation flag is mechanically confounded (metals
settle ~1:30 p.m. ET, *before* the 2:00 p.m. statement, so the flag fires trivially every
announcement day). Open-interest roll-migration velocity is decoration for a float that
rarely spans a roll cycle.

**Retro-point-in-time constructs — non-reconstructable (P3).** Pre-release survey-dispersion
sizing needs the *last-pre-release* analyst panel, but a historical Data License pull returns
the *post-release* final panel, so the very history needed to validate it is contaminated.
A Fed-information-effect residual embeds full-sample information and has no live input past
2023-12. A paid pre-2015 news archive fails on four axes at once: duplicative regimes (P2),
a richer version of the killed sentiment class (P1), retro-generated point-in-time sentiment
(P3), and an enterprise/personal-use seat that fails the commercial + train + cache gate
(P5).

**One empirical landmine to check before shipping anything Greysheet-spread-based.** Confirm
the Greysheet *ask* moves independently of the bid. If dealer sheets set the ask as a
mechanical fixed markup over the bid, the spread is static and any spread-stress gauge is
dead on arrival.

---

## Build order

Sequenced so value lands early and nothing blocks on the pending ledger or on any purchase.

1. **Free derived signals from data already in the database** — GVZ ingest + tail-vol
   floor; realized-volatility term structure per metal; the premium wedge *once Greysheet
   is subscribed*. Days of work, zero licensing.
2. **The buy — rhodium/PGM history** (~$200–500) — unlocks the rhodium tail and converter
   basket, the strongest single build, and it is cheap and on-mission.
3. **The CME-derived engine** (once the Databento backfill is pulled) — lease-rate/squeeze
   alarm, implied-vol surface, self-computed FOMC surprise. All $0 on the owned licence.
4. **The ledger joins** (the moment AMC's ledger exports land) — dollar-VaR on the book,
   realized-premium reconciliation. The proprietary edge; spec'd separately.
5. **Account-gated and conditional** — log a wholesale two-way feed forward if AMC holds a
   dealer account; defer Norgate until the tail work demonstrates pre-2010 crises help;
   treat the dealer stock-out panel as a forward-capture-under-consent project.

## Caveats

- **Prices and ToU are as of the paid review (2026-07-12) and the 2026-07-16 audit.**
  Re-read any vendor's terms for AMC's specific commercial + cache + train use before
  ingesting — free is not cleared, and a paid API is not automatically clear either (the
  CME lesson).
- **The ledger is the gate.** The single most valuable dataset in the whole program is
  AMC's own ledger, and the joins that constitute the proprietary edge cannot be *populated*
  until AMC's exports arrive. The market-derived components (Parts B, decisions 1–3) do not
  wait on it; the dollar-VaR-on-book join does.
- **Build, don't buy, is the default — not a slogan.** Each derived construct was checked
  against the four research priors; the ones that survive do so because they are operational
  or tail-exceedance objects, not fitted regime features. The moment a "derived feature" is
  really a sentiment or regime signal in disguise, P1 applies and it starts from "no."
