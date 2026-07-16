# The AMC Data Program: Five Collectors to Start Now

**Prepared 2026-07-12.** Companion to `results/phase5_amc_business_implications.pdf`
(what the Phase 5 causal findings mean for AMC Company) and to the Phase 7 research
portfolio (`plans/phase_7_amc_program.md`); the paid-side counterpart — what money
could add to these free collectors — is `results/amc_paid_data_review.md`. Audience:
deep business experience, above-average but non-expert statistics. Acronyms are
spelled out on first use.

---

## Bottom line

The most valuable inputs to AMC's next round of research share one property: **they
cannot be backfilled**. Coin premiums, search interest, and AMC's own transaction
records exist only if someone records them as they happen.
Each of the five collectors below takes roughly one to three days to build; together
they are about seven to ten working days. **Every week of delay is a week of
irreplaceable data lost.** None of them requires any modelling to pay for itself —
three of the five deliver useful business information from the first week of
operation.

**Correction (2026-07-15):** this note originally listed *futures open interest*
among the non-backfillable series. That was wrong. Databento retains CME's
`statistics` schema (settlement, open interest, cleared volume) permanently, so the
series can be pulled retroactively at any time — see Collector 4 below. The error
mattered: it manufactured urgency around the one collector that never needed it, and
that urgency is what motivated an approach that turned out to violate CME's Terms of
Use.

## About this note

AMC Company buys scrap gold, silver, platinum, and palladium — assaying each lot to
establish its fine-metal content — and buys and sells gold coin and specie. It is
structurally long physical metal over a days-to-weeks holding float, which makes
event-driven price risk (documented in the Phase 5 research) its core inventory risk.

A structured brainstorm (2026-07-12) produced a ten-project research portfolio for
AMC — spread-floor models, hedging playbooks, crisis indices, liquidation alarms,
demand forecasting. Under adversarial review, nearly every high-value project traced
back to the same bottleneck: **a data series that does not exist yet and cannot be
reconstructed later.** This note specifies those series as five concrete "collectors"
— small, scheduled data-capture jobs feeding the project's single research database
(DuckDB, the local analytical database that already holds prices, macroeconomic
series, news, and futures positioning).

A reading note: "backfillable" means a series can be downloaded later with full
history (like prices from an exchange). "Non-backfillable" means history is either
never published, rewritten after the fact, or paywalled — so the only honest record
is the one you capture in real time.

---

## Collector 1 — AMC's own ledger (highest priority)

**What it is.** A database schema and import routine for AMC's own transaction
records, exported periodically (for example weekly, as a spreadsheet) from AMC's
point-of-sale or bookkeeping system. Three tables:

- `amc_scrap_lots` — one row per scrap purchase: date, metal, gross weight, assayed
  fineness, fine ounces, price paid, spot price at purchase, and — when the lot is
  sold or refined — disposition date, type, and proceeds.
- `amc_coin_trades` — one row per coin/specie trade: date, side (buy/sell), product
  (e.g., 1 oz American Gold Eagle), quantity, unit price, spot at trade — from which
  the realized premium over melt value is computed.
- `amc_till_daily` (optional) — daily counts: walk-ins, offers made, offers accepted.

**Why start now.** Every research question that matters to AMC ends at this join.
Without it, the research can only price risk *per unit of metal*; with it, risk
becomes *dollars on AMC's actual book*. The float-duration distribution (how long
metal actually sits between purchase and sale) is the single number that converts the
Phase 5 event findings into hedge sizes. Realized coin premiums are the ground truth
that no scraped benchmark can replace. Paper records may allow partial
reconstruction, but consistent capture starts only when the schema exists.

**What AMC gets on day one.** Descriptive answers before any model: which metals sit
longest, realized margin per lot by size and metal, premium earned by product, and
the float-duration histogram — each individually useful for pricing and staffing.

**What it unlocks.** True inventory Value-at-Risk (VaR — a standard measure of how
much a position could plausibly lose) in the spread-floor project; the target
variable for scrap-inflow forecasting; validation data for the premium panel
(Collector 2).

**Build notes.** One database migration; a validating importer
(`metals.data.amc_ledger`) run on each export; everything stays on the local
machine — no cloud service touches AMC's books. Effort: one to two days of build,
plus setting up the export on AMC's side.

## Collector 2 — Retail coin-premium panel

**What it is.** A once-daily capture of posted retail prices for a fixed basket of
benchmark bullion products — 1 oz American Gold Eagle and Silver Eagle, 1 oz Canadian
Maple Leaf, 90% "junk" silver, generic 1 oz rounds and bars — from two large online
dealers (e.g., APMEX and JM Bullion), storing the dealer's ask, its buyback bid where
published, the spot price at capture, and the timestamp; from these, the premium over
melt value per product per day.

**Why start now.** No reliable free history of retail premiums exists. Internet
Archive snapshots are sporadic and biased toward famous episodes, so they can
validate but never train a model. Premium blowouts — September 2015 rationing, March
2020, February 2021, March 2023 — are exactly the episodes AMC monetizes, and the
next one will only be in the dataset if the collector is already running.

**What AMC gets on day one.** A daily benchmark: are AMC's coin prices and buyback
spreads in line with the national online market? That is pricing intelligence with no
modelling at all.

**What it unlocks.** Ground truth for the physical-tightness index (built from
futures-versus-ETF pricing gaps already in the database); eventually, a premium
forecasting model; a calibrated "crisis playbook" for the coin desk.

**Build notes.** A polite scraper — a handful of product pages, once daily,
respecting each site's terms and robots.txt, for internal research use — plus a
database migration and a breakage alert (retailers redesign pages; the collector must
fail loudly, not silently). Effort: one to two days, plus occasional maintenance.

## Collector 3 — Search-interest archiver (Google Trends)

**What it is.** A weekly (daily during spikes) pull of a fixed set of search terms —
"sell gold," "cash for gold," "gold price," "sell silver," "coin shop near me" — for
the United States (state-level where feasible), storing the raw response and the pull
timestamp.

**Why start now.** This is the subtlest of the five. Google Trends *rescales* its
index on every request, so a series downloaded later is not the series a real-time
observer would have seen. A model trained on retroactively downloaded data quietly
overstates how well it would have worked. The only honest history is an archive of
as-pulled snapshots — which therefore starts accruing value on the day the collector
is turned on, and not one day before. (History pulled at setup time is kept, but
permanently flagged "not captured in real time.")

**What AMC gets on day one.** A same-week gauge of public interest in *selling* —
the best available leading indicator for walk-in scrap volume.

**What it unlocks.** The scrap-wave anticipation model (pre-positioning cash and
assay staffing ahead of inflow surges), with the ledger (Collector 1) as its
eventual target.

**Build notes.** A scheduled pull via the public interface, stored verbatim with
request parameters. Effort: about one day.

## Collector 4 — Futures open-interest collector (CME)

**What it is.** A daily record of settlement volume and open interest — the number
of futures contracts outstanding — per metals contract (gold, silver, platinum,
palladium), licensed from Databento.

**Superseded 2026-07-15 — read this before building anything here.** The original
design scraped CME's website. Both of its premises were wrong:

- *"Forward capture is free; hindsight is not."* **False.** Databento retains the
  `statistics` schema (settlement price, open interest, cleared volume, block volume)
  permanently. The series is fully backfillable at any later date, so there is no
  clock on this collector and never was.
- *"Each day's figures are public as they appear."* **Publicly visible, but not
  licensed for this use.** CME's Data Terms of Use prohibit using "scripts, software,
  spiders, robots... to navigate, access... retrieve, harvest... any portion of the
  Website" absent prior written permission, and limit access to "personal use for
  non-commercial purposes" — expressly excluding software development, model training,
  and the creation of "archived or cached data sets." AMC's use fails all three
  independently, and manual download does not cure it. The Akamai block encountered
  in build is the enforcement of Advisory Chadv23-364, effective 2024-01-08.

**Why not now.** No urgency exists. Pull it when convenient.

**Replacement source.** Databento `statistics` on `GLBX.MDP3`, `stype_in=parent`,
symbols `GC.FUT,SI.FUT,PL.FUT,PA.FUT`. Roughly **$1/month** for the forward leg
(a `StatMsg` is 64 bytes; ~27 MB/month all-in), and **no market-data licence is
required** — Databento embargoes historical data at 24 hours precisely to stay
outside real-time licensing. The trade-off is freshness: T+1 for settlement and
T+2 for final open interest, versus the scrape's same-evening preliminary figures.
Buying that day back means the live feed at roughly $900/month (GC/SI are COMEX,
PL/PA are NYMEX — two DCMs, so non-display "Research and Analysis" licensing
doubles), which is indefensible against a days-to-weeks inventory float.

**An unintuitive benefit.** The licensed replay is *better* for this codebase than
the live capture it replaces. `ts_recv` timestamps the exact nanosecond each
statistic became knowable and `update_action` preserves the full revision sequence,
so what-was-known-when reconstructs exactly. The scrape's coarse `pulled_at` was
only ever a proxy for that.

**Before funding the backfill.** GLBX.MDP3 reaches back to 2010-06-06, but
2010-06 → 2017-05 is reconstructed from CME's legacy MDP2 feed (timestamps from
tag 52 with `F_BAD_TS_RECV` set; pre-2015-01-20 `stat_flags` do not match the
current spec). Whether `statistics` is *complete* across that era is unconfirmed —
and it is the leg being paid for. Ask Data Sales before signing up, since the free
credits expire ~6 months from signup, not from first use.

**What AMC gets on day one.** Little by itself — this one is pure seed corn.

**What it unlocks.** Within months: the "shadow positioning" nowcast (estimating
where speculative positioning is *today* rather than at the last weekly government
report, which arrives three to ten days stale) and the labels for the
platinum/palladium liquidation alarm — open-interest contraction is the signature of
the forced-selling episodes that hit PGM (platinum-group metals) inventory hardest.

**Build notes.** Pull Databento `statistics` into one table (per contract month plus
a roll-neutral aggregate — per-contract open interest contracts mechanically at every
quarterly roll). Effort: about one day. The existing `src/metals/data/cme_daily.py`
keeps its parsing, provenance, and splice discipline; only the source changes.

## Collector 5 — Event-calendar and Fed-surprise upkeep

**What it is.** Three small maintenance jobs. (a) Add the 2024–26+ Federal Open
Market Committee (FOMC — the Federal Reserve committee that sets interest rates)
meeting calendar to the events table; it is public and takes an hour. (b) Add the
Bureau of Labor Statistics (BLS) release calendar for the Consumer Price Index (CPI —
the main U.S. inflation report). (c) Extend the monetary-surprise series — which
currently ends December 2023 and left the research's single most actionable finding
inoperable for live meetings — using the FOMC-day change in the 2-year Treasury
yield, a standard stand-in already present in the database daily; then keep scoring
each new meeting the same evening.

**Why start now.** The Phase 5 anchor finding (a hawkish Fed surprise knocks roughly
1.4% off gold and 2.9% off silver over the following week) is only usable if each
meeting can be scored as it happens. Unlike the other collectors this one *is*
partially backfillable — but it gates the highest-value quick win in the portfolio,
and the calendar entries are prerequisites for the event-risk work.

**What AMC gets.** A per-meeting hedge playbook that is live again — including for
the current cutting cycle the original data never covered.

**Build notes.** Calendars: hours. Surprise-series extension and re-estimation:
about three days (it is the first analysis job once the calendar lands).

---

## What unlocks what

| Collector | Effort | Feeds (portfolio project) | AMC decision improved |
|---|---|---|---|
| 1. AMC ledger | 1–2 days + AMC-side export | Inventory VaR in dollars; scrap-wave target; premium ground truth | Hedge sizing, spread floors, staffing |
| 2. Premium panel | 1–2 days | Tightness-index validation; premium model; crisis playbook | Coin pricing, buyback spreads |
| 3. Trends archiver | ~1 day | Scrap-inflow surge model | Cash and assay staffing |
| 4. CME open interest | ~1 day | Shadow positioning; PGM liquidation alarm | PGM float caps, de-risking triggers |
| 5. Calendars + Fed surprises | ~0.5 day + 3 days | Live FOMC hedge playbook; event-vol risk card | Event hedging |

## Engineering and operations notes

- **One database.** Every collector lands in the existing DuckDB file through a
  numbered migration (next free number 008), append-only, timestamps in UTC
  (Coordinated Universal Time), with `source` and `pulled_at` provenance columns on
  every row.
- **Real-time honesty stamps.** Any row captured after the fact (setup-time history,
  Internet Archive material) carries a permanent flag; models must treat flagged
  history as second-class. This is the same look-ahead discipline the research
  already enforces on features.
- **Fail loudly.** Scrapers rot. Each collector alerts on schema drift, empty
  responses, or a missed day, so gaps are noticed the week they occur — a silent
  three-month hole discovered next year is precisely the non-backfillable loss this
  program exists to prevent.
- **Scraping etiquette.** Low frequency (once daily), a handful of pages, identified
  user agent, respect for robots.txt and site terms, internal research use only.
- **Backup becomes urgent.** The laptop database is currently the *sole* copy of the
  research corpus. These collectors add data that is irreplaceable by construction —
  an off-site backup (already planned, deferred) should land in the same sprint as
  the first collector.

## Suggested sequence

1. **Ledger schema + export agreement with AMC** (Collector 1) — longest lead time
   on the business side, so start the conversation first.
2. **Premium panel and Trends archiver** (Collectors 2–3) — the clock is ticking on
   both; build in the same week.
3. **Calendars** (Collector 5a/5b) — an hour, unblocks event work.
4. **CME open interest** (Collector 4).
5. **Fed-surprise extension** (Collector 5c) — the first *analysis* job on top of
   the new plumbing, and the fastest route to a live, business-usable deliverable.

Total: roughly seven to ten working days of build, most of it parallelizable.

## What these collectors do not do

- **They are not models.** Value compounds with time; most of the downstream models
  need six to twenty-four months of accrued labels before an honest evaluation is
  even possible. The point of starting now is that the clock only runs forward.
- **Posted online premiums are not AMC's premiums.** The panel is a national
  benchmark; AMC's local market, product mix, and condition grading differ. The
  ledger (Collector 1) is the ground truth; the panel is context.
- **Search interest is relative, not absolute.** Trends measures share of searches,
  rescaled per request; it indicates surges, not volumes — useful as a leading
  indicator, unusable as a count.
- **No collector predicts anything.** Each one records reality cheaply so that
  later models can be tested honestly — the same evaluation-first discipline that
  produced the Phase 5 findings AMC can actually trust.
