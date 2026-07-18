# Paid Data for AMC: A Verified Buy/Skip Review

**Prepared 2026-07-12.** Companion to `results/amc_data_acquisition_program.md` (the five
free, start-now collectors) and to the Phase 7 research portfolio
(`plans/phase_7_amc_program.md`). Audience: deep business experience, above-average but
non-expert statistics. Acronyms are spelled out on first use. All prices are as verified
on 2026-07-12; each carries its evidence status.

---

## Bottom line

Only **two purchases** clear the bar, together roughly **$425–725 in year one**:

1. **Databento** — a one-time backfill of official Chicago Mercantile Exchange (CME)
   history (~$0–125, likely free inside signup credits; an options add-on ~$50–300),
   which closes the program's one explicitly paywalled gap — historical daily open
   interest — sixteen years at a stroke.
2. **Greysheet (CDN "Coin Dealer Digital")** — $299/yr for the dealer-to-dealer
   wholesale coin bid/ask benchmark, the one side of the coin market the free
   collectors cannot see.

Everything else surveyed is **free, deferrable, or a documented skip** — including
several free sources found along the way that are strong enough to join the Phase 7
collector program. A separate short list of *dealer-operations* data (converter
databases, trade credit ratings, wholesale feeds, probate leads) is flagged for AMC's
owner: valuable, but business decisions rather than research ones.

## About this review

The five collectors in the companion program are all free or self-built; this note asks
what *money* could add. A structured survey (2026-07-12) covered six categories —
exchange and futures data, coin and numismatic services, physical-market assessments,
macro consensus data, news and sentiment feeds, and options / implied volatility — with
every price checked against the vendor's live page that day and classified
**verified-on-page**, **reported-secondhand**, or **quote-only**. Every recommended
candidate was then re-verified by an independent adversarial pass (re-fetch the pricing
page; attack the claimed benefit). Two candidates failed that pass and their verdicts
were reversed — both are documented below, because *why* something was rejected is as
reusable as what was bought.

Four principles governed the verdicts:

1. **Don't pay for what the free collectors already capture** (posted retail premiums,
   search interest, forward open interest, calendars, AMC's own ledger).
2. **One-time historical purchases are backfillable by definition** — they can be bought
   whenever a project actually needs them. Zero urgency unless trivially cheap; the
   urgency argument that drives the collector program does not apply.
3. **The program's own pre-registered null rules out news-sentiment feeds.** Text
   sentiment features added zero forecasting lift and hurt out-of-sample. A product
   class the research has already falsified starts from "no."
4. **Small-business budget.** Hundreds to low thousands of dollars a year. Anything
   quote-only with an enterprise sales motion is presumptively out.

---

## Buy 1 — Databento: the CME data, both legs

**Scope widened 2026-07-15.** This was written as a *historical* backfill, on the
assumption that ongoing daily figures would be captured free by scraping CME's
website. That assumption failed twice over: CME's Data ToU bar the scrape for AMC's
use outright (commercial, model-training, and cached-dataset grounds — see the
revised Collector 4 in `amc_data_acquisition_program.md`), and the series turned out
to be **backfillable anyway**, since Databento retains `statistics` permanently.
Databento is therefore the source for both legs. The forward leg is ~**$1/month**
(a `StatMsg` is 64 bytes; ~27 MB/month for four parent symbols) and needs **no
market-data licence** — the 24-hour embargo keeps historical pulls outside real-time
licensing, and the $35,220/yr historical distribution fee is Databento's vendor
obligation, not a subscriber's. Live data would cost ~$900/month (GC/SI on COMEX and
PL/PA on NYMEX = two DCMs; non-display "Research and Analysis" doubles accordingly)
to buy back one day of freshness — indefensible for a days-to-weeks float.

**One check before signup.** The 2010-06 → 2017-05 leg is reconstructed from CME's
legacy MDP2 feed (tag-52 timestamps with `F_BAD_TS_RECV`; pre-2015-01-20 `stat_flags`
off-spec). Whether `statistics` is complete across that era is unconfirmed, and it is
the leg being paid for. Ask Data Sales first — the free credits expire six months from
**signup**, not from first use, so signing up before you are ready to pull burns them.

**What it is.** Databento resells the official CME market-data feed at usage-based
(per-gigabyte) prices with no subscription required for historical downloads. Its
`statistics` data schema carries the exchange's **official daily settlement price, open
interest, and volume per instrument** — futures *and* options strikes — for gold (GC),
silver (SI), platinum (PL), and palladium (PA), plus 30-day federal-funds futures (ZQ),
back to **June 2010**. The same feed serves 1-minute price bars.

**Price.** Usage-based; new accounts get **$125 in free credits** (six-month expiry),
and the portal's cost estimator quotes exact dollars before any purchase — so there is
no cost risk. The daily statistics backfill for four metals plus ZQ is a few hundred
megabytes; 1-minute bars for ~90 Federal Open Market Committee (FOMC) announcement days
are tens of megabytes. Realistic total: **$0–125**, very likely $0. The options-level
statistics (settlement + open interest per strike, 2010+) add an estimated **$50–300**
one-time. *(Verified on databento.com/pricing, 2026-07-12; a fallback flat-rate plan is
$199/mo for new signups — not needed for a one-time backfill.)*

**Why it clears the bar despite principle 2.** It is the one paid item that converts a
*wait* into a *now*: Collector 4 (forward open-interest capture) was scoped as "pure
seed corn" precisely because history is paywalled. This purchase is that history — at a
price near zero:

- **PGM liquidation-alarm labels today** (project 7.6): the ~38 forced-selling onsets
  since 2010 become trainable now instead of after years of forward accrual.
- **Shadow-positioning nowcast labels** (7.6) for the same reason.
- **Kuttner-style monetary surprises** (7.2): 1-minute GC/SI/PL/PA + ZQ bars around the
  2:00 p.m. ET FOMC statement upgrade the surprise series from the daily 2-year-yield
  stand-in — the exact "small new ingest" the Phase 7 plan flagged as optional.
- **Self-computed implied volatility** (7.3/7.4): from daily option settles per strike,
  one day of code (the standard Black-76 formula) yields at-the-money implied
  volatility and 25-delta skew for gold and silver, 2010–2026 — a forward-looking
  benchmark for the spread-floor tail engine and the event-window volatility card. This
  kills any case for OptionMetrics, ORATS (which carries no futures options at all), or
  paid CVOL history.

**Caveats.** Coverage starts 2010-06 (~125 FOMC meetings); pre-2017 data is
reconstructed from CME's FIX flat files — spot-check early-2010s completeness during the
free-credit test before paying anything. License permits internal use, no
redistribution. Collector 4 still runs forward regardless: the backfill and the
collector meet in the middle.

## Buy 2 — Greysheet: the wholesale coin benchmark

**What it is.** CDN Publishing's Greysheet is the U.S. coin trade's **dealer-to-dealer
wholesale bid/ask benchmark** — the sheet every counterparty quotes against. The "Coin
Dealer Digital" tier is **$30/mo or $299/yr** *(verified digit-for-digit at
greysheet.com/about/subscribe, 2026-07-12)*: twelve digital monthly issues,
hourly-updated online wholesale values across U.S. coin series plus proof sets, type
coins, world bullion gold, and bags; CAC (Certified Acceptance Corporation) values;
two users; basic API (application programming interface) access. Heavier API use is
metered on top (~$25/mo minimum; $95/mo at 50k calls).

**Why it clears the bar.** The verification pass confirmed this is a genuine gap, not a
duplicate: nothing free covers the **wholesale bid** — the side AMC actually trades
against when exiting inventory. Value arrives on day one with no modelling: are the coin
desk's buy/sell quotes and inventory marks in line with the wholesale market? It also
anchors generic/bullion-coin bid levels for the spread-floor work.

**Scope corrected 2026-07-16 — Greysheet is bigger than this section assumed, and
Collector 2 is gone.** The ToU audit (journal.md, 2026-07-16) found that Collector 2's
retail scrape was never licensed for AMC's use and retired it — so the "Collector 2
scrapes posted retail asks" premise no longer holds. It also found the CDN **Public API
V2** exposes **CPG retail values in addition to wholesale bid/ask**, so this subscription
is not only the wholesale benchmark: it is the licensed path to *both* sides. Two
consequences: (1) Greysheet is now the replacement for most of the coin-premium panel,
not a complement to it; (2) one construct caveat — CPG retail is CDN's *published
benchmark*, not APMEX's/JM Bullion's *posted asks*, so dealer-specific spread
intelligence is not reconstructable from it (better-grounded for spread floors, but a
different number; dealer-specific asks need written consent from the dealers). **Before
subscribing, read the API Terms of Use / License Agreement for commercial use, storage
into a local database, and model training** — a paid API is not automatically clear on
any of them (the CME lesson). The `licensing/1_greysheet_cdn.md` draft asks sales those
three questions in writing.

**Caveats.** Skip the Pro tier ($1,850/yr, a dealer trading network — overkill at AMC's
volume). The depth of *downloadable historical* bid data is not stated publicly — worth
one email to sales, because a deep bid archive would be premium-history gold. Do not pay
anyone else in this category: PCGS and NGC price guides, population reports, and
cert-verification APIs are free (register the free PCGS API key), and CAC values arrive
inside the Greysheet subscription.

---

## Where verification changed the answer

Three candidates survived the survey but not the adversarial pass — or survived it
with a materially different verdict. Recorded so nobody re-litigates them from the
same starting claims.

**Norgate Data ($270/yr — price verified).** Daily settlement + volume + open interest
back to **1978–1982** (GC from 1979, SI from 1978, PA from 1982): the only budget source
covering the 1980 silver collapse and the 1997–2001 palladium squeeze. The surveyed
framing — "buy six months for $148.50, export everything, cancel" — was **refuted**: the
license requires deleting all data (and derived data) on lapse, so compliant use is a
recurring $270/yr. Decision rule: revisit only if the liquidation-alarm work
demonstrates that pre-2010 episodes materially improve it; start with Databento's
2010+ sample. (Tooling is Windows-only; workable via the Windows host, another
friction.)

**FXMacroData ($250–1,000/yr — price verified, benefit refuted).** Advertised as
point-in-time macro consensus history — exactly the shape needed to extend the FOMC
event study to CPI (Consumer Price Index) and payrolls surprises. The verifier tested
the vendor's own free API and found the historical "consensus" is **retro-generated**:
predictions attached to a 2002 CPI announcement carry a generation timestamp of
2026-07-11. That is precisely the look-ahead contamination this codebase's leakage
guards exist to catch — a model trained on it would quietly overstate real-time
performance. **Skip**, and note the general lesson: *paid data can carry leakage too;
"point-in-time" is a claim to test, not a label to trust.* The in-house alternative is
free: first-print actuals from ALFRED (the St. Louis Fed's real-time data archive) plus
a scraped consensus column reconstructs surprise series, and the published
Gürkaynak–Sack–Swanson and Acosta FOMC-surprise series are free downloads.

**CME CVOL implied-volatility history (quote-only).** CME's official 30-day implied-vol
indices for gold, silver, and platinum (with skew), up to ~9 years licensable. Demoted
to *try-free-first*: a free DataMine registration may already grant ~2 years of
downloads, the silver vol index VXSLV relaunched free in mid-2025, and the Databento
options backfill makes self-computed measures the primary source anyway — CVOL becomes
a convenience benchmark, worth at most a small one-time quote.

## Free upgrades found along the way

The survey's most valuable output may be free. Each of these fits the Phase 7 collector
pattern (append-only, provenance columns, real-time flags where relevant):

- **Johnson Matthey PGM base prices — including rhodium, iridium, ruthenium.** Free CSV
  downloads, decades of history, and a verified programmatic pull. Rhodium is typically
  the **dominant value component in catalytic-converter scrap** and has no exchange
  price; nothing in the current stack prices it. This is a strong candidate for a
  sixth collector (one-time historical pull + small forward capture), and it makes paid
  minor-PGM feeds (Fastmarkets, Argus) unnecessary.
- **World Gold Council Goldhub — India/China local premium series.** Free, daily
  (5-day rolling average, updated ~weekly), India from 2012, China from 2003: a regional
  physical-demand thermometer behind coin-premium intelligence. **ToU gap corrected
  2026-07-17 — see the Addendum below: "free" was mistaken for "cleared." Goldhub's
  terms were never run against the AMC gate (commercial + model-training + cached-local)
  and plausibly fail it as CME did; reclassified to barred-pending-written-consent.
  Quarantine — do not build a loader — until a licence clears.**
- **Cboe GVZ gold implied-volatility index** — free via the existing FRED collector,
  2008+; silver sibling VXSLV relaunched 2025.
- **eBay Terapeak** — realized (not posted) bullion transaction prices, free with a
  seller account, but a **rolling three-year window that ages out permanently**. Same
  logic as the collectors: start periodic snapshots of the benchmark-product queries
  now. **WorthPoint** ($30/mo, cancel after) can one-time-backfill realized premiums to
  ~2006 — buy one or two months only when the historical premium study actually runs.
- **Auction archives** — Heritage (1997+) and Stack's Bowers (2002+) realized prices,
  free with registration: the cross-check for rarer material walked into the shop.
- **Factiva via university library access** — free, for hand-dating PGM supply events
  (the 7.5 event ledger); manual verification only under the personal-use license. If
  trade-press paywalls block the sprint, one month of The Northern Miner + MINING.COM
  is ~USD $11.

## The skip list

Documented with representative prices so the conclusion is reusable.

| Product class | Representative price | Why skipped |
|---|---|---|
| News-sentiment feeds (RavenPack, Dow Jones DNA, Benzinga, Marketaux, Tiingo) | ~$20–40k/yr (RavenPack academic); five-figure+ (DNA); $30–200/mo (budget APIs) | The program's own pre-registered null: sentiment features added zero lift and hurt out-of-sample. Budget APIs also lack archive depth for the one surviving use (event dating). |
| Enterprise physical-market assessments (Metals Focus, Fastmarkets, Argus, SFA Oxford, LSEG/GFMS) | $1,500/report/yr (Metals Focus, verified); rest quote-only enterprise | Annual-frequency structural data at terminal-lite prices; free substitutes cover the need — Silver Institute World Silver Survey (same producer's silver numbers, free), WGC Goldhub, Johnson Matthey's annual PGM report, WPIC quarterlies. |
| Macro consensus / calendars (Trading Economics, Econoday, MNI, Haver, Consensus Economics, Citi surprise index) | $2,388/yr + 500 req/mo (TE, secondhand); rest quote-only or Bloomberg-locked | ALFRED first prints + a scraped consensus column reconstruct surprise series free; official BLS/BEA/Fed calendars are already Collector 5. |
| Other futures/options vendors (CME DataMine direct, TickData, Portara, FirstRate Data, Barchart Premier, OptionMetrics, ORATS) | Portara ~$1.1–1.65k one-time (pre-2010 intraday); Barchart $199.95/yr; others quote-only or no-OI | Dominated by Databento for this use. Portara is the one principled *later*: 1990s intraday if the FOMC study ever needs 300+ meetings. ORATS carries no futures options. |
| CPM Group yearbooks | $170/book | Fine one-time references (PGM volume most differentiated, for rhodium detail); backfillable — buy the book when a specific table is needed. |

## Dealer-operations data (outside the research scope, flagged for the owner)

An adversarial completeness pass asked what a *quant* survey structurally misses about a
*dealer*. Four finds — real prices, real fit, but business decisions rather than
research inputs:

- **Catalytic-converter PGM content databases** (Eco Cat, AutoCatalystMarket;
  ~$20–115/mo): serial-number → recoverable Pt/Pd/Rh content and value, refreshed
  against current prices. Arguably the highest *operational* value per dollar on the
  scrap side — per-piece buy pricing at the counter.
- **Jewelers Board of Trade** ($195/yr): the jewelry trade's credit bureau — ratings and
  bankruptcy alerts on refiners/wholesalers. AMC's metal sits unsecured at
  counterparties for days-to-weeks; the 2018 Republic Metals bankruptcy burned exactly
  this class of dealer. Doubles as local-market competition intelligence.
- **Wholesale market-maker price feeds** (Dillon Gage FizTrade/FizConnect; similarly
  A-Mark, MTB): live two-way wholesale prices on hundreds of coin/bar products, free to
  nominal with an approved trading account. This is the **exit side** of the premium —
  retail and wholesale premia decouple exactly in the stress episodes the premium
  program targets. If AMC holds such an account, logging the feed daily is a
  non-backfillable series and a natural Collector 7.
- **County probate feeds** (~$20–150/mo): estates are the canonical origin of coin
  collections and scrap lots — buy-side deal flow plus a measurable local leading
  indicator for walk-in supply. Test one county before scaling.

## Summary

| Item | Cost (evidence) | Decision improved | Action |
|---|---|---|---|
| Databento CME backfill | ~$0–125 one-time (verified) | PGM alarms, shadow positioning, FOMC surprises | Buy now, inside free credits; estimator first |
| Databento options add-on | ~$50–300 one-time (verified model) | Spread-floor tail engine, event vol card | Buy with the same pull |
| Greysheet Coin Dealer Digital | $299/yr (verified) | Coin desk pricing, premium ground truth | Subscribe now; email sales re: bid history |
| JM PGM prices (incl. rhodium) | $0 (verified) | Converter-scrap pricing, PGM alarms | Add as Collector 6 |
| GVZ, Terapeak snapshots, auction archives, PCGS API | $0 (verified) | Premium intelligence, vol card | Fold into Phase 7 collectors |
| WGC Goldhub India/China premia | $0 but **ToU unverified** | Premium intelligence | **Quarantine pending written consent** (2026-07-17 gap; Addendum) |
| WorthPoint | $30–60 total, 1–2 months (secondhand) | Realized-premium backfill study | Defer until that study runs |
| Norgate deep history | $270/yr recurring (verified; license refuted one-time framing) | Pre-2010 liquidation episodes | Defer; revisit on demonstrated need |
| Portara pre-2010 intraday | ~$1.1–1.65k one-time (verified) | 1990s FOMC meetings | Defer indefinitely |
| FXMacroData | $250–1,000/yr (verified; benefit refuted — retro-generated consensus) | — | Skip; build surprises from ALFRED + scrape |
| Sentiment feeds, enterprise assessments, macro terminals | five figures / quote-only | — | Skip; free substitutes documented above |
| Converter DB, JBT, FizTrade logging, probate feeds | $20–195/mo-yr range (mixed) | Counter pricing, counterparty risk, exit premia, deal flow | Owner's call — operations, not research |

## Caveats

- **Prices dated 2026-07-12.** Vendor pricing moves; "verified-on-page" means read from
  the vendor's live page that day, "reported-secondhand" from a credible third-party
  source, "quote-only" means no public price exists — expect institutional pricing.
- **Licensing is part of the price.** Every recommendation was checked for
  single-business internal research use; none permit redistribution. Norgate's
  delete-on-lapse term is the cautionary example — a license can void an otherwise
  correct purchase plan.
- **Paid data can carry leakage.** The FXMacroData refutation generalizes: any vendor
  claiming "historical consensus/point-in-time" data must prove generation timestamps
  precede the events. The same real-time honesty stamps the collectors use apply to
  purchased history.
- **Nothing here displaces the five collectors.** Purchased history complements forward
  capture; the single most valuable dataset in the program remains AMC's own ledger,
  and it costs nothing.

---

## Addendum 2026-07-17 — Phase 8 (SSL / representation) data & methods review

Scoping Phase 8 (`plans/phase_8_ssl_probing.md` — a self-supervised low-rank
representation of the daily price + GDELT-news state, framed as *insight*, not
prediction) raised two questions this review had not asked: **(Q1)** does paid data
relax the phase's "four hard facts," and **(Q2)** would a LoRA or distillation *method*
be a better approach than buying data? A four-track researched-and-critiqued survey
(web-verified where possible) answers both.

**Bottom line.** **No in-budget, ToU-clean dataset relaxes the phase's binding
constraint, and no LoRA/distillation method overturns it either.** The two buys above
(Databento CME backfill, Greysheet) are unchanged and **no new purchase is justified.**
The representation framing mints **three near-free builds and one compliance
correction** — all below. The two governing truths: the binding constraint (the *joint*
price+news sample) is unbuyable within budget + ToU + the Phase-6 prior; and LoRA /
distillation change *capacity/transfer*, not *information*, so they cannot manufacture
signal on an information-constrained problem.

### The four hard facts (from the Phase-8 plan)

1. Prices are **daily OHLCV only** (Yahoo), no intraday/tick.
2. **No per-metal news** — GDELT themes are industry-wide and collapse to one shared
   daily `market` news-state; the corpus has no article bodies, only titles/slugs (real
   titles from 2019-09-22) + theme codes + tone.
3. **Tiny joint sample** — news overlap caps the price+news model at ~2,800 daily rows
   (GDELT 2015+), ~1,700 with real titles; ~99% autocorrelation ⇒ **~40–50 effective
   independent regimes**, not thousands.
4. **Adverse prior** (`phase6_validation.md`): classical baselines beat ML and
   regime/sentiment features *hurt* OOS.

### Q1 — per-fact data verdict

- **FACT 1 (daily-only prices).** The fix — Databento full 1-minute history — is already
  licensed (extending the FOMC-window slice is tens of dollars of usage, not a purchase),
  but it is **decoration for AMC's decisions**: the float is days-to-weeks, FACT 2 caps
  the joint sample regardless of price resolution, and the FOMC-window slice that matters
  is already in Buy 1. Full intraday only feeds a *price-only* encoder — the class FACT 4
  penalizes. `build`, but gated behind the price-only arm surviving Stage A.
- **FACT 2 (no per-metal news).** No budget/ToU-clean feed exists. Entity-tagged
  sentiment (RavenPack / Bigdata.com, ~2003+, ~$20–40k/yr academic = **non-commercial**,
  the CME/personal-use trap) is a **richer version of the exact sentiment class Phase-6
  falsified** (`lgbm_sentiment` Diebold-Mariano t = +2.90) — it starts from "no."
  Reaffirms this review's sentiment skip. The only lever that touches FACT 2 is a
  *method* over owned text — the LLM annotator (Q2), not a dataset.
- **FACT 3 (tiny joint sample) — the binding constraint, and it is unbuyable.** Pre-2015
  news that could extend the *joint* panel is enterprise-priced and ToU-barred
  (Factiva/DNA five-figure+, LexisNexis; a university seat is personal-use only — no model
  training, no caching), and every candidate lands on the falsified sentiment class.
  GDELT 1.0 (1979+) is feature-incompatible (no GKG themes/tone before ~2013). And the
  arithmetic dissolves the benefit anyway: ~3,000 extra rows buy only **~20–30 additional
  *independent* regimes**, largely duplicating what free price-only history already spans.
  A genuine, money-saving **null-for-buying**.
- **FACT 4 (adverse prior).** No data "fixes" an empirical finding, but the class most
  likely to *move* it is **orthogonal physical-market data Phase-6 never tested — mostly
  free/already-flagged, not richer price or news** (two builds below).

### The three near-free builds (zero new purchase; all gated by baseline-first)

1. **LLM-as-annotator** — Claude reads each day's GDELT *titles* and emits per-metal,
   event-typed features (~**$30–150 one-time**, clean ToU: Anthropic commercial terms
   permit it, features are AMC's, nothing third-party redistributed/cached). *The only
   lever on either question that adds information* — it recovers per-metal structure the
   `market`-row collapse discards. Non-negotiable caveats: **(i)** titles-only ~1,700-row
   run primary, and the PGM channel (where AMC most needs it) will be sparse-to-empty;
   **(ii) parametric leakage** — a dated title lets the model "know what happened next";
   date-blind the prompt, treat labels as hindsight-colored, cross-check a date-blinded
   re-run (LLM output is not bit-reproducible even at temp 0 — a provenance liability);
   **(iii)** its output *is* sentiment polarity + regime flags (two falsified classes), so
   it must clear the same incremental-IC + block-permutation null as any news arm, and a
   **clean null is the modal, shippable outcome** — its value is *closing* "should AMC buy
   a news feed?" empirically.
2. **Databento-derived lease/forward-rate alarm** — GC/SI calendar spread + ZQ (**$0** on
   the owned licence, point-in-time clean — settlements are as-of-dated, avoiding the
   FXMacroData retro-generation trap). Its durable value is an **operational tightness
   alarm for a physically-long dealer** (silver 1-month lease ~35–40% in Oct 2025 is a
   rare regime *event*), not a predictor: the instant the backfill adds contract months,
   the calendar spread is inside `X_price` and the Phase-8 §2.1 incremental-IC residualizer
   eats most of its "orthogonality." Term/calendar spreads *are* new versus the Yahoo-built
   `spreads.py` panel, so a candidate feature starts from a **neutral** prior (not "no") —
   pre-register a possible null IC. **Build from raw per-contract settlements with a
   point-in-time roll**; a back-adjusted/stitched series injects roll look-ahead.
3. **US Mint bullion-sales collector** — **$0**, US-gov open data, clean on all three ToU
   grounds. Marginal (monthly collapses against FACT 3) and a supply-rationed *noisy
   demand proxy* — a slow thermometer for the coin/premium arm, not a daily feature.

### Compliance correction — WGC Goldhub reclassified

This review listed **WGC Goldhub India/China premia** under "free upgrades adopted" with
only "Free, daily" as justification — it **never ran WGC's ToU against the AMC gate.**
Goldhub's commercial/scrape/derivative/cached-use terms plausibly fail the same three
grounds CME failed. **Reclassified to barred-pending-written-consent — quarantine before
any loader reads it** (no Goldhub table exists yet; do not build one). Also flagged in
`plans/phase_7_amc_program.md` (collector 7 + the physical-tightness nowcast) and
`plans/phase_8_ssl_probing.md`. The general lesson: *"free" is not "cleared" — a free
source's ToU must still be run against AMC's actual use.* **Unverified:** the specific
T&C wording was not independently re-fetched this session (PLAUSIBLE-pending-confirmation);
the quarantine default holds either way.

### Q2 — LoRA / distillation: capacity, not information

**The distinction the question conflates.** LoRA's low-rank *weight update* (ΔW = B·A,
parameter-efficient fine-tuning of a pretrained net) is a different object from the
Phase-8 low-rank *representation* (the `PLSCanonical`/PCA axes you probe). LoRA does not
produce that representation — so "use LoRA" is a proposal to fine-tune some external
pretrained encoder, which re-imports every leakage/prior problem below. **These methods
change how much of a pretrained prior you can transfer and how cheaply you can serve it;
they add zero information about the metals that is not already in AMC's data or an
external model's (undisclosed, probably contaminated) corpus.**

Ranked verdict:

| Method | Verdict | Why |
|---|---|---|
| **LLM-as-annotator** | **build** | Only lever that adds information; extracts latent per-metal structure from owned titles. Caveats above. |
| TSFM frozen encoder (IBM Granite TTM; Datadog Toto as low-leakage control) | **defer** — one gated experiment | Apache-2.0, weights local. Best on-point study restates Phase-6 (TSFMs don't uniformly beat HAR). One null-tolerant *insight* run gated behind Stage A; never a forecaster, never fine-tuned on the backtest window. |
| TSFM via LoRA (Chronos-2 / TimesFM-2.5) | **defer → skip** | Its one claimed edge — "the prior never saw your test window" — is unprovable and probably false (undisclosed corpora, plausibly include gold/commodities → FXMacroData-class contamination). Moirai is CC-BY-NC (fails commercial); TimeGPT sends data off-machine (fails). |
| LoRA on a text encoder (MiniLM / FinBERT) | **skip** | The plan freezes the text tower by design; LoRA only sharpens a channel Phase-6 showed *hurts*. |
| Distillation (all variants) | **skip** | Label-distillation is an inference-cost play at 2,800 rows; cross-modal text→price distillation *is* CoMPASS Stage B relabeled; TSFM→student re-imports the adverse prior. |

**Dominant trap:** a pretrained external TSFM cannot be re-pretrained per walk-forward
fold, so freezing-then-probing it re-imports the full-history pretrain leak (Phase-8
§5.1) by construction. On the leakage-strict standard, unverifiable contamination = treat
as contaminated — condemning every finance-touching corpus at meaningful power.

### Action delta vs this review

**No change to BUY.** New **BUILD** (near-free, gated): LLM-as-annotator (~$30–150),
Databento lease-rate alarm ($0), US Mint collector ($0); conditional/gated: full-intraday
extension and one TTM/Toto frozen experiment, both only after Stage A greenlights.
**DEFER add:** WGC Goldhub (quarantine). Everything previously skipped stays skipped, now
reinforced by the SSL/Phase-6 framing; LoRA-on-text and all distillation join the skip
list. The modal outcome remains a **pre-registered null** — which is itself the shippable
recommendation "use the classical vol baseline, don't buy a news feed, keep news out of
the scenario context," reached for ~$30–150 rather than wondered about at ~$20–40k/yr.

### Unverified flags (2026-07-17)

- Two TSFM-vs-classical citations (arXiv 2511.18578; "Brini 2026") — **not independently
  verified**; carried from the method tracks.
- The TSFM "leak-safe prior" claim — asserted, **probably false**; corpora undisclosed.
- A Chronos-2 partial-eval-contamination claim — carried as stated, not re-verified.
- **WGC Goldhub T&C wording** — PLAUSIBLE-pending-confirmation; quarantine default holds.
- Stooq / FirstRate ToU, Kronos licence — **unverified**; barred-pending-verification per
  the CME default.
- Databento pre-2017 `statistics` completeness — **unconfirmed** (the leg being paid for);
  spot-check inside the free credits before paying (as in Buy 1).
