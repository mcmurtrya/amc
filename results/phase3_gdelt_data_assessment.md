# GDELT data assessment (Phase 3)

**Date:** 2026-06-25 · **Corpus:** `headlines` table, 63,267,343 rows ·
**Source:** GDELT 2.0 GKG, filtered to 14 curated themes at ingestion.

This documents structural limits of the GDELT GKG corpus found while making the
Phase 3 pipeline runnable on the server. Each finding was verified against the
live DuckDB; the queries are included so they can be re-run. Two findings are
severe enough to change the research design (§1, §2).

---

## 1. 🔴 No per-metal news signal (the metal axis on text features is redundant)

All four metals receive **byte-identical** daily text features. There is no
theme that distinguishes silver / platinum / palladium (GDELT has no per-metal
theme except gold), and **gold's only theme adds nothing**:

- `ECON_GOLDPRICE` tags 993,710 articles (1.57%).
- Of those, **0** occur *without* an all-metal theme — so gold's article set is
  identical to silver/platinum/palladium's on **every one of the 2,315 days**.

Empirical confirmation on the one aggregated day in the DB (2026-05-01):
gold, silver, platinum, palladium all have `n_articles=24007`,
`embedding_dispersion=0.458`, `mean_tone_overall=-0.787`, and **byte-identical
`mean_embedding`**.

**Independently re-verified (2026-06-25)** against the full corpus, not just the
one aggregated day: across all 63,267,343 rows / 2,315 days the gold-only count is
**0**, so the four metals' article sets — and therefore every aggregated text
feature — are identical on *every* day. The mechanism is theme saturation: every
`ECON_GOLDPRICE` article carries **≥2 curated themes** (minimum observed = 2)
because `WB_1699_METAL_ORE_MINING` (56.7% of the corpus, mapped to all four
metals) tags essentially every gold-price article.

**Why:** `THEME_TO_METALS` maps only `ECON_GOLDPRICE` to gold-only; all 13 other
curated themes map to all four metals. Since `ECON_GOLDPRICE` never appears
alone, the per-metal split collapses.

**Implication:** News cannot explain *cross-metal divergence* (e.g. platinum
rallying while gold is flat) — that signal is not in this data. Per-metal text
attribution in Phase 4/5 would be spurious.

**Options:**
- **(a)** Collapse text features to a single shared daily *news-state* series and
  drop the metal axis on text. **Recommended, and exactly lossless** — the four
  series are byte-identical, and the one gold-specific scalar that exists
  (`ECON_GOLDPRICE` prevalence) already lives in the shared `daily_topic_prevalence`.
  Also removes the 4× metal fan-out in `aggregate_daily` (the GPU encode is unaffected).
- **(b)** Build a genuine per-metal signal from a real text source. The cheapest
  real path is *not* article-URL/body keyword matching on this corpus — see the
  verified source menu in **§5** (GDELT DOC 2.0 API titles + mining.com RSS).

Reproduce:
```sql
-- gold-only articles (the only rows that could differentiate gold): expect 0
SELECT count(*) FROM headlines
WHERE list_contains(from_json(themes,'["VARCHAR"]'),'ECON_GOLDPRICE')
  AND NOT list_has_any(from_json(themes,'["VARCHAR"]'),
      ['ECON_CENTRALBANK','WB_1235_CENTRAL_BANKS','EPU_POLICY_MONETARY_POLICY',
       'WB_444_MONETARY_POLICY','ECON_INTEREST_RATES','EPU_POLICY_INTEREST_RATES',
       'WB_1125_INTEREST_RATE_POLICY','ECON_INFLATION','WB_442_INFLATION',
       'WB_1164_COMMODITY_PRICES_SHOCKS','WB_1699_METAL_ORE_MINING','SANCTIONS',
       'ECON_TRADE_DISPUTE']);
```

---

## 2. ~~🔴 Coverage is 2020-01-01 → 2026-06-19 only~~ → ✅ RESOLVED 2026-07-02

**The 2015–2019 wide backfill ran 2026-07-02** (laptop DB): coverage is now
**2015-02-18 → 2026-06-19 continuous** at day granularity, 139.9M rows, with
exactly one hole — **2017-08-29 is empty upstream in GDELT itself** (BigQuery
returns 0 rows for the whole day; the neighbouring days are visibly depressed
too). The 2015–2019 pull itself scanned ~1.35 TB. The regime table:

| Regime | In range? |
|---|---|
| 2011 gold peak | ❌ |
| 2013 taper tantrum | ❌ |
| 2015–16 commodity bust | ✅ (from 2015-02-18) |
| 2018 trade war | ✅ |
| 2020 COVID flight-to-safety | ✅ |
| 2022 inflation shock | ✅ |
| 2023 banking stress | ✅ |

The **server DB has none of this** — it diverged from the laptop DB (git syncs
code, not data). Run the migrations there, then re-run the backfill (or copy
the DuckDB file) before any Phase 3 compute on the server.

Reproduce: `SELECT min(timestamp_utc)::DATE, max(timestamp_utc)::DATE, count(*) FROM headlines;`

---

## 3. 🟠 The embedded "documents" are URLs, not headline text

GKG carries no article title *field*, so the pipeline embeds `article_url` (the
surviving URL column after migration 005 dropped the redundant
`document_identifier` copy). URL embeddings are a weak, domain/slug-driven
signal — a day's `embedding_dispersion` reflects URL-string variety more than
news disagreement.

**2026-07-02 correction — the Extras `PAGE_TITLE` has a hard start date.**
Migration 007 pulls real titles from the GKG `Extras` XML into `page_title`,
but the backfill proved GDELT only began publishing `PAGE_TITLE` on
**2019-09-22** (a step function: 0% through 2019-09-21, 37% on the switch-on
day, ~99.2% steady from 2019-09-23). Verified both in landed rows and by
BigQuery `COUNTIF(Extras LIKE '%<PAGE_TITLE>%')` probes across 2017–2019.
Consequences:

- "Titles for free" held only for the last ~3.3 months of the 2015–2019
  backfill (~3.1M of ~76.6M rows). **2015-02-18 → 2019-09-21 can never get
  titles from GKG**; the DOC 2.0 API (below) is the only title source there.
- `src_lang` has no such limit: 100% populated for all backfilled rows
  (TranslationInfo exists from GKG 2.0's start). English share 2015–2019 is
  32.4% (24.9M of 76.6M rows).
- The title UPDATE-backfill **2020→2026 ran the same evening** (2026-07-02):
  63.27M rows now carry `page_title` (99.3–99.6% per year) + `src_lang` (~100%).
  The naive path — plain wide `refresh()` — was killed after measuring
  ~10 min/chunk: per-row `ON CONFLICT DO UPDATE` through the 140M-row ART PK
  index is a 30–40 h job. `scripts/backfill_titles.py` replaced it: extract
  PAGE_TITLE *inside* BigQuery (download ~100-byte strings, not multi-KB
  Extras blobs; zero-mismatch parity vs the python extractor on 323K rows),
  land per-chunk parquet, then yearly bulk `UPDATE … FROM` (63.27M rows in
  31 s). The parquet (`data/raw/title_backfill/`, 7.6 GB) is portable — the
  server applies it in ~30 s without re-scanning BigQuery. July 2026 total
  BigQuery spend across both backfills: 2.533 TB → $9.58.

This is why **themes-via-SQL is the correct Phase 3 default**, and why the
`mean_embedding` / `embedding_dispersion` features deserve skepticism.

**Recommendation:** Lean on **themes + tone**. For real semantics (and the
per-metal signal from §1), add a source that carries the article **title** —
which GKG lacks entirely. The lowest-friction fix is the **GDELT DOC 2.0 API**
(returns titles + supports per-metal keyword queries, back to 2017), supplemented
by **mining.com** per-metal RSS; see the verified menu in **§5**. (The roadmap
names Kitco, but Kitco exposes no per-metal feeds — mining.com does.) Real titles
are also the only thing that makes embeddings/BERTopic worth their cost.

---

## 4. 🟡 Secondary (quality / hygiene)

- **Theme skew / dead themes.** `WB_1699_METAL_ORE_MINING` = 56.7% (all-metal,
  near-saturated → low discriminative value, effectively an "is-this-metals-news"
  flag). `WB_1164_COMMODITY_PRICES_SHOCKS` = **8 rows** (dead, topic_id 10);
  `WB_1125_INTEREST_RATE_POLICY` = 0.13% (near-dead, topic_id 6). Real signal:
  `ECON_INFLATION` ~21%, `SANCTIONS` 16%, rates ~12%, central banks ~10%.
  Consider dropping the two dead themes and down-weighting mining.
- **Counts are outlet-weighted, not story-weighted.** One syndicated story = many
  URL rows, so `n_articles` / prevalence measure *volume of coverage*, not number
  of events. Fine as an attention proxy; dedup by domain / near-duplicate URL for
  story-level counts.
- **Corpus is aggregator-heavy; quality is uneven.** The top outlets are
  Chinese/Vietnamese/Indian finance aggregators (`sina.com.cn` 735K, `eastmoney.com`
  482K, `baomoi.com` 463K, `cnfol.com` 447K); `reuters.com` is only #15 (209K). Of
  48,539 distinct sources, the top 1,000 hold 54.7%. A **source whitelist** of
  reputable finance/metals outlets is a cheap quality lever. Specialist metals
  outlets are thin or sparse: `kitco` appears on only 441 / 2,315 days, `mining.com`
  on 2,130 days (but supply-focused).
- **Tone is a free precomputed sentiment signal** (V2Tone columns:
  `tone_overall/positive/negative/polarity/ard/sgrd`) — present on **100%** of rows
  (mean −0.61, sd 3.66) — arguably the best cheap news feature alongside theme
  prevalence; no embedding needed.
- **Leakage scrutiny for text.** Confirm `timestamp_utc` is first-seen/publish
  time and that a day's text strictly precedes the forward returns used in
  clustering / local projections — the price pipeline has a leakage guard; give
  text the same.
- **Disk reclaim.** Migration 005 dropped the redundant URL column but DuckDB did
  not shrink the file (still 23.8 GB). Run `scripts/compact_headlines.py --replace`
  before the full embed/aggregate — disk is tight (~72 GB free).

Theme distribution (per article, % of 63.3M):

| theme | % | topic_id |
|---|---|---|
| WB_1699_METAL_ORE_MINING | 56.7 | 11 |
| ECON_INFLATION | 21.5 | 7 |
| WB_442_INFLATION | 20.9 | 8 |
| SANCTIONS | 15.7 | 12 |
| ECON_INTEREST_RATES | 12.3 | 4 |
| ECON_CENTRALBANK | 9.8 | 0 |
| WB_1235_CENTRAL_BANKS | 9.7 | 1 |
| EPU_POLICY_INTEREST_RATES | 8.4 | 5 |
| WB_444_MONETARY_POLICY | 2.8 | 3 |
| EPU_POLICY_MONETARY_POLICY | 2.6 | 2 |
| ECON_GOLDPRICE | 1.6 | 9 |
| ECON_TRADE_DISPUTE | 1.5 | 13 |
| WB_1125_INTEREST_RATE_POLICY | 0.13 | 6 |
| WB_1164_COMMODITY_PRICES_SHOCKS | ~0 (8 rows) | 10 |

(topic_id per `results/phase3_theme_topic_map.csv`.)

---

## 5. Options for a better text signal (verified)

§1 says the metal axis is dead and §3 says the embedded docs are URLs. This is the
verified menu for getting a *real* signal, split by goal. Grounded against the GKG
2.1 codebook, a live GDELT DOC 2.0 API probe, and the live DuckDB.

**Correction to an earlier hypothesis.** `AllNames` / `V2Persons` /
`V2Organizations` are **proper nouns only** — the common nouns "silver / platinum /
palladium / PGM / catalytic converter" do **not** appear in them. The only GKG
fields carrying raw body strings are `V2.1Amounts.Object` and `V2.1Quotations.Quote`,
both sparse and noisy. **There is no clean per-metal axis anywhere inside GKG.** The
cleanest per-metal text in all of GDELT is the **DOC 2.0 API article title**, which
GKG (and the GEG) lack entirely.

### Goal A — a genuine per-metal signal

| Option | Per-metal | History | Effort | Verdict |
|---|---|---|---|---|
| **GDELT DOC 2.0 API** (titles + keyword query) | yes | 2017→now | med | **Recommend.** Free HTTP API (*not* BigQuery). Returns real titles + lets you query `"platinum"`, `"palladium price"`. Fixes §1 *and* §3 at once. Catch: rolling ~3-mo default window → windowed/throttled backfill; no per-article tone (aggregate `TimelineTone` only). |
| **mining.com per-metal RSS** | yes | forward-only | low | **Recommend.** `mining.com/tag/{gold,silver,platinum,palladium}/feed/`, real titles, republish-with-link-back license. Supply-side editorial (S. Africa PGM strikes, autocatalyst demand) = exactly where Pt/Pd diverge from gold. Metal known from the feed — no keyword guessing. |
| **WPIC / WGC / Silver Institute** quarterly | yes | 2014/2010→ | med | **Consider.** Authoritative per-metal *fundamentals* + commentary, but quarterly/annual → low-freq overlay (lag to release date), not a daily feature. |
| GKG `Amounts.Object` / `Quotations.Quote` regex | weak | re-pull | med | **Test.** Only genuine per-metal strings inside GKG; sparse/messy but cheap to bolt onto a re-pull and test before committing. |
| URL→title scrape + FinBERT on the 1.25M metal-URL subset | weak | n/a | high | **Avoid as primary.** DOC API hands you the same titles without scraping 1.25M URLs (link rot, ToS, dedup). |
| `AllNames` / org miner watchlist | weak | re-pull | high | **Avoid.** Proper-noun only; high effort, thin payoff. |

### Goal B — better sentiment / semantics (still shared across metals)

| Option | Verdict |
|---|---|
| Lean on **V2Tone + themes** (already 100% coverage) + **source whitelist** + story-dedup | **Do now (free, in-hand).** Tone is reduced to 3 means today; add dispersion / percentiles / polarity. Whitelisting reputable outlets cuts the aggregator noise (§4). |
| **V2GCAM** (curated subset of ~2,300 dims) | **Consider.** Richer than V2Tone but document-level → still identical across metals; pull 10–30 dims, not all 2,300. |
| **GEG `geg_gcnlapi`** (Google-NL entities + sentiment) | **Only if needed.** Cleaner sentiment/entities but a second large BigQuery scan, no titles, weak per-metal. |

**Skip:** free news APIs (NewsAPI / GNews / Marketaux / Alpha Vantage) — effectively
forward-only; Alpha Vantage's news history starts ~2022-03, so none align to the
2020+ corpus.

### The strategic point

Caveats §1 (no per-metal), §2 (coverage), and §3 (URLs not titles) **share one
solution**: re-source the text from the **GDELT DOC 2.0 API** (per-metal keyword →
real titles → FinBERT), backfilled 2017→present, with **mining.com RSS** forward for
the PGM supply axis and **WPIC/WGC** as a quarterly fundamentals overlay. That turns
the text stream from "shared theme/tone counts on URL slugs" into a genuine per-metal
headline corpus — and it is almost entirely off-BigQuery (DOC API + RSS are free HTTP).

*Uncertain (not hard-confirmed):* DOC-API exact rate limits (undocumented — throttle
empirically); mining.com RSS archive depth (likely shallow → forward-only); BigQuery
re-ingest byte costs (no creds configured, so modeled not measured — see §6).

### Reproduce (run against the live DuckDB)
```sql
-- §1 mechanism: ECON_GOLDPRICE never appears as the sole curated theme
SELECT count(*) FROM headlines
WHERE list_contains(from_json(themes,'["VARCHAR"]'),'ECON_GOLDPRICE')
  AND len(from_json(themes,'["VARCHAR"]')) = 1;          -- 0

-- corpus is syndication-heavy: top outlets are CN/VN/IN aggregators, reuters #15
SELECT source, count(*) c FROM headlines GROUP BY 1 ORDER BY 2 DESC LIMIT 15;

-- per-metal URL keyword probe: numerically present, too noisy/sparse to trust
SELECT
  count(*) FILTER (WHERE lower(article_url) LIKE '%silver%')    AS silver,     -- 226,606
  count(*) FILTER (WHERE lower(article_url) LIKE '%platinum%')  AS platinum,   --  11,972
  count(*) FILTER (WHERE lower(article_url) LIKE '%palladium%') AS palladium   --   4,173
FROM headlines;   -- palladium dominated by the heraldpalladium.com newspaper domain
```

---

## 6. BigQuery cost of the §5 plan

**Anchor (measured).** The 5-column GKG query (`DATE`, `SourceCommonName`,
`DocumentIdentifier`, `V2Themes`, `V2Tone`), partition-pruned, scanned **0.65 TB for
57 months** (2021-09→present) on a free dry run — i.e. **~11 GB/month, ~0.14 TB/year**
(journal, 2026-06-25). Pricing: **$6.25/TB**, first **1 TiB/month free**
(`scripts/backfill_gdelt.py`). On-demand bills **bytes scanned in referenced
columns** after partition pruning — the `REGEXP_CONTAINS` theme filter does *not*
reduce bytes, so cost = (columns) × (date range), **not** (rows returned).

**The recommended plan is ≈ $0 in BigQuery — by design.** The per-metal fix is the
**DOC 2.0 API** (free HTTP, *not* BigQuery); mining.com RSS and WPIC/WGC are also
off-BigQuery. The only GKG backfill the plan needs — **2015–2019 (~0.5 TB)** — fits
under the 1 TB/month free tier.

| Component | BigQuery? | Modeled scan | Modeled cost |
|---|---|---|---|
| Collapse metal axis (§1a) | no | — | $0 |
| **DOC 2.0 API per-metal re-source (2017→)** | **no (free HTTP)** | — | **$0** |
| mining.com RSS / WPIC / WGC overlay | no | — | $0 |
| 2015–2019 GKG backfill (§2) | yes | ~0.5 TB | ~$0 (< free tier) |
| *opt.* re-pull + `Amounts` / `Quotations` (per-metal test) | yes | ~0.7–2 TB extra | ~$0–7 |
| *opt.* re-pull + `GCAM` (sentiment upgrade) | yes | ~2.5–4 TB | ~$0–25 (≈$0 if spread) |
| *opt., not recommended* GEG `geg_gcnlapi` | yes | multi-TB | tens of $ — dry-run first |

**Notes / confidence.** No BigQuery creds are configured (`.env`
`GOOGLE_APPLICATION_CREDENTIALS` is empty), so the optional rows are **modeled** from
the single measured datapoint + GDELT column sizes; `GCAM`/GEG are lower-confidence
(`GCAM` is the largest GKG column → roughly 2–3× the per-month bytes). Get exact
numbers for free: `backfill_gdelt.py --estimate` for any date range, plus a one-off
`dry_run=True` query that adds the candidate columns to measure the multiplier
(dry runs are never billed). The free tier resets per calendar month, so spreading a
large re-pull across ≥2 billing cycles keeps each month < 1 TB → ≈ $0. Keep the
`--max-gb` chunk cap (a `GCAM` month ≈ 25–35 GB, well under the 100 GB default). The
Storage Read API used for result download is negligible next to the query scan.

## 7. Maintain the 4-metal text axis vs. collapse — the decision (verified)

Should we keep a genuine per-metal text axis instead of collapsing? Examined in depth
(live DOC-API measurement + method design + an adversarial 3-lens panel that did not
refute the conclusion). **Answer: collapse for all four metals; a per-metal text axis
is viable only for gold and silver, and only as a CV-gated experiment.**

### The volume cliff (decisive)

A per-metal *daily* text feature needs daily article volume. Measured two independent
ways — the live corpus (cleaned URL keyword) and the live GDELT DOC 2.0 API (titles):

| Metal | Cleaned URL/day | DOC-API raw/day | Daily signal? |
|---|---|---|---|
| gold | 133 (median 134) | ~6,300 | ✅ viable |
| silver | 32 (median 27) | ~3,000 | ✅ viable |
| platinum | 1.1 (**median 0**) | ~268 (single digits *real*) | ⚠️ weekly at best |
| palladium | 0.6 (**median 0**) | ~43 (≈0 *real*) | ❌ not viable |

The DOC API does **not** rescue Pt/Pd: contamination is *semantic*, so real titles
carry the same false positives — "platinum" → Amex/Chase cards, LEED certs, RIAA "goes
platinum", Xeon Platinum CPUs; "palladium" → the London Palladium theatre, Mr Bean,
catalytic-converter-*theft*. `"palladium price"` ≈ 0/day. On ~96% of days Pt/Pd have
no genuine metal articles.

### Why collapse is the right null (not just the cheap one)

- **Lossless today** — the four series are byte-identical (`gold-only = 0`).
- **Removes a degeneracy** — on Pt/Pd's ~96% no-article days `context.py` leaves
  `embedding_dispersion` / `text_pca_*` NaN; imputing them creates a degenerate point
  mass that manufactures a spurious "no-news" HDBSCAN cluster and makes the rare
  1-article days random outliers. A daily Pt/Pd text axis **degrades** clustering.
- **Causally safe** — no per-metal text term ⇒ Phase 5 DoubleML cannot fit a spurious
  per-metal "news effect" on low-support, name-contaminated regressors.
- **Metals already differentiated** — `context.py` carries per-metal `ret_5d/20d`,
  `rvol_20d`, and `cot_managed_money_z`. The collapse touches only the *text* axis.

### Options vs. collapse

| Option | Real for | Pt/Pd daily | Effort | Cost | Verdict |
|---|---|---|---|---|---|
| **Collapse (baseline)** | shared | n/a | low | $0 | **Ship now.** |
| **(d)** URL/title disambiguation lexicon (existing corpus) | gold, silver | no | low | $0 | Best-ROI maintain path; full history. |
| **(a)** DOC-API titles + FinBERT | gold, silver | partial | high | $0 | Recall lift over (d); optional. |
| **(e)** WPIC/WGC/Silver Institute quarterly fundamentals | all 4 | no (quarterly) | med | $0 | Honest Pt/Pd per-metal channel (not text). |
| **(f)** Hybrid (gold/silver text; Pt/Pd shared + fundamentals) | 2+2 | no | med | $0 | Recommended **end-state**, built incrementally. |
| (b) mining.com RSS | per-feed | no | low | $0 | Forward-only → no backfill; live overlay only. |
| (c) in-GKG `Amounts`/`Quotations` | weak | no | med | ~$3–9 BQ | Skip. |
| (g) relevance-weighting | illusory | no | low | $0 | Avoid (fabricates collinear pseudo-variance). |

### Recommendation & sequencing

1. **Ship collapse now, all four metals** (`text_daily.aggregate_daily` → one shared
   `market` row/day; `context.py` reads it for every metal). Pin the lossless invariant
   with a regression test.
2. **Per-metal text is a gold/silver-only experiment** — build option **(d)** first
   ($0, automatable, full history) before any manual fundamentals work.
3. **Pt/Pd per-metal = quarterly fundamentals (e)** — MVP one body (WGC gold) first.
4. **End-state = hybrid (f)**, with the comparability fix: *not* a provenance flag —
   cluster per-metal, or exclude the per-metal-text columns from the Pt/Pd distance
   metric (a flag does not fix a distance-based clusterer).

### The bar to ever build per-metal text

Per metal, never pooled, on walk-forward CV vs collapse: DBCV/silhouette lift ≥ ~0.02
with no inflated cluster share; higher between-cluster forward-return dispersion;
**incremental IC ≥ ~0.02 after partialling out price + COT** (residual IC, since
silver's news corr with gold is 0.65) with stable sign across ≥70% of folds; and a
DoubleML CI excluding zero with stable sign. Gold/silver may pass; Pt/Pd are expected
to fail every bar.

## Decisions needed before the full Phase 3 run (~9 h GPU)

Doing these *first* avoids redoing the 9-hour aggregate:

1. **Coverage** — backfill GDELT 2015–2019, or commit to 2020+ scope? *(Recommend
   backfill.)* Note: the DOC-API per-metal re-source (§5) only reaches 2017, so the
   GKG 2015–2019 BigQuery backfill remains the only route to 2015–16.
2. **Text axis** — collapse the metal axis now for all four metals (§7; lossless).
   Per-metal text is a **gold/silver-only, CV-gated** experiment via option (d) (§5);
   platinum/palladium stay on shared text plus a quarterly fundamentals overlay (§7).
   *(Recommend: collapse now; defer per-metal until it beats collapse on walk-forward CV.)*
3. **Disk** — run `scripts/compact_headlines.py --replace` to reclaim space.
4. Then: `aggregate` → `context` → `cluster` → `analyze`.
   (`daily_topic_prevalence` is already populated via themes-via-SQL.)

Lower priority: rename migration `005_phase3_artifacts` → `006` to remove the
duplicate-prefix fragility; prune dead themes `WB_1164` / `WB_1125`.
