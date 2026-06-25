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

**Why:** `THEME_TO_METALS` maps only `ECON_GOLDPRICE` to gold-only; all 13 other
curated themes map to all four metals. Since `ECON_GOLDPRICE` never appears
alone, the per-metal split collapses.

**Implication:** News cannot explain *cross-metal divergence* (e.g. platinum
rallying while gold is flat) — that signal is not in this data. Per-metal text
attribution in Phase 4/5 would be spurious.

**Options:**
- **(a)** Collapse text features to a single shared daily *news-state* series and
  drop the metal axis on text (simpler, honest, 4× less storage). Recommended.
- **(b)** Build genuine per-metal signal from article **title/body** keyword
  matching — requires a real text source (see §3).

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

## 2. 🔴 Coverage is 2020-01-01 → 2026-06-19 only (~6.5 years)

The roadmap's own regime sanity-checks are mostly **out of range**:

| Regime | In range? |
|---|---|
| 2011 gold peak | ❌ |
| 2013 taper tantrum | ❌ |
| 2015–16 commodity bust | ❌ |
| 2018 trade war | ❌ |
| 2020 COVID flight-to-safety | ✅ |
| 2022 inflation shock | ✅ |
| 2023 banking stress | ✅ |

**Recommendation:** GKG 2.0 supports back to **2015-02-18** (the pipeline's
default `--start` already assumes it). Backfill **2015–2019** if you want those
regimes and a real pre/post-2015 robustness split (Phase 5). Otherwise scope
every news-based claim explicitly to 2020+.

Reproduce: `SELECT min(timestamp_utc)::DATE, max(timestamp_utc)::DATE, count(*) FROM headlines;`

---

## 3. 🟠 The embedded "documents" are URLs, not headline text

GKG carries no article title, so the pipeline embeds `article_url` (the surviving
URL column after migration 005 dropped the redundant `document_identifier` copy).
URL embeddings are a weak, domain/slug-driven signal — a day's
`embedding_dispersion` reflects URL-string variety more than news disagreement.

This is why **themes-via-SQL is the correct Phase 3 default**, and why the
`mean_embedding` / `embedding_dispersion` features deserve skepticism.

**Recommendation:** Lean on **themes + tone**. For real semantics (and the
per-metal signal from §1), add a text source — Kitco RSS (per the roadmap) or
resolve URLs → titles. That is also the only thing that makes BERTopic worth its
cost.

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
- **Tone is a free precomputed sentiment signal** (V2Tone columns:
  `tone_overall/positive/negative/polarity/ard/sgrd`) — arguably the best cheap
  news feature alongside theme prevalence; no embedding needed.
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

## Decisions needed before the full Phase 3 run (~9 h GPU)

Doing these *first* avoids redoing the 9-hour aggregate:

1. **Coverage** — backfill GDELT 2015–2019, or commit to 2020+ scope? *(Recommend backfill.)*
2. **Text axis** — collapse text features to one shared news-state (§1 option a),
   or invest in a real per-metal text source (§1 option b)? *(Recommend a now, b later.)*
3. **Disk** — run `scripts/compact_headlines.py --replace` to reclaim space.
4. Then: `aggregate` → `context` → `cluster` → `analyze`.
   (`daily_topic_prevalence` is already populated via themes-via-SQL.)

Lower priority: rename migration `005_phase3_artifacts` → `006` to remove the
duplicate-prefix fragility; prune dead themes `WB_1164` / `WB_1125`.
