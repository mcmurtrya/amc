# Phase 3 write-up: GDELT text → scenario clustering → lift gate

**Status: Phase 3 closed 2026-07-11.** Branch `phase3-streaming-themes`;
final clustering lineage `phase3_optC_tone_lag1_2024split`. This document
consolidates the phase's three deliverables — the corpus, the scenario
clustering with LLM labels, and the pre-registered predictive-lift test —
and states what the phase did and did not establish.

## Headline result

Unsupervised scenario clustering of daily market context (news tone +
themes + macro/price state) produces **interpretable, historically coherent
regimes** — but those regimes **add no forecastable information** over the
Phase-1 feature set at the pre-registered primary target (GC=F 5-day-ahead
realized vol). The result comes from a design whose decision rules were
fixed before any run (`phase3_cluster_lift_design.md`), so it is a genuine
null, not a garden-of-forking-paths artifact. Per those same rules, the
corpus-scale embedding investment (assessment §7) is **not justified**.

## 1. What was built

- **Corpus**: 139.9M GDELT GKG headlines, 2015-02-18 → 2026-06-19,
  day-continuous (one upstream hole: 2017-08-29). `page_title` ~99.5%
  within its 2019-09-22+ era (two-phase BigQuery backfill, 2026-07-02);
  `src_lang` ~100% everywhere. Portable title parquets in
  `data/raw/title_backfill/` (7.6 GB).
- **Pipeline**: 8 stages (`gdelt → embed → aggregate → topics → context →
  cluster → analyze → label`), month-chunked, resumable
  (`scripts/phase3_pipeline.py`). Option C mode (`--no-text-embeddings`)
  runs end-to-end on CPU in minutes.
- **Daily text features**: tone means (V2Tone, 100% coverage — the signal
  the data assessment trusts), article counts, BERTopic theme prevalences;
  all joined to trading days **lagged one trading day** (a same-day join
  found and fixed during adversarial review — see journal 2026-07-02).
- **Clustering**: UMAP(7) + HDBSCAN on the daily context vector, trained
  through 2024, assigned out to 2026-05 via `approximate_predict`.
- **Labels**: LLM-assisted (Opus 4.8), cause-based vocabulary with
  confidence grades, upserted into `cluster_centroids`.

## 2. Data reality (constraints that shaped everything)

From `phase3_gdelt_data_assessment.md`:

1. **There is no per-metal news signal.** Metal-specific filtering leaves
   too little volume; text features are one shared "market" row per day.
2. **PAGE_TITLE exists only from 2019-09-22.** Pre-2019 rows can never get
   titles from GKG; URL slugs are the only text there.
3. **Tone (V2Tone) is the trustworthy full-history signal** — 100%
   coverage, no era break. Option C leans on it.

## 3. The scenario taxonomy

Final model `phase3_optC_tone_lag1_2024split`: 2,718 trading days, 7
clusters + noise (27 days unassigned). Opus 4.8 labels (confidence in
brackets; cost of the full taxonomy: ~$0.06):

| id | days | label |
|---|---|---|
| 0 | 208 | fed-rate-hike-expectations [low] |
| 1 | 445 | diffuse-macro-noise-baseline [low] |
| 2 | 505 | unclear [low] |
| 3 | 379 | mixed-newsflow-crude-uptrend [low] |
| 4 | 103 | covid-recovery-stimulus-rebound [medium] |
| 5 | 67 | trade-war-dovish-fed-tailwind [high] |
| 6 | 414 | unclear [low] |

Reading: the small, historically distinctive clusters (4, 5) are real and
crisply labelable — the labeller found the 2019 trade-war/dovish-Fed gold
rally and the 2020 stimulus rebound without being told about them. The
large clusters are regime mixtures the labeller honestly marked low
confidence / unclear, consistent with the no-per-metal-signal constraint:
GDELT world news describes the *macro backdrop*, not metals-specific
catalysts. 9 of 14 labels across both model versions were graded low —
the honest-confidence prompt design worked as intended.

## 4. The lift gate (pre-registered)

**Design** (`phase3_cluster_lift_design.md`, registered 2026-07-03, before
any run): paired walk-forward comparison on one binding `shared` row set.
Arm A = Phase-1 features; Arm B = A + per-fold regime features (UMAP+HDBSCAN
refit per fold at `split.train_end`, one-hots + confidence + purged target
encoding — `metals/features/regimes.py`, leakage-tested). Decision rule:
B beats A iff mean-RMSE improves ≥1.0% AND B wins ≥60% of splits.

**Readout** (2026-07-11, AYMStation, 11 folds, test 2020-08 → 2026-01;
runner `scripts/phase3_cluster_lift.py`, per-split table
`phase3_cluster_lift_readout.csv`):

| comparison | rel ΔRMSE | split wins | verdict |
|---|---|---|---|
| **GC=F h=5 (primary)** | **−0.37%** (bar −1.0%) | **4/11** (need 7) | **B does not beat A** |
| SI=F h=5 (secondary) | +1.80% | 5/11 | B worse (report-only) |
| GC=F h=20 (secondary) | −2.12% | 7/11 | suggestive (report-only) |

The B_notext attribution ablation was gated on B beating A and correctly
did not run. Harness run ids for all six runs are recorded in the journal
(2026-07-11 entry 2).

**Caveats**, as pre-registered: no train/val or val/test embargo (identical
across arms; the paired Δ largely cancels it — an assumption, not a
theorem); fold count came out 11 vs the design's ≈9 estimate (parameters
applied verbatim; the estimate was off); the 2025–2026 test windows carry
2–4× the RMSE of earlier folds in every arm — the per-split win rule exists
to keep that from dominating, and it also says no.

## 5. What Phase 3 established

**Positive:** a reproducible, leak-audited text pipeline at 139.9M-row
scale; a defensible daily market-context vector; an interpretable scenario
taxonomy whose distinctive clusters align with known macro episodes; and
per-fold regime-feature machinery (`build_regime_features`) reusable for
any future clustering variant.

**Null (the load-bearing result):** at the primary target, regime
membership derived from tone+themes+macro context carries no incremental
predictive information over conventional price/macro features. The two
honest interpretations: (a) GDELT's shared world-news signal is too diffuse
for metals-specific 5-day vol; (b) whatever regime information exists is
already spanned by the Phase-1 macro/price features the clusters are
partly built from.

## 6. Consequences

1. **Embedding gate closed** — no corpus-scale PAGE_TITLE embedding run,
   no GPU provisioning for it (assessment §7 resolved in the negative).
2. **Phase 4's multimodal premise is weakened**: the roadmap's
   cross-attention text stream starts from a demonstrated text-lift null.
   Scope decision recorded separately.
3. **Open lead, not pursued**: the h=20 secondary (−2.1%, 7/11) would have
   passed a primary-style bar. It is report-only by pre-registration.
   Chasing it requires a fresh pre-registered design *before* any further
   runs at that horizon.
