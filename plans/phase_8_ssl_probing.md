# Phase 8: Self-supervised low-rank representation of the daily price + news state

Added 2026-07-17. **Status: Stage A scaffolded (2026-07-18).** Wired into
`00_roadmap.md` as Phase 8. The §7 step-1 library layer is built and tested
(`features/ssl_views.py`, `models/factor_ssl.py`, `eval/probes.py`); the walk-forward
driver, harness wiring, and embedding backfill are not.

**Companion (general-design treatment).** A deliberately modality-agnostic version of
this design lives in the separate *brain2* research wiki
(`C:\Users\mcmur\OneDrive\Documents\Claude\Projects\brain2`, i.e.
`/mnt/c/Users/mcmur/OneDrive/Documents/Claude/Projects/brain2` from WSL —
`wiki/synthesis/ssl-for-market-structure-probing.md`, `wiki/synthesis/cross-asset-pretraining.md`,
`wiki/reference/amc-metals-case-study.md`), developed from the SSL / vision literature without this
project's data constraints. **This plan is the grounded instantiation and governs for
the metals problem.** As of 2026-07-18 the two are reconciled: the brain2 pages carry a
"Grounding" section pointing back here and to `amc-metals-case-study.md` (which captures this
project's four hard facts, the Phase-6 classical-beats-ML / sentiment-hurts-OOS result, the
triangulated hawkish-FOMC finding, and the paid-data null), and both sides adopt the same
baseline-first gate. Where they differ, the daily / market-wide / ~40–50-effective-regime
reality here wins; brain2 is the design library, not the runnable spec.
(Paths verified 2026-07-20. They were originally written as `../../brain2/…`, which resolves
only from the OneDrive checkout of this repo — not from the canonical WSL working copy at
`/home/mcmur/projects/amc`; hence the absolute form above.)

This phase asks a narrow, honest question: *does the joint daily state of the gold
price complex and the GDELT news stream contain a low-dimensional, fold-stable axis
whose meaning we can name and whose existence survives a block-permutation null —
even if it adds zero predictive lift over a classical baseline?* It is a
**representation/insight** program, not a bid to beat the Phase-1 vol baseline.

It inherits Phase 7's governing discipline verbatim: **baseline-first; any deep
encoder runs only as a gated bake-off behind a pre-registered kill criterion**
(`plans/phase_7_amc_program.md`). All standing conventions hold — walk-forward CV
only, the `features/leakage.py` guard before training, harness logging, UTC
timestamps, a `journal.md` entry per session.

---

## 0. The recommendation in one paragraph

Build the **classical low-rank joint factorization first** (`LRJ-Metals`, the
highest-scored proposal at 6.5) as *both* the primary self-supervised representation
*and* the bar a deep model must clear. It is genuinely self-supervised (no labels; it
learns a shared price↔news latent by cross-view covariance), it reuses the repo's
audited leakage guards rather than reinventing them, and it adds **no torch** to a
sklearn/lgbm/duckdb box. Treat a deep encoder (a `CoMPASS`-style frozen-text-tower
contrastive net) as a **gated Stage B** you build only if Stage A surfaces a
fold-stable cross-modal axis that a *linear* method visibly leaves on the table. Graft
the disciplined probe stack from `PanelMAE`, the effective-sample-size honesty and
price-only-first sequencing from `TempoContrast`, and the frozen-MiniLM tower plus
per-split-prefix pretraining from `CoMPASS`. Given the Phase-6 prior,
**pre-register for a null** and treat "no incremental news lift, use the classical
vol baseline, don't buy a sentiment feed" as a first-class, shippable deliverable.

---

## 1. Honest framing — what ambition is actually right here

Confront the four hard facts before writing any code, because they set the ambition.

1. **Daily-only, multi-asset panel.** "Detailed gold-price data" here is the daily
   OHLCV panel (`GC=F/SI=F/PL=F/PA=F`, the ETFs, `^GSPC/^VIX/HG=F/CL=F`, FRED
   `DTWEXBGS`) plus the engineered features in `features/assemble.py` (multi-horizon
   log returns, 5/20/60d realized vol, 20d skew/kurt, 60d drawdown, cross-metal
   ratios, log-spread z-scores) and `features/macro.py` (TIPS/DXY/VIX/GPR levels and
   changes). There is **no intraday/tick** structure for a temporal-contrastive net to
   exploit at sub-daily resolution, and all series end 2026-05-22.
2. **No per-metal news signal.** GDELT themes are industry-wide; text collapses to
   **one shared daily `MARKET` row** (`features/text_daily.py`, `MARKET = "market"`).
   Any "news axis" is a *market-wide* axis by construction. Per-metal differentiation
   (gold vs PGM) must come from the price/COT block, never the text block.
3. **N is tiny and autocorrelated.** Text overlap caps the joint sample at ~2,800
   daily rows (2015-02-18 → 2026-06-19), ~1,700 if *real* titles are required
   (`page_title` is a URL slug before 2019-09-22). After a 252-day warmup and
   forward-target truncation it is smaller. Worse, daily financial rows are ~99%
   autocorrelated over the window lengths that matter: the **effective independent
   sample is on the order of a few dozen macro regimes, not thousands of rows**
   (~2,800 / 64 ≈ 44 non-overlapping 64-day windows). Every parameter budget, every
   significance claim, and any "hundreds of thousands of contrastive pairs" defense
   must be sized against *that* number.
4. **The Phase-6 prior is adverse and specific.** `results/phase6_validation.md`
   (63-day hold-out): **classical baselines beat ML, and regime/sentiment features
   *hurt* OOS.** A learned low-rank *regime* coordinate is exactly the artifact that
   prior predicts will fail out-of-sample.

**Conclusion on ambition.** The right goal is **insight / representation, not
prediction.** The honest modal outcome is *null incremental lift*. A clean
pre-registered *no* (mirroring the Phase-3 pre-registered null in
`results/phase3_writeup.md`) is a real result and a real business recommendation. It
also dictates the architecture: when the deliverable is an *interpretable, stable*
low-rank basis on ~40 effective samples, a 150k-parameter torch encoder is the wrong
tool for the primary; a whitened PCA → PLS/CCA is the right one.

---

## 2. The architecture decision

### 2.1 Primary: classical low-rank joint factorization (`LRJ-Metals`)

**What it is.** Two views of each trading day:

- **View A (price/macro/COT), as-of close, unlagged:** `assemble.build_price_features`
  + `macro.compute_macro_features` + a 252-day COT z (the `spreads.py` / `context.py`
  `rank_window=252` convention).
- **View B (news), lagged one trading day:** the tone means
  (`mean_tone_overall/positive/negative`), `embedding_dispersion`, the 16-dim
  train-only PCA of the day mean-embedding, and topic prevalences.

Reduce each view to its own train-fitted whitened PCs, then fit **`PLSCanonical`**
(covariance-maximizing; prefer it over `sklearn.cross_decomposition.CCA`, which is
numerically unstable at these dims) on the **train prefix only** to extract K≈2–4
canonical pairs. The frozen representation `Z` is the concatenation of the canonical
scores (and optionally the block PCs). This is the encoder-agnostic `Z` the probing
methodology consumes.

**Why it is the primary, not a fallback:**

- **It IS self-supervised.** No labels touch the factorization; it learns the shared
  latent by cross-view covariance. It is the honest SSL for this data regime.
- **Highest infra fit + no new heavy dependency.** It reuses
  `context.py::_pca_fit_transform` (`PCA(whiten=True)`, train-mask),
  `clustering.py::_standardize`, `cv.walk_forward_splits` / `check_no_leakage`, and the
  `leakage.py` guards verbatim. No torch, no GPU determinism burden, no
  seed-sensitivity liability on the tight WSL2 box.
- **It is the bar.** Phase-6 says classical wins; the correct experimental design is to
  *make the classical low-rank model the thing the deep model must beat*, and build it
  first so the deep model has a real, OOS-scored opponent rather than a strawman.

**The one non-negotiable fix `LRJ-Metals` was missing** — build it in from day one: an
**incremental-IC tautology guard**. View A already contains realized vol, returns, and
spread z-scores, so a factor loading on `rvol_20d` "predicting" forward vol is
near-tautological. Every *news-arm* claim must be evaluated as **incremental IC after
residualizing both the target and the news canonical score on the full `X_price` panel**
(train-fit residualizer, applied forward). "Factor→target IC" and "beats the price-only
baseline metric" are insufficient and will credit recovered price structure as
discovery.

### 2.2 When a deep encoder is worth it (Stage B gate) — `CoMPASS`-style contrastive

Only build the deep encoder if **all three** gates are green after Stage A:

1. Stage A finds ≥1 **fold-stable** cross-modal canonical axis (loading cosine-stable
   across folds, survives block-permutation null on test).
2. That axis's **incremental** signal over `X_price` is nonzero *and* a linear method
   plausibly under-exploits it (e.g. the relationship is visibly nonlinear /
   threshold-like in the exemplar-day scatter).
3. You can absorb torch's maintenance + reproducibility cost.

If green, the deep candidate is **`CoMPASS`** (scored 6, best of the deep three): a
small price tower (~25k params) over a W-day window, a **frozen** MiniLM
day-mean-embedding text tower, InfoNCE contrastive alignment. It scored highest among
the deep proposals precisely because the frozen text tower and tiny price tower keep
capacity sane. Adopt it **with its required fixes baked in** (per-split-prefix
pretraining, titles-only primary run, time-structure-preserving null). A masked-feature
autoencoder (`PanelMAE`) is **not** the deep choice — see §2.4.

**When the classical baseline is simply the honest answer:** if any Stage-A gate is red
— no stable cross-modal axis, or the incremental news IC is ≤0 (the Phase-6-consistent
expectation) — **stop.** The classical factorization *is* the representation; a deep
encoder would only be a more expensive way to relearn `X_price` and overfit ~40 regimes
with 150k parameters.

### 2.3 The graft — best idea from each runner-up

| Source | Idea grafted into the plan | Why |
|---|---|---|
| `PanelMAE` (5.5) | Fair full-window raw-feature null; **BLOCK** permutation (not iid); adding the actual Phase-6 classical champion (`lgbm_vol` HAR-style) as the forward-vol baseline; the `materialize_day_embeddings` assertions (own-day articles only, train-prefix PCA) | Most disciplined guard stack of the four; its "reconstruction is degenerate on this panel" lesson is why we avoid MAE |
| `TempoContrast` (5) | Report **effective independent sample size** (~40–50 windows / regimes), not pair counts; **price-only first**, defer the news term; lagged realized vol as the named forward-vol benchmark; **Procrustes** for rotation-unidentifiability; **multi-seed** cross-variance for any deep run | Kills the "hundreds of thousands of pairs" inflation and the seed-sensitivity liability |
| `CoMPASS` (6) | **Frozen MiniLM** text tower; **per-split-prefix pretraining** (the leak fix); **titles-only primary** variant; the staged `--only/--resume-from` pipeline mirroring `phase3_pipeline.py` | The soundest deep design; its candor about null-to-negative lift is the model to follow |
| Probing methodology (Appendix B) | The whole playbook: CCA/CKA cross-modal test, linear probes, unsupervised axis interpretation (loadings/exemplars/theme-lift/stability), block bootstrap + BH-FDR + pre-registration | This is the actual deliverable machine |

### 2.4 Why not `PanelMAE` or `TempoContrast` as primary

- **`PanelMAE`** has a **degenerate pretext**. The 64 panel cells are heavily
  algebraic/statistical functions of one another — 1/5/20d returns are nested,
  5/20/60d vols overlap, spread z-scores are deterministic transforms of the spreads,
  pctiles track their levels — and W=20 daily rows are ~99% autocorrelated.
  Mask-and-reconstruct is then solved by neighbor-copy or an arithmetic identity, so
  train MSE looks great while `z` learns smoothing/arithmetic, not market structure. If
  you ever want a reconstruction term, **de-redundify first** (drop derived cells or
  mask correlated groups jointly) — but it is the wrong primary.
- **`TempoContrast`** carries the worst overfit / parameter-vs-effective-sample ratio
  (150–250k params vs ~40 independent windows) and the most seed sensitivity, and its
  CPC forward-predictive loss makes any later "z predicts forward vol" probe partly
  **self-fulfilling**. Both are viable-with-fixes as *experiments*, never as the thing
  you lead with.

---

## 3. Staged implementation plan (module paths, shapes, loops)

### 3.0 Prerequisite: rematerialize `daily_text_features.mean_embedding`

**Blocking fact:** `daily_text_features.mean_embedding` is currently **all NULL**
(confirmed: `context.py` reads it via `sub["mean_embedding"].notna()`, and with
all-NULL, `present.any()` is False → no `text_pca_*` columns). The tone-mean and topic
channels work *now*; the **embedding channel does not** until this backfill runs.

- **New:** `scripts/materialize_day_embeddings.py`. For each UTC calendar day, average
  **only that day's own** headline embeddings from the sharded Parquet cache
  (`features/embeddings.py`, `ParquetEmbeddingCache` keyed by `sha256(text)`),
  L2-normalize, and write via `text_daily.upsert_daily` (whose
  `ON CONFLICT ... COALESCE(EXCLUDED.mean_embedding, ...)` upsert is fine at ~2,800-row
  scale — this is **not** the large-table per-row `refresh()` anti-pattern the CLAUDE.md
  sharp edges warn about).
- **Hard requirements** (each an assertion, not prose): (a) the day boundary is the
  *identical* UTC calendar cut that `text_daily`/`context.py` assume, so the downstream
  `shift(1)` lag stays correct; (b) the **same text** (slug vs real title) that
  inference will see is what gets embedded — a drift between cached and served text
  silently reintroduces leakage; (c) **no cross-day pooling** and **no global PCA** —
  the embedding PCA fits strictly on the train prefix per split, which
  `build_context(pca_fit_until=...)` already enforces.
- **First cut can skip this entirely:** run `ContextConfig(include_embeddings=False)`
  to get tone+topic text views immediately, and add the embedding block once the
  backfill lands. Disk is tight; do **not** persist per-day headline embedding *sets* —
  only the (384,) day-means.

### 3.1 New files & reuse map

**New:**

- `src/metals/features/ssl_views.py` — assemble `Z_p` (View A) and `Z_t` (View B) with
  the correct lag + missing-news-day handling.
- `src/metals/models/factor_ssl.py` — the classical joint factorization (`FactorSSL`
  dataclass with `save/load`, mirroring `ClusterPipeline` and
  `MODEL_DIR = data/processed/factor_ssl/<version>`).
- `src/metals/models/ssl_encoder.py` — **Stage B only**, torch **lazy-imported**
  exactly as `clustering.py` lazy-imports umap/hdbscan so `pytest` runs without it.
- `src/metals/eval/probes.py` — linear/logistic probes, incremental-IC
  residualization, block permutation, block bootstrap, BH/BY-FDR.
- `scripts/ssl_pipeline.py` — staged, `--only/--resume-from`, mirroring
  `scripts/phase3_pipeline.py`.
- `src/metals/data/migrations/014_ssl_axis_cards.sql` — **optional** (next free id is
  `014`; `013` is now `013_spread_floor`). Probe *runs* log through the existing harness (`runs`/`run_predictions`), so a
  new table is only needed if you want to persist per-axis interpretation cards. Prefer
  harness + `results/` CSVs first.

**Reuse (verified signatures):**

- `features/assemble.py`: `build_price_features(prices)` (:46),
  `build_feature_matrix(prices, macro_wide, target_ticker, target_kind, target_horizon,
  realized_vol_window, min_warmup=252)` (:75), `shift_target(series, horizon)` (:63).
  Note `build_feature_matrix` already computes `nan_tail = h + w − 1` for `realized_vol`
  — reuse it rather than hand-rolling targets.
- `features/context.py`: `build_context(prices, macro_wide, cot_positioning,
  text_daily, topic_prevalence, pca_fit_until, config) -> (df, artifacts)`; the
  `sub = sub.shift(1)` text lag at **line 183** (which lags tone means *and* embeddings
  — the tone cells are already lagged in the reference); `_pca_fit_transform`
  (`PCA(whiten=True)`, train-mask, :45); `artifacts["text_pca"]` for forward
  re-application.
- `features/leakage.py`: `assert_chronological` (:30),
  `assert_target_strictly_future(features, target, target_horizon, min_nan_tail)` (:39),
  `assert_features_have_history(features, min_warmup)` (:72).
- `eval/cv.py`: `walk_forward_splits(timestamps, train_start, val_days, test_days,
  step_days, min_train_days=5*365)` → `Split(train_idx, val_idx, test_idx, train_end,
  val_end, train_start)`; `check_no_leakage(splits)`.
- `models/clustering.py`: `_standardize` (:59), `MODEL_DIR` pattern (:26),
  `save_pipeline/load_pipeline` (:183/:205).
- `eval/harness.py`: `register_run(name, model_type, target_type, config, notes)
  -> run_id` (:75), `log_predictions(run_id, df)` (requires cols `timestamp_utc,
  ticker, horizon, prediction, actual`, :95), `compute_metrics` (rmse/mae/**ic**/
  hit_rate, :133), `compare_runs(run_ids, metric="rmse")` (:171),
  `log_feature_importances`/`aggregate_feature_importances` (:242/:310 — reuse to get
  cross-split loading-cosine stability "for free").
- `features/text_daily.py`: `load_daily(metal)` (:289), `MARKET` sentinel (:43);
  `topics.load_topic_prevalence_wide()`; `configs/scenarios.yaml::hawkish_fomc`;
  `data/cot.py`, `data/fomc_surprises.py`, `data/fomc_dgs2.py` /
  `fomc_yield_surprises` (migration 012, Phase 7.2).

### 3.2 Views: `Z_p` and `Z_t` (lag enforcement)

The cheapest correct path is to **reuse `build_context()` and partition its columns**,
because it already bakes in (a) the one-trading-day text lag (line 183) and (b) the
train-only embedding PCA (`pca_fit_until`). Per fold:

```python
ctx, artifacts = build_context(
    prices, macro_wide, cot_positioning, text_daily, topic_prevalence,
    pca_fit_until=split.train_end,          # per-split train-only PCA  <-- leakage-critical
    config=ContextConfig(target_metal="gold", include_embeddings=True),
)
assert_chronological(ctx)                    # leakage guard (1)
Z_p = ctx[[c for c in ctx if is_price_macro_cot(c)]]   # as-of close, UNLAGGED
Z_t = ctx[[c for c in ctx if is_text(c)]]              # tone/dispersion/text_pca_*/topic_* — ALREADY shift(1)
```

`is_text` = columns in `{n_articles, mean_tone_*, embedding_dispersion, text_pca_*,
topic_*}`. Everything else is `Z_p`.

**Missing-news-day handling** (a leak surface `LRJ-Metals` under-specified):
`context.py` deliberately leaves tone/embedding **NaN** on no-article days ("a missing
news day must not fake a neutral tone of 0"). PLS/CCA cannot ingest NaN. The imputer
(mean-fill / dispersion-default) **is a fitted statistic** and must be fit on the
**train prefix only** — a global mean-fill leaks. Put this in `ssl_views.py`, fit on
`split.train_idx`, apply forward, and unit-test it.

### 3.3 Stage A: classical factorization — per-fold protocol + shapes

Shapes (illustrative; F/G depend on the assembled panel):

- `Z_p`: `(T, F)`, F ≈ 40–60 columns.
- `Z_t`: `(T, G)`, G ≈ 20–40 (3 tone + 1 dispersion + 16 embedding-PCA + ~15 topic).
- Per-block train-fitted whitened PCA: `Z_p → P_p (T, k_p)`, `k_p ≈ 10–15`;
  `Z_t → P_t (T, k_t)`, `k_t ≈ 8–12` (never larger than the train-fold scree supports;
  k ≲ 10–20 is the honest ceiling on a few-hundred effective samples).
- `PLSCanonical(n_components=K)`, `K ≈ 2–4`: canonical scores `U (T, K)` (price side),
  `V (T, K)` (text side).
- Frozen `Z = concat([U, V], axis=1)` → `(T, 2K)`, i.e. `d ≈ 4–8` (optionally append
  block PCs).

```python
for split in walk_forward_splits(ctx.index, train_start, val_days=180, test_days=180,
                                 step_days=180, min_train_days=5*365):
    tr, va, te = split.train_idx, split.val_idx, split.test_idx
    # 1) per-block standardize + whitened PCA, TRAIN-PREFIX ONLY
    sp, pp = fit_scaler_pca(Z_p.iloc[tr], k_p)      # clustering._standardize + PCA(whiten=True)
    st, pt = fit_scaler_pca(Z_t.iloc[tr], k_t)
    Pp = {x: pp.transform(sp.transform(Z_p.iloc[x])) for x in (tr, va, te)}
    Pt = {x: pt.transform(st.transform(Z_t.iloc[x])) for x in (tr, va, te)}
    # 2) PLSCanonical on TRAIN ONLY; K chosen on VAL block-permutation, never test
    pls = PLSCanonical(n_components=K).fit(Pp[tr], Pt[tr])
    Utr, Vtr = pls.transform(Pp[tr], Pt[tr])
    Ute, Vte = pls.transform(Pp[te], Pt[te])
    rho_test = canonical_corr(Ute, Vte)             # report on TEST, not train
    Z_te = np.concatenate([Ute, Vte], axis=1)
    # 3) probes on Z_te (see §4); log to harness
```

Persist per-fold `components_`, `mean_`, `explained_variance_`, PLS loadings, and
`rho_test`; feed the loadings to `aggregate_feature_importances` for cross-fold cosine
stability.

### 3.4 Stage B: contrastive deep encoder — shapes + training-loop sketch

Only after the §2.2 gate. Price window tensor `X ∈ ℝ^{B×W×F}` (W=20, F≈40–60); frozen
text view `e ∈ ℝ^{B×384}` (MiniLM day-mean, **lagged**); both towers project to
`d ≤ 16`.

```python
for split in walk_forward_splits(...):
    tr = split.train_idx                                  # PER-SPLIT-PREFIX ONLY (leak fix)
    scaler = fit_standardizer(F_num.iloc[tr])             # train-prefix stats
    Xtr = windowize(scaler(F_num.iloc[tr]), W=20)         # (Ntr-W+1, 20, Fdim)
    etr = mean_emb.iloc[tr].shift(1)                      # TEXT LAG; frozen MiniLM day-mean
    Xtr, etr = drop_or_train_impute_news_nan(Xtr, etr)    # news-day subset or train-prefix impute
    enc = PriceTower(d=16)                                # ~25k params; decay + dropout
    proj = TextProjector(384, d=16)                       # small; MiniLM stays FROZEN
    for epoch in range(E):
        for xb, eb in loader(Xtr, etr, batch=256):
            zp, zt = enc(xb), proj(eb)                    # (B,16),(B,16)
            loss = info_nce(zp, zt, temp=0.1)             # positive = same day; negatives = other days
            loss.backward(); opt.step()
        if early_stop(val_contrastive_loss(split.val_idx)):  # val, never test
            break
    freeze(enc)
    Z_te = enc(windowize(scaler(F_num.iloc[split.test_idx]), W=20))   # frozen forward
    # probes on Z_te (§4)
```

**Report EFFECTIVE sample size** (non-overlapping W-windows / distinct regimes, order
~40–50) and justify the ≤25k-param budget against *that*, not the O(B²) pair count.
**Multi-seed** the per-split tower (≥5 seeds) and report cross-seed variance of every
probe metric — a non-deterministic encoder must beat deterministic linear baselines to
claim anything. Ship **price-only first**; the cross-modal news term is the
ablate-by-default per Phase-6.

### 3.5 Harness logging

Every probe run:
`run_id = register_run(name="ssl_factor_probe_fwdvol_h5", model_type="factor_ssl_probe",
target_type="realized_vol", config={...frozen Z hash, K, block_len, grid...})`. Pool
test-fold predictions into a frame with `timestamp_utc, ticker, horizon, prediction,
actual` and `log_predictions(run_id, df)`; `compute_metrics(run_id)` for
RMSE/MAE/IC/hit_rate; `compare_runs([ssl_run, lgbm_vol_run, lagged_rvol_run],
metric="rmse")` for the load-bearing comparison. Log loadings via
`log_feature_importances` and roll up stability with `aggregate_feature_importances`.

### 3.6 Pipeline / CLI

`scripts/ssl_pipeline.py`, stages `materialize-embeddings → build-views → fit-factors →
probe → interpret → report`, each `--only`/`--resume-from`, mirroring
`phase3_pipeline.py`'s idiom.

---

## 4. The low-rank probing playbook (summary — full methodology in Appendix B)

### 4.1 Supervised linear probes

Linear/logistic **only** (a nonlinear head on ~2,000 rows conflates "information
present" with "my head overfit it" — the Phase-6 failure mode). `Ridge` for
regression, `LogisticRegression(penalty="l2")` for classification, **L2 tuned on
`split.val_idx`, refit per fold** (the probe's weights are a fitted projection; leakage
rule (3) applies).

| Target | Type | Construction | Guard |
|---|---|---|---|
| Forward realized vol | reg | `build_feature_matrix(target_ticker, target_kind="realized_vol", target_horizon=h, realized_vol_window=w)` | `min_nan_tail = h+w−1` |
| Forward return sign | bin | `sign(shift_target(ret_1d, h))` | `min_nan_tail = h` |
| Hawkish-FOMC | bin | `scenarios.yaml::hawkish_fomc` = `fomc_surprises:mps_orth` tercile_high, **in-window** thresholds, rolled to next trading day | base-rate line (33%) |
| FOMC surprise sign | bin | `sign(fomc_surprises.mps_orth)`, cross-check `fomc_yield_surprises` ΔDGS2 | **sparse: ~88 FOMC days total, ~a dozen/fold** |
| COT extreme | bin | `cot` 252d z beyond ±thr | threshold on **train prefix** |
| VIX regime | bin | `^VIX` above rolling quantile | quantile **in-window** |

**The two claims that actually matter** (everything else is a sanity check):

- **Predictive:** the forward-vol probe must be compared not to "probe > 0" but to the
  **Phase-6 classical champion** — the HAR-style `models/lgbm_vol.py` and, as the
  simplest named floor, **lagged realized vol (vol clustering)**. Beating a bare linear
  raw-feature probe is too weak to engage "classical beats ML."
- **Insight (news):** **incremental IC.** Residualize *both* the target and the news
  canonical score on the **full `X_price` panel** (train-fit residualizer, applied
  forward); test whether the news factor explains the *residual*. Without this, every
  "insight" reduces to price structure the panel already held — and Phase-6 predicts
  the incremental news content is ≤0.

Fair null for "windowing/structure helps": the baseline probe must be the **same
flattened W×F window** (or a matched window summary), not day-t features alone —
otherwise you only prove "a 20-day window helps."

### 4.2 Unsupervised axis interpretation (from the **train** fold only)

For each retained PC / canonical variate, converge four lines of evidence:

1. **Top standardized loadings** — the loading vector, not one top feature.
2. **Exemplar days** — 10 highest/lowest-scoring days; join back via `connection()` and
   read `page_title` (real only 2019-09-22+; slugs before) and macro prints; verify
   they are nameable events, not glitches.
3. **Correlation with series NOT in `Z`.** Correlating an axis with
   `DTWEXBGS/TIPS/^VIX` is **circular** — those are input columns of View A. Name an
   axis by external series (`GPR_DAILY`, COT net-position *changes*) *or* explicitly
   label the correlation a tautology-check.
4. **Theme lift at extremes** — GDELT `themes` JSON frequency in the top/bottom decile
   vs corpus base rate, with a permutation null. This is the only principled way to
   attach headline *content*; remember it is market-wide, never per-metal.

**Stability:** an axis "means" something only if sign-aligned loading cosine and
top-loading rank-correlation are reproducible across folds; use **Procrustes** to
handle rotation-unidentifiability. Loadings that flip fold-to-fold get **no**
interpretation.

### 4.3 Statistical guards on ~40 effective samples

- **Block permutation null** (block-permute the text block against the price block,
  ≥500–1,000 shuffles, re-fit the *entire* fold pipeline per shuffle). A naive iid
  shuffle destroys autocorrelation and is anti-conservative. For sparse events
  (FOMC/COT), permute event *assignment* preserving the event count. This is also the
  null that stops a drift-only "temporal position" axis from masquerading as genuine
  alignment.
- **Moving/stationary block bootstrap** CIs (block ≈ 10–20 trading days; report
  block-length sensitivity) for probe metrics and axis-macro correlations.
- **BH (or BY, given positive dependence) FDR** across the **pre-declared** grid
  (6 targets × horizons × axes × theme-lifts). Report raw and adjusted p.
- **CKA** (linear, double-centered, on held-out rows) between `Z` and the existing
  `build_context` vector — if CKA≈1, `Z` is merely re-expressing the context basis and
  there is no new representation.
- **Power reality:** ~88 FOMC days total → a dozen per fold; a logistic probe on that
  many positives is nearly inert. Pre-commit minimum event counts per fold and expect a
  null permutation result on FOMC.

### 4.4 Pre-registration (before touching any test fold)

Write to `journal.md`: the frozen `Z` and its **hash**, the axis-extraction recipe, the
exact probe grid, primary metric per target, block length, correction method, and the
decision rule: *"insight is claimed iff OOS metric beats the block-permutation null at
FDR-adjusted p<0.05 AND the axis loadings are fold-stable AND (for news) incremental IC
over `X_price` excludes 0 in block bootstrap."* This is the Phase-3 discipline that
makes a negative result publishable and a positive one credible.

---

## 5. Leakage traps & failure modes specific to THIS design

**Leakage traps (each becomes an assertion/test):**

1. **Full-history pretrain leak** — the fatal, self-contradictory hole in
   `CoMPASS`/`TempoContrast`/`PanelMAE`: pretraining on 2007–2026 absorbs every later
   test fold's covariance/vol-scale into the weights, then freezing and probing early
   folds evaluates a rep that has "seen" the future via unsupervised statistics.
   **Pretrain strictly within each split's train prefix** (or cap to data before the
   earliest test fold and never reuse weights across folds). **Add a test that fails if
   any encoder/PCA weight was fit on data ≥ a split's `train_end`.**
2. **`mean_embedding` rematerialization drift** — must average **only each day's own**
   articles, on the **same UTC day boundary** and the **same text** (slug vs real
   title) served at inference, PCA on train prefix only, no cross-day pool, no global
   PCA. A day-boundary mismatch shifts View B by a day and the `shift(1)` guard cannot
   catch it because the frame still "looks" lagged. **Unit-test: each rebuilt day-mean
   equals the mean of exactly that day's cached vectors.**
3. **Tone lag** — `context.py` line 183 already lags tone means; any *re-derivation*
   outside `build_context` must `shift(1)` all text (tone, dispersion, embeddings,
   topics). Price stays as-of close, unlagged.
4. **Slug→title modality break at 2019-09-22** — MiniLM maps slugs and real titles to
   different regions of the sphere; a projection fit on pre-2019 slugs applied forward
   to real titles is a fitted-projection-on-a-different-modality. Make the
   **titles-only (~1,700-row) run the primary**, or add a slug/title indicator and
   demonstrate axis stability across the break; do not pool two modalities through one
   forward-applied projection.
5. **Missing-news-day imputation** — train-prefix imputer only; a global mean-fill
   leaks (and crashes PLS on NaN otherwise).
6. **Standardizer scope** — train-prefix mean/std only, especially the 384-dim news std
   on ~1,400 rows; `StandardScaler().fit_transform(Z)` over all rows is leakage even if
   the probe is walk-forward.
7. **Window-target tail leakage** — `min_nan_tail = h+w−1` for realized-vol targets;
   passing `h` silently leaks the most recent `w−1` days (the guard's own docstring
   warning).
8. **In-window threshold labels** — COT-extreme / VIX-regime thresholds on the **train
   prefix**, not the full sample (the `scenarios.yaml` in-window convention).
9. **CCA degeneracy** — never fit CCA/PLS on raw high-dim blocks with small N
   (manufactures spurious unit correlations); reduce each block to its own PCs first,
   and report canonical ρ on **test**, not the train rows where it is maximized by
   construction.
10. **CPC self-fulfillment** — a forward-predictive contrastive loss makes "z predicts
    forward vol" partly circular; benchmark against lagged rvol, not the encoder's own
    objective.
11. **External pretrained-model contamination** (§8) — a pretrained TSFM cannot be
    re-pretrained per walk-forward fold, so freezing-then-probing it re-imports trap 1:
    its weights have "seen" the backtest era via an undisclosed corpus. On the
    leakage-strict standard, unverifiable contamination = treat as contaminated. The same
    applies to the LLM annotator's **parametric** weights — a dated title lets the model
    "know what happened next"; date-blind the prompt, treat labels as hindsight-colored,
    and cross-check a date-blinded re-run.
12. **Back-adjusted futures roll leakage** — the lease/forward-rate feature (§8) must be
    scored from raw per-contract settlements with a point-in-time roll; a
    continuously-stitched/back-adjusted series injects future roll information.

**Failure modes:**

- **Tautology / input-recovery** — axis-naming against inputs is renaming, not
  discovery; the Phase-5 scenario silhouette is *partly circular* because those labels
  derive from the same price channel. Anchor insight on the incremental-IC and
  external-label probes.
- **Degenerate reconstruction** (if a MAE term is ever added) — arithmetic identities /
  neighbor-copy; de-redundify or drop it.
- **Effective-N inflation** — pair-counting / O(B²) contrastive pairs are not
  independent information; size claims to ~40–50 regimes.
- **Seed sensitivity** of the deep encoder on the WSL2 box — multi-seed or don't claim.
- **Disk pressure** — persist only (384,) day-means, never per-day headline sets;
  `DROP COLUMN` doesn't reclaim space (`compact_headlines.py --replace`).
- **Quarantine** — if any collector table
  (`coin_premiums/macro_consensus/search_interest/pgm_prices`) is ever joined for the
  coin-premium probe, filter `quarantine_reason IS NULL`; today those return empty by
  design.

---

## 6. What success looks like / what a null looks like

**Success (the ambitious, less-likely outcome):** a **pre-registered, fold-stable
cross-modal canonical axis** (e.g. text-negativity ↔ forward realized vol) whose OOS
canonical ρ on **test** rows survives the block-permutation null at BH-FDR<0.05, whose
**incremental IC over `X_price` excludes 0** in block bootstrap, and whose loadings are
Procrustes-stable across folds — and which either **beats the `lgbm_vol`/lagged-rvol
baseline** on the AMC-relevant 5–20d horizon *or* ties it while **localizing** the
signal to one nameable axis. That axis then maps to exactly one pre-registered AMC
decision rule with an OOS lift-vs-baseline number and a block-bootstrap CI — most
plausibly the **FOMC-hedge-timing** rule (the Phase-5 hawkish-FOMC finding has the
strongest support), stated as *correlational monitoring*, not a causal extension.
CKA(`Z`, `build_context`) is reported so you know `Z` is not merely the old basis
renamed.

**Null (the modal, fully acceptable, publishable outcome):** every canonical ρ
collapses into the block-permutation band on test; no probe beats its label-shuffle
null after FDR; the incremental news IC is ≤0 (consistent with Phase-6's "sentiment
hurts OOS"); axes are input-recovery. The deliverable is then a clean pre-registered
negative mirroring Phase-3 **plus three concrete business calls**: (1) use the
classical rolling-vol / `lgbm_vol` baseline — not a news axis — to set buy-spread
floors; (2) **do not buy a sentiment feed** (a direct input to
`results/amc_paid_data_review.md`); (3) **keep the news channel out** of the
`clustering.py` scenario-discovery context — do not re-inject the exact feature Phase-6
found harmful. That is a decision, not a failure.

---

## 7. Execution order

1. **Classical baseline runs first, no torch:** `ssl_views.py` + `factor_ssl.py` on the
   tone+topic text block (`include_embeddings=False`) → probe grid → pre-register in
   `journal.md`.
2. **Rematerialize `mean_embedding`** (`scripts/materialize_day_embeddings.py` +
   assertions), add the embedding block, re-run Stage A titles-only as primary.
3. **Gate check** (§2.2). If any gate red → stop; the classical factorization is the
   answer.
4. **Only if green:** `ssl_encoder.py` (lazy torch), price-only first, per-split-prefix
   pretrain, multi-seed, same probe harness; must beat the classical baseline *and*
   Stage A on identical folds to earn its keep.
5. Log every run through `eval/harness.py`; append the session to `journal.md`. Run
   `ruff check` / `ruff format` / `mypy` / `pytest` before considering any change done.

---

## 8. Data & methods: can paid data or LoRA/distillation relax the four facts?

Reviewed 2026-07-17. Full record, costs, ToU verdicts, and unverified flags in
`results/amc_paid_data_review.md` ("Addendum 2026-07-17"). Short answer to both
"buy better data?" and "use LoRA/distillation?": **no.** Paid data cannot relax the
binding constraint (FACT 3, the joint sample) within AMC's budget + ToU gate + the
Phase-6 prior; and LoRA/distillation change *capacity/transfer*, not *information*, so
they cannot overturn information constraints. The review mints **no new purchase** — the
two existing buys (Databento CME backfill, Greysheet) stand — but three near-free builds
and one compliance correction follow.

**Per-fact (data):**

- **FACT 1** — full Databento intraday is ~free (owned licence) but decoration for AMC's
  days-to-weeks decisions; it only feeds a *price-only* encoder, the class FACT 4
  penalizes. Build gated behind Stage A.
- **FACT 2** — no budget/ToU-clean feed exists; entity sentiment (RavenPack) is a richer
  version of the falsified class. The only lever is the LLM annotator below.
- **FACT 3** — unbuyable: pre-2015 news is enterprise-priced + ToU-barred + falsified-class,
  and ~3,000 rows buy only ~20–30 near-duplicate independent regimes.
- **FACT 4** — the escape is orthogonal physical-market data Phase-6 never tested — mostly
  free (below), not richer price/news.

**Three near-free builds (gated by the baseline-first discipline of §7):**

1. **LLM-as-annotator** — Claude reads each day's GDELT *titles* → per-metal, event-typed
   features (~$30–150 one-time, clean ToU). The one lever that adds *information*.
   NON-NEGOTIABLE: titles-only ~1,700-row run primary; the PGM channel is
   sparse-to-empty; date-blind the prompt and treat labels as hindsight-colored
   (parametric leakage, trap 11); its output *is* sentiment + regime, so it must clear the
   same incremental-IC + block-permutation null (§4.1), and a **clean null is the modal,
   shippable outcome** — its value is closing "should AMC buy a news feed?".
2. **Databento-derived lease/forward-rate alarm** — GC/SI calendar spread + ZQ ($0,
   point-in-time clean). Value is an operational tightness *alarm* for a physically-long
   dealer, not a predictor (once the backfill adds contract months the spread is inside
   `X_price` and the §2.1 residualizer eats its orthogonality). Build from RAW per-contract
   settlements with a point-in-time roll — a back-adjusted series injects roll look-ahead
   (trap 12). Term/calendar spreads are new vs the Yahoo-built `spreads.py` panel, so a
   candidate feature starts from a **neutral** prior; pre-register a possible null IC.
3. **US Mint bullion-sales collector** — $0, clean; monthly, supply-rationed noisy demand
   proxy for the coin arm.

**Compliance correction — WGC Goldhub reclassified.** The paid review listed Goldhub
India/China premia as a free "adopted" upgrade but never ran its ToU against the AMC gate
(commercial + model-training + cached-local); its terms plausibly fail it as CME did.
**Barred-pending-written-consent — quarantine before any loader reads it** (no table
exists; do not build one). T&C wording is PLAUSIBLE-pending-confirmation; the quarantine
default holds regardless.

**Methods verdict (capacity, not information).** LoRA's low-rank *weight update* is not
this plan's low-rank *representation* — different objects; LoRA is a transfer technique,
not a replacement for the probing program. A pretrained TSFM as a **frozen** encoder (IBM
Granite TTM; Datadog Toto as a low-leakage control) is worth ONE null-tolerant experiment
gated behind Stage A — never a forecaster, never fine-tuned on the backtest window.
LoRA-adapting a finance TSFM (Chronos/TimesFM) is defer→skip: its only claimed edge (an
uncontaminated prior) is unprovable and probably false. LoRA-on-text and all distillation
variants are skip (sharpen a falsified channel / relabel CoMPASS Stage B / pure
inference-cost play). Full ranked verdict + unverified flags in the paid-review addendum.

### 8.1 LLM-as-annotator — schema v2 & the Stage-0 pilot

The one method-lever that adds *information* (§8) is an LLM annotator over the day's
metals-relevant GDELT **titles** (no article bodies exist — §5 ceiling). Design record from
a scoped brainstorm (2026-07-17); marginal cost of extra fields in a pass you already pay
for is ~zero, so the useful split is **research-infrastructure** (value independent of any
forecast — dodges the Phase-6 prior) vs **predictive** (faces it, starts from "no").

**Flagship byproduct:** a per-title `event{}` object (datable occurrence + typed vocab +
entity + confidence + verbatim evidence) that **auto-drafts the Phase-7.5 PGM supply-event
ledger** (currently scoped for manual hand-dating) and supplies clean event dating for
Phase-2 local projections. Close co-anchor: `relevant` (a metal-as-financial-asset flag —
the corpus has **no relevance filter**, `text_daily.py:165`, so "gold medal"/"Platinum
Jubilee" are admitted) + `metals` (per-title attribution — breaks the byte-identical
`market`-row collapse, `text_daily.py:12-16`).

**Schema v2 (near-free wins).** Per-title: `relevant`, `metals`+`direction`, `event`
(`type`/`entity`/`supply_demand_side`/`framing`), `monetary_stance` (Phase-5-only,
firewalled). Per-day/market: `gold_narrative_regime`, `monetary_stance_day`. Derived in SQL
(not LLM tokens): `corpus_offtopic_fraction`, `physical_content_share`. **Dropped as
redundant/not-title-extractable:** generic sentiment (=V2Tone), generic geopolitical risk
(=GPR index), "monetary-policy-present" (=theme codes), EPU, and any magnitude/body-detail
field. The full field-by-field menu (Groups 1–3, leakage handling, the JSON schema) is the
2026-07-17 workflow record; the machine-readable schema lives in
`src/metals/annotate/schema.py`.

**Schema v3.0 (2026-07-21) — added before the pilot ran, while changes were still free.**
Four **conditional** per-title fields, emitted only when `event_type != "none"` and omitted
entirely otherwise (output tokens dominate this job, and ~90% of titles are recaps), plus one
new `event_type` value:

- `novelty` (`first_report`/`followup`/`recap`/`unclear`) — per-day dedupe is *within-day
  only* (`titles.py`), so without this a five-day strike reads as five events. Phase 10 is a
  **dating** exercise expecting only ~10–30 clean PGM shocks, where miscounting is fatal. The
  annotator sees one day in isolation and can only read linguistic cues, so this is partial
  signal by construction — and it is the field most likely to invite parametric recall, so
  watch it in the date-blind A/B.
- `event_time_ref` (`past`/`today`/`days_ahead`/`weeks_plus_ahead`/`unspecified`) — `framing`
  separates anticipatory from reaction but not *how far*; a title previewing an FOMC meeting
  three weeks out must not date an event to today.
- `physical_tightness` (`tightening`/`easing`/`none`) — premiums, delivery delays, mint
  suspensions, backwardation. The external premium panel is **licence-blocked**, so headlines
  may be the only legally usable premium signal until that clears.
- `region` (normalised enum) — `event_entity` is verbatim free text and hard to join on; gold
  demand is an India/China story, PGM supply a South Africa/Russia one.
- `EVENT_TYPES` gains **`scrap_recycling_flow`** — consumers selling jewellery, pawn/resale
  volume, refiner throughput. AMC's own *supply side*, which the v2 vocabulary had no home
  for, and the target for the scrap-inflow nowcast (`plans/research_backlog.md` E2).

Deliberately still excluded: numeric magnitude extraction (rarely in the title, noisy),
per-title confidence, and anything resembling a sentiment score beyond `direction` — that is
the Phase-6-falsified class and must not re-enter by the back door. `severity` was considered
and **held** pending a decision to commit to Phase 10.

`TASK_VERSION` → `v3.0` (invalidates the v2 cache and any pre-registration built on it).
Report card gains `novelty_fill` / `event_time_ref_fill` (gated at ≥80% of event-bearing
titles — the prompt demands them, so a low rate means the instruction was ignored) and
report-only `physical_tightness_informative` / `region_informative` /
`scrap_recycling_fires`. Cost impact ~+9%: 80-day pilot ×2 variants **$32.66** Opus batch;
full 1,678-day single-variant run **$342.54** Opus / **$205.52** Sonnet / **$68.51** Haiku.

**Schema v3.1 (2026-07-21, adversarial review pass — still before any run.)** Prompt
clarifications: the `event_time_ref` boundary is now exclusive ("up to and including one
week" / "more than one week"), titles naming a calendar date or month ("ahead of the June
FOMC") must use `unspecified` — computing the distance requires today's date, exactly the
knowledge the date-blind design withholds — and `region` gets a precedence rule for
actor-vs-affected titles (use where the supply/demand effect lands, not the sanctioning
actor). Report card gains three review-driven checks: **`results_current` (gated)** — the
card refuses to trust parquet whose `task_version`/`prompt_hash` stamps don't match the
current instrument (previously the stamps were written but never read back, so a schema
bump silently passed stale results); **`novelty_ab_drift` / `event_time_ref_ab_drift`
(report-only)** — per-title date-blind A/B drift on the dating fields, closing the gap where
the day-level drift gate never watched `novelty`, the field most at risk of parametric
recall; **`v3_spurious_emission` (report-only)** — share of non-event titles emitting a
conditional key, the number that says whether the 60-tokens/title cost model holds.

**Firewall:** `monetary_stance` and `gold_narrative_regime` are directional
sentiment/regime signals — scoped to Phase-5 triangulation / Phase-3 clustering, **never**
fed to a return forecaster without first clearing the §4 incremental-IC + permutation null
(default expectation: null).

**Stage-0 pilot** (`src/metals/annotate/`, driver `scripts/annotate_pilot.py`) — the cheap
feasibility test *before* annotating all ~1,678 title-era days. It draws an ~80-day
stratified sample (FOMC/CPI event days with external ground truth · random days for
coverage · known PGM-stress windows), filters each day's titles to the ~15 metal-relevant
GDELT theme codes and de-duplicates syndication, then runs the frozen **date-blinded**
prompt and computes five checks against pre-registered thresholds: (1) coverage — any-metal
and PGM-specific; (2) human-audit accuracy (manual gold set); (3) known-event recall vs
`fomc_surprises`; (4) **date-blind A/B drift** (date-visible vs date-stripped — the
parametric-leakage control, trap 11); (5) reproducibility across seeds. A red gate
(PGM coverage ~0, or material date-blind drift, or the wrong-date counterfactual anchoring
on parametric memory) stops the program for at most the pilot spend (~$33 Opus batch under
schema v3.x; a mid-pilot abort costs less) instead of the full-run spend. Cost
estimate and model comparison: `scripts/annotate_pilot.py estimate` (dry-run token
measurement; see the Phase-8 cost note).

**Filter review (2026-07-17) — verdict ADJUST.** An adversarial review of the title
pre-filter (`titles.py`) against measured corpus stats found the foundations sound
(theme-OR correct, date-blind intact, LLM-defer defensible) but flagged real issues.
**Applied** (high-value, low-risk): (a) hardened the de-dup key — HTML-decode + NFKC +
outlet-suffix and punctuation stripping, unicode-safe — because the deliverable is *counts*
and weak dedup double-counts one wire story; (b) a stop-phrase veto *before* the cap
("silver alert", "gold medal/coast/rush/standard", "platinum jubilee", …) so junk can't evict
real stories from the 250 slots; (c) recall vocabulary — iridium/ruthenium, `PGM(s)`,
producers (Stillwater/Northam/Zimplats/Anglo American Platinum/Johnson Matthey/Heraeus),
`comex/lbma`, tickers `xpt/xpd/pplt/pall`, and *anchored* coin terms (krugerrand, gold/silver
eagle, maple leaf, proof/bullion coin, numismatic, specie — bare `coin`/`sovereign`/`eagle`
deliberately excluded); (d) a coverage flag (`DayTitles.pre_title_era`, `n_titled`) so a
structurally-titleless pre-2019 day or a mid-era corpus gap is not read as a quiet news day.

**Time-stratified cap (applied 2026-07-17).** The cap *selection* was earliest-250-by-timestamp,
which discarded the US-afternoon session (FOMC 2pm ET, COMEX/LBMA close) — the most
price-relevant slice for a US dealer. Now `_select_capped` reserves ≥50% of the budget for the
US window (13–22 UTC) when it has enough titles, hands unused slack to whichever side has more,
and evenly strides each side's picks across time. The cap still discards ~2.8× of titles/day on
average (mean ~696, growing to ~1,400/day by 2026); *removing* it instead would raise the
full-run batch cost roughly 2.7× over the capped figure (v2-era estimate: ~$860 Opus /
$520 Sonnet / $175 Haiku uncapped vs ~$314/$188/$63 capped; both sides scale ~+9% under
schema v3.x — the capped v3 figures are in the v3.0 addendum above) — a separate
cost/coverage call, but the clock-bias is now fixed regardless.

**Standing limitations (documented, not bugs):** (1) the keyword gate is **English-centric** —
~64% of gold-relevant news is non-English (Arabic 24%, Chinese 14%, …), and only gold has a
non-English lifeline (the `ECON_GOLDPRICE` theme-OR); silver/PGM lose their non-English
stratum. **Measured 2026-07-23** (`scripts/lang_gate_count.py`, 62.2M title-era rows): the
current gate admits ~576 unique candidates/day (278 English; Arabic already gets ~84/day via
the gold theme), and first-cut native-script terms for 20 languages would add **~520 unique
titles/day** — zho +119, spa +90, vie +74 (the Vietnamese domestic gold market is nearly
invisible today at 1.0/day), rus +38. Because the 250/day cap binds, admission is a
*composition* trade (needs a language-stratified reserve in `_select_capped`, template = the
US-session reserve) or a cap raise (~400 ⇒ full run ~$530 Opus / ~$318 Sonnet). Terms are
recall-first (vàng = yellow, plata = money slang, złota ↔ złoty) — a per-language stop-list
pass plus a per-language `corpus_offtopic_fraction` split in the pilot must precede adoption;
the annotator itself is language-capable, so the only prompt change is a "titles may be in
any language" line. **Precision measured 2026-07-23** (mini-batch, 2,100 titles, judge
v1.0, ~$0.75): six languages pass with no stop-listing (kor .89, zho .78, tha .76,
vie .74, ara .73, tur .66) vs the **current English gate's own 0.58** under the same
judge; jpn fails hard (.02, ゴールド=fashion); the rest have one dominant, stop-listable
FP pattern each (AUR the Romanian party, La Plata place names, medaglia d'oro). Full
table in journal.md 2026-07-23; results parquet in data/processed/. **Retest under terms v2
(stop-lists + case fixes, same day): jpn 0.02→0.68, ron 0.22→0.61, ind 0.58→0.62 promoted;
11 languages dropped (diffuse residual noise — Romance bare-metal sports metonymy, Slavic/
Greek gold-idioms, German surnames). Final bridge set: 9 languages, ~+195 relevant
titles/day (~doubles the English gate's ~161). v3.2 freeze pending.** (2) The `page_title` slug boundary at 2019-09-22 is an **upstream GDELT feature** (0%
before, 99.47% after — GDELT never emitted the GKG `<PAGE_TITLE>` tag earlier), so the
annotator is titleless before then, unfixably. (3) **Corpus INGESTION gaps** within the era —
mapped by `scripts/coverage_audit.py`: **48 missing days in 4 windows**. One (2017-08-29) is the
known GDELT *upstream* empty day; the other **47 are our-DB ingestion gaps** (contiguous blocks
bounded by full months — **2024-01 except the 15th**; **2025-06-15→07-01**), re-pullable via a
targeted `backfill_gdelt.py` + Extras/`page_title` pull (one process per month). Title-era
`page_title` completeness is 98.8–99.6%/yr. The sampler is now **coverage-aware**
(`draw_sample(require_coverage=True)` → gap days are never drawn; a FOMC on a gap day is dropped),
and `load_day_titles` flags any residual gap via `n_titled` rather than reading it as a quiet day.

---

## Appendix A — Architecture scorecards

Four architectures were designed independently and each adversarially critiqued for
leakage, overfitting, tautological probing, and honesty against the Phase-6 prior.
Scores are out of 10 for soundness *in this codebase and data regime*.

### A.1 `LRJ-Metals` — classical low-rank joint factorization — **6.5, viable-with-fixes (SELECTED PRIMARY)**

Two views (price / news), each reduced to train-only whitened PCs, joined by
`PLSCanonical` on the train prefix. Strongest on leakage (reuses the real guards) and
honesty (turns a null into the "don't buy a sentiment feed" decision).

- **Leakage:** strongest of the four — reuses `context.py` `shift(1)` + train-mask PCA
  and `cv.walk_forward_splits`. Open surfaces: missing-news-day imputation scope, the
  384-dim news standardizer scope, and the embedding-rebuild day boundary.
- **Overfit:** the "~600 loading params" accounting is misleadingly optimistic — the
  dominant fitted object is the 384-dim news whitening PCA + standardizer (thousands of
  stats on ~1,400 train rows). Power on the news arm is marginal; modal outcome is a
  well-guarded null.
- **Probing:** central weakness — price-side scores recover vol/return structure already
  in the panel, and naming axes via DXY/TIPS/VIX (panel inputs) is circular. **Fix: the
  incremental-IC residualization guard** (now §2.1) is mandatory.
- **Key required fixes:** incremental-IC tautology guard; train-restricted
  imputer+standardizer with a unit test; honest parameter accounting for the news PCA;
  interpret only against non-input series; assert the embedding-rebuild day boundary;
  quarantine the "append factors to clustering context" step behind the null outcome;
  prefer `PLSCanonical` over `CCA`; pre-register K/λ on val.

### A.2 `CoMPASS` — cross-modal contrastive (frozen MiniLM text tower) — **6, viable-with-fixes (SELECTED STAGE B)**

Small price tower (~25k) + frozen MiniLM text tower, InfoNCE alignment. The soundest
deep design; best candor about null-to-negative lift.

- **Leakage:** the big one is a self-contradiction — proposes full-2007+ price-tower
  pretraining while also claiming train-prefix-only fitting. **Must** pretrain
  per-split-prefix. Plus the slug→title modality break and the embedding-rebuild
  drift.
- **Overfit:** encoder overfit well-mitigated (frozen text, ~25k params, d≤16); the
  real risk is interpretation-layer overfit on ~40 effective samples.
- **Key required fixes:** fix the pretrain leak (per-split or pre-first-test-fold cap +
  a weight-provenance test); time-structure-preserving null; titles-only primary; treat
  axis-naming-by-input as a sanity check; drop the operating AMC rules unless
  OOS-backtested; report effective sample size, not pair counts.

### A.3 `PanelMAE` — masked-cell reconstruction autoencoder — **5.5, viable-with-fixes (NOT chosen; probe-stack grafted)**

Masks panel feature-cells and reconstructs, news-conditioned. Most disciplined probe
stack of the four (block permutation, block bootstrap, BH-FDR, raw-feature null) — those
are grafted into the plan. **Not** the primary because the pretext is degenerate: the 64
cells are algebraic functions of each other, so mask-and-reconstruct is solved by
identities and the representation learns arithmetic, not market structure.

- **Concrete leakage bug in the proposal:** it fed *unlagged* tone cells into the
  reconstruction target ("as-of close") — tone is text and must be `shift(1)`.
- **Key required fixes:** lag the tone cells; make the raw-feature null a fair
  full-window null; add `lgbm_vol` as the forward-vol baseline; use BLOCK permutation;
  de-redundify the reconstruction target; assert train-prefix embedding PCA.

### A.4 `TempoContrast` — temporal-contrastive price encoder + news auxiliary view — **5, viable-with-fixes (NOT chosen; sample-size honesty grafted)**

TS2Vec/CPC-style temporal contrast on price windows, news as an auxiliary view. Worst
parameter-vs-effective-sample ratio (150–250k params vs ~40 independent windows) and
most seed-sensitive; its CPC loss makes later forward-vol probes self-fulfilling. Its
lasting contribution is the honest **effective-sample-size accounting** (~2,800/64 ≈ 44
windows) and the **price-only-first** sequencing, both grafted in.

- **Key required fixes:** resolve the same full-history-pretrain contradiction as
  CoMPASS; retire the pair-count defense; drop axis-interpretation-against-inputs; power
  analysis + minimum event counts for FOMC/scenario probes; lagged rvol as the named
  benchmark; ship price-only first; multi-seed for seed sensitivity.

---

## Appendix B — Full probing methodology (encoder-agnostic)

*This is the standalone methodology the plan's §4 summarizes. It is encoder-agnostic:
wherever it says "the frozen representation `Z`," substitute the PCA-reduced context
vector of `features/context.py::build_context()`, an SSL encoder's daily embedding, or
any other `(T × d)` matrix with one row per trading day.*

**Governing priors (do not relitigate them; design around them).** Tiny N (~2,800, or
~1,700 with real titles); classical beats ML OOS and regime/sentiment features hurt OOS
(`results/phase6_validation.md`), so the honest default hypothesis is **null lift** and
probing is justified as *insight generation*; and there is **no per-metal news signal**
(`features/text_daily.py`, `MARKET`), so any "text axis" is market-wide by construction.

### B.1 Extracting low-rank axes from a frozen daily representation

**The object.** `Z ∈ ℝ^{T×d}`, indexed by a strictly-increasing, unique
`DatetimeIndex` (enforce with `assert_chronological(Z)` first). Where the representation
admits it, partition columns into a **price-side block** `Z_p` (returns, realized
vol/skew/kurt, drawdown, ratios, spread z-scores from `assemble.py`; macro
TIPS/DXY/VIX/GPR from `macro.py`; COT z) and a **text-side block** `Z_t` (tone means,
topic prevalences, the 16-dim day-mean-embedding PCA). Keeping this partition explicit
is what makes the CCA/CKA steps meaningful.

**PCA / whitening — the low-rank axes.** The PCA must be fit on the training prefix only
(exactly what `context.py::_pca_reduce` does for the embedding block). Inside each
walk-forward fold: (1) standardize using train-prefix mean/std only
(`clustering.py::_standardize`); (2) `PCA(n_components=k, svd_solver="full",
random_state=42).fit(Z_train)`, choosing `k` by the train scree (~90% cumulative EV, and
never larger than `N_train` can estimate — `k ≲ 10–20` is the honest ceiling); (3)
`transform` val/test with the *train-fitted* PCA. Persist `components_`, `mean_`,
`explained_variance_` per fold. **Whiten** for CCA/probing (makes coefficients
comparable and CCA scale-invariant); keep un-whitened scores for variance-ranked
interpretation. Fix a deterministic PCA sign convention so PC1 means the same thing
across folds.

**Why low-rank at all.** (a) With N≈2,800 a full-dimensional probe overfits; the
low-rank projection is the regularizer Phase-6 says you need. (b) Interpretation only
survives if the axis is stable — a handful of high-variance PCs are estimable and
reproducible; the 200th direction is noise.

**CCA — aligning price-side and text-side latents.** To ask "do the headline state and
price state share a common low-dimensional signal, and what is it?", run CCA between the
*train-fitted PCA scores* of `Z_p` and `Z_t` (reduce each block to its own PCs first —
CCA on raw high-dim blocks with small N manufactures spurious unit correlations). Fit on
the train prefix; apply the canonical directions forward; report canonical correlations
`ρ_1 ≥ ρ_2 ≥ …` **on held-out test rows**. Every ρ is tested against a **block
permutation** null (B.4). Expect most to collapse to the null band; a single surviving
pair (e.g. tone-negativity ↔ realized-vol) *is* the insight.

**CKA — representational similarity.** Where CCA asks "is there a shared linear
subspace," linear CKA (`‖YᵀX‖_F² / (‖XᵀX‖_F‖YᵀY‖_F)`, on double-centered blocks, on
held-out rows) asks the coarser "how similar are the two representations overall," and
is the right tool for comparing *whole representations* (price-side vs text-side, or
SSL-`Z` vs the existing `build_context` vector). It is invariant to orthogonal rotation
and isotropic scaling — stable under the PCA sign/rotation ambiguity. Give it the same
block-permutation null.

### B.2 Supervised linear probes

A probe is a **linear** (or logistic) head fit on frozen `Z` (or its PCs). Linearity is
deliberate: it measures *what is linearly decodable*, the interpretable quantity; a
nonlinear head on N≈2,000 overfits (the Phase-6 failure mode). Targets and their repo
construction are the table in §4.1. For every window target, call
`assert_target_strictly_future(features, target, target_horizon, min_nan_tail)` with the
correct `min_nan_tail` — a window target with `min_nan_tail=h` (instead of `h+w−1`)
silently leaks the most recent `w−1` days.

`Ridge` for regression, `LogisticRegression(penalty="l2")` for classification, L2 chosen
on each fold's `Split.val_idx`, never on test. Report OOS R²/RMSE vs a persistence
baseline (regression) or OOS AUC/balanced-accuracy vs the **base rate** (a tercile label
is 33% positive — accuracy is meaningless without the base-rate line). The load-bearing
comparison is never "probe > 0" but "probe > the classical baseline on the same folds"
(`compare_runs`).

**Why the probe itself must be walk-forward, not merely evaluated walk-forward:** a
probe fit once on the full sample leaks the future into its coefficients — it learns the
label geometry of a regime it is then "predicting," and a null relationship looks real.
`Z` may be frozen, but the probe's weights are a fitted projection and leakage rule (3)
applies. Re-fitting per fold also yields the coefficient-stability diagnostic for free.

### B.3 Unsupervised axis interpretation

For each retained PC / canonical variate, assemble **four independent lines of
evidence** (all from the **train fold** — interpretation fit on test rows is itself a
form of leakage) and assign a meaning only where they converge: (1) top-loading
features (the loading vector, not one feature); (2) exemplar days (10 highest/lowest,
joined back via `connection()`, `page_title`s read — real only 2019-09-22+); (3)
correlation with external macro series **not in `Z`** (`DTWEXBGS`, `GPR_DAILY`, TIPS,
COT net-position *changes*) — correlating against panel inputs is circular; (4)
over-represented GDELT `themes` at the axis extremes vs corpus base rate, with a
permutation null (market-wide, never per-metal).

**Stability.** An axis "means" something only if reproducible: track across folds the
sign-aligned cosine of loading vectors, the rank-correlation of top-loading features,
and the sign/magnitude of the probe coefficient. Loadings that flip fold-to-fold get no
interpretation. Use Procrustes to handle rotation-unidentifiability.

### B.4 Statistical rigor on ~2,800 autocorrelated rows

Treat every "N=2,800" as a few hundred independent observations; never use an iid test.
**Permutation/label-shuffle:** build the null by shuffling target labels and re-running
the *entire* fold pipeline (≥500–1,000 shuffles); for sparse events permute event
assignment preserving the event count. **Block bootstrap:** moving/stationary block
bootstrap (block ≈ 10–20 trading days; report sensitivity) for probe-metric CIs,
axis-macro correlation CIs, and the CCA/CKA null (block-permute text against price).
**Multiple-testing:** pre-declare the full grid (6 targets × horizons × axes × theme
lifts), then control FDR (Benjamini–Hochberg, or Benjamini–Yekutieli given positive
dependence); report raw and adjusted p. **Null-lift pre-registration:** before touching
test folds, write down in `journal.md` the frozen `Z` and its hash, the axis recipe,
the probe grid, primary metric per target, block length, correction method, and the
decision rule. This is the Phase-3 discipline that makes a negative result publishable
and a positive one credible. **Budget for and be content with a null.**

### B.5 Leakage traps specific to probing

(1) Fitting the projection on the full sample; (2) fitting the probe head on the full
sample; (3) full-sample standardization; (4) the text one-day lag (`context.py:183`
`sub.shift(1)` — all text lagged, price as-of-close unlagged); (5) window-target tail
(`min_nan_tail = h+w−1`); (6) in-window threshold leakage on re-derived labels (compute
COT/VIX thresholds on the train prefix); (7) real-title regime break (restricting to
titles silently changes the sample to 2019-09-22+ — test both spans and say which you
pre-registered).

### B.6 What "market insight" means for AMC's decisions

A probed axis becomes *business* insight only when it maps to a lever AMC pulls. AMC is
structurally **long physical metal over a days-to-weeks float**. Four translations:

- **Buy-spread floor.** If an axis (the risk/vol PC, or a surviving text-negativity ↔
  forward-realized-vol canonical pair) gives *calibrated* OOS lift on forward realized
  vol at the held metal — and **beats the classical rolling-vol baseline** on the 5–20d
  horizon — high scores widen the minimum buy spread AMC quotes on scrap (the spread
  must cover expected price drift over the float, which scales with forward vol). If it
  merely ties, use it as a confirming flag, not the primary input.
- **FOMC hedge timing.** The hawkish-FOMC probe (`scenarios.yaml::hawkish_fomc`, backed
  by `results/phase5_triangulation.md`) translates directly: a pre-FOMC axis raising the
  probability of a hawkish surprise (validated against `fomc_surprises.mps_orth` /
  `fomc_yield_surprises`) tells AMC to lighten or hedge its float into that meeting.
  Most likely to clear pre-registration — but sign and horizon must be pinned first.
- **PGM risk alarm.** `PL=F`/`PA=F` carry AMC's fattest tail. An axis flagging elevated
  forward PGM vol — even with modest but *real* (block-bootstrap-significant)
  discrimination — caps PGM scrap intake or accelerates offload. The PGM-specific
  content must come from the price/COT block; text contributes only the market-wide
  backdrop.
- **Coin-premium intelligence.** Retail-demand axes (search-interest and coin-premium
  collectors, *when a licence clears them* — today ToU-quarantined; read only
  `quarantine_reason IS NULL`) probed against realized premium moves inform when AMC
  widens or holds coin/specie sell premiums. Design-ready but not-yet-runnable; the
  probe scaffold must filter the quarantine flag so it returns empty rather than
  training on barred data.

For each, the deliverable is a one-line pre-registered decision rule, a back-tested OOS
lift-vs-baseline number with a block-bootstrap CI, and an explicit statement of whether
the lift survived — including "it did not, use the classical baseline" as a first-class
outcome.

### B.7 Reference protocol (one fold)

```
assert_chronological(Z)                                   # leakage guard (1)
for split in walk_forward_splits(Z.index, train_start, …):# expanding train→val→test
    Ztr, Zva, Zte = Z[train_idx], Z[val_idx], Z[test_idx]
    scaler = fit_standardizer(Ztr)                        # train-prefix stats only
    pca    = PCA(k, random_state=42).fit(scaler(Ztr))     # train-prefix only
    Ptr,Pva,Pte = pca.transform(scaler(Zx)) for x in tr,va,te
    # optional: CCA(P_price_tr, P_text_tr) → canonical variates; CKA on test
    for target in TARGETS:                                # forward vol/return, hawkish-FOMC, …
        y = build_target(...)                             # assemble.*; correct min_nan_tail
        assert_target_strictly_future(features, y, h, min_nan_tail=h+w-1 if window else h)
        head = Ridge/Logit; α tuned on val_idx            # never on test
        head.fit(Ptr, ytr)
        log_predictions(run_id, head.predict(Pte))        # eval/harness
# after all folds: block-permutation null + block bootstrap CI + BH/BY across the pre-registered grid
```

**Grounding files:** `src/metals/features/leakage.py`, `src/metals/eval/cv.py`,
`src/metals/features/context.py` (the `shift(1)` text lag at line 183),
`src/metals/features/assemble.py`, `src/metals/models/clustering.py`,
`src/metals/eval/harness.py`, `configs/scenarios.yaml` (`hawkish_fomc`, in-window
thresholds), `results/phase3_writeup.md` (pre-registered null precedent),
`results/phase5_triangulation.md` (hawkish-FOMC finding), `results/phase6_validation.md`
(classical-beats-ML prior).
