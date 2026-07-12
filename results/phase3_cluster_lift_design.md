# Cluster → forward-vol lift: pre-registered design

**Status: pre-registered 2026-07-03, BEFORE any lift run** (revised same day
after an adversarial design review, still before any run). The decision rules
below were fixed before seeing any A/B/C results; changing them after seeing
results voids the gate (note it in the journal if it ever happens).

## Question

Do Phase 3 regime labels (Option C clustering: tone + themes + macro/price
context, `phase3_optC_tone_lag1_2024split` lineage) improve an out-of-sample
volatility forecast over the same forecast without them — and, downstream,
would PAGE_TITLE-embedding clusters improve it enough to justify the GPU
spend (assessment §7 gate)?

## Row policy (binding, executable)

Let `context_optC` = `build_context(include_embeddings=False)` on the live DB
and `fm` = `build_feature_matrix(feature_set="full")` for the target below.

```
shared = fm.X.index ∩ context_optC.dropna().index      # computed once
```

Every arm's clustering input is `context_*.loc[shared]` — no arm may derive
its own row set via a private `dropna()`. `fm` rows are likewise restricted to
`shared` for **all** arms, A included, so ΔRMSE is paired row-for-row.

## Arms

| Arm | Clustering input (all restricted to `shared`) |
|---|---|
| **A** | none — Phase-1 features only. |
| **B** | `context_optC.loc[shared]` → per-fold regime features: cluster one-hots + `regime_confidence` + purged `regime_target_mean`. Nothing else (in particular no raw cluster id — `build_regime_features` does not emit one). |
| **B_notext** (ablation) | `context_optC.loc[shared]` minus the text-derived columns (`mean_tone_*`, `n_articles`, `topic_*`) — same rows as B by construction. Run only if B beats A. |
| **C** (embedding gate) | Title-era sub-experiment, see below. Run only if the B/B_notext readout justifies it. |

## Target and metric

- **Primary target**: `GC=F` forward realized vol — `target_kind=realized_vol`,
  `target_horizon=5`, `realized_vol_window=20` (the Phase-1 baseline target:
  vol over trading days `[t+5, t+24]`).
- **Primary metric**: per-split test RMSE. Headline = mean across splits.
  Decision statistic, literally: `Δ_s = RMSE_arm2,s − RMSE_arm1,s` per split
  `s` where arm2 is the richer arm (B vs A; C vs B). Negative Δ favours the
  richer arm.
- **Secondary (report, never decide)**: per-split IC; `SI=F`; `horizon=20`.
  Each target uses its own purge: `purge_days_for(h, w)` from
  `metals.features.regimes` (44→45 for h=5/w=20; **65** for h=20/w=20 — a
  flat 45 provably under-purges the h=20 target, whose window spans ≥53
  calendar days).

## Folds

`walk_forward_splits(train_start="2015-02-19", val_days=180, test_days=180,
step_days=180, min_train_days=5*365)`, computed **once on `shared`** and
reused identically for arms A, B, B_notext. Expected ≈ 9 splits, test windows
≈ 2021-02 → 2026-05.

**Arm C is a separate paired sub-experiment on the title era** (its context
columns do not exist before 2019-09-22, so sharing A/B's folds would either
silently shrink C's row set or make early folds degenerate):
`shared_C = shared ∩ [2019-09-23, end]`, folds =
`walk_forward_splits(train_start="2019-09-23", val_days=180, test_days=180,
step_days=180, min_train_days=3*365)` (≈ 6 splits, test ≈ 2023-03 → 2026-05).
The C gate compares **C vs B re-run on these same `shared_C` folds** — B is
refit there so the comparison is paired; the main-fold B numbers are never
compared against C.

## Leakage rules (binding)

1. **Per-fold refit**: for each split, regime features are built with
   `boundary = split.train_end` (exclusive, matching the split's train
   window): clustering fit on rows `< boundary` only; all rows assigned via
   `approximate_predict` — one uniform assignment path for train and test.
   For arm C the context PCA must also be fold-fit: `build_context`'s
   `pca_fit_until` mask is **inclusive**, so pass the last trading day
   strictly before `boundary`, never `boundary` itself.
2. **Target encoding purged**: the cluster-conditional target mean uses train
   rows `< boundary − purge` only, purge = `purge_days_for(h, w)` per target
   (rule above). Clustering itself needs no purge (context features are
   strictly backward-looking; text is lagged one trading day).
3. Feature frames pass `assert_chronological`; regression tests must prove
   post-boundary perturbations (context rows or target values) cannot move
   pre-boundary regime features (`tests/test_features_regimes.py`).

## Determinism and bookkeeping (binding)

- All arms use `DEFAULT_LGBM_PARAMS` verbatim plus the pinned overrides
  `{"seed": 42, "deterministic": True, "force_row_wise": True,
  "num_threads": 8}`. **No per-arm tuning of any kind.**
- UMAP/HDBSCAN via `ClusteringConfig()` defaults (random_state=42) for every
  arm and fold.
- The gate is decided by runs executed **on AYMStation**; a re-run elsewhere
  is diagnostic only (LightGBM results can drift with core count, and the
  bars are 1–1.5%).
- Every arm runs through the eval harness (`register_run` +
  `log_predictions` per split); run ids recorded in the journal next to the
  readout.

## Accepted caveats (pre-existing in the Phase-1 CV, identical across arms)

1. No train/val embargo: train targets near `train_end` overlap the val
   window.
2. No val/test embargo either, and early stopping selects the boosting round
   on val targets whose forward windows reach into the test window — so the
   stopping round sees some test-period return information, in every arm.
   Absolute RMSE numbers carry this caveat; the paired ΔRMSE largely cancels
   it, but that cancellation is an assumption, not a theorem — reported as
   such.

## Decision rules (fixed in advance; S = number of splits in the comparison)

- **B beats A** iff
  `(mean_s RMSE_B − mean_s RMSE_A) / mean_s RMSE_A ≤ −0.010`
  AND `#{s : Δ_s < 0} ≥ ceil(0.6 · S)` (S=9 → 6 splits).
- **Attribution** (only if B beats A): let
  `gap_total = mean RMSE_A − mean RMSE_B` and
  `gap_text = mean RMSE_B_notext − mean RMSE_B`.
  The lift is attributed to **text** iff `gap_text ≥ 0.5 · gap_total`;
  if `gap_text ≤ 0` it is attributed entirely to macro clustering.
- **C beats B (embedding gate)** iff, on the `shared_C` folds,
  `(mean_s RMSE_C − mean_s RMSE_B) / mean_s RMSE_B ≤ −0.015`
  AND `#{s : Δ_s < 0} ≥ ceil(0.6 · S)` (S=6 → 4 splits).
  Below the bar → no corpus-scale embedding spend (assessment §7).
- A negative or null B − A readout is a reportable result and caps arm C's
  priority.

## Machinery

`metals/features/regimes.py` (`build_regime_features`, `purge_days_for`),
tests in `tests/test_features_regimes.py`.
