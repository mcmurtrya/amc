# Phase 1 — LightGBM Volatility Baseline

First-run baseline that every later phase must beat. One LightGBM regressor per metal, retrained on every walk-forward split, predicting realized vol over a strictly-forward window.

## Target specification

For each metal *m* and date *t*, the target `y[t]` is the annualised realized
volatility of one-day log returns over the **forward window** `[t+h, t+h+w-1]`,
with `h = 5` (target horizon) and `w = 20` (vol-window width). The last
`h + w - 1 = 24` rows of `y` are by construction NaN, which is enforced by
`assert_target_strictly_future(min_nan_tail=24)`.

This convention was chosen after the first run produced an implausibly high
mean IC ≈ +0.64 — see "Three surprises" below.

## Configuration

- **Feature matrix:** price-derived (multi-horizon log returns, realised vol at 5/20/60 days, skew/kurt, max drawdown, four log spreads and their z-scores) and macro (real yield, breakevens, DXY level/change/percentile, VIX level/change/percentile, curve slope, GPR level/change). Macro is aligned to the price index and forward-filled at feature-build time only.
- **Walk-forward CV:** expanding train ≥ 5 years; 6-month validation; 6-month test; 6-month step. Train start `2010-01-01` (sidesteps the 2007–2009 thin-trading gaps in PL=F and PA=F).
- **LightGBM:** 600 trees, lr 0.03, num_leaves 31, feature/bagging fraction 0.85, min_data_in_leaf 50, early stopping at 50 rounds on the validation split.
- **Cross-fitting:** none at this phase — each split's model is trained independently.

## Results

| Metal | Ticker | Splits | Mean RMSE | Mean IC | Eval-harness run_id |
|---|---|---|---|---|---|
| Gold      | GC=F | 22 | 0.0563 | **+0.070** | `0104a52b-a30d-4f93-9b95-3dd52d25d565` |
| Silver    | SI=F | 22 | 0.1330 | −0.097 | `3369d71c-8c8d-4673-b6d1-81d12f4364b6` |
| Platinum  | PL=F | 22 | 0.0987 | −0.071 | `f3c32db5-44f5-4871-bd8a-b0de00e8eaf1` |
| Palladium | PA=F | 22 | 0.1349 | −0.055 | `eda77804-d3eb-4191-b682-fa2282a3d315` |

Plan step 1.12's plausibility band for a 5-day vol forecast is IC ∈ [0.05, 0.20]; only gold lands in it. The other three metals' mean ICs are slightly negative — i.e. the conditional model is doing fractionally worse than predicting the unconditional mean over a 6-month test window.

## Three surprises

1. **The first run looked great because it was leaking.** Mean IC came in at +0.643 across 22 splits with per-split IC reaching +0.94. The plan step 1.12 sanity check explicitly says "IC > 0.30 → look hard for leakage", which it was. Cause: `build_feature_matrix` constructed the realised-vol target via `shift_target(trailing_rvol, h)`, so the target window at row *t* was `[t-w+h+1, t+h]` — 15 of 20 days observable at *t*. The forward-window fix above is what produced the honest numbers in the table. Lesson: a structural leakage guard that only checks the NaN tail (`assert_target_strictly_future` pre-fix) is too weak; the strengthened version requires `min_nan_tail = h + w - 1` for window-valued targets.

2. **Gold is the *only* metal with positive mean IC.** The plan flagged this directionally — "expect silver harder than gold, platinum/palladium harder still" — but it's striking that the LightGBM with the current feature set doesn't even reach IC > 0 on silver, despite Au/Ag being correlated. Likely explanations to test in Phase 2+: feature mix is gold-centric (heavy macro/monetary, light supply/industrial); the metals require different conditioning variables; or there isn't enough signal in close-to-close vol for these three with this feature set.

3. **Split 21 is uniformly the hardest split for every metal.** RMSE on split 21 is 4–10× the per-metal mean (0.198 for Au, 0.671 for Ag, 0.449 for Pt, 0.371 for Pd). Split 21 covers the most recent six months and contains both the 2026-01-30 silver/gold/copper move (audit flagged for human verification) and the 2025-04-09 S&P move. Walk-forward CV doesn't shrink this kind of regime-shift tail; methods sensitive to recent state (online updating, scenario conditioning from Phase 3/5) may address it.

## Caveats documented from the data audit (`notebooks/01_data_audit.ipynb`)

- **`BAMLH0A0HYM2` (HY credit spread) only goes back to 2023-05-23.** It is a feature input to `compute_macro_features`. Until the FRED-side issue is resolved (likely a series-ID rename), credit-spread information is effectively missing for >85% of the training window.
- **ETF / futures close-to-close correlations** are 0.886–0.918, below the plan's 0.95 bar. Almost certainly the COMEX-1:30 vs ETF-4:00 close mismatch; documenting rather than fixing for the baseline (we model the futures directly, not the cross-relationship).
- **Pt/Pd 2007–2009 thin-trading gaps** are sidestepped by `train_start = 2010-01-01`.

## What this baseline does *not* tell us

- Whether ICs hold up on an untouched hold-out window — Phase 6 will check.
- Whether a tuned LightGBM beats this baseline. The defaults are deliberately untuned.
- Whether news/text features (Phase 3) or learned representations (Phase 4) lift Pt/Pd above the gold-set bar.
- Whether a *direction* model (`--target return`) would tell a different story. Only the vol target was run.

## Reproducing

```bash
$env:UV_PROJECT_ENVIRONMENT = $null
uv run python -m metals.data.migrations.runner
uv run python -m metals.data.prices
uv run python -m metals.data.fred
uv run python -m metals.data.gpr
foreach ($t in "GC=F","SI=F","PL=F","PA=F") {
  uv run python -m metals.models.lgbm_vol --ticker $t --target realized_vol --horizon 5
}
```

Run records live in the eval harness (`runs` and `run_predictions` tables in `data/processed/metals.duckdb`) — use `metals.eval.harness.compute_metrics(run_id)` to recompute the headline numbers.
