# Metals Research Journal

A running log of work, learnings, surprises, and open questions. Add an entry at the end of every working session.

---

## Lessons learned (updated periodically)

_Nothing yet — add the most important meta-lessons here as they accumulate._

---

## 2026-05-14

### What I did

- Set up Phase 0 scaffolding: project layout, `pyproject.toml`, `.env.example`, `.gitignore`, README, journal.
- Implemented DuckDB connection helpers (`src/metals/data/db.py`) and migration runner (`src/metals/data/migrations/runner.py`).
- Wrote initial schema migration (`001_initial_schema.sql`) with tables for prices, macro, events, positioning, headlines.
- Implemented the evaluation harness (`src/metals/eval/harness.py`) supporting register_run, log_predictions, compute_metrics, compare_runs, list_runs.
- Implemented walk-forward CV utility (`src/metals/eval/cv.py`) with expanding-window splits and a within-split leakage check.
- Wrote pytest coverage for both CV and harness.

### What I learned

- DuckDB's `ON CONFLICT … DO UPDATE` is the cleanest way to handle idempotent prediction logging; cleaner than DELETE+INSERT.
- The proper invariant for expanding-window walk-forward CV is *within-split* disjointness — test data of split i legitimately appears in train of split i+1.

### What confused me

- _Nothing yet._

### Next session

- Phase 1: implement `yfinance` and FRED ingestion, build the price/macro panel, train the LightGBM volatility baseline.

---

## 2026-05-14 (Phase 1)

### What I did

- Configs: `configs/universe.yaml` (price universe with metals/ETFs/benchmarks) and `configs/fred_series.yaml` (FRED series with frequencies).
- Data ingestion modules:
  - `metals.data.config` (YAML loader)
  - `metals.data.prices` (yfinance, idempotent upsert, coverage report)
  - `metals.data.fred` (FRED API)
  - `metals.data.gpr` (Caldara-Iacoviello daily GPR)
- Feature engineering:
  - `metals.features.loaders` (read prices/macro from DuckDB into wide pandas frames)
  - `metals.features.returns` (log returns, realized vol, skew/kurt, max drawdown)
  - `metals.features.spreads` (Au/Ag, Pt/Pd, Au/Cu, Au/Oil ratios; log changes; rolling z-scores)
  - `metals.features.macro` (real yield, breakeven changes, DXY/VIX changes and percentiles, curve slope, GPR change, HY OAS)
  - `metals.features.leakage` (chronological, target-strictly-future, warmup checks)
  - `metals.features.assemble` (build_feature_matrix with FeatureMatrix dataclass)
- Models: `metals.models.lgbm_vol` — walk-forward LightGBM regressor with CLI, early stopping, per-split logging to eval harness.
- Tests: 33 new tests covering returns, spreads, macro, leakage, assembly, config, and end-to-end LightGBM. All 50 tests pass.

### What I learned

- Decided to standardize on pandas throughout (vs. Polars in the plan) to reduce friction with DuckDB and LightGBM boundaries. Easy to swap later if needed.
- `lgb.early_stopping` callback API is cleaner than the old `early_stopping_rounds` kwarg.
- The leakage guard's "target has NaN tail of length `horizon`" heuristic catches the most common forward-shift bugs without needing to track every feature's lineage.

### What confused me

- OneDrive on Windows sometimes truncated `.py` files written through the file tool. Workaround used during scaffolding: write to the scratch folder and `cp` via bash. Should not affect normal local development.

### Next session

- Phase 2: economic event calendar (FOMC, CPI, NFP, ECB), surprise measures, COT ingestion with the Friday-close lag fix, and the Jordà local-projection module with HAC standard errors.

---

## 2026-05-24

### What I did

- First end-to-end run of the Phase 1 pipeline against real data. Surfaced and fixed several bugs:
  - `prices.py` was silently dropping `timestamp_utc` because the newer `yfinance` no longer names the date index; switched to `reset_index(names=...)`.
  - `fred.py` aborted the whole batch on the first missing series; now skips with a warning.
  - Removed `DX=F` (delisted on yfinance — DTWEXBGS from FRED covers the same signal) and `DFII2` (never existed as a FRED series).
  - Added `xlrd>=2.0` for the Caldara-Iacoviello GPR `.xls` download.
- Built `notebooks/01_data_audit.ipynb` (plan step 1.5): per-series coverage, gap analysis, return outliers, futures/ETF correlations. Surfaced two real blockers and three lesser ones — see notebook's Findings cell.
- Traced a recurring `RuntimeWarning: invalid value encountered in log` to **CL=F closing at −$37.63 on 2020-04-20** (real WTI futures negative-price event). Masked non-positive values to NaN in `compute_log_returns`, `compute_max_drawdown`, and `compute_log_spread_changes`. Added regression tests for each.
- **Caught a target-leakage bug** in `build_feature_matrix`: realised-vol target was a trailing-window vol shifted by `h`, so the measurement window `[t-w+h+1, t+h]` was 75% observable at time *t*. The first gold baseline came in at mean IC = +0.643 (plan's red-line is 0.30). Switched to a strictly-forward window `[t+h, t+h+w-1]` and strengthened `assert_target_strictly_future` to require `min_nan_tail = h + w - 1`. Honest re-run lands at IC = +0.070.
- Ran the LightGBM baseline for all four metals. Wrote `results/phase1_baseline.md`.

### What I learned

- "Passes the leakage guard" is not the same as "no leakage" if the guard only checks structure, not source-window membership. The NaN-tail check was too coarse for window-valued targets.
- np.log on a series with a single negative value blows up the whole feature path — defensive masking is cheap and worth it at every np.log site.
- yfinance's `yf.download` output schema changed; rely on `reset_index(names=...)` rather than renaming whatever happens to land in the index column.
- The plan's IC = 0.05–0.20 plausibility band for 5-day vol forecast is a useful tripwire — the buggy run's IC = 0.64 looked like a great result until I checked it against this.

### What confused me

- Only **gold** has positive mean IC. Silver/Pt/Pd come in slightly negative even after the leakage fix. Plan anticipated "harder than gold" but I'd expected at least mild positive IC across all four. Possibly the feature set is too gold-centric (macro/monetary heavy, light industrial/supply), or close-to-close vol is genuinely harder to predict for these. Will revisit when Phase 2 events and Phase 3 text features come online.
- Split 21 is uniformly the hardest split for all four metals (RMSE 4–10× the per-metal mean). It covers the most recent six months, including the audit-flagged 2026-01-30 silver/gold/copper move and the 2025-04-09 S&P move. Walk-forward CV doesn't dampen this kind of regime-shift tail.

### Open items not resolved today

- `BAMLH0A0HYM2` only goes back to 2023-05-23 (786 rows vs. ~4,800 expected). Likely a FRED series-ID rename. Effectively no credit-spread feature for the bulk of training.
- ETF / futures close-to-close correlations all 0.886–0.918, below the plan's 0.95 audit threshold. Almost certainly the COMEX-1:30 vs. ETF-4:00 close-time mismatch. Documenting rather than fixing — the model trains on the futures directly.
- 2026-01-30 SLV at −33.6% — flagged for manual verification. If real, this is the largest move in the panel and may need its own handling.

### Next session

- Either Phase 2 (events + local projections, per the plan) or a focused Phase 1 cleanup pass: investigate `BAMLH0A0HYM2` rename, extract LightGBM feature importances per metal, and check whether the negative IC on Ag/Pt/Pd is feature-mix-driven or genuinely the data.
