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
