# Phase 1 — Price Foundation and Baseline

## Goal
Build a clean panel of metals prices and macro covariates, then train a LightGBM baseline that future phases must beat. By the end of this phase you have a re-runnable data pipeline and a meaningful number to compare against.

## Prerequisites
- Phase 0 complete (env, DuckDB, eval harness, walk-forward CV)
- FRED API key in `.env`

## Steps

### 1.1 Define the canonical price universe
In `configs/universe.yaml`:
```yaml
metals:
  - {ticker: GC=F, name: gold_futures, source: yfinance}
  - {ticker: SI=F, name: silver_futures, source: yfinance}
  - {ticker: PL=F, name: platinum_futures, source: yfinance}
  - {ticker: PA=F, name: palladium_futures, source: yfinance}
etfs:
  - {ticker: GLD, name: gold_etf}
  - {ticker: SLV, name: silver_etf}
  - {ticker: PPLT, name: platinum_etf}
  - {ticker: PALL, name: palladium_etf}
benchmarks:
  - {ticker: ^GSPC, name: sp500}
  - {ticker: ^VIX, name: vix}
  - {ticker: HG=F, name: copper_futures}
  - {ticker: CL=F, name: wti_oil_futures}
  - {ticker: DX=F, name: usd_index}
date_range:
  start: 2007-01-01
  end: null   # null = today
```

### 1.2 Price ingestion module
`src/metals/data/prices.py`:
- `fetch_yfinance(tickers, start, end) -> pl.DataFrame`
- `upsert_prices(df)` writing to the `prices` DuckDB table with idempotent upsert (use a composite primary key `(timestamp_utc, ticker)`)
- CLI: `uv run python -m metals.data.prices --refresh`

Validate by counting rows per ticker per year: expect ~250 trading days/year. Flag any year missing more than 5%.

### 1.3 FRED ingestion module
`src/metals/data/fred.py`. Configure `configs/fred_series.yaml`:
- `DFII10`, `DFII5`, `DFII2` (TIPS yields)
- `DGS10`, `DGS2`, `DGS3MO` (Treasury yields)
- `DTWEXBGS` (USD broad index, daily)
- `VIXCLS`
- `T10YIE`, `T5YIE` (breakeven inflation)
- `BAMLH0A0HYM2` (HY OAS, credit spread proxy)
- `WALCL` (Fed balance sheet, weekly)

Use `fredapi.Fred(api_key=...)`. Store in `macro` table. Important: each series has its own update frequency — forward-fill at use time, not at ingestion time, and never fill a value into a date before its publication date.

### 1.4 GPR index
The Caldara–Iacoviello Geopolitical Risk index is at https://www.matteoiacoviello.com/gpr.htm. Build `src/metals/data/gpr.py` that downloads both the daily and monthly series and stores into `macro`.

### 1.5 Data quality audit
`notebooks/01_data_audit.ipynb`:
- Per-series coverage plot (rows per month)
- Gap analysis flagging gaps > 5 business days
- Outlier detection: returns > 6σ from rolling mean
- Cross-source sanity: daily return correlation of `GLD` vs `GC=F` should exceed 0.95 on overlapping dates; same for the other pairs
- Document anomalies in `journal.md`

### 1.6 Feature engineering — returns and volatility
`src/metals/features/returns.py`:
- `compute_log_returns(prices, horizons=[1,5,20])`
- `compute_realized_vol(returns, windows=[5,20,60])`
- `compute_realized_skew_kurt(returns, window=20)`
- `compute_max_drawdown(prices, window=60)`

Polars DataFrames in, Polars out, with a `timestamp_utc` column.

### 1.7 Spreads and ratios
`src/metals/features/spreads.py`:
- Au/Ag, Pt/Pd, Au/Cu, Au/Oil ratios
- Log-spread changes
- Z-scored spreads over a 252-day rolling window

### 1.8 Macro features
`src/metals/features/macro.py`:
- Real yield = `DGS10 - T10YIE` and 5/20-day changes
- DXY level, 5/20-day change, 252-day percentile
- VIX level, change, percentile
- Yield curve slope `DGS10 - DGS2`
- GPR level, 20-day change

### 1.9 Feature matrix assembly
`src/metals/features/assemble.py` exposes `build_feature_matrix(as_of, target_metal, horizon) -> (X, y)` returning features known strictly before `as_of` and a target measured after. Write a `_check_no_leakage` helper used in tests to assert no feature column contains data with a timestamp ≥ as_of.

### 1.10 Walk-forward CV configuration
Use the Phase 0 utility:
- Train: expanding, minimum 5 years
- Validation: 6 months
- Test: 6 months
- Step: 6 months between splits
- 6–10 splits depending on data start date

### 1.11 LightGBM baseline for volatility
`src/metals/models/lgbm_vol.py`:
- Target: 20-day realized vol of gold returns, predicted 5 days ahead
- Features: everything from 1.6–1.8
- Per split: train, validate (early stopping), predict on test, log to harness
- CLI: `uv run python -m metals.models.lgbm_vol --target gold --horizon 5`

### 1.12 Sanity-check baseline metrics
Rough plausibility ranges (not hard rules):
- IC: 0.05–0.20 for 5-day vol forecast
- RMSE / mean(realized vol): 0.3–0.5
- IC > 0.30 → look hard for leakage
- IC ≤ 0 → check feature stationarity and lag alignment

### 1.13 Replicate for silver, platinum, palladium
Same model, different targets. Compare metrics. Expect silver vol harder than gold (more retail flow), platinum/palladium harder still (illiquid, more regime breaks).

### 1.14 Document the baseline
`results/phase1_baseline.md`:
- Metrics table per metal × horizon
- LightGBM feature importance per model
- Three things that surprised you, written before you re-tune

## Deliverables
- One-command data refresh from yfinance, FRED, GPR
- DuckDB populated with prices and macro through current date
- Feature matrix builder with leakage tests
- Trained LightGBM baselines with metrics in the eval harness
- `results/phase1_baseline.md`

## Common pitfalls
- `yfinance` adjusts for splits and dividends silently. Verify on a known date (e.g., GLD's annual expense impact).
- FRED series with mismatched frequencies (weekly `WALCL`, daily `DGS10`) — forward-fill at feature build time only.
- Mixing future-revised values with first-print values for surprise calculations later. For Phase 1 (vol forecasting) this is OK; remember the distinction matters for Phase 2.
- Cross-asset returns at different close times — e.g. COMEX closes at 1:30 ET, ETFs at 4:00 ET. Use the same ticker's own close-to-close return when computing per-ticker features.
