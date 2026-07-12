# Metals Research Journal

A running log of work, learnings, surprises, and open questions. Add an entry at the end of every working session.

---

## Lessons learned (updated periodically)

- **Reproducibility / data hygiene**
  - Prefer FRBSTL-calculated FRED series (BAA10Y, T10Y2Y, T10YIE) over licensed
    third-party indices, which can be truncated to a short license window without
    notice (cost us the HY-OAS feature). The FRED coverage audit now catches this.
  - A structural leakage guard that only checks the target's NaN tail is too weak
    for window-valued targets — require `min_nan_tail = h + w - 1`. A 5-day vol IC
    much above ~0.2 is a leakage tripwire, not a great result.
  - OneDrive can corrupt files written through the editor tools — truncating
    mid-content or appending NUL bytes. Author code/SQL via bash heredoc (`cat >`)
    and verify with `wc -l` plus a NUL scan before trusting the file.
- **DuckDB gotchas**
  - A semicolon inside a `--` comment silently truncates whole-file multi-statement
    execution (`conn.execute(text)` stops early, no error). Never put `;` in a
    migration comment.
  - `ALTER TABLE ... DROP COLUMN` is refused while any index exists on the table;
    drop and recreate the index around it. DuckDB also does not shrink the data
    file in place on DROP COLUMN — rebuild a fresh DB to reclaim bytes.
  - `runs` / `run_predictions` are created lazily by the harness, not by a
    migration; rebuilding from migrations alone loses them.

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

---

## 2026-05-25 (Phase 2 — events + local projections)

### What I did

- Built `src/metals/models/lp.py`: per-horizon OLS of cumulative h-step-ahead log return on treatment + controls, Newey-West HAC standard errors. 8 tests including a recovery test against a known synthetic IRF.
- Curated FOMC calendar 2007-2026 from federalreserve.gov pages, committed at `configs/fomc_calendar.csv`. 176 events. `src/metals/data/events.py` loader + upsert + CLI.
- CFTC disaggregated COT positioning, 2010-2026, 3,420 rows. Critical Friday-close-of-release timestamp convention (not Tuesday positioning date). Excludes E-MINI / MICRO gold variants. Module + 5 tests.
- Bauer-Swanson MPS_ORTH FOMC surprises via SF Fed XLSX. Migration 002 added `fomc_surprises` table. 361 rows 1988-2023.
- Wrote five Phase 2 notebooks:
  - `02_fomc_indicator_irf.ipynb` — smoke test, indicator only. Weak as the plan predicted.
  - `03_fomc_surprise_irf.ipynb` — **headline result**. Hawkish IRFs are sharp negative across all four metals (gold -1.5% at h=5, silver amplified). Dovish IRFs are small positive and insignificant.
  - `04_geopol_dxy_irf.ipynb` — GPR mostly null on top-5% threshold; DXY +2σ weak; DXY −2σ surprisingly negative (risk-off sample contamination).
  - `05_fomc_robustness.ipynb` — 100-trial placebo, 2015 subsample, alt thresholds. Hawkish IRF survives for Au/Ag/Pt, weakens for palladium post-2015.
- Wrote `results/phase2_scenarios.md` — durable Phase 2 record, scenario-by-scenario tables, methodology notes, Phase 5 hand-off.

### What I learned

- **Bauer-Swanson MPS_ORTH was the right pivot.** The plan flags consensus history as the binding constraint; pivoting to a public high-frequency-identified surprise series (already orthogonalised for the Fed information effect) turned the hardest piece of Phase 2 into a 50-line ingestion module.
- **Indicator-only vs surprise spec is a real signal multiplier.** Pooled FOMC indicator: 3/24 borderline-significant cells (notebook 02). Tercile-split surprise: 11/24 significant cells. The plan's "indicator alone is mostly priced in" was vindicated numerically.
- **Cross-metal sign consistency is a cheap, powerful filter.** For monetary scenarios Au/Ag/Pt should track. They do for hawkish FOMC. They don't for GPR spikes (palladium goes positive at h=20 while others stay near zero), which is itself a yellow flag on that scenario.
- **Placebo distributions are tighter than I expected.** With ~3,400 trading days × 35 random "events" each trial, the placebo SD at h=5 is < 1% for gold. Makes the p=0.000 for real Au and Ag a strong robustness signal rather than a coincidence.
- **2σ DXY -shock landed in unexpected territory.** The down-shock subsample is heavily contaminated by risk-off regimes where USD weakness and metal selling co-occur, inverting the textbook FX-pricing channel. Useful reminder that "weakening dollar" means different things in different macro regimes.

### What confused me

- **Palladium's regime instability.** Hawkish FOMC -4.68% at h=5 pre-2015, near-zero and insignificant post-2015. Most likely the 2018-22 supply squeeze dominating, but worth a Phase 5 second opinion.
- **The dovish/hawkish asymmetry.** Hawkish surprises produce sharp negative IRFs; dovish surprises produce mirror-image-shaped but insignificant positive IRFs. Could be tail-risk hedging asymmetry, regime composition of the 2010-2023 sample, or just sample noise on 35 events per tercile. Not resolved.

### Open items not resolved today

- CPI / NFP / ECB / BoE calendars and surprises. Consensus-history acquisition is the binding constraint (Bauer-Swanson is FOMC-only).
- Refresh Bauer-Swanson XLSX — ~10 FOMC meetings 2024-2026 are missing.
- Robustness on GPR and DXY scenarios — currently deferred until treatment definitions sharpen.
- The Phase 1 open items still open: BAMLH0A0HYM2 truncation, ETF/futures correlation note, 2026-01-30 SLV verification.

### Next session

- Phase 3 (text + clustering) or Phase 2 follow-up (CPI/NFP/ECB ingestion). Phase 3 is the bigger pivot and where the multimodal narrative starts; CPI/NFP is incremental and adds robustness to the FOMC story.

---

## 2026-06-15 (Phase 1 cleanup)

### What I did

- **BAMLH0A0HYM2 diagnosis** — fetched the FRED series page directly. Smoking gun: "Starting in April 2026, this series will only include 3 years of observations. For more data, go to the source." ICE Data Indices imposed a 3-year license window on FRED publication, leaving only 2023-05-15 onward via the free API. Not an ingestion bug, not a rename.
- **Substituted `BAA10Y`** (Moody's Baa - 10Y Treasury, daily, freely published, history from 1986-01-02). Updated `configs/fred_series.yaml`, `src/metals/features/macro.py` (feature renamed `hy_oas_chg_{h}d` -> `baa_spread_chg_{h}d`), and the fixtures in `tests/test_features_macro.py` and `tests/test_features_assemble.py`. All tests still green.
- **FRED coverage audit** — added `coverage_report()` to `src/metals/data/fred.py`. Each refresh now prints a per-series table of (rows, expected, coverage %) and warns loudly when any series drops below 50% of expected. Sorted ascending so the worst offenders are at the top. Six unit tests in `tests/test_data_fred.py`.
- **Feature-importance schema** — migration `004_run_feature_importances.sql` with `(run_id, split_id, feature_name, importance, importance_type)`. Eval harness gained `log_feature_importances`, `fetch_feature_importances`, and `aggregate_feature_importances` (the last with normalize-by-split-sum so cross-split comparison isn't dominated by raw scale). Eight tests in `tests/test_harness_importances.py`.
- **Wired importance capture into the LightGBM baseline** — `train_one_split` now returns `(predictions, result, importances)` and `run()` logs both `gain` and `split` importance types after every walk-forward split. End-to-end smoke test extended to confirm.
- **Diagnostic script** for the negative-IC question — `scripts/phase1_diagnose.py`. For each metal, trains seven feature configurations (full panel, three ablations dropping {returns+vol, spreads, macro}, three marginal-only variants), logs every run to the harness, and writes a markdown report to `results/phase1_negative_ic_diagnosis.md` with per-config mean IC, std, and the fraction of splits with IC > 0. Reading guide is embedded in the report.
- **115 tests, all passing**, +24 since the Phase 2/3 baseline.

### What I learned

- **License-driven series truncation is a recurring risk in macro pipelines.** The FRED coverage audit catches it automatically going forward, but the conceptual takeaway is to prefer FRBSTL-calculated series (BAA10Y, T10Y2Y, T10YIE) over licensed third-party indices wherever the analytical content is comparable. The Bauer-Swanson FOMC surprises pivot in Phase 2 was the same pattern resolved differently.
- **Booster `feature_importances_` defaults to split-count, not gain.** Easy gotcha. We capture both and normalize-per-split in aggregation so the gain-vs-split discussion is one column away rather than a model rerun.
- **Ablation > raw importance for "is this feature carrying its weight"** — high gain with low ablation impact is the textbook signature of fitting noise. The diagnostic script is built around the ablation/marginal pair so this comparison is one report away.

### What confused me

- OneDrive sync still occasionally truncates file writes mid-content; the workaround is to compose via bash heredoc + `cat >` or `python -c` and verify with `wc -l`. Should not affect normal local development.

### Open items not resolved today

- The diagnostic script still needs to be **run** on the user's box against real data. Output goes to `results/phase1_negative_ic_diagnosis.md`. Expected runtime ~5-10 minutes for 4 metals x 7 configurations x ~8 splits.
- 2026-01-30 SLV at -33.6% (Phase 1 audit) still unverified.
- ETF/futures correlation note (close-time mismatch) still documented-not-fixed.

### Next session

- User runs `scripts/phase1_diagnose.py` and we read the resulting report together. Then either deepen the lowest-IC subset, or move on to Phase 3 (text + clustering) which was the natural next step before this cleanup detour.

---

## 2026-06-18 (Phase 1 diagnostic + DuckDB housekeeping)

### What I did

- **Ran `scripts/phase1_diagnose.py` against real data** (4 metals x 7 feature
  configs x 22 walk-forward splits). Wrote `results/phase1_negative_ic_diagnosis.md`.
  Headline result answers the long-open question: the negative IC on Ag/Pt/Pd is
  **feature-mix-driven, not intrinsic to the data**, and the culprit is the
  108-feature `returns_and_vol` block. `only_returns_and_vol` is negative for all
  four metals; dropping it (`no_returns_and_vol`, 34 feats) lifts every metal —
  Au +0.064 -> +0.164, Ag -0.064 -> +0.111, Pt -0.071 -> +0.074, Pd -0.056 -> -0.016.
  The compact spreads (16) + macro (18) blocks carry the signal; for gold,
  `only_macro` (+0.101) and `only_spreads` (+0.089) each beat the full model.
- **Diagnosed the 7GB DuckDB.** It is one table: `headlines`, 14,117,541 GDELT GKG
  rows spanning 2020-01-01..2024-01-15. Everything else combined is <150K rows.
  Zero duplicate ids (clean, no idempotency bug). The cost is redundant URL
  storage: the article URL (~93 chars) is stored 3x per row — `article_url`,
  `headline` (exact copy, 100% of rows), and `headline_id` (timestamp + URL[:200]).
- **Removed a stray empty `-Force` directory** at the repo root (a PowerShell
  `mkdir`/`-Force` mishap). Needed the Cowork file-delete grant; the mount blocks
  `rm` by default.
- **Slimmed `headlines` (conservative): dropped the redundant `headline` column.**
  - `migrations/005_drop_redundant_headline.sql` (drop index -> drop column ->
    recreate index).
  - `gdelt.py` no longer writes `headline` (parse output + upsert columns).
  - `scripts/compact_headlines.py`: rebuilds a fresh, compacted copy with the
    column absent; read-only on the source, verifies per-table row counts, and
    `--replace` swaps in a timestamped `.bak`. Tested end-to-end on synthetic data
    incl. harness tables; all 33 light tests + the fresh migration build pass.

### What I learned

- **OneDrive write corruption is real and bit twice this session.** The `Write`
  tool truncated `005_*.sql` mid-comment (statements never landed) and appended
  39 trailing NUL bytes to `gdelt.py` (grep flagged it "binary"; Python refused to
  parse). Fix/standing rule: author code/SQL via bash heredoc (`cat >`), then
  verify with `wc -l` and a NUL scan (`open(p,'rb').read().count(b'\x00')`).
- **A semicolon inside a SQL comment silently truncates DuckDB whole-file
  execution.** `conn.execute(text)` runs multiple statements, but a stray `;` in a
  `--` comment makes it stop early and skip later statements with no error. This
  cost an hour (the column "wouldn't drop"). Rule: no semicolons in migration
  comments. Migration 003 only survived because it is comment-light.
- **DuckDB refuses `ALTER TABLE ... DROP COLUMN` while any index exists on the
  table** (even an index on a different column). Drop the index first, then
  recreate it. The PK does not block.
- **`runs` and `run_predictions` are created lazily by the harness
  (`_ensure_schema`), not by any migration.** Anything that rebuilds the DB from
  migrations alone silently loses them. The compaction script iterates *source*
  tables and calls `_ensure_schema`, so harness tables keep their PKs (the
  `log_predictions` ON CONFLICT upsert depends on them).

### What confused me

- The diagnostic ran against the 2026-05-25 DB snapshot, which **predates the
  BAA10Y substitution**, so `baa_spread_chg_{5,20}d` were 100% NaN. LightGBM just
  ignores all-NaN columns, so the headline finding stands, but the macro subset's
  true ceiling is understated until a FRED refresh lands BAA10Y.

### Open items not resolved today

- **Run `scripts/compact_headlines.py` locally** to actually reclaim the bytes —
  migration 005 + the code change are in, but the existing 7GB file is unchanged
  on disk (DuckDB does not shrink in place; the OneDrive mount is too slow to
  rebuild a multi-GB DB from a sandbox). Expect ~24% / ~1.7GB back; `--replace`
  keeps a `.bak`.
- **Re-run the diagnostic after a FRED refresh** (BAA10Y) so the macro subset is
  evaluated with its credit-spread feature live. Then consider pruning the
  `returns_and_vol` block in the production baseline and re-benchmarking against
  `results/phase1_baseline.md`.
- The diagnostic's 28 harness runs were logged to a throwaway slim DB, not the
  canonical one, so those run_ids will not resolve locally — re-run if you want
  them persisted.
- Still open from prior sessions: 2026-01-30 SLV -33.6% verification; ETF/futures
  close-time correlation note.

### Next session

- Compact the DB locally, refresh FRED for BAA10Y, re-run the diagnostic, then
  decide between a lean-feature production rerun and moving on to Phase 3
  (text + clustering).

---

## 2026-06-18 (later — compaction done, diagnostic re-run with BAA10Y live)

### What I did

- **Compacted the canonical DuckDB** via `scripts/compact_headlines.py --replace`.
  DuckDB block accounting (authoritative): **6.9 GiB -> 4.5 GiB used, ~2.4 GiB /
  ~35% reclaimed**, 0 free blocks. All row counts preserved (headlines
  14,117,541). A 6.9 GiB `metals.duckdb.bak-20260618_123519` remains as backup.
  (The OneDrive mount still reports the old ~7.4 GB file size until it re-syncs;
  `PRAGMA database_size` is the source of truth.)
- **Refreshed FRED (BAA10Y now live) and re-ran `phase1_diagnose.py`** against the
  compacted DB. `results/phase1_negative_ic_diagnosis.md` regenerated.

### What I learned (refined finding)

- **Core diagnosis holds with the credit-spread feature live.** `only_returns_and_vol`
  is negative for all four metals (Au -0.076, Ag -0.087, Pt -0.050, Pd -0.083), and
  dropping that 108-feature block lifts Au/Ag/Pt: `all` -> `no_returns_and_vol` is
  +0.022 -> +0.098 (Au), -0.068 -> +0.079 (Ag), -0.054 -> +0.047 (Pt). Silver and
  platinum flip negative -> positive.
- **BAA10Y added real signal.** `only_macro` improved most for platinum
  (-0.015 -> +0.033) and silver (+0.026 -> +0.035); gold steady (+0.100).
- **Palladium is the clear exception.** Dropping returns/vol does NOT help it
  (-0.016 -> -0.043) and macro hurts it (`only_macro` -0.075; its best subset is
  `no_macro` +0.040). Consistent with the Phase 2 note that palladium is dominated
  by its 2018-22 supply squeeze / industrial dynamics, not macro/monetary drivers.

### Open items not resolved today

- **Delete `metals.duckdb.bak-20260618_123519`** (6.9 GiB) once confident in the
  compacted DB.
- **Lean-feature production rerun**: prune `returns_and_vol` to own-ticker + a
  couple horizons, keep spreads + macro, re-run `lgbm_vol`, and compare against
  `results/phase1_baseline.md`. Handle palladium separately (different drivers).
- Minor: the regenerated report's reading guide has an em-dash mojibake from the
  Windows run encoding (cosmetic).

### Next session

- Lean-feature baseline rerun, or pivot to Phase 3 (text + clustering).

---

## 2026-06-18 (lean-feature baseline)

### What I did

- Added a `--feature-set {full,lean,lean_own}` switch to `lgbm_vol` (plus a
  `feature_columns` helper and 3 unit tests). `lean` drops the returns-and-vol
  block entirely (spreads + macro, 34 feats); `lean_own` keeps only the *target*
  metal's own returns/vol (43 feats).
- Re-ran all three sets x 4 metals on current data (BAA10Y live), 22 splits.
  `full`/`lean` reproduce the diagnostic to the digit. Updated
  `results/phase1_baseline.md` with the comparison table, findings, recommendation.

### What I learned

- **`lean` is the IC winner for Au/Ag/Pt**: Au +0.022 -> +0.098, Ag -0.068 ->
  +0.079, Pt -0.054 -> +0.047 (silver and platinum flip negative -> positive).
- **`lean_own` does not help — and usually hurts.** Adding the target metal's own
  returns/vol takes silver (+0.079 -> -0.069) and platinum (+0.047 -> -0.056) back
  negative; gold is flat. Even own-series vol clustering is net noise for this
  target/horizon once spreads + macro are present. Surprising, but consistent.
- **Palladium is the exception**: `lean` makes it *worse* (-0.016 -> -0.043); its
  drivers aren't in this feature set (supply/industrial, per Phase 2).
- Current-data `full` gold IC fell vs the original May-25 baseline (+0.070 ->
  +0.022) after the FRED refresh + newer data. The lean-vs-full comparison is on
  identical current data, so that shift doesn't affect the conclusion.

### Open items not resolved today

- Operationalize `lean` as the default for Au/Ag/Pt (e.g. per-metal feature_set in
  the run config); give palladium a bespoke feature study. `lean_own` can retire.
- (carryover) Phase 3 (text + clustering) remains the next roadmap phase.

### Next session

- Either wire `lean` in as the Au/Ag/Pt default + start a palladium-specific
  feature study, or pivot to Phase 3.

---

## 2026-06-18 (Phase 2 review)

### What I did

- Reviewed Phase 2 (events + local projections): read `lp.py`, `cot.py`,
  `events.py`, `fomc_surprises.py`, their tests, and notebooks 03/05.
- **Independently reproduced** the gold hawkish-FOMC IRF from raw data:
  -1.50% (h=5), -1.78% (h=20) — matches `phase2_scenarios.md` to the bp. Extended
  to all four metals: the full hawkish table reproduces.
- **New robustness result**: the hawkish IRF is invariant to control specification
  (contemporaneous macro controls vs macro-lagged-1d vs no controls), within
  ~0.3% across all four metals. The contemporaneous macro controls are NOT
  absorbing the effect. Added as section 4 of `notebooks/05_fomc_robustness.ipynb`
  (saves `results/phase2/fomc_hawkish_control_robustness_h5.png`) and documented in
  `results/phase2_review.md`.
- **Probed and dismissed two suspected weaknesses**: (1) unscheduled-meeting
  contamination of the dovish tercile — only one unscheduled event in-window
  (2019-10-11), and March 2020 is absent from Bauer–Swanson, so the dovish/hawkish
  asymmetry is genuine; (2) control mediation — see robustness above.
- **Fixed the COT release-date offset**: `cot.py` now uses a holiday-aware
  `release_date()` (delays the nominal Friday by in-week federal holidays, snaps to
  next business day) instead of a fixed `+3 days`. Two tests added
  (Thanksgiving / July-4 weeks -> Monday). COT is still not consumed in the LPs, so
  this is preventive.
- Wrote `results/phase2_review.md` as the durable review record.

### What I learned

- Phase 2 is the strongest part of the project: the estimator is correct and
  tested, and the headline result both reproduces and is robust to control
  specification (a check it had not previously been through).
- The event-day-return convention (cumulate from t+1, excluding the FOMC-day
  close-to-close move) is *supported* by the Phase 1 audit's ~0.89 futures/ETF
  close-to-close correlation, i.e. a COMEX ~1:30 PM settlement that precedes the
  2 PM FOMC, so the reaction lands in r_{t+1}.

### Open items not resolved today

- In-sample tercile thresholds (data snooping) -> holdout-based cut in Phase 6.
- Refresh Bauer–Swanson XLSX (~10 FOMC meetings missing since 2023-12).
- HAC `maxlags=h` is marginally optimistic at long horizons (minor).
- Add a one-line event-day-convention note to `phase2_scenarios.md`.

### Next session

- Phase 5 can take hawkish-FOMC as the first triangulation target with confidence,
  or backfill CPI/NFP scenarios.

---

## 2026-06-18 (Phase 3 review + backfill plan)

### What I did

- Reviewed Phase 3 status. **Built** (steps 3.1–3.6 scaffolding): GDELT GKG
  fetcher, embeddings wrapper (+ tests), themes config. **Not built**: the entire
  clustering core (3.7–3.15) — daily text aggregation, BERTopic, contextual
  vector, UMAP, HDBSCAN, cluster taxonomy — plus `kitco.py` (3.4) and
  `text_prep.py` (3.5). The embeddings cache and `data/raw/` are empty (the embed
  step has never run on the real corpus).
- **Found the blocker**: the `headlines` corpus has a 28-month hole. Continuous
  coverage is only 2020-01..2021-08 (~14.1M rows); 2021-09..2023-12 is empty;
  2024-01 is a 32k-row fragment; nothing after. 116 of ~138 months in 2015–2026
  are missing.
- Wrote `scripts/backfill_gdelt.py` (gap report / free dry-run cost estimate /
  capped `--execute`, reusing gdelt's pure functions) + 5 tests; validated
  `--gaps` against the canonical DB. Wrote `results/phase3_backfill_plan.md`
  (date-range decision, gap inventory, cost methodology, guards, runbook).

### What I learned

- GKG provides URLs, not headline text — embedding raw URLs is weak signal;
  `text_prep.py` (3.5) should extract the URL slug first.
- Only `ECON_GOLDPRICE` (~4.4k rows) is metal-specific; the corpus is dominated by
  generic mining + macro themes, so per-(date, metal) news vectors will be nearly
  identical across metals.
- GKG 2.0 starts 2015-02-18; the 2011 peak / 2013 taper regimes are unreachable in
  this schema.

### Open items not resolved today

- Run the backfill on the user's box (GCP creds): `--estimate` then `--execute`.
  Decide full-history (2015→) vs minimum-viable (2021-09→) after the dry-run bytes.
- Then build `text_prep.py` (URL-slug) + step 3.7 daily aggregation *before*
  clustering. Do not cluster on the 20-month COVID-only window.

### Next session

- Backfill GDELT to continuous coverage, then build the daily text-feature
  aggregation.

---

## 2026-06-18 (Phase 3 backfill kickoff + text_prep)

### What I did

- **GCP/BigQuery set up** (new project `amc-metals`, service account `gdelt-reader`,
  key wired into `.env`; the old `metals-research` key path was stale).
- **BigQuery cost (recorded per plan 3.3).** Dry-run estimate of the post-2021 gap
  (2021-09 .. present, 57 months): **0.65 TB scanned -> $0.00**, under the 1 TB/month
  free tier. On-demand is **$6.25/TB** (US) beyond the free tier (corrected from the
  $5 I'd first put in the tooling). Per-month chunks scan ~9-15 GB, well under the
  100 GB `--max-gb` cap. The full 2015->present fill would add ~0.5 TB.
- **Test pull succeeded**: 2021-09 pulled 695,316 rows and upserted (consistent with
  the ~700k/month rate). Full post-2021 `--execute` (~56 months, ~30M rows) still to
  run; expect the DuckDB to grow back to ~13-14 GiB.
- **Built `src/metals/data/text_prep.py` (step 3.5)**: recovers readable text from
  GDELT article URLs (pick richest path segment, strip extensions / numeric ids /
  query strings, split camelCase, drop id-like tokens). On real sampled URLs it
  yields clean headlines, e.g. "sensex nifty end higher as reliance shares gain".
  11 tests. This is the fix for "GKG stores URLs, not headline text" — the embedding
  step should run on this, not raw URLs.

### What I learned

- The whole post-2021 fill is free (0.65 TB < 1 TB/mo). The `--estimate` dry-run uses
  the same `build_query` as the real pull, so the estimate equals the real scan.
- URL-slug recovery is strong for the rich-slug majority; section-only paths
  (`/topic/123`, `/article/456`) degrade to a single section word and numeric-only
  paths yield empty -- both acceptable (weak signal, averaged out in daily
  aggregation). Many slugs are non-English (GDELT is global): recovered as words,
  not translated.

### Open items not resolved today

- Run the full post-2021 `--execute` (then optionally 2015-2019 in a later calendar
  month to keep it free). After the corpus lands: step 3.7 daily aggregation built on
  `text_prep` + `embeddings`.
- Consider a language and/or per-metal relevance filter before clustering.

### Next session

- Full backfill, then the daily text-feature aggregation (3.7).

---

## 2026-06-19 (Phase 3 build-out)

### What I did

- **Migration 005** added persistence tables: `daily_text_features` (per-(date, metal) mean embedding, dispersion, tone), `daily_topic_prevalence` (long-format topic vectors), `cluster_assignments` and `cluster_centroids` (versioned by `model_version`).
- **`metals.features.text_daily`**: theme→metal mapping, headline aggregator, embedding centroid + dispersion, BLOB-pack of mean embedding, idempotent upsert + loader.
- **`metals.features.topics`**: lazy BERTopic wrapper with sentence-transformers-free embeddings path (pass pre-computed embeddings); persistence to `data/processed/topic_models/`; per-day prevalence pivot.
- **`metals.features.context`**: daily contextual feature builder — macro + own-metal returns/vol + PCA-reduced text mean embedding + topic prevalences + COT z-scores. Returns `(context_df, artifacts)` so the PCA can be re-applied at inference.
- **`metals.models.clustering`**: standardization + UMAP + HDBSCAN with deterministic seeds, `ClusterPipeline` dataclass for fit artifacts, save/load + sidecar JSON, idempotent upsert of assignments and centroids.
- **`metals.eval.clusters`**: `forward_returns`, `cluster_forward_stats` (per (cluster, metal, horizon): n, mean, std, hit rate), `dominant_topics`, `representative_dates` (HDBSCAN-confidence-sorted), `example_headlines`, and a `cluster_summary` bundler.
- **66 new tests** across five files. All pass; 2 skip when UMAP/HDBSCAN/BERTopic aren't installed (importorskip-gated). Total now 176 passing.
- **`scripts/phase3_pipeline.py`**: end-to-end orchestration with stages `gdelt → embed → aggregate → topics → context → cluster → analyze` and `--only` / `--resume-from` flags for chunked execution. Auto-generates `model_version` if not supplied.

### What I learned

- BERTopic accepts pre-computed embeddings, which means the expensive sentence-transformers pass happens once (cached) and BERTopic refitting is fast.
- HDBSCAN's `approximate_predict` requires `prediction_data=True` at fit time. Caught early.
- UMAP transforms (not just fits) need standardized inputs — applying the *training* mean/std to new data is the right invariant. Codified in `_standardize(X, mean=..., std=...)`.
- DuckDB's BLOB type is the cleanest way to persist NumPy embeddings without committing to a separate Parquet sidecar. Round-trips through `np.frombuffer`.

### What confused me

- Nothing major. The OneDrive truncation issue stayed quiet this session — files written via the Write tool all arrived intact.

### Open items not resolved today

- The end-to-end pipeline has not been executed against real GDELT data. The user's box (with GCP creds + GPU) is the natural place to run it.
- LLM-assisted cluster labeling is wired schematically (`cluster_centroids.label`, `label_source`) but the labeling helper module is not yet written. Deferred to next session.
- Kitco RSS supplement (Phase 3 step 3.4) is not built — GDELT only for now.
- Sanity-check vs known regimes (Phase 3 step 3.13) is a post-run verification; nothing to build, just a checklist to apply once clusters exist.

### Next session

- Either: run `scripts/phase3_pipeline.py` on user's machine, then drill into the cluster taxonomy and label it (LLM or hand). The labeling helper module would be ~$1 of API and 100 lines of code.
- Or: Phase 5 prep — `metals.models.causal` (DoubleML) and `metals.models.svar` (sign-restricted VAR). Can be built against the existing Phase 2 IRFs without waiting for Phase 3 to land.

---

## 2026-06-23 (Phase 3 — LLM cluster labelling)

### What I did

- **`metals.eval.cluster_labeling`** — pure-function prompt builder and JSON parser plus an `Anthropic`-SDK-backed `label_cluster` with retry/backoff. The `caller` argument is a thunk, so tests mock it without network calls.
- **`build_cluster_context`** assembles per-cluster context: representative dates (sorted by HDBSCAN confidence), example headlines from each, dominant topics, and mean forward returns per metal/horizon. Optional inputs tolerated — context still builds when headlines/topics/forward stats are absent.
- **`upsert_labels`** writes back to `cluster_centroids` with `label_source = "llm:<confidence>"` so downstream analysis can filter by labelling provenance.
- **14 new tests** covering: prompt construction, JSON parse edge cases (plain, markdown-fenced, embedded in prose, garbage rejection), confidence normalisation, retry logic, end-to-end with mocked caller, persistence round-trip. All pass.
- **`scripts/phase3_pipeline.py` gained a `label` stage** — runs after `analyze`, gated on `ANTHROPIC_API_KEY` (silently skips otherwise), uses Haiku 4.5 by default. New `--llm-model` flag for overriding.
- **190 total tests passing**, 2 skipped (UMAP/HDBSCAN/BERTopic importorskip).

### What I learned

- BERTopic ships its own UMAP/HDBSCAN — fine to reuse the libraries directly in our clustering pipeline without conflict. They share state-of-init configurations between fits.
- Anthropic's content blocks are a list; concatenating `.text` across them is the safe extraction pattern.
- The JSON-parse fallback (extracting the first `{...}` block on raw failure) is worth the ~5 lines — LLMs occasionally add "Sure, here's the label:" preamble despite the system prompt asking them not to.

### What confused me

- Nothing major. The orchestration patch landed cleanly via `python -c "..."` — that scripting pattern has been bulletproof against the OneDrive truncation issue.

### Open items not resolved today

- Pipeline still not executed against real GDELT. That's a you-side action — GCP creds + (preferably) local GPU.
- Kitco RSS supplement (Phase 3 step 3.4) deferred.
- Sanity check vs known regimes (Phase 3 step 3.13) is a post-run task, no code needed.

### Next session

- Either: run `scripts/phase3_pipeline.py` end-to-end (you-side) and we walk through the labels together. Likely cost: $0–10 for embeddings (free with local GPU), ~$1 for labels on Haiku, $0 for BQ if chunked across two billing cycles.
- Or: Phase 5 prep against Phase 2 outputs — `metals.models.causal` (DoubleML) and `metals.models.svar` (sign-restricted VAR). Unblocked now.

---

## 2026-06-23 (Phase 3 — embedding cache rewrite)

### What I did

- **Rewrote `metals.features.embeddings`** with a sharded-Parquet on-disk cache. 4,096 shards (first-3-hex-of-sha256 keys), Parquet-per-shard, in-memory LRU of the 32 most-recently-touched shards, atomic write via tmp + os.replace.
- **Switched default model** to `all-MiniLM-L6-v2` (384-dim) from mpnet. Roughly 5× embedding throughput on the same GPU, ~37 GB cache footprint for the 48.5M-headline backfill (down from ~150 GB at mpnet/fp32).
- **fp16 on-disk, fp32 to callers**. Storage halved again, downstream consumers see the precision they expect.
- **Cache location moved OFF OneDrive.** New default: `%LOCALAPPDATA%\metals\embeddings` on Windows, `~/.cache/metals/embeddings` elsewhere — both outside any sync engine's purview. `METALS_EMBEDDING_CACHE_DIR` env var overrides. The resolver warns at runtime if the resolved path contains tokens like `OneDrive`, `Dropbox`, `GoogleDrive`, `iCloud`, `Box`.
- **Public API preserved**: `embed_texts(...)`, `embed_dataframe(...)`, `EmbedConfig`, `DEFAULT_MODEL`, `CACHE_ROOT`. All existing call sites work unchanged.
- **29 new tests**: default model, dtype, fingerprint variance, cache-dir resolution priority + OneDrive warning, hash math + shard distribution uniformity, Parquet round-trip (fp32 exact and fp16 within tolerance), same-shard write merging, atomic-write hygiene, public API smoke with a mocked SentenceTransformer.
- **212 total tests passing**, 2 skipped (UMAP/HDBSCAN/BERTopic gated). Zero regressions.

### What I learned

- Parquet's `pa.list_(pa.float16())` is the cleanest way to store variable-dim embeddings without committing to a fixed-list schema (which would lock the cache to a single dim per fingerprint anyway). The fp16 conversion happens at write time and round-trips cleanly to fp32 on read.
- The OneDrive token-sniffing heuristic is crude but catches the obvious cases. Path components like `OneDrive`, `Dropbox`, etc. are case-insensitively matched against a frozenset.
- Shard-prefix uniformity: 2000 hashes across 4096 shards → max bucket size of ~10. Healthy.

### What confused me

- The OneDrive write-truncation issue struck again — the file tool emitted 168 lines instead of 320. The bash-heredoc workaround landed the file intact. Pattern is now well-established: anything > ~200 lines goes via bash heredoc + `wc -l` verification.

### Open items not resolved today

- Pipeline still not executed end-to-end against real data. Now unblocked: with the new defaults, the embed stage should run in ~3–5 hours on local GPU at ~37 GB cache, safely outside OneDrive.
- `data/processed/embeddings/` in the repo is now empty / abandoned; could add a `MOVED.md` stub but it's harmless to leave alone.

### Next session

- You run `--only embed` against the 48.5M-headline corpus. Estimated runtime 3–5 h on a 4090, ~37 GB cache landing in `%LOCALAPPDATA%\metals\embeddings`. Then aggregate → topics → cluster → analyze → label in one evening.

---

## 2026-06-25 (Phase 3 — server verification + streaming/themes redesign)

### Context
Now on a Linux server (4 cores, 32 GB RAM, RTX A6000 48 GB, CUDA 13). The repo
was built on Windows + a 4090 + OneDrive. Task: verify training won't break
here, then fix the Phase 3 pipeline.

### What I did
- **Verified the environment** (multi-agent workflow). Env builds on Py 3.11,
  torch 2.12.0+cu130 sees the A6000, GPU matmul + MiniLM encode work, all heavy
  deps import, OS-portability clean (`db_path`/`resolve_cache_dir` resolve to
  Linux paths), and Phase 1 LightGBM trains end-to-end (harness runs 34-35).
- **Migrations**: `005_phase3_artifacts` had never been applied (two files share
  the `005` prefix; the runner keys idempotency on the full filename stem, so
  both are tracked independently — phase3_artifacts had simply not run). Applied
  it; the four Phase 3 tables now exist.
- **Found three escalating Phase 3 break points and fixed them:**
  1. *Immediate crash*: `run_embed/aggregate/topics` all `SELECT
     document_identifier`, a column migration 005 dropped. The live URL column
     is `article_url`. Fixed 7 refs; added `scripts/phase3_smoke.py` to
     bind-check stage queries against the live schema (would have caught this
     before a multi-hour run).
  2. *OOM*: `embed_texts` materialized the whole 63.3 M-row corpus
     (`np.vstack` ~97 GB; confirmed adversarially). Added streaming
     `cache_embeddings()` (no vstack) and rewrote `run_embed`/`run_aggregate`
     to process one calendar month at a time. Months align to day boundaries,
     so per-chunk daily aggregates concatenate losslessly (locked by a test).
  3. *BERTopic intractable* over 63 M docs on 4 cores. Replaced with
     **themes-via-SQL** (`topics.compute_theme_prevalence`): a streaming DuckDB
     GROUP BY over the curated GDELT theme set, writing the same
     `daily_topic_prevalence` table via a stable `theme->topic_id` map. **Ran
     over the full 63.3 M corpus in 8 s** (30,079 prevalences / 2,315 days /
     14 themes). BERTopic kept as optional `--topics-method bertopic` (sample-
     bounded so it can't OOM).
- **Perf fix**: the hash-sharded embedding cache is slow for bulk sequential
  reads — 51.7 s for 24k cache *hits* (~4,000 tiny Parquet opens) vs ~9 s to
  re-encode. So `run_aggregate` now encodes on the fly (`use_cache=False`):
  faster AND drops the 48 GB disk requirement. The `embed` stage is now optional
  (pre-warm for bertopic/Phase 4 only).
- 227 tests pass (+13 new). ruff clean on touched code.

### What I learned
- Dispersion has a closed form for L2-normalized embeddings: `1 - ||mean(e_i)||`,
  so it needs only a running sum — fully streamable (aggregate_daily already
  computes the equivalent).
- GDELT corpus is **2020-01-01 -> 2026-06-19** only; pre-2020 regime checks
  (2011 peak, 2013 taper) are out of range for *any* text method.
- Theme prevalence tracks regimes cleanly (ECON_INFLATION ~0.30 in mid-2022).

### Open / next session
- Full `aggregate` run is ~9 h single GPU pass (~6.4 h encode @ ~2,753 texts/s +
  ~2.9 h aggregate_daily). Then context -> cluster -> analyze.
- `aggregate_daily` uses per-row `iterrows` (~0.17 ms/row); fine, vectorizable.
- Rename `005_phase3_artifacts` -> `006` to kill the duplicate-prefix fragility.
- DB mutations this session: applied 005_phase3_artifacts; harness runs 34-35;
  `daily_topic_prevalence` fully populated (themes); `daily_text_features` holds
  1 day (2026-05-01) from a smoke (idempotent — a full aggregate overwrites it).

---

## 2026-06-26 (Phase 3 — text-quality audit → GKG headline discovery)

### Context
Audited how good the URL-slug recovery (`text_prep.url_to_text`) actually is as a
proxy for article headlines, since the whole text channel rides on it. The audit
cascaded into a much bigger finding: **GDELT has had the real article titles all
along, in a GKG column we never ingest.** No code shipped to `src/` yet — this is
a decision/handoff entry. The plan below is ready to implement.

### What I did
- **Audited slug recovery, whole-corpus.** Wrote `scripts/phase3_slug_quality.py`
  (read-only, repeatable `--seed`, `--n`). Uniform 8k-row sample of all 63.3M
  rows: only **61% recover to headline-like text (6+ tokens)**; **~30% are
  degenerate (≤2 tokens** — section words `news`/`story`/`item`, or empty); only
  **27% English-ish**. The earlier curated English-news spot-check (near-perfect)
  was survivorship-biased. Systematic loss: `clean_slug_text` drops numeric
  tokens, so prices/percentages/years/quarters are gone from every recovered slug.
- **Found the theme filter is a no-op.** `text_daily.metals_for_themes` keeps
  **100%** of rows — the corpus was *ingested* on these exact themes, and
  `WB_1699_METAL_ORE_MINING` alone tags 57% of GKG, so the union is everything.
  The daily aggregate genuinely averages over the full, noisy, ~70%-non-English mix.
- **Tested whether degeneracy is missing-at-random — it is NOT.** Degenerate URLs
  are overwhelmingly Asian/non-Latin finance portals with numeric-ID paths (`.cn`
  13.5% of degenerate vs 0.0% of rich; sina/sohu/163/eastmoney; 67% numeric-id
  paths). Degeneracy correlates with three model regressors: **time** (degen share
  35.8%→26.1%, 2020→2026), **tone** (degen mean −0.106 vs rich −0.806, Cohen
  d=+0.19), and **theme mix** (METAL_ORE +17pt, INFLATION −10pt). So a naive
  min-token filter (drop degenerate rows) would bias tone/theme/count features in
  a time-varying way — a selection-bias trap.
- **Assessed "use Haiku to recover headlines" — rejected.** There is no headline
  text in the DB (the `headline` column was a byte-copy of the URL, dropped in
  migration 005). For the 67% numeric-ID degenerate URLs there are *no words in
  the input*, so Haiku would hallucinate, not recover. Cost to run Haiku over 63M
  URLs ≈ $6–13k for near-zero gain. LLMs belong at the cluster-label scale (already
  built), not per-row.
- **Probed two un-ingested GKG columns — the payoff.** Wrote
  `scripts/phase3_gkg_enrichment_probe.py` (reuses `gdelt.build_query`'s exact
  theme+date predicate; always free-dry-runs first). Pulled `Extras` and
  `TranslationInfo` for a 3-day themed sample (67k rows, 1.4 GB scan, free):
  - **`Extras` contains `<PAGE_TITLE>` on 99.6% of rows** — the real scraped
    article title. **99.1% of the degenerate-slug rows are rescued** by it.
  - Titles are HTML-entity-encoded; `html.unescape` decodes cleanly to native
    Unicode (verified: Serbian/Latvian/Finnish/Persian).
  - **`TranslationInfo` gives a free per-row source-language code** (empty =
    English-original). Confirms corpus is **29.6% English**, ~70% native-language;
    PAGE_TITLE is in the *original* language.

### What I learned
- **The real headlines were one un-pulled GKG column away the whole time.**
  `gdelt.build_query` selects only 5 of ~27 GKG columns; `Extras` (the XML blob)
  carries `<PAGE_TITLE>` near-universally. This **supersedes slug recovery** and
  **dissolves** the degeneracy + missing-at-random + feature-gating problems: the
  embedding becomes defined for ~100% of rows, so there's no non-random
  "text-bearing" subset left to bias anything.
- With real native-language titles in hand, a **multilingual encoder is now clearly
  worth it** (genuine foreign headlines to embed, not garbage slugs) — and
  `TranslationInfo` is the clean language label for routing/filtering.
- Other un-ingested GKG fields worth a later look: `GCAM` (2,300 content dims vs
  our 6 tone components), `Quotations` (real extracted sentences), `Amounts`
  (recovers the numbers slug recovery drops).

### Environment / housekeeping this session
- **Fixed `.env`**: `GOOGLE_APPLICATION_CREDENTIALS` was a stale Windows path
  (`~\.gcp\gdelt-reader.json`); rewrote to the Linux absolute path
  `/home/mcmur/.gcp/gdelt-reader.json`. BigQuery dry-run + execute both work now.
- **No DuckDB mutations** this session. The probe is read-only BQ; nothing written
  to `headlines` or any table. New files: `scripts/phase3_slug_quality.py`,
  `scripts/phase3_gkg_enrichment_probe.py` (both standalone, no `src/` changes).

### Next session — implement the PAGE_TITLE plan (ready to go)
1. **Add two columns to the GKG ingest** (`src/metals/data/gdelt.py`):
   - `build_query`: also select `Extras`, `TranslationInfo`.
   - `parse_gkg_rows`: derive `page_title = html.unescape(<PAGE_TITLE> regex from
     Extras)` and `src_lang` (from `TranslationInfo`; empty → `eng`). The probe
     script already has the exact regexes (`_PAGE_TITLE_RE`, `_SRCLC_RE`).
   - New migration `006_*` adding `page_title VARCHAR`, `src_lang VARCHAR` to
     `headlines`; extend the upsert column lists in `gdelt.upsert_headlines`.
     (Also: rename the duplicate-prefix `005_phase3_artifacts` → `006` while here,
     per the 2026-06-25 note — pick distinct numbers.)
2. **Backfill** `Extras`+`TranslationInfo` over 2020-01→2026-06 to populate the new
   columns. Cost ≈ 1.4 GB / 3 days → **~1.1 TB total scan ≈ $0 if chunked across
   two billing months** (~$7 at full on-demand). Can reuse `backfill_gdelt.py`'s
   chunking; needs a column-update path (rows already exist — update, don't insert).
3. **Switch the embedding input** in `text_daily`/the pipeline from
   `text_prep.url_to_text(url)` → `html.unescape(page_title)`, with slug recovery
   kept only as the <0.4% fallback. The degeneracy/gating/MNAR work then becomes a
   tiny safety net for the residual empties, not a bias mitigation.
4. **Adopt the multilingual encoder** in `features/embeddings.py`:
   `paraphrase-multilingual-MiniLM-L12-v2` (384-dim — drop-in, no BLOB/PCA/cache
   schema change; ~2× slower encode). Re-warm/rebuild the embedding cache.
5. Keep `phase3_slug_quality.py` + `url_to_text` as fallback/diagnostic. Re-run the
   slug audit's `[D]` cross-tab against the new `page_title` column to confirm
   coverage on the full corpus before deleting any slug path.

### Watch-outs for the next agent
- `Extras` is a fat XML column — the dry-run scan is wider than the 5-narrow-column
  monthly pull. Always `--dry-run` (the probe does this by default) and check GB
  before a full `--execute`. Stay under 1 TB/month to keep it free.
- PAGE_TITLE is native-language + entity-encoded → `html.unescape` is mandatory
  before embedding; don't embed raw `&#x...;` strings.
- `TranslationInfo` is populated only for GDELT-translated (non-English-source)
  docs, so empty ≠ missing — empty means English-original. Map empty → `eng`.
- This is a schema change + full re-backfill (bigger commit). The probe proved the
  premise (99.6% coverage); the user OK'd the direction but hadn't green-lit the
  backfill itself — confirm scope before kicking off the multi-hour pull.

---

## 2026-06-26 (later — Phase 3 correctness foundation + ruff adoption)

### Context
New agent onboarding: read the whole journal + roadmap + GDELT assessment, then
mapped every subsystem against the actual source (5-way parallel read) to ground
the model and catch doc-vs-code drift. Agreed plan with the user: **correctness
before compute** — land the in-flight collapse, fix the one real leak, ship a
defensible clustering baseline on the signals we trust (themes + tone), and gate
the expensive PAGE_TITLE/embedding work on a measured CV win. User decided:
**extend coverage to 2015** and do the **migration-006 schema work first** so the
2015–2019 backfill can be pulled *wide* (titles for free, no re-scan).

### What I did
- **Committed the in-flight work** in clean units: the per-metal text-axis
  **collapse** to a shared `market` row (code + tests + assessment §7), and the
  untracked **Docker/CUDA env + CLAUDE.md + the two audit scripts**.
- **Fixed a real look-ahead leak in `context.build_context`.** The text-embedding
  whitening PCA was `fit_transform`-ed on the *entire* date range, so `text_pca_*`
  leaked future covariance into past coordinates — and (unlike Phase-1
  `assemble`) the function ran no leakage guard. Added a `pca_fit_until` boundary
  (`_pca_fit_transform`: fit on rows ≤ boundary, transform the full series),
  threaded `--train-until` into `run_context`, and added an `assert_chronological`
  guard. Regression test proves changing post-boundary embeddings cannot move
  train-window `text_pca` coords; the full-sample fit is asserted to leak as a
  control.
- **Tidied the pipeline** in the same pass: run `analyze` before `label` (matched
  `STAGES`; they were inverted in `main()`); bounded `run_label`'s headline pull
  to the assignment date range with a per-day cap (was materializing the whole
  ~63 M-row corpus); dropped the always-`None` `date_range` print; deduped the
  `model_version` fallback. Corrected `leakage.py`'s docstring (it advertised a
  nonexistent `check_no_lookahead`).
- **Adopted ruff as the project standard** (user call). The repo was neither
  `ruff format`-clean nor `ruff check`-clean tree-wide (93 check errors at HEAD).
  Three isolated commits: `ruff format` repo-wide (mechanical), then a check-clean
  pass (safe autofixes; `strict=False` on 8 bare `zip`s; `raise … from None`;
  wrapped 4 long lines; **ignore N803/N806** since capitalised math-notation vars
  `X`/`E`/`Z`/`X_std`/… are idiomatic here), and a `.git-blame-ignore-revs` for the
  two bulk commits. CLAUDE.md now states both are enforced.
- **229 tests pass** (+1 leak regression); `ruff check` + `ruff format --check`
  both clean.

### What I learned
- The leak was structural, not a typo: the PCA fit lived upstream of the
  clustering train/test split, so the split couldn't protect it. Fixing it needed
  the train boundary pushed *into* `build_context`, not just into `run_cluster`.
- `context.build_context` carries **no tone** today — only `n_articles`,
  `embedding_dispersion`, and `text_pca_*`. The signals the assessment trusts most
  (V2Tone, 100% coverage) aren't in the clustering vector yet. The Option-C
  baseline should *add* tone while dropping the weak URL embeddings, not just
  remove things.
- `ruff format` is safe to bulk-adopt (it reduced check errors 93→75, introduced
  none); the only judgment call was N803/N806, correctly resolved by ignoring them
  for math code rather than renaming.

### Open / next session
1. **Migration 006/007 + wide GKG ingest** (the data no-regret, do before the pull):
   add `page_title` + `src_lang` columns; `gdelt.build_query` to also select
   `Extras`/`TranslationInfo`; `parse_gkg_rows` to derive them (regexes already in
   `phase3_gkg_enrichment_probe.py`); extend `upsert_headlines`; rename
   `005_phase3_artifacts` → `006` to kill the duplicate prefix; new `007` for the
   columns. Tests for the parse. Then backfill **2015–2019 wide** (titles free for
   that range); a later UPDATE-backfill adds titles to existing 2020–2026 rows.
2. **Option C clustering**: add a `ContextConfig.include_embeddings` flag, *add
   tone* to the context vector, drop `text_pca`/dispersion when off. Run
   `aggregate` (embeddings-free, fast) → `context` → `cluster` → `analyze` and
   sanity-check the 2020/2022/2023 regimes. Cheap, leak-free, defensible.
3. **Then** gate the multilingual PAGE_TITLE re-embed on the §7 CV bar.
- DB untouched this session (code + tests + git only; no pulls, no GPU).

---

## 2026-07-02 (Phase 3 — migration 006/007 + wide GKG ingest)

### Context
Step 1 of the agreed "correctness before compute" sequencing: land the schema +
ingest code so the 2015–2019 backfill can be pulled *wide* (titles for free).
Session ran on **AYMStation (WSL2 laptop, RTX A1000)** — NOT the A6000 server.
Important discovery: the two machines' DuckDB files had **diverged** (git syncs
code, not data): this laptop's DB was missing `005_phase3_artifacts` and the
four Phase 3 tables until today, while the server has them populated.

### What I did
- **Renamed `005_phase3_artifacts.sql` → `006_phase3_artifacts.sql`** (kills the
  duplicate-005 prefix). Safe on every live DB state because the file is fully
  `IF NOT EXISTS`: the server (applied under the old stem) re-runs it as a
  no-op and keeps a stale tracking row; fresh/laptop DBs apply it normally.
  Locked by `tests/test_migrations_runner.py` simulating all three states.
- **New `007_headlines_page_title.sql`**: `page_title VARCHAR`, `src_lang
  VARCHAR` on `headlines` (`ADD COLUMN IF NOT EXISTS` works on duckdb 1.5.3
  with the index present).
- **Widened the ingest** (`metals/data/gdelt.py`): `build_query` also selects
  `Extras` + `TranslationInfo`; new pure extractors `extract_page_title`
  (html.unescape once, collapse whitespace, 512-char cap, absent → None) and
  `extract_src_lang` (empty → `'eng'`, `srclc:xx` → code, malformed → None);
  `parse_gkg_rows` derives both and tolerates narrow (pre-007) frames by
  landing NULLs; `upsert_headlines` upserts them with **COALESCE on conflict**
  so a narrow re-pull can never clobber landed titles. Semantics rule:
  **NULL `src_lang` = "not pulled wide", never English.**
  `phase3_gkg_enrichment_probe.py` now imports the extractors (no regex drift).
- **BigQuery dry-run estimates** (per convention, bytes + cost): backfill
  2015-02-18→2019-12-31 **wide = 1.352 TB** ($8.45 on-demand full price; $0 if
  split across 2 billing months; ~$2.20 in one month). Narrow same range =
  1.125 TB → the wide increment is only **0.227 TB (~20%, $1.42)** — pulling
  titles with the backfill is confirmed near-free vs a later 1.15 TB re-scan.
  Title UPDATE-backfill 2020-01→2026-06-19 = **1.154 TB** ($7.21 / $0 across
  2 months). Heaviest year: 2016 at 0.347 TB wide. NB the assessment's ~0.5 TB
  backfill estimate was ~2.4× optimistic.
- **Live landing verification** (one-day pull 2024-06-01, 0.41 GiB, through
  the real build_query→BQ→parse→upsert path into a throwaway DB): schema match
  exact; 18,633 rows; **99.53% title coverage; 0 undecoded entities; 27.1%
  English** (assessment said ~30%); **100.000% headline_id match** against the
  live 63.3M-row table (ON CONFLICT hits existing PKs — the future title
  backfill can use the plain upsert); COALESCE guard held under a simulated
  narrow re-pull; 1 intra-batch PK collision (pre-existing last-write-wins
  dedup semantics, headline_id = ts + url[:200]).
- **Adversarial review (3 lenses + verification agents)** confirmed one major:
  with migration 007 absent from a checkout (e.g. forgotten `git add`),
  `compact_headlines.py` would rebuild the canonical schema without the new
  columns and its column-intersection copy would **silently discard populated
  titles while row-count verification passes**. Fixed two layers: all files
  explicitly tracked in the commit, and `compact_headlines.py` now **refuses
  to drop source columns** missing from the canonical schema unless
  `--allow-column-drop` is passed (guard verified live both ways). Also
  future-proofed the runner tests against a future 008 migration.
- CLAUDE.md sharp edges rewritten (005-conflict resolved → stem-tracking note,
  page_title/src_lang facts, NULL-src_lang rule). 241 tests pass (+12), ruff
  check/format clean, mypy at HEAD parity (41 pre-existing stub errors).

### What I learned
- `compact_headlines`-style "rebuild schema from the migration glob, copy
  intersecting columns" passes row-count verification while losing columns —
  a failure class invisible to working-tree tests (the untracked file exists
  locally). Guard in the tool, not just the process.
- Wide-vs-narrow GKG scan is only ~+20% bytes: `V2Themes` (in the WHERE) is
  the fat column, so `Extras` rides along cheaply.
- The title UPDATE-backfill may not need a dedicated column-update path at
  all: the plain wide `refresh()` upsert updates titles on conflict, and its
  extra scan cost over a minimal 4-column query is just V2Tone +
  SourceCommonName. Measure both dry-runs before building anything.

### DB mutations this session
- **Laptop DB only**: applied `006_phase3_artifacts` + `007_headlines_page_title`
  (Phase 3 tables now exist here, empty; `headlines` gained two NULL columns).
- **Server DB untouched** — run `uv run python -m metals.data.migrations.runner`
  there BEFORE any ingest: `upsert_headlines` now requires 007 and fails with a
  Binder error on a pre-007 DB.
- Real `headlines` data untouched on both machines. The verification pull
  landed only in a deleted throwaway DB — deliberately, so `backfill_gdelt.py`
  month-gap detection stays clean (a stray 2015 sliver would mask a whole
  month from the backfill).
- BigQuery spend: ~0.8 GiB scanned (2× one-day pulls; dry-runs are free) —
  negligible against the 1 TB/month free tier.

### Next session
1. **Decide + run the 2015–2019 wide backfill** (user green-light + machine
   choice pending): target the *server* DB presumably (it's the compute box);
   split across the Jul/Aug billing boundary for $0 or accept ~$2.20 in one
   go. `backfill_gdelt.py` already pulls wide via the shared `build_query`;
   its default `--max-gb 100` per chunk is comfortably above the ~30 GB/month
   wide scan. Remember: migrations on the server first.
2. **Title UPDATE-backfill 2020–2026** (1.154 TB, $0 across 2 months): compare
   dry-runs of plain wide `refresh()` vs a minimal 4-column pull before
   deciding whether any new code is even needed.
3. **Option C clustering baseline** (unchanged): `include_embeddings` flag,
   add tone to the context vector, aggregate→context→cluster→analyze, regime
   sanity checks.

---

## 2026-07-02 (later — crash recovery: 2015–2019 backfill landed)

### Context
The machine overheated and restarted mid-way through the 2015–2019 wide
backfill (it launched right after the 11:52 commit and died ~13:08). Session
goal: diagnose, resume safely, land the whole thing. AYMStation again — all
backfill data lives in the **laptop** DB; the server still has none of it.

### What I did
- **Forensics**: BigQuery job history showed all 55 chunk queries `DONE` — the
  machine died *between* chunks and DuckDB closed clean (no WAL). Landed data
  stopped at 2016-11-21, mid-November: exactly the case where the
  month-granularity gap detection in `backfill_gdelt.py` would have resumed at
  December and silently left Nov 22–30 empty forever.
- **Fixed gap detection to day granularity** (`present_days` / day-level
  `gap_ranges`). Exact by construction: chunk upserts are atomic and
  day-aligned, so a day with rows is a complete day. Regression test encodes
  the crash (present through 11-21 → gap must start at 11-22).
- **Resumed the pull** (~0.86 TB remaining). The resumed process was
  **OOM-killed** at ~15.3 GB RSS after ~70 chunks (`dmesg` oom-kill; the WSL2
  VM has 15 GB): memory accumulates across chunks in the BQ→pandas path and
  never returns to the OS. DuckDB rolled back cleanly — zero damage. NB: OOM
  is a plausible contributor to the original "overheat" crash too.
- **Restarted as a month-windowed driver** — one fresh process per calendar
  month, 16 windows, each idempotent thanks to day-level gaps. All 16
  completed without incident (~1 chunk/min, ~3–5 GB RSS steady).
- **Verified**: coverage **2015-02-18 → 2026-06-19, day-continuous**, 139.9M
  rows (was 63.3M). Exactly one hole: **2017-08-29 is empty upstream in GDELT
  itself** (a re-pull returns 0 rows; neighbouring days depressed too — left
  as documented). `src_lang` 100% on backfilled rows, 32.4% English.
- **PAGE_TITLE has a hard onset: 2019-09-22.** Coarse-probed via
  `COUNTIF(Extras LIKE '%<PAGE_TITLE>%')` over a 2017–2019 day grid, then
  confirmed in landed rows: 0% through 09-21, 37% on the switch-on day
  (14:30 UTC), ~99.2% steady after. So "titles for free" held only for the
  last ~3.3 months of the backfill (~3.1M titled rows); **2015→2019-09-21 can
  never get titles from GKG** — the DOC 2.0 API remains the only title source
  there. Corrected the assessment (§2 → resolved, §3 correction block), the
  `gdelt.py` docstring (~99.6% claim), and the CLAUDE.md sharp edge.
- **Title UPDATE-backfill 2020–2026 needs no new code**: dry-runs put plain
  wide `refresh()` at 1.233 TB vs a minimal 4-column variant at 1.154 TB —
  $0.49 apart, because V2Themes is scanned by the WHERE clause either way.

### What I learned
- Day-granular gap detection paid for itself twice in one session: the crash
  resume (9 days would have vanished silently) and surfacing the 2017-08-29
  upstream hole that month granularity structurally cannot see.
- Long BigQuery→pandas→DuckDB pulls on this box must run one process per
  window: RSS grows ~200 MB/chunk regardless of chunk size, so the OOM
  killer — not thermals — is the binding constraint. The driver-loop pattern
  (fresh process per month, script skips present days) is now the standard.
- Never open the DuckDB file (even read-only) while a backfill is writing:
  single-writer locking means a stray reader can kill the writer's next
  connect. Verify only between runs.

### DB mutations this session
- **Laptop DB only**: `headlines` 63.3M → 139.9M rows (this session pulled
  2016-11-22 → 2019-12-31 wide; the pre-crash run had landed 2015-02-18 →
  2016-11-21). File 35 → 51.4 GB (disk fine: 769 GB free).
- **Server DB untouched** — still 2020+, narrow, pre-007. Migrations first,
  then either re-run the backfill there (~1.35 TB again) or copy the file.
- BigQuery July total: **1.373 TB billed → $2.33** past the free tier
  (projection was $2.39).

### Next session
1. **Option C clustering baseline** (unchanged, now unblocked on 2015+ data):
   `ContextConfig.include_embeddings` flag, add tone to the context vector,
   aggregate→context→cluster→analyze; regime sanity checks can now include
   the 2015–16 commodity bust and 2018 trade war.
2. **Title UPDATE-backfill 2020–2026** via plain `refresh()` after Aug 1
   ($0 across the billing boundary; ~1.23 TB).
3. Decide the server-DB sync path (re-pull vs copy the 51 GB file) before any
   server-side Phase 3 compute.

---

## 2026-07-02 (evening — 2020–2026 title backfill: the fast way, after the slow way)

### Context
User call: finish the title UPDATE-backfill now, in July, at the quoted
~$7.71 rather than waiting for August's free tier. What was supposed to be
"plain `refresh()`, no new code" turned into a redesign — and a much better
tool.

### What I did
- **Started the naive path** (`refresh()` per month window, the OOM-safe
  driver pattern) and watched it crawl: ~10 min per 10-day chunk vs ~60 s in
  the afternoon. Diagnosis: this workload *updates* every row (all PKs exist),
  and DuckDB's per-row `ON CONFLICT DO UPDATE` through the 140M-row ART PK
  index is the bottleneck — a 30–40 h job. Killed it after chunk 3 of 236
  (the applied updates are idempotent, no cleanup needed). A second, additive
  problem: title-era `Extras` blobs (PAGE_LINKS etc.) made even the pure pull
  ~8.5 min/month.
- **Two-phase replacement** (`scripts/backfill_titles.py`, promoted from
  scratchpad with tests):
  1. `pull` — extract PAGE_TITLE **inside BigQuery** (`REGEXP_EXTRACT` on
     Extras) so the download is ~100-byte strings, not multi-KB blobs. Same
     scan billing (columns scanned, not bytes downloaded), ~5× faster wall
     clock. Wrote per-chunk parquet, atomic + skip-if-exists (crash-resume
     for free), **zero DuckDB contact** (no writer-lock hazard). Client-side
     normalisation reuses `extract_src_lang` and replicates
     `extract_page_title`'s post-regex steps; **parity-verified: 0 mismatches
     on 323K live rows** against the python-extracted January parquet.
  2. `apply` — yearly bulk `UPDATE … FROM read_parquet(...)` (hash join, PK
     index untouched). Pilot: 907K rows in **0.77 s**.
- **Ran it**: 77 month-window processes pulled 2020-02 → 2026-06-19 in
  ~98 min (~50–75 s/month); apply updated **63,266,028 rows in 31 s** total.
  End-to-end ~1.7 h vs the 30–40 h naive path (~1000× on the update step).
- **Verified**: 2020–2026 titled 99.3–99.6% per year, `src_lang` ~100%
  everywhere; 2019 at 24.7% titled = exactly the 2019-09-22-onset fraction of
  the year; 2015–2018 correctly 0% titled / 100% lang. Sample titles are real
  headlines. ~99.99% of parquet rows matched existing PKs.
- Docs: CLAUDE.md sharp edge rewritten (title state, the parquet artifact,
  "never conflict-update a big indexed table" rule); assessment §3 correction
  block updated to "ran the same evening" with the method + numbers.

### What I learned
- **Upsert ≠ update.** The identical `ON CONFLICT` upsert that ingests fresh
  rows at ~60 s/chunk collapses to ~10 min/chunk when every row conflicts.
  For column fills on existing rows: pull to parquet, then bulk
  `UPDATE … FROM` — 63M rows in 31 s.
- **Push extraction into BigQuery when the raw column is fat.** Billing is
  per column *scanned* either way; downloading the regex group instead of
  `Extras` cut transfer ~5× and memory to trivial.
- The per-chunk parquet directory doubles as a **portable artifact**: the
  server gets identical titles in ~30 s (`backfill_titles.py apply`) with $0
  BigQuery spend. Worth keeping `data/raw/title_backfill/` (7.6 GB) around
  until the server is synced.

### DB mutations this session
- **Laptop DB only**: `page_title`/`src_lang` filled on all 63.27M 2020+
  rows (plus the 907K 2020-01 rows via the pilot). Row count unchanged
  (139.9M). File 51.7 → 54.4 GB (update row-group rewrites); disk fine
  (758 GB free), compaction skipped as unnecessary.
- `data/raw/title_backfill/`: 81 parquet files, 7.6 GB (gitignored), keep for
  the server sync.
- **BigQuery July 2026 final: 2.533 TB billed → $9.58** past the free tier
  ($2.33 for the 2015–2019 backfill, ~$7.25 for the title pull — under the
  $7.71 quote since the fast query skips V2Tone/SourceCommonName).

### Next session
1. **Option C clustering baseline** (unchanged; text signals now maximal:
   tone+themes 2015+, titles 2019-09-22+, languages everywhere).
2. **Server DB sync**: migrations runner, then either copy the 54 GB DuckDB
   file, or re-run the 2015–2019 backfill (~$8.45 or split months) + apply
   the title parquet. Decide before server-side Phase 3 compute.
3. Forward-fill note: corpus still ends 2026-06-19; next `refresh()` pull of
   recent weeks lands wide (titles included) by default.

---

## 2026-07-03 (Option C clustering baseline — landed)

### Context
The long-queued Option C: a defensible, embeddings-free scenario clustering
on the signals the GDELT assessment trusts (tone + themes), now running on
the full 2015+ corpus backfilled earlier today. Laptop session, no GPU.

### What I did
- **`ContextConfig.include_embeddings` flag** (default True = old behaviour):
  off drops `text_pca_*`/`embedding_dispersion`; the V2Tone daily means
  (`mean_tone_overall/positive/negative`) now always join the context vector.
  `--no-text-embeddings` threads it end-to-end: `aggregate` runs tone-only
  (no torch import, no GPU), `embed` is skipped, `context` sets the flag.
  `run_cluster` now **registers every fit with the eval harness** (config,
  train range, feature names, git hash); the `label` stage feeds the LLM
  `COALESCE(page_title, article_url)` — real titles where they exist.
- **Aggregate + topics on the laptop**: 139,904,911 headlines → 4,092 daily
  `market` rows (8.5 min) + 52,524 topic-prevalence rows (seconds). Tone
  face-validity on famous days: COVID crash Monday 2020-03-16 the most
  negative (-1.76), 2018-06-19 trade-war escalation -0.96, Brexit -0.79.
- **Adversarial review before committing** (31-agent workflow: 3 lenses ×
  2 refuters per finding): 14 raw findings, **3 confirmed** (all minor),
  11 refuted. Fixed all three:
  1. *Same-day text vs forward returns* (inherited by every text feature,
     newly load-bearing through tone): daily text aggregates span the full
     UTC day incl. post-close hours, violating "a day's text must strictly
     precede the forward returns it predicts". Fix at the input layer —
     **all text-derived context features (tone, counts, topics, embeddings)
     now join lagged one trading day**; day-t price features stay as-of the
     close. Regression test pins the lag.
  2. *Tone-only re-runs NULLing embedding aggregates*: `upsert_daily` now
     COALESCEs the embedding columns (and lands NaN dispersion as SQL NULL
     so the guard is real) — same pattern as the headlines title upsert.
  3. *Nonstationary tone levels under train-anchored z-scores*: measured
     before acting — OOS (2024+) tone shift is +0.83σ but the yearly path
     tracks true stress years (2022 -0.97, 2020 -0.85, calm 2024 -0.30),
     i.e. mostly signal, not corpus drift; and the feared OOS noise-dump
     did not occur (2024 assigns 87% to an existing cluster at normal
     confidence). Kept levels; documented here.
- **Final model `phase3_optC_tone_lag1_2024split`** (harness run
  cb2e33a7…): 39 features, 2,148 train rows 2015-02-19 → 2023-12-29,
  assignments through 2026-06-19 (2024+ strictly OOS via
  `approximate_predict`), 7 clusters + 4.7% noise, mean confidence 0.86.
- **Regime sanity — all five targets recovered**:
  | regime | dominant cluster |
  |---|---|
  | 2015–16 commodity bust | 0 (69%; pure-2015 cluster) |
  | Brexit window | 2 (100%; 2016–17 regime) |
  | 2018 trade war | 6 (98%) |
  | 2020 COVID crash | 4 (48%, an 84%-2020 cluster) + 44% noise on the crash weeks |
  | 2022 inflation shock | 1 (99%) |
  | 2023 banking stress | 1 (100%; same high-rates regime) |
  OOS 2024 → 87% cluster 1 (regime continuation); OOS 2025–26 fragments
  (25% c6 / 24% c3 / 24% noise) — the record gold bull reads as genuinely
  novel rather than force-fitted, which is the honest behaviour.
- **Forward-return separation (gold, 20d, descriptive)**: bust cluster 0 is
  the only negative regime (-0.9%, 40% hit); easing-2019 cluster 5 the
  strongest bull (+4.1%, 86% hit); COVID cluster 4 +2.9% (73%). Not a
  trading claim — analyze-stage description, now free of same-day text
  leakage after the lag fix.

### What I learned
- The adversarial-review pattern earns its cost: the same-day-text finding
  was invisible to tests (alignment is a convention, not a crash) and the
  cheap fix landed *before* the baseline's numbers went anywhere.
- Measure before stationarizing: the "drift" in tone is mostly regime
  signal; a reflexive trailing-z transform would have thrown away exactly
  the level information the clusters use to separate stress years.
- 139.9M rows → 4,092 daily aggregates in 8.5 min on a laptop once
  embeddings are out of the loop: the Option-C iteration cycle is minutes,
  which is what makes the CV-gated embedding decision (assessment §7)
  actually testable.

### DB mutations this session
- **Laptop DB**: `daily_text_features` 0 → 4,092 rows (tone-only, embeddings
  NULL); `daily_topic_prevalence` 0 → 52,524 rows; `cluster_assignments`
  2 model versions (superseded `phase3_optionC_tone_2024split` + final
  `phase3_optC_tone_lag1_2024split`); `cluster_centroids` populated; 2 runs
  registered in `runs`. Headlines untouched.
- Server DB still untouched/behind (see 2026-07-02 entries).

### Next session
1. **CV gate for embeddings** (assessment §7): with the Option-C baseline in
   the harness, wire the cluster→forward-return lift into walk-forward CV and
   test whether PAGE_TITLE embeddings (2019-09-22+) buy anything over
   tone+themes before any GPU spend.
2. **LLM cluster labels**: set ANTHROPIC_API_KEY and run the label stage on
   the final model (now title-fed).
3. Server sync decision still open (copy 54 GB file vs re-pull).

---

## 2026-07-11 — LLM cluster labels landed (Opus 4.8, both model versions)

### What happened
- Set up `ANTHROPIC_API_KEY` (documented in `.env.example`), added the
  missing `anthropic>=0.60` dependency to pyproject (the label stage would
  have ImportError'd — the SDK was never declared), and flipped the label
  stage default from Haiku 4.5 to **Opus 4.8** (`cluster_labeling.py`,
  `phase3_pipeline.py`).
- Corrected the stale cost docstring in `cluster_labeling.py`: measured
  reality is ~1,350 input / ~95 output tokens per cluster, not the
  ~$0.50–15/run the old comment claimed. Actual: **$0.13 for 14 clusters**
  across both model versions on Opus 4.8.
- Ran the label stage on both `cluster_assignments` model versions via a
  budget-guarded runner ($1.00 hard cap, worst-case pre-check per call;
  never approached). Labels upserted into `cluster_centroids`.

### Labels
- `phase3_optionC_tone_2024split`: mostly low-confidence —
  2× `unclear`, 3× diffuse-macro-noise variants,
  `covid-crash-recovery-rebound` (medium), `summer-2019-gold-rally` (low).
- `phase3_optC_tone_lag1_2024split` (the final/lagged version): more
  distinct — `trade-war-dovish-fed-tailwind` (**high**),
  `covid-recovery-stimulus-rebound` (medium),
  `fed-rate-hike-expectations`, `mixed-newsflow-crude-uptrend` (low),
  2× `unclear`, 1× diffuse baseline.

### What I learned
- The honest-confidence prompt design works: Opus marked 9/14 labels
  low-confidence rather than confabulating themes for the big
  regime-mixture clusters (up to 681 days) — consistent with the
  no-per-metal-signal caveat in the GDELT assessment.
- The lag1 (final) clustering labels noticeably cleaner than the
  superseded same-day version — weak supporting evidence that the lagged
  text alignment sharpened the clusters, not just de-leaked them.

### Next session
1. CV gate for embeddings (unchanged from 2026-07-02 plan).
2. Server sync decision still open.

### Addendum (same day): local DB backup
- A6000 Thunder server turned out to be **deleted** (`tnr status`: no
  instances) — laptop DB was the sole copy of the corpus. Backed up to the
  Windows side: `C:\amc-backup\metals-2026-07-11.duckdb` (54.4 GB,
  byte-verified, opens read-only: 139.9M headlines, migrations ✓, 14 LLM
  labels ✓) + `title_backfill-2026-07-11\` parquets (8.1 GB). Protects
  against WSL VHDX loss, not laptop loss — GCS Coldline off-site copy
  (~$0.24/mo in `amc-metals`) discussed and deferred.

---

## 2026-07-11 (2) — Cluster→forward-vol lift experiment: gate readout

Ran the pre-registered A/B experiment (results/phase3_cluster_lift_design.md,
runner scripts/phase3_cluster_lift.py). Shared rows 2,718 (2015-02-19 →
2026-05-22); folds came out at **11** (design estimated ≈9 — parameters were
applied verbatim, the estimate was just off; test windows 2020-08 → 2026-01).
All runs on AYMStation, pinned LGBM + ClusteringConfig defaults, per-fold
regime refits at split.train_end.

### Primary (GC=F rvol h=5): **B does NOT beat A**
- rel ΔRMSE **−0.37%** (bar −1.0%); B wins **4/11** splits (need 7).
- mean RMSE: A 0.05438 / B 0.05418. Mean IC: A +0.001 / B +0.057
  (IC is report-only).
- B_notext ablation correctly not run (gated on B beating A).
- run ids: A 5f236194-99fa-4f0e-b502-76b70a239923,
  B 8af4cdef-be84-44a8-b286-ce167b036ec6.

### Secondaries (report-only, never decide)
- SI=F h=5: B **worse** (+1.80% rel RMSE, 5/11 wins).
  A 381b16b1-a640-40a7-910f-582128851f00,
  B c0e6e66e-76a5-41cf-bd95-add923057621.
- GC=F h=20: B better on paper (−2.12% rel RMSE, 7/11 wins) — would have
  passed a primary-style bar, but h=20 was pre-registered as report-only;
  it does not decide anything. If regime-at-longer-horizon is worth chasing,
  it needs its own pre-registration first.
  A fbcb1c5c-ef2c-442d-b685-31a2c213fd26,
  B 1f97199c-9b1e-4546-b98a-bd57e9a5e93a.

### Consequence (per the pre-registered decision rules)
- **No corpus-scale embedding spend**: the null B−A readout "caps arm C's
  priority" — the assessment §7 GPU gate stays closed. No new A6000 instance
  needed for now (server was deleted anyway, see earlier entry).
- Per-split table: results/phase3_cluster_lift_readout.csv (66 rows).

### Honest caveats
- Fold count 11 vs the design's estimated ≈9 (estimate error, not a rule
  change; recorded here for transparency).
- The known val/test embargo caveats from the design apply identically to
  both arms; the paired Δ mostly cancels them (assumption, not theorem).
- Splits 9–10 (2025–2026 test windows) have 2–4× the RMSE of earlier splits
  in every arm — the recent vol regime dominates the mean; the per-split win
  count was the guard against exactly this and it also says no.

### Addendum: Phase 3 write-up
- Consolidated the phase into results/phase3_writeup.md (corpus, taxonomy,
  pre-registered null, consequences). Phase 3 considered closed; branch
  merging to main next.

### Addendum: merge to main + recovered Phase 5 scaffolding
- PR #1 merged: phase3-streaming-themes → main (690ca84). Phase 3 closed on
  the mainline.
- Merging surfaced two never-pushed local-main commits from 2026-06-23:
  a CLAUDE.md dedupe + **Phase 5 causal scaffolding** (models/causal.py
  with DoubleML ATE/placebo/CATE, features/scenarios.py, loaders, ~530
  lines of tests — steps 5.1–5.4). Preserved verbatim on branch
  `phase5-causal-scaffolding` (pushed); local main realigned to origin.
  Integrating that branch (rebasing onto post-Phase-3 main, reconciling
  CLAUDE.md/journal, re-running its tests) is the natural first task of
  Phase 5.
