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
