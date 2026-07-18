# Precious Metals Research Roadmap

A phased research program to study what drives precious metals prices (gold, silver, platinum, palladium) using a combination of statistical methods, machine learning, and causal inference. Three scenario-discovery methods are integrated across the phases: event-driven local projections (Phase 2), unsupervised clustering (Phase 3), and causal ML (Phase 5).

## Phase 0: Scoping and setup

Get the environment and habits in place. Python with `pandas`, `polars`, `lightgbm`, `statsmodels`, `linearmodels`, `pytorch`, `transformers`, `sentence-transformers`, `HDBSCAN`, `UMAP`, `BERTopic`, `arch` (GARCH), and `econml` or `DoubleML`. Use `uv` or `conda` and pin versions — reproducibility matters more than you think on a project of this size.

Storage: DuckDB or SQLite is plenty; Parquet on disk works too. Standardize everything to UTC timestamps and keep a single canonical "as-of" timestamp on every row so you can prevent leakage by construction.

Start a research journal (a single `journal.md` you append to) and a single evaluation harness — every model run, no matter how toy, writes its metrics there. The harness is empty now but you'll thank yourself in Phase 4.

Designate gold as the primary "first model" target even though all four metals are in scope. It has the deepest data and clearest macro drivers, so it's the right metal to fail on first. Silver, platinum, palladium come in once the pipeline works.

## Phase 1: Price foundation and baseline

Pull daily OHLCV for `GC=F`, `SI=F`, `PL=F`, `PA=F` and the ETFs `GLD`, `SLV`, `PPLT`, `PALL` via `yfinance`. From FRED pull `DFII10` (10Y TIPS), `DGS10`, `DGS2`, `DTWEXBGS` (USD index), `VIXCLS`, `T10YIE` (breakeven inflation), `DCOILWTICO` (oil), copper futures, and the **GPR daily index**.

Build features: log returns, rolling realized vol (5/20/60-day), the spreads (Au/Ag, Pt/Pd, Au/Cu), z-scored macro variables. Train a LightGBM that predicts next-week realized vol. Vol is much easier than direction and gives you a meaningful baseline. Validation is walk-forward with expanding window — never random split.

Deliverable: an end-to-end re-runnable pipeline and a baseline RMSE / information coefficient that future phases must beat.

## Phase 2: Events + event-driven scenarios via local projections

This is where the **first of the three scenario methods** comes online.

Add scheduled economic releases (FOMC, CPI, PPI, NFP, ECB, BoE) and their "surprise" component (actual minus consensus). Free consensus archives are sparse — the Federal Reserve Bank of Atlanta's GDPNow archive, MarketWatch calendar scrapes, or Investing.com's calendar are workable; if UChicago gives you Bloomberg access, use that. Add the weekly CFTC COT report and lag it correctly (Tuesday positioning released Friday afternoon — using it as a Tuesday feature is the most common data leak in this field).

Implement **Jordà local projections**: for each event type, regress the h-step-ahead return on the event indicator and controls for h = 1, 5, 20, 60 days, with HAC standard errors. `statsmodels` and `linearmodels` make this clean. Produce IRF plots with confidence bands for "hawkish FOMC surprise," "dovish FOMC surprise," "CPI upside surprise," "GPR spike," "DXY shock."

Deliverable: a folder of impulse-response charts you can defend. You should be able to write a paragraph per scenario describing what each metal does, with statistics behind it.

## Phase 3: Text data + unsupervised scenario clustering

This is where the **second scenario method** comes online.

Pull GDELT 2.0 events and GKG themes from BigQuery (free under quota) filtered to relevant themes — central banking, mining, geopolitics, commodity markets. Supplement with Kitco RSS scraped to disk for sector-specific commentary.

Embed headlines with FinBERT or `all-mpnet-base-v2`. Aggregate to daily features per metal: mean embedding vector, article count, embedding dispersion (proxy for news disagreement), mean sentiment, topic prevalences from BERTopic.

Now the scenario discovery step: build a daily feature vector combining macro state + news features + recent returns, UMAP-reduce to 5–10 dimensions, and cluster with HDBSCAN. Examine the conditional distribution of forward returns and realized vol by cluster. Label clusters by hand using their constituent days and dominant topics.

Sanity check: your clusters should pick up known regimes — 2011 gold peak, 2013 taper tantrum, 2020 COVID flight-to-safety, 2022 inflation shock, 2023 banking stress. If they don't, your features are wrong before your method is.

Deliverable: a cluster taxonomy with descriptive labels, per-cluster forward-return distributions, and code that assigns new days to clusters.

## Phase 4: Multimodal transformer

Now upgrade the model. Start with **iTransformer** or **PatchTST** treating each metal and macro variable as a variate. Train on a multi-horizon multi-task objective — predict t+1d, t+5d, t+20d returns and realized vol jointly. Multi-task acts as a regularizer and forces representations that work across horizons.

Add news embeddings: first by simple concatenation per timestep, then by cross-attention between numeric and text streams. Compare lift at each step. Anything below ~10% improvement over LightGBM should make you suspicious — financial ML papers routinely overstate transformer gains because of look-ahead bugs.

Re-run the Phase 3 clustering on the transformer's learned hidden representations rather than hand-engineered features. Usually cleaner clusters because the model has learned what's predictive. Compare the two clusterings — agreement is reassuring, disagreement is interesting.

Add attribution: integrated gradients or `captum` for numeric features, attention rollout for the text stream.

Deliverable: a trained model with per-date predictions and attributions, plus an updated cluster taxonomy from learned representations.

## Phase 5: Causal ML and method triangulation

The **third scenario method** comes online here and ties everything together.

For each scenario type — whether defined by an event (Phase 2) or by a cluster (Phases 3–4) — estimate its causal effect on returns using **double/debiased ML** (`DoubleML` or `econml`). Your transformer predictions serve as nuisance estimators for the outcome model; the scenario indicator is the treatment. This gives you confidence intervals that account for the flexibility of the ML model.

Where the scenario maps to an economic shock, validate with **sign-restricted structural VAR**: identify "real yield shock," "risk-aversion shock," "USD shock" via sign restrictions on yields/equities/FX, and check that your scenario's effect on gold matches the theoretical sign.

Then triangulate. A scenario found by clustering (Phase 3) should produce a similar IRF when estimated as a local projection (Phase 2 method) and a similar treatment effect under double ML (Phase 5 method). Agreement across methods is much stronger evidence than any single method alone — and disagreement points you at the most interesting analysis questions.

Robustness checks every scenario must pass: placebo (random "scenario" dates should produce zero effect), subsample stability (pre/post 2015 split), alternative-metal consistency (does silver react in the predicted way?), and sensitivity to the lookback window.

Deliverable: a master table of scenarios with their causal effect estimates, confidence intervals, robustness diagnostics, and method-agreement scores.

## Phase 6: Validation and writeup

Out-of-sample year you never touched — typically the most recent 12 months. Quantify lift over benchmarks (random walk, GARCH(1,1), unconditional mean, sentiment-only model, LightGBM-only). Write up methodology, scenarios identified, and an honest limitations section. The writeup is where the learning consolidates.

## Phase 7: AMC program (added 2026-07-12)

Phases 0–6 answered *what moves metals prices*; Phase 7 turns the answers into operating decisions for **AMC Company**, the dealer the research serves (scrap Au/Ag/Pt/Pd buying with assay; gold coin & specie; structurally long inventory over a days-to-weeks float). Two tracks: (a) a start-now **seven-collector data-acquisition program** for non-backfillable series — AMC's own ledger, retail coin premiums, search interest, event calendars (`results/amc_data_acquisition_program.md`), plus JM PGM/rhodium prices and premium-side forward capture added 2026-07-12 with a small paid sprint (Databento CME data, Greysheet; ≤ ~$725/yr — `results/amc_paid_data_review.md`); CME open interest was removed from the non-backfillable list on 2026-07-15 (Databento retains it permanently; the website route is barred by CME's Data ToU regardless); (b) a decision-support portfolio (spread floors, a live FOMC hedge playbook, PGM liquidation alarms, crisis indices, demand models, macro-release movers §7.8), baseline-first with transformers only as pre-registered, kill-criterioned bake-offs. Full plan: `plans/phase_7_amc_program.md`; business translation of Phase 5: `results/phase5_amc_business_implications.pdf`.

**Status (2026-07-16 ToU audit):** the automated collectors were found to bar AMC's commercial use, so the data-acquisition track is **paused** — timers disabled, already-captured rows quarantined (migration 010) — pending licences and AMC-ledger access (the gate, no ETA); the baseline-first analysis portfolio continues off the clean Phases 0–6 data (see §7.7 and `journal.md`).

---

Each phase produces something usable on its own, so you can pause anywhere without throwing away effort. The three scenario methods come online at Phases 2, 3, and 5 respectively, and Phase 5 is where they get reconciled.

**What I'd push back on if you skip:**
- Phase 0 feels tedious but the leakage and reproducibility scaffolding pays for itself many times over.
- Phase 4 is the flashiest phase but often the least pedagogically valuable — financial ML is one of those fields where a tuned LightGBM beats most transformers in honest evaluations. Do it for learning, not because you expect it to dominate.
- Phase 5 is where the actual research-grade insight lives. Don't skip it because it's the least visual.

---

## Status as of 2026-07-18

| Phase | Status | Notes |
|---|---|---|
| 0 — Scoping and setup | Complete | DuckDB, eval harness, walk-forward CV |
| 1 — Price + LightGBM baseline | Complete + cleanup | `BAA10Y` replaced license-restricted `BAMLH0A0HYM2`. Feature-importance pipeline added. Lean / lean_own feature sets reflect diagnostic finding that cross-ticker returns/vol block is net-negative for IC. |
| 2 — Events + local projections | Complete | 176 FOMC events, Bauer-Swanson MPS_ORTH surprises, COT with Friday-close lag. Headline result: hawkish-FOMC IRF -1.5% on gold at h=5, sign-consistent across Au/Ag/Pt. |
| 3 — Text + clustering | **Complete (merged to main 2026-07-11, PR #1)** — 139.9M-row GDELT corpus (2015-2026, titles backfilled), Option C tone+themes clustering, Opus-labelled 7-cluster taxonomy, pre-registered cluster-lift experiment | Lift readout: NO predictive lift at the primary target (GC=F h=5 rvol): rel ΔRMSE -0.37% vs -1.0% bar, 4/11 wins. Embedding gate (assessment §7) closed. Full narrative in results/phase3_writeup.md. |
| 4 — Multimodal transformer | **Deferred (re-scoped)** | Re-scoped to a numeric-only optional experiment and deferred; off the critical path. Consistent with the program's finding that a tuned LightGBM beats transformers in honest OOS evaluation here (Phase 6). |
| 5 — Causal ML + triangulation | **Complete (2026-07-11)** — DoubleML ATE/placebo (5.2-5.3), regime-CATE (5.4), sign-restricted SVAR (5.5), subsample stability (5.8), master scenario table (5.7/5.9), write-up (5.10) | Anchor finding triangulated 3 ways: hawkish FOMC → gold −1.4% at h=5 (LP/DML/SVAR agree; sign-stable across eras; amplified in the rate-hike-expectations regime). GPR + DXY-down fail robustness for documented measurement reasons. See results/phase5_triangulation.md + scenario_master.parquet. |
| 6 — Validation and writeup | **Complete (2026-07-16) — v1.0 tagged** (annotated; local, unpushed) | 63-day hold-out: classical baselines (GARCH/VAR) beat ML; regime/sentiment features hurt OOS. 6.10 repro package built + tested (`metals.refresh` / `metals.train` orchestrators, harness Parquet export, README repro section); 6.11 cleanup done (mypy 8→0, journal lessons extended). Validation `results/phase6_validation.md`; methodology 6.8; findings 6.9. |
| 7 — AMC program | **Scoped; data-acquisition track PAUSED (2026-07-16 ToU audit)** — `plans/phase_7_amc_program.md` | Seven-collector start-now data program (`results/amc_data_acquisition_program.md`), but the ToU audit found the automated collectors bar AMC's commercial use → timers disabled, 171k captured rows quarantined (migration 010, filter `quarantine_reason IS NULL`), pending licences; AMC-ledger access is the gate (no ETA). The baseline-first analysis portfolio is unblocked and runs off the clean Phases 0-6 data; first job 7.2 (FOMC hedge playbook) — ΔDGS2 same-evening surprise leg built (migration 012). Phase 5 business translation: `results/phase5_amc_business_implications.pdf`. |
| 8 — Self-supervised representation | **Scoped 2026-07-17 (design only, no code)** — `plans/phase_8_ssl_probing.md` | Low-rank joint factorization of the daily price + GDELT-news state, framed as insight not prediction (classical `LRJ-Metals` primary; deep `CoMPASS` encoder a gated Stage-B bake-off). Pre-registers a clean null as a first-class, shippable outcome. Blocking prereq: `daily_text_features.mean_embedding` is currently all-NULL and must be rematerialized. Stage-A scaffold built (`features/ssl_views.py`, `models/factor_ssl.py`, `eval/probes.py` + 16 tests, 2026-07-18). |
| 9 — Real-yield event study | **Scoped 2026-07-18 (design only, no code)** — `plans/phase_9_realyield_event_study.md` | Re-specifies the Phase-5 hawkish-FOMC anchor by making the **real-yield surprise** the treatment (the SVAR showed the effect runs through real yields), on a broadened FOMC+CPI+Employment event set (~450–550 event-days vs 35 MPS_ORTH) so it can finally clear the 2024–26 hold-out the anchor failed. Three treatments: a GSS target-vs-path factor split (no new data), a same-evening ΔDFII2/ΔDFII10 surprise (new `realyield_surprises` table), and a CPI breakeven-vs-real-yield split. Reuses LP/DoubleML/SVAR; sharpens AMC's FOMC/CPI hedge-timing rule. |
| 10 — PGM supply-shock event study | **Scoped 2026-07-18 (design only, no code)** — `plans/phase_10_pgm_supply_shocks.md` | Builds the correct treatment for Pt/Pd, which failed the monetary channel in Phase 5 (Pd sign-flipped post-2020, only metal to fail placebo): a dated, typed `pgm_supply_events` ledger (LLM annotator + hand-verified anchors + trade-press), then an LP/DoubleML event study net of risk-off. Only ~10–30 clean shocks → wide CIs / underpowered null expected. Flagship byproduct of the Phase-8.1 annotator; targets AMC's fattest inventory tail. Needs migration 014. |

**Cumulative test count: 559 — all pass (2026-07-18); ruff + mypy clean. (507 at the v1.0 tag, 2026-07-16; +52 since = Phase 7.2/7.3 + the Phase 8 Stage-A scaffold.)**
