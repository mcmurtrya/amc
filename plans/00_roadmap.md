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

---

Each phase produces something usable on its own, so you can pause anywhere without throwing away effort. The three scenario methods come online at Phases 2, 3, and 5 respectively, and Phase 5 is where they get reconciled.

**What I'd push back on if you skip:**
- Phase 0 feels tedious but the leakage and reproducibility scaffolding pays for itself many times over.
- Phase 4 is the flashiest phase but often the least pedagogically valuable — financial ML is one of those fields where a tuned LightGBM beats most transformers in honest evaluations. Do it for learning, not because you expect it to dominate.
- Phase 5 is where the actual research-grade insight lives. Don't skip it because it's the least visual.
