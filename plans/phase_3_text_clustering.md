# Phase 3 — Text Data and Unsupervised Scenario Clustering

## Goal
Bring news and topic information into the feature set and produce a first cut of data-driven "scenarios" via clustering of daily contextual feature vectors. This is the second scenario-discovery method.

## Prerequisites
- Phase 2 complete
- Google Cloud account with BigQuery enabled (free tier sufficient)
- GPU helpful for embedding throughput but not required

## As-built notes (updated 2026-06-23)

The implementation in this repo diverges from the planning doc in a few
deliberate places, all driven by the real GDELT scale (~48.5M filtered
headlines after backfill):

- **Default embedding model is `all-MiniLM-L6-v2` (384d)**, not mpnet (768d).
  The storage and throughput win is decisive at our corpus size; cluster
  quality on news-style text is essentially indistinguishable for our
  scenario-discovery purpose.
- **Embedding cache lives outside the repo folder**, at
  `%LOCALAPPDATA%\metals\embeddings` on Windows (or
  `~/.cache/metals/embeddings` on Unix), to avoid OneDrive trying to sync
  tens of GB of cache files. `METALS_EMBEDDING_CACHE_DIR` env var overrides.
- **Cache format is sharded Parquet**, not per-text `.npy` (4,096 shards by
  first-3-hex of sha256(text), atomic writes via tmp+rename, fp16 on disk,
  fp32 to callers).
- **An eighth pipeline stage `label` was added** after `analyze` for
  LLM-assisted cluster naming via the Anthropic API (default Haiku 4.5,
  ~$1 for the full taxonomy). Gated on `ANTHROPIC_API_KEY` presence.
- **End-to-end orchestration**: `scripts/phase3_pipeline.py` with stages
  `gdelt -> embed -> aggregate -> topics -> context -> cluster -> analyze -> label`
  and `--only` / `--resume-from` flags for chunked execution.

## Steps

### 3.1 GDELT BigQuery access
- Create a GCP project, enable BigQuery API, create a service account, download JSON credentials, set `GOOGLE_APPLICATION_CREDENTIALS` in `.env`
- Read the GDELT 2.0 documentation, especially the GKG (Global Knowledge Graph) schema and CAMEO event codes
- Test with a small query against `gdelt-bq.gdeltv2.gkg` for a single day

### 3.2 Theme and event filter set
Themes to capture (subset of GKG taxonomy):
- `ECON_INTEREST_RATES`, `ECON_INFLATION`, `CENTRAL_BANK`, `ECON_HOUSE_PRICES`
- `COMMODITIES_GOLD`, `COMMODITIES_SILVER`, mining-related codes
- `WB_*` (geopolitical risk variants), `GEOPOLITICAL`
- `MILITARY`, `KILL`, `PROTEST`
- `TRADE_DISPUTE`, `TARIFF`

Build a parameterized SQL query that filters GKG records to these themes and exports daily aggregated counts plus a sample of representative articles per day per theme.

### 3.3 Bulk extract GDELT
- Pull 2015–present in monthly chunks
- Save to `data/raw/gdelt/YYYY-MM.parquet`
- Track query cost in `journal.md` — well-filtered queries stay near zero in the free tier

### 3.4 Supplement with Kitco
`src/metals/data/kitco.py`:
- Scrape Kitco news RSS, archive to `data/raw/kitco/YYYY-MM-DD.json`
- Schedule to run daily going forward
- For historical backfill, Kitco URLs are structured by date — write a polite scraper that respects robots.txt and rate-limits

### 3.5 Text preprocessing
`src/metals/data/text_prep.py`:
- Deduplicate near-duplicate headlines (MinHash or lowercased exact-match)
- Drop articles outside the project timeframe
- Truncate to 256 tokens for embedding efficiency
- Store cleaned headlines in DuckDB `headlines` table: `(timestamp_utc, source, headline, themes JSON, article_url)`

### 3.6 Embedding model setup
Default (as-built): `sentence-transformers/all-MiniLM-L6-v2` (384-dim).
Alternative: `all-mpnet-base-v2` (768-dim, slightly better cluster fidelity but 2x storage and 5x slower).
Selected MiniLM after the GDELT backfill confirmed 48.5M rows; storage and throughput became the binding constraint.

`src/metals/features/embeddings.py`:
- Batch encode headlines on GPU if available
- Cache embeddings to `data/processed/embeddings/{date}.parquet` so re-runs are cheap
- Store a config hash so cache invalidates when the model changes

### 3.7 Daily aggregation of text features
For each (date, metal) pair:
- Mean embedding vector
- Article count
- Embedding dispersion (mean cosine distance from the day's centroid — proxy for news disagreement)
- Mean FinBERT sentiment score
- Theme-prevalence vector (count of articles per major theme)

### 3.8 Topic modeling with BERTopic
- Fit BERTopic on the full article corpus, 2015–present
- Target 30–50 topics
- Inspect top words per topic and manually label the top 20 (e.g., "FOMC commentary," "Russia-Ukraine," "China demand," "S. Africa mining strikes," "EV/auto demand")
- For each day, compute topic-prevalence vector

Save the fitted BERTopic model to disk so downstream code can apply it to new days without refitting.

### 3.9 Assemble the daily contextual vector
For each date, concatenate:
- Macro state (TIPS 10Y, DXY, VIX, GPR — levels and changes)
- Recent returns (5/20-day, all four metals)
- Recent vol (5/20-day, all four metals)
- Text mean-embedding (PCA-reduced to 16 dims to control dimensionality)
- Topic-prevalence vector
- Positioning state (COT z-scores for each metal)

Store as `data/features/daily_context_YYYY-MM-DD.parquet` indexed by date.

### 3.10 UMAP reduction
- Reduce to 5–10 dimensions
- Fit on the training segment only (everything pre-hold-out from Phase 6 design — define the hold-out now even if you won't use it until later)
- Save the fitted UMAP model so new days can be projected consistently

### 3.11 HDBSCAN clustering
- Tune `min_cluster_size` to land at 8–15 clusters
- Inspect noise points (label = -1) — these are often the most informative outliers
- Save cluster centroids and the fitted model

### 3.12 Cluster analysis
Per cluster:
- Date list with example dates and headlines
- Forward return distribution at h = 1, 5, 20, 60 days, per metal (mean, median, std, hit rate of positive return)
- Dominant topics (top 3 by mean prevalence)
- Macro-state characterization (mean TIPS, DXY, VIX percentiles)

Then hand-label clusters. Aim for descriptive names: "hawkish-Fed-strong-USD," "geopolitical-flight-to-safety," "China-demand-pulse," "industrial-cyclical-rally," "QE-easy-money," "stagflation-fear."

### 3.13 Sanity check against known regimes
Confirm your clusters identify (or contain) these well-known episodes:
- 2011 gold peak
- 2013 taper tantrum
- 2020 COVID flight-to-safety
- 2022 inflation shock
- 2023 banking stress (SVB/CS)
- Palladium 2018–2022 supply squeeze (if Pt/Pd-specific clusters emerge)

Missing several of these signals a problem with feature mix, not clustering.

### 3.14 Persist cluster assignments and labels
- Per-date cluster label + assignment confidence
- Cluster centroids
- Cluster → human label mapping
- Loader function that takes a new date's contextual vector and returns its nearest cluster — used in Phase 5
- **LLM-assisted labeling**: `metals.eval.cluster_labeling` builds a per-cluster context bundle (representative dates, example headlines, dominant topics, mean forward returns) and calls Anthropic Haiku 4.5 to produce a short label + one-sentence description. Persisted with `label_source = 'llm:<confidence>'` so downstream analysis can filter by labelling provenance.

### 3.15 Document the cluster taxonomy
`results/phase3_clusters.md`:
- One section per cluster: label, definition, headline examples, forward-return statistics per metal, dominant topics, economic interpretation
- Comparison table: clusters vs known regimes
- Notes on which clusters are "macro" vs "industrial" vs "geopolitical" vs "supply" — this typology matters in Phase 5 cross-metal consistency checks

## Deliverables
- GDELT and Kitco ingestion with caching
- Headline embedding feature module with cache
- BERTopic topic model and per-day prevalences
- UMAP + HDBSCAN cluster model with persisted artifacts
- Per-date cluster assignments stored in DuckDB
- `results/phase3_clusters.md`

## Common pitfalls
- GDELT GKG timestamps reflect when GDELT processed the article, which usually matches publication but not always. For high-stakes claims, cross-check against the article URL.
- News-coverage density grows substantially over time. Don't interpret a cluster's modern prevalence increase as a real-world change without controlling for total daily article count.
- BERTopic is sensitive to UMAP and HDBSCAN hyperparameters used internally. Save the fitted model and don't refit casually between phases — you'll lose comparability.
- Re-using the same embedding model in Phase 4 — pin the version. A new release of `sentence-transformers` can shift embeddings enough to invalidate clusters.
- Cluster labels become anchors that bias later interpretation. Re-examine them after Phase 5 causal analysis; some will need renaming.
