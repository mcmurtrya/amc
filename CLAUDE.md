# CLAUDE.md

Quantitative research on drivers of precious-metals prices (gold, silver, platinum,
palladium). This is a **research codebase, not a deployed service**: code exists to
produce defensible empirical results, so correctness and the absence of look-ahead
leakage matter more than latency or API ergonomics.

Work is organized into 7 sequential phases (`plans/00_roadmap.md`, one plan file per
phase). Current state: mid–**Phase 3** (GDELT news text → embeddings → unsupervised
scenario clustering), on branch `phase3-streaming-themes`. Phases 4–6 (multimodal
transformer, causal ML, validation) are planned but largely unbuilt.

## Commands

Everything runs through `uv` (Python pinned to 3.11). `src/` is on the path via
`pythonpath` in pyproject, so import as `metals.*`.

```bash
uv sync --extra dev                                   # install deps + dev tools
uv run pytest                                         # full test suite (-v --tb=short)
uv run pytest tests/test_phase3_streaming.py          # one file
uv run pytest tests/test_features_context.py -k name  # one test
uv run ruff check src tests                           # lint (line length 100)
uv run ruff format src tests                          # format
uv run mypy                                           # type-check (src/metals only)
uv run python -m metals.data.migrations.runner        # apply DB migrations
```

Run lint, mypy, and the relevant tests before considering a change done.

## Architecture

Single source package `src/metals/`, layered data → features → models → eval. A single
local **DuckDB** file (`data/processed/metals.duckdb`) is the source of truth for
everything; all DB I/O goes through `data/db.py`.

- **`data/`** — ingestion, one module per source: `prices.py` (Yahoo OHLCV),
  `fred.py` (macro), `cot.py` (CFTC positioning), `fomc_surprises.py`, `gpr.py`
  (geopolitical risk), `gdelt.py` (GKG news via BigQuery → the ~63M-row `headlines`
  table that dominates the DB). Schema changes live in `migrations/` (apply via
  `runner.py`).
- **`features/`** — `returns.py`/`macro.py`/`spreads.py` build features; `assemble.py`
  + `loaders.py` turn DuckDB tables into wide ML matrices; `leakage.py` is the
  mandatory look-ahead guard. Phase 3 text: `embeddings.py` (sentence-transformers,
  Parquet-cached), `text_daily.py`, `topics.py` (BERTopic), `context.py` (daily
  contextual vector).
- **`models/`** — `lgbm_vol.py` (Phase 1 realized-vol baseline), `lp.py` (Jordà local
  projections / IRFs, Phase 2), `clustering.py` (UMAP + HDBSCAN, Phase 3).
- **`eval/`** — `harness.py` (every model run logs here), `cv.py` (walk-forward CV),
  `clusters.py` + `cluster_labeling.py` (LLM-assisted cluster labels).

**Phase 3 entry point:** `scripts/phase3_pipeline.py` — 8 ordered stages
`gdelt → embed → aggregate → topics → context → cluster → analyze → label`, each
runnable via `--only` / `--resume-from`, chunked by calendar month to cap memory.
The embedding stage expects a CUDA GPU (`Dockerfile` / `docker-compose.yml` provide
a CUDA 12.4 image); CPU runs are slow.

## Conventions (non-negotiable)

- **All timestamps stored in UTC.**
- **No look-ahead leakage.** Every feature pipeline must pass `features/leakage.py`
  before training; a day's text must strictly precede the forward returns it predicts.
- **Walk-forward CV only — never a random split.**
- **Every model run logs to the eval harness** (`eval/harness.py`).
- **Append a `journal.md` entry after every working session** (it's the running
  research log; ~45 KB and growing).
- Match surrounding style; ruff governs (line length 100, rules E/F/I/N/W/B/UP).

## Sharp edges

- **DuckDB does not reclaim space on `DROP COLUMN`.** After dropping columns, run
  `uv run python scripts/compact_headlines.py --replace` to actually shrink the file
  (keeps a `.bak`). Disk is tight on the research box.
- **Two conflicting `005_*.sql` migrations** exist (`005_drop_redundant_headline.sql`,
  `005_phase3_artifacts.sql`). Confirm runner ordering before adding migration 006+;
  the plan is to rename the artifacts one to `006`.
- **GDELT corpus limits** are documented in
  `results/phase3_gdelt_data_assessment.md` — read it before doing anything with text
  features. Key facts: embedded "documents" are article **URLs, not titles**; there is
  **no per-metal news signal** (all four metals get byte-identical daily text
  features — the metal axis on text is being collapsed); coverage is **2020+ only**.
- **Secrets** live in `.env` (copy from `.env.example`); needs at least `FRED_API_KEY`.
  BigQuery (GDELT) needs `GOOGLE_APPLICATION_CREDENTIALS`.
- **Windows/OneDrive:** keep the venv outside OneDrive (`UV_PROJECT_ENVIRONMENT`) to
  avoid file-lock failures during `uv sync`; see README.
