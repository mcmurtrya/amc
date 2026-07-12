# CLAUDE.md

Quantitative research on drivers of precious-metals prices (gold, silver, platinum,
palladium), in service of **AMC Company** — a small dealer that buys scrap Au/Ag/Pt/Pd
(assaying fine content) and buys/sells gold coin and specie, and is therefore
structurally long physical metal over a days-to-weeks inventory float. This is a
**research codebase, not a deployed service**: code exists to produce defensible
empirical results and translate them into AMC's operating decisions (buy-spread
floors, FOMC hedging, PGM risk alarms, coin demand/premium intelligence), so
correctness and the absence of look-ahead leakage matter more than latency or API
ergonomics.

Work is organized into sequential phases 0–7 (`plans/00_roadmap.md`, one plan file
per phase). Current state: **Phases 0–3 and 5 complete** (2026-07-11), working on
`main`. Phase 3 closed with a labelled scenario taxonomy and a pre-registered null
lift result (`results/phase3_writeup.md`); Phase 5 closed with the
three-way-triangulated hawkish-FOMC finding and the master scenario table
(`results/phase5_triangulation.md`). Phase 6's validation core and long-form
write-ups are done (`results/phase6_validation.md`: 63-day hold-out, classical
baselines beat ML, regime/sentiment features hurt OOS; `phase6_methodology.md`,
`phase6_findings.md`); remaining: 6.10 repro entry points, 6.11 cleanup + v1.0 tag.
Phase 4 (transformer) was re-scoped to a numeric-only optional experiment and
deferred. **Phase 7 — the AMC program — is scoped** (`plans/phase_7_amc_program.md`):
Phase 5 translated into AMC's decisions in
`results/phase5_amc_business_implications.pdf`, and the start-now five-collector
data-acquisition program (non-backfillable series: AMC ledger, coin premiums, search
interest, CME open interest, event calendars) in
`results/amc_data_acquisition_program.md`. Collectors are append-only with
`source`/`pulled_at` provenance and real-time flags; anything from AMC's ledger
stays on the local machine.

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

Run lint, format, mypy, and the relevant tests before considering a change done
(`ruff check` and `ruff format` are both enforced; the codebase is kept clean of both).

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
- **Migration IDs are tracked by filename stem.** The old duplicate-005 conflict was
  resolved by renaming the artifacts migration to `006_phase3_artifacts.sql`
  (2026-07-02); DBs that ran it under the old stem keep a stale
  `005_phase3_artifacts` row and re-run the renamed file as a no-op (it is fully
  `IF NOT EXISTS`). Next free number: `008`. Never put a `;` inside a migration
  comment — whole-file execution silently truncates at it.
- **GDELT corpus limits** are documented in
  `results/phase3_gdelt_data_assessment.md` — read it before doing anything with text
  features. Key facts: there is **no per-metal news signal** (text features are a
  single shared `market` row per day). Coverage on **this (laptop) DB** is
  **2015-02-18 → 2026-06-19** (backfilled 2026-07-02; the server DB is still 2020+
  and needs migrations before any ingest). One known upstream hole: **2017-08-29 is
  empty in GDELT itself**. GKG `Extras` carries `PAGE_TITLE` **only from
  2019-09-22** (0% before — those rows can never get titles from GKG, only URL
  slugs). On this laptop DB the title backfill is **done** (2026-07-02 evening):
  `page_title` is ~99.3–99.6% within 2019-09-22 → 2026-06-19 and `src_lang` is
  ~100% everywhere (~32% English 2015–2019). NULL `src_lang` still means "not
  pulled wide", **not** English (`'eng'`) — relevant for the server DB and future
  pulls. The pulled titles live in `data/raw/title_backfill/*.parquet` (7.6 GB) —
  the server can apply them in ~30 s with `scripts/backfill_titles.py apply`, no
  BigQuery re-scan. Backfill gap detection is **day-granular**
  (`scripts/backfill_gdelt.py`); long pulls should run **one process per month
  window** — a single long-lived process accumulates RAM and gets OOM-killed on
  the 15 GB WSL2 VM. **Never fill columns on existing rows via the `refresh()`
  upsert**: per-row `ON CONFLICT` through the ART index runs ~1000× slower than
  `scripts/backfill_titles.py`'s pull-to-parquet + yearly bulk `UPDATE … FROM`.
- **Secrets** live in `.env` (copy from `.env.example`); needs at least `FRED_API_KEY`.
  BigQuery (GDELT) needs `GOOGLE_APPLICATION_CREDENTIALS`.
- **Windows/OneDrive:** keep the venv outside OneDrive (`UV_PROJECT_ENVIRONMENT`) to
  avoid file-lock failures during `uv sync`; see README.
