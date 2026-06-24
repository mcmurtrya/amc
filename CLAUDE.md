# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

`metals` — quantitative research on the drivers of precious-metals prices (gold,
silver, platinum, palladium). A phased program (see `plans/`) that combines
classical stats, ML, and causal inference. Python ≥ 3.11, managed with `uv`,
package source under `src/metals`, storage in DuckDB.

Gold is the primary "first model" target; the other three metals come online once
a pipeline works.

## Environment & commands

This repo uses **`uv`**. Always run code through `uv run` so the project venv and
`src/` path are picked up.

```bash
uv sync --extra dev                              # install incl. dev tools
cp .env.example .env                             # then fill in secrets (see below)
uv run python -m metals.data.migrations.runner   # apply DB migrations
uv run pytest                                     # full test suite
uv run pytest tests/test_features_embeddings.py  # a single test module
uv run ruff check src tests                       # lint (line-length 100)
uv run ruff format src tests                      # format
uv run mypy                                        # type-check (files = src/metals)
```

Test baseline: **214 tests, all pass** after `uv sync --extra dev`. `umap-learn`,
`hdbscan`, and `bertopic` are **core** dependencies, so the two `importorskip`-gated
clustering/topic tests **run** (not skip) in a complete install — a green run is
**214 passed / 0 skipped** (~20 min; the UMAP/HDBSCAN tests dominate). You only see
"2 skipped" in a degraded env where those heavy deps failed to import. Note: bare
`uv run pytest` needs the dev extras synced first (`uv sync --extra dev`), otherwise
it falls back to a non-venv `pytest` that can't import `duckdb` (13 collection errors).

## Repo layout

```
src/metals/
  data/        ingestion (prices, fred, gdelt, cot, events, gpr, fomc_surprises),
               db.py (DuckDB conn), migrations/ (numbered .sql + runner.py)
  features/    feature engineering — embeddings, topics, text_daily, context,
               assemble, returns, spreads, macro, leakage, loaders
  models/      lgbm_vol.py (baseline), clustering.py (UMAP+HDBSCAN)
  eval/        harness.py (run logging), cv.py (walk-forward), clusters.py,
               cluster_labeling.py (LLM labels)
scripts/       backfill_gdelt.py, compact_headlines.py, phase1_diagnose.py,
               phase3_pipeline.py (the Phase 3 orchestrator)
configs/       YAML run configs (universe, fred_series, gdelt_themes, fomc_calendar)
plans/         phased research plan; 00_roadmap.md has the live status table
data/          raw/ processed/ (DuckDB lives here) features/  — all gitignored
results/       write-ups, charts, per-cluster CSVs
journal.md     research log — append an entry after every working session
```

## Non-negotiable conventions

These are the project's core discipline. Violating them invalidates results.

- **No leakage, by construction.** Every feature pipeline must pass the leakage
  check (`src/metals/features/leakage.py`) before use in training. For
  window-valued targets the structural guard requires `min_nan_tail = h + w - 1`
  — checking only the target's NaN tail is too weak. A 5-day-vol IC much above
  **~0.2 is a leakage tripwire, not a good result.** Be suspicious, not pleased.
- **UTC everywhere.** All stored timestamps are UTC; keep a single canonical
  as-of timestamp per row.
- **Walk-forward CV only.** Expanding-window splits (`src/metals/eval/cv.py`).
  Never a random split. Correct invariant is *within-split* disjointness (test of
  split i may legitimately appear in train of split i+1).
- **Every model run logs to the harness** (`src/metals/eval/harness.py`):
  register_run, log_predictions, compute_metrics, compare_runs.
- **Lag positioning data correctly.** CFTC COT is Tuesday positioning released
  Friday afternoon — using it as a Tuesday feature is the classic leak.
- **Journal after every session.** Append to `journal.md` (what I did / learned /
  confused me / next).

## Gotchas (learned the hard way)

- **DuckDB — never put `;` inside a `--` comment in a migration.** A semicolon in
  a comment silently truncates whole-file multi-statement execution with no error.
- **DuckDB `DROP COLUMN`** is refused while any index exists on the table (drop &
  recreate the index around it), and does **not** shrink the file in place —
  rebuild a fresh DB to reclaim bytes (`scripts/compact_headlines.py`).
- **`runs` / `run_predictions` tables are created lazily by the harness**, not by a
  migration. Rebuilding from migrations alone loses them.
- **Prefer FRBSTL-calculated FRED series** (e.g. `BAA10Y`, `T10Y2Y`, `T10YIE`)
  over licensed third-party indices, which can be silently truncated to a short
  license window (this cost the HY-OAS feature; the FRED audit now catches it).

## Embeddings & cache (Phase 3)

- Default model `all-MiniLM-L6-v2` (384-dim). fp16 on disk, fp32 returned.
- Cache is **chunked Parquet**: one shard per `sha256(text)[:3]` → 4,096 shards.
  ~48 GB for the full 63.3M-row corpus.
- **Cache lives OUTSIDE the repo**: `~/.cache/metals/embeddings` (Linux/WSL),
  `%LOCALAPPDATA%\metals\embeddings` (Windows). Override with
  `METALS_EMBEDDING_CACHE_DIR`. The resolver warns if the path contains a
  sync-folder token (OneDrive/Dropbox/GoogleDrive/iCloud/Box) — never put the
  cache in a synced folder.
- **Known limitation:** GDELT GKG carries no headline text, so the embed source
  is `document_identifier` (the article URL), i.e. a URL-based heuristic. Stable
  across runs but not true headline semantics — don't silently "fix" this without
  discussing the downstream impact.

## Phase 3 pipeline

`scripts/phase3_pipeline.py` runs 8 ordered stages:
`gdelt → embed → aggregate → topics → context → cluster → analyze → label`.

```bash
# full run
uv run python scripts/phase3_pipeline.py --start 2015-02-18 --end 2026-05-12
# one stage
uv run python scripts/phase3_pipeline.py --only embed
# resume from a stage onward
uv run python scripts/phase3_pipeline.py --resume-from context
```

Flags: `--target-metal {gold,silver,platinum,palladium}`, `--train-until ISO`,
`--min-topic-size`, `--nr-topics`, `--model-version`, `--llm-model`. Heavy
artifacts (BERTopic model, clustering pipeline pickle, embeddings cache) persist
across invocations. The `label` stage is gated on `ANTHROPIC_API_KEY`.

## Environment variables

- `FRED_API_KEY` — required for FRED ingestion (free).
- `GOOGLE_APPLICATION_CREDENTIALS` — path to GCP service-account JSON for GDELT
  BigQuery (Phase 3).
- `ANTHROPIC_API_KEY` — enables the Phase 3 `label` stage.
- `METALS_DB_PATH` — override the DuckDB path (default
  `data/processed/metals.duckdb`).
- `METALS_EMBEDDING_CACHE_DIR` — override the embeddings cache location.

## Where things stand (2026-06-23)

Phases 0–2 complete. Phase 2 headline result: hawkish-FOMC IRF ≈ −1.5% on gold at
h=5, sign-consistent across Au/Ag/Pt. Phase 3 is **code-complete**, the GDELT
backfill is done (**63.3M `headlines` rows**), the 23.8 GB DuckDB has been migrated
to WSL and verified intact, and the **embedding pass is the next step**:

```bash
uv run python scripts/phase3_pipeline.py --only embed   # GPU, ~48 GB cache, ~6–12 h on a 6 GB A1000, $0
```

Phases 4–6 not started. See `plans/00_roadmap.md` for the full status table and
per-phase "As-built notes".

## Working copy

The canonical working copy is this WSL repo (`/home/mcmur/projects/amc`). The
OneDrive mirror is backup only — OneDrive's sync engine corrupted files during
earlier rewrites (mid-content truncation / NUL bytes). Keep the canonical copy on
WSL ext4; never let the embeddings cache or DuckDB file live in a synced folder.
