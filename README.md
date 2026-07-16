# metals

Quantitative research on drivers of precious metals (gold, silver, platinum, palladium) prices, combining classical statistical and causal-inference methods with machine learning. Since July 2026 the program serves **AMC Company** — a small dealer that buys scrap precious metals (with assay) and deals in gold coin and specie — whose operating decisions (spread floors, event hedging, inventory risk, demand intelligence) set the research priorities.

## Roadmap

The work is organized into sequential phases. See `plans/00_roadmap.md` for an overview; each phase has its own step-by-step plan in `plans/`.

| Phase | Subject |
|-------|---------|
| 0 | Scoping and setup |
| 1 | Price foundation and LightGBM baseline |
| 2 | Macro events and local projections |
| 3 | Text data and unsupervised scenario clustering |
| 4 | Multimodal transformer |
| 5 | Causal ML and method triangulation |
| 6 | Validation and writeup |
| 7 | AMC program: data acquisition and dealer decision support |

## Setup

```bash
# Install uv if not present (https://docs.astral.sh/uv/getting-started/installation/)
pip install uv

# Pin Python and create the environment
uv python pin 3.11
uv sync --extra dev

# Configure secrets
cp .env.example .env
# Edit .env to add at least FRED_API_KEY (free signup).

# Apply database migrations
uv run python -m metals.data.migrations.runner

# Run tests
uv run pytest
```

### Windows + OneDrive venv location

If the project lives inside a OneDrive-synced folder (e.g.
`C:\Users\<you>\OneDrive\...`), put the virtual environment **outside** OneDrive
to avoid file-lock errors during `uv sync` (OneDrive holds files open while
syncing, which breaks atomic dist-info replacement). One-time setup:

```powershell
# In any new PowerShell session for this project
$env:UV_PROJECT_ENVIRONMENT = "C:\Users\mcmur\.venvs\amc-research"

uv sync --extra dev --link-mode=copy
```

Every subsequent `uv` command in the session inherits the variable, so the same
project-local venv is reused. Do **not** set this user-scoped — that breaks
other projects.

## Reproducing the results

Three entry points rebuild the research from a clean checkout. All model steps are
seed-pinned, so a rerun reproduces the published numbers.

```bash
# 1. Rebuild the Phases 0-6 research inputs (idempotent; per-source failure isolation).
#    The 7 core sources are seconds-to-minutes and need at most FRED_API_KEY.
uv run python -m metals.refresh
#    GDELT (the ~63M-row text corpus) is opt-in: a billed TB-scale BigQuery scan
#    that needs GOOGLE_APPLICATION_CREDENTIALS and an explicit window.
uv run python -m metals.refresh --gdelt --start 2015-01-01 --end 2026-06-30

# 2. Retrain the models in dependency order (Phase 1 -> 3 -> 5 -> 6).
uv run python -m metals.train --dry-run    # show the ordered plan + gate decisions
uv run python -m metals.train --all        # CPU steps: the Option-C pipeline Phase 6 validated

# 3. Export / re-import the eval-harness run records (they ship as Parquet, not the 54 GB DB).
uv run python scripts/export_harness.py            # -> results/harness_export/
uv run python scripts/export_harness.py --load     # re-import into a fresh DB
```

**No model weights are shipped, by design.** The models are cheap and deterministic
to refit — LightGBM (seed-pinned), the DoubleML/SVAR estimators (`random_state`/
`seed=42`), and the UMAP+HDBSCAN clustering (`random_state=42`) all regenerate from
the seed-pinned configs in seconds to minutes, so `metals.train` reproduces them
rather than loading a checkpoint. What *is* irreplaceable travels with the repo: the
eval-harness records (`results/harness_export/*.parquet`, re-derivable only by
re-running every model) and the scenario master table (`results/phase5_scenario_master.csv`).
Exact dependency versions are frozen in the tracked `uv.lock`.

**GPU note.** The reproduced pipeline is *Option C* throughout — tone/theme text
features, no neural embeddings — which is exactly what Phase 6 found best out of
sample (embeddings and regime/sentiment features hurt OOS). So `metals.train --all`
is complete on a CPU box. The neural embedding stage runs only under `--with-gpu`
on a machine with CUDA and is exploratory, not part of the shipped result.

**The Phase 7.1 AMC collectors are not part of this reproduction.** They capture
live business data, are separately governed (`scripts/run_collectors.py`,
`plans/phase_7_amc_program.md` §7.7), and — following the 2026-07-16 Terms-of-Use
audit — are mostly barred or manual; `metals.refresh` refuses them with a pointer.

## Maintenance

### Reclaiming database space

The `headlines` table (GDELT GKG, ~14M rows) dominates the DuckDB file. Migration
005 drops a redundant per-row copy of the article URL from the schema, but DuckDB
does **not** shrink the data file in place on `DROP COLUMN`. To actually reclaim
the bytes, rebuild a densely-packed copy with `scripts/compact_headlines.py`:

```bash
# Dry run: writes data/processed/metals.duckdb.compact and reports bytes saved
uv run python scripts/compact_headlines.py

# Rebuild and swap in place (keeps a timestamped .bak alongside the original)
uv run python scripts/compact_headlines.py --replace
```

The source database is opened read-only and every table's row count is verified
before anything is swapped. Run this **locally** — a OneDrive-synced DB is too
large to rebuild over a network mount.

## Project layout

```
amc/
├── data/
│   ├── raw/         # original downloads (gitignored)
│   ├── processed/   # cleaned, joined; DuckDB lives here
│   └── features/    # ML-ready feature matrices
├── src/metals/
│   ├── data/        # ingestion, DB, migrations
│   ├── features/    # feature engineering
│   ├── models/      # baseline, statistical, transformer
│   └── eval/        # harness, walk-forward CV
├── notebooks/       # exploratory only
├── configs/         # YAML run configs
├── plans/           # phased research plan
├── results/         # write-ups and outputs (incl. harness_export/ Parquet)
├── licensing/       # data-source licence-request drafts (Phase 7.1 ToU)
├── tests/
├── journal.md       # research log (append after every session)
└── pyproject.toml
```

## Conventions

- All timestamps stored in UTC.
- All feature pipelines must pass the leakage check before being used in training.
- Every model run logs to the evaluation harness (`src/metals/eval/harness.py`).
- Walk-forward CV only. Never a random split.
- A journal entry follows every working session.
