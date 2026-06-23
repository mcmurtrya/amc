# metals

Quantitative research on drivers of precious metals (gold, silver, platinum, palladium) prices, combining a multimodal transformer with classical statistical and causal-inference methods.

## Roadmap

The work is organized into seven phases. See `plans/00_roadmap.md` for an overview; each phase has its own step-by-step plan in `plans/`.

| Phase | Subject |
|-------|---------|
| 0 | Scoping and setup |
| 1 | Price foundation and LightGBM baseline |
| 2 | Macro events and local projections |
| 3 | Text data and unsupervised scenario clustering |
| 4 | Multimodal transformer |
| 5 | Causal ML and method triangulation |
| 6 | Validation and writeup |

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
├── results/         # write-ups and outputs
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
