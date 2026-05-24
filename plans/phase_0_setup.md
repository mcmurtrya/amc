# Phase 0 — Scoping and Setup

## Goal
Set up a reproducible project foundation: environment, storage, evaluation harness, and habits that will compound across the project's lifetime. Skipping this phase will cost more time later than doing it.

## Prerequisites
- Python 3.11+ installed
- Git installed
- A code editor (VS Code recommended)
- A FRED API key (free at https://fred.stlouisfed.org/docs/api/api_key.html)
- A Google Cloud account, deferred until Phase 3

## Steps

### 0.1 Create the project directory structure
```
metals-research/
  data/
    raw/         # original downloads, never modified
    processed/   # cleaned, deduplicated, joined
    features/    # ML-ready feature matrices
  notebooks/     # exploratory analysis only
  src/metals/
    __init__.py
    data/        # ingestion modules
    features/    # feature engineering
    models/      # model code
    eval/        # evaluation harness
  configs/       # YAML configs for runs
  results/       # model outputs, metrics, write-ups
  plans/         # this folder
  tests/
  journal.md
  pyproject.toml
  README.md
  .gitignore
```

Create with `mkdir -p` once and commit the empty skeleton.

### 0.2 Initialize the Python environment with uv
Install uv (`curl -LsSf https://astral.sh/uv/install.sh | sh` or `pip install uv`). Then:
```
uv init --package
uv python pin 3.11
```

### 0.3 Install core dependencies
```
uv add pandas polars duckdb pyarrow numpy scipy scikit-learn lightgbm \
       statsmodels linearmodels arch \
       torch transformers sentence-transformers \
       hdbscan umap-learn bertopic \
       econml doubleml \
       yfinance fredapi requests \
       matplotlib seaborn plotly \
       pyyaml python-dotenv tqdm
uv add --dev pytest jupyterlab ipykernel ruff mypy
```
Smoke test: `uv run python -c "import pandas, torch, transformers, lightgbm, statsmodels; print('OK')"`.

### 0.4 Configure environment variables
Create `.env` (gitignored):
```
FRED_API_KEY=...
GOOGLE_APPLICATION_CREDENTIALS=...   # for Phase 3
```
Load via `python-dotenv` at the top of any script that needs credentials.

### 0.5 Set up DuckDB as canonical storage
Create `src/metals/data/db.py` exposing `get_connection()` that returns a connection to `data/processed/metals.duckdb`. Use SQL migration files in `src/metals/data/migrations/`, numbered `001_…sql`, `002_…sql`. Write a small migration runner that tracks applied migrations in a `_schema_migrations` table.

Initial schema (migration 001):
- `prices(timestamp_utc, ticker, open, high, low, close, volume, source)`
- `macro(timestamp_utc, series_id, value, source)`
- `events(timestamp_utc, event_type, metadata JSON, source)`

### 0.6 Build the evaluation harness scaffold
Create `src/metals/eval/harness.py`. Minimum interface:
- `register_run(name, model_type, config_dict) -> run_id`
- `log_prediction(run_id, timestamp, ticker, horizon, prediction, actual)`
- `compute_metrics(run_id) -> dict` (RMSE, MAE, IC, directional hit rate, Diebold-Mariano vs benchmark)
- `compare_runs(run_ids) -> DataFrame`

Persist to DuckDB tables `runs` and `run_predictions`. Every future model writes here.

### 0.7 Walk-forward CV utility
`src/metals/eval/cv.py` with `walk_forward_splits(timestamps, train_start, val_period_days, test_period_days, n_splits)` yielding `(train_idx, val_idx, test_idx)` tuples. Write 3–5 pytest tests covering edge cases (single-day windows, exactly-fitting periods, leakage between val and test).

### 0.8 Git, .gitignore, and a first commit
```
git init
git add .
git commit -m "Phase 0: project scaffold"
```
`.gitignore` should exclude `data/raw/`, `data/processed/*.duckdb`, `.env`, `__pycache__/`, `.ipynb_checkpoints/`, `*.parquet`, `.uv-cache/`. Raw downloads do not belong in git.

### 0.9 Configure linting and typing
Add `[tool.ruff]` (line length 100) and `[tool.mypy]` (strict on `src/metals/`) to `pyproject.toml`. Run `uv run ruff check src/` and `uv run mypy src/` — they should pass on empty modules.

### 0.10 Initialize the research journal
Create `journal.md` with a date-stamped template:
```
## YYYY-MM-DD
### What I did
### What I learned
### What confused me
### Next session
```
Write in it after every working session. The journal is the single biggest determinant of how much you actually learn.

### 0.11 Validation checklist before moving on
- `uv run python -c "import metals"` works
- DuckDB connection function works and migrations apply cleanly
- Eval harness can register a run, log a fake prediction, and compute metrics
- Walk-forward CV tests pass
- `.env` is gitignored
- Empty repo committed

## Deliverables
- Working Python environment with pinned versions
- DuckDB with migration framework
- Evaluation harness (empty but functional)
- Walk-forward CV utility with tests
- Linting configured
- Journal template
- First git commit

## Common pitfalls
- Skipping the eval harness because it feels premature — by Phase 4 you'll have eight models and no way to compare them honestly.
- Putting credentials in code instead of `.env`.
- Over-engineering the DuckDB schema. Add columns when you need them, not preemptively.
- Mixing POSIX and Windows paths in shared code. Use `pathlib.Path` everywhere and avoid string joins.
- Treating the journal as optional. It isn't.
