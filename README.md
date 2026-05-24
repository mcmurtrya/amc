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
