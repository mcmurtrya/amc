# Phase 6 — Validation and Writeup

## Goal
Stress-test every model and scenario claim against a hold-out year that was never touched during development, and produce a clean record of methodology, findings, and limitations.

## Prerequisites
- All prior phases complete
- A 12-month hold-out window of recent data that has not been used for any training, tuning, clustering, or threshold selection

## Steps

### 6.1 Freeze the hold-out
Designate the most recent 12 months as the hold-out. Verify by tracing every dataset and feature pipeline:
- No model was trained on hold-out dates
- No cluster model was fit on hold-out dates
- No threshold (e.g., "top tercile") was computed using hold-out dates
- No hyperparameter was tuned using any hold-out data

If you discover the hold-out has been contaminated, designate a new hold-out further back and accept that you have less recent OOS coverage.

### 6.2 Score every model on the hold-out
Re-run, without retraining, every model from Phases 1–4:
- LightGBM vol baseline (Phase 1)
- LightGBM with event features (Phase 2 extension)
- Numeric-only transformer (Phase 4)
- Concat multimodal transformer (Phase 4)
- Cross-attention multimodal transformer (Phase 4)

Log all hold-out predictions and metrics to the evaluation harness.

### 6.3 Benchmark suite
For honest comparison, score these as well:
- Random walk: `pred_{t+h} = value_t`
- Historical unconditional mean
- GARCH(1,1) for vol forecasting (use the `arch` package)
- VAR(2) on the same numeric features
- "Sentiment-only": LightGBM using only news features from Phase 3

The transformer should clearly beat random walk and unconditional mean. The question is whether it beats GARCH and LightGBM by enough to justify its cost.

### 6.4 Lift tables
Per metal, per horizon, compute:
- RMSE relative to random walk
- Information coefficient
- Diebold-Mariano test of forecast accuracy vs LightGBM baseline
- 95% bootstrap CI on RMSE (resample dates with replacement)

Present as a clear table in `results/phase6_validation.md`.

### 6.5 Re-estimate scenarios on hold-out
For each scenario in `scenario_master.parquet`:
- Identify hold-out dates where the scenario fires
- Compare realized forward returns to the training-period ATE
- Report sign agreement (binary: same sign as training) and magnitude ratio (hold-out point estimate / training ATE)

With only 12 months of OOS data most scenarios won't fire enough times for statistical significance. That's expected. The right question is "do the signs hold up?" not "are they significant?"

### 6.6 Component ablation
Re-train the transformer on training data only, dropping one component at a time:
- No news features
- No COT positioning
- No GPR index
- No event indicators
- No cross-attention (concatenation only)
- Reduced lookback (30 instead of 60 days)

For each variant, evaluate on hold-out and compute lift contribution = `OOS_IC(full) − OOS_IC(ablated)`. This is the most honest "what mattered" answer.

### 6.7 Limitations analysis
Write a one-paragraph honest assessment for each potential failure mode:
- **Data quality**: timestamp accuracy, look-ahead leakage risk, survivorship in news coverage
- **Identification**: where causal claims rest on assumptions (which assumptions, how plausible)
- **Regime instability**: scenarios that depend on regime (e.g., post-2022 inflation dynamics) and what would invalidate them
- **Modeling choices**: hyperparameter sensitivity, architectures not tried (state-space models, mixed-frequency VAR, BERT variants)
- **External validity**: would the methodology transfer to oil, copper, equities? What's gold-specific?
- **Statistical power**: number of effective observations per scenario, multiple-comparisons concerns

### 6.8 Methodology write-up
Structure (target: 15–25 pages):
1. Research question and approach
2. Data sources and processing — including provenance, frequency, known issues
3. Feature engineering
4. Modeling architecture (baseline through transformer)
5. Scenario identification — three methods, why three, what each adds
6. Causal estimation
7. Validation and ablation
8. Limitations
9. Future work

### 6.9 Findings write-up
Lead with the scenarios, not the model. The model is the tool; the scenarios are the substance.
- What scenarios reliably move which metals (with quantified effects and CIs)
- How the three methods agreed or disagreed
- Cross-metal patterns confirmed and surprised by
- Open questions
- One paragraph each on the most counterintuitive finding, the most robust finding, and the most fragile-seeming finding

### 6.10 Reproducibility package
Final deliverable for someone (including future-you) coming back in a year:
- README with setup instructions
- Single-command data refresh (`uv run python -m metals.refresh`)
- Single-command model retrain (`uv run python -m metals.train --all`)
- Pre-trained model weights checkpoint (Git LFS or external storage)
- Frozen `pyproject.toml` with exact dependency versions
- Eval harness with all run records
- The scenario master table

### 6.11 Cleanup pass
- Review `journal.md` end-to-end and extract a "lessons learned" section to the top
- Remove dead code, stale notebooks, and abandoned experiments — keep them in a `_archive/` folder rather than deleting outright
- Tag the final git commit with `v1.0`

## Deliverables
- Hold-out evaluation report with metrics for every model
- Component ablation table
- Updated scenario master table with hold-out validation columns
- Final methodology + findings + limitations write-up
- Reproducibility package
- `results/phase6_validation.md`

## Common pitfalls
- Looking at the hold-out before final evaluation. Even once. If you did, designate a new hold-out — don't pretend.
- Highlighting the impressive findings and burying the failures. Failures are where the next project starts.
- Optimizing the write-up before the analysis finishes. The write-up should reflect what you found, not what you hoped to find.
- Conflating "did not reject H0" with "no effect." Especially with one year of OOS data.
- Forgetting to write the limitations section honestly. This is the section that distinguishes research from PR.
- Not tagging a clean commit at the end. Future-you will want a known-good snapshot to return to.
