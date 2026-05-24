# Phase 4 — Multimodal Transformer

## Goal
Replace the LightGBM baseline with a transformer that can ingest both numeric and text modalities and learn its own representations. Use those learned representations to re-do the Phase 3 clustering and compare.

## Prerequisites
- Phases 1–3 complete
- GPU access (RTX 3060+ or cloud) recommended; CPU-only is feasible but slow

## Steps

### 4.1 Pick a base architecture
Two strong starting points:
- **PatchTST** (Nie et al. 2023) — patches a series, treats each patch as a token. Simpler, robust.
- **iTransformer** (Liu et al. 2023) — treats each *variate* as a token; cross-attention learns variable relationships. Better for multi-asset modeling.

Recommendation: start with iTransformer for the multi-asset case, fall back to PatchTST if training is unstable.

### 4.2 Sequence dataset preparation
`src/metals/data/sequences.py`:
- For each prediction timestamp `t`, build a `(lookback × n_features)` numeric tensor and a text-embedding tensor for the same window
- Lookback: start at 60 trading days (~3 months)
- Implement as a `torch.utils.data.Dataset` reading from DuckDB and caching to disk

### 4.3 Implement iTransformer
Options:
- Use the official repo at github.com/thuml/iTransformer with light modifications
- Reimplement in `src/metals/models/itransformer.py` for educational value (it's not a large model)

Either way: write unit tests verifying shapes for a toy 4-variate input.

### 4.4 Multi-horizon, multi-task objective
Single model, multiple output heads:
- Head A: t+1d return (regression)
- Head B: t+5d return (regression)
- Head C: t+20d realized vol (regression)
- Head D: t+5d directional move > 1σ (binary classification)

Loss = weighted sum of MSE / BCE per head. Tune weights on validation, starting at equal weights and rebalancing toward heads that under-converge.

### 4.5 Train numeric-only model
- Use walk-forward splits from Phase 1
- Log every split's metrics to the eval harness
- Compare to LightGBM baseline
- Honest expectation: a small but real improvement on vol prediction; near-parity on return prediction (returns are mostly noise at this horizon)

### 4.6 Add news features via concatenation
Concat the daily text feature vector (mean embedding + topic prevalences) to numeric features at each timestep, then retrain. Compare lift vs numeric-only.

### 4.7 Cross-attention fusion
Separate numeric and text encoders, cross-attention layers letting numeric tokens attend to recent text and vice versa. Reference: MulT (Tsai et al. 2019) or LXMERT-style fusion.

Compare three variants in the eval harness:
- Numeric-only
- Numeric + concat text
- Numeric + cross-attention text

Honest expectation: cross-attention beats concat by a small margin if data is sufficient, equals or loses if not.

### 4.8 Hyperparameter tuning
Use Optuna (or grid for simplicity). Tune:
- Lookback length (30, 60, 120 days)
- Model depth (2, 3, 4 layers)
- Attention heads (4, 8)
- Learning rate (1e-4, 5e-4, 1e-3)
- Weight decay
- Loss weights between heads

Tune on validation. Never on test. If you tune any hyperparameter on the test set, you've contaminated it — designate a new test set or accept reduced credibility.

### 4.9 Re-cluster on learned representations
- Extract penultimate-layer activations for every date
- Re-run UMAP + HDBSCAN (matching Phase 3 hyperparameters where possible)
- Build a cluster comparison: contingency table between hand-engineered (Phase 3) and learned (Phase 4) clusters
- Examine cases where they disagree — these are interesting

Are the learned clusters more interpretable? More predictive of forward returns? Worse on either dimension? Don't assume the transformer wins.

### 4.10 Attribution
- Numeric: integrated gradients via `captum`
- Text: attention rollout to identify which historical headlines the model attended to for each prediction
- Generate per-prediction attribution reports for:
  - 20 randomly chosen dates
  - 5 high-impact event days (largest |return|)
  - 5 dates the model got most wrong

### 4.11 Honest comparison vs baseline
`results/phase4_transformer.md`:
- Side-by-side metrics: LightGBM vs numeric-only transformer vs concat vs cross-attention
- Per metal, per horizon, per target
- Compute time and inference cost
- Did the transformer earn its complexity? Be honest. In financial ML this answer is often "barely" or "no."

## Deliverables
- Trained iTransformer (numeric and multimodal variants)
- Cross-attention fusion implementation
- Learned-representation cluster taxonomy
- Per-prediction attribution reports
- `results/phase4_transformer.md` with honest comparison

## Common pitfalls
- Insufficient walk-forward discipline. Each split must fully retrain; reusing weights across splits leaks future information through the optimizer state.
- Confusing "test" with "out-of-sample." If anything was tuned using the test set, it isn't OOS anymore.
- Loss-weight instability with multi-task heads. If training diverges, temporarily reduce the dominant head's weight until balance returns.
- Reading too much into transformer-discovered clusters before Phase 5's causal validation.
- Calling success on small-sample-size lift. With ~250 trading days/year and noisy returns, "the transformer improved IC by 0.02" is barely distinguishable from random.
- Forgetting to set deterministic seeds where it matters for reproducibility (`torch.manual_seed`, `np.random.seed`, `random.seed`, `torch.use_deterministic_algorithms(True)` where feasible).
