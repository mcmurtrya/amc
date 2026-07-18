"""Low-rank probing playbook for Phase 8 Stage A (§4 / Appendix B).

Encoder-agnostic probes over a frozen daily representation ``Z`` (the
``factor_ssl`` scores, the ``build_context`` vector, or any ``(T x d)`` matrix).
Four load-bearing pieces:

- :func:`incremental_ic` — **the tautology guard.** View A already carries
  realized vol / returns / spread z-scores, so a news factor that "predicts"
  forward vol may just be recovering price structure. This residualizes *both*
  the target and the news score on the full price panel (train-fit residualizer,
  applied forward) and measures the IC on the residual — the only honest test of
  incremental news content.
- :func:`linear_probe` — a Ridge / logistic head (never nonlinear on ~2k rows —
  that is the Phase-6 failure mode), fit on train, regularization tuned on val,
  scored on test.
- :func:`block_permutation_pvalue` — the autocorrelation-aware null (block-shuffle
  one series against the other); a naive iid shuffle is anti-conservative on
  ~40 effective regimes.
- :func:`block_bootstrap_ci` — moving-block bootstrap CI for a paired statistic.

Pure functions on numpy/pandas. Targets are constructed by the caller
(``assemble.build_feature_matrix`` / ``shift_target``); harness logging is wired
by the pipeline driver. Integer position arrays (``train_idx`` / ``test_idx``)
index rows, matching ``eval.cv.Split``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

FloatArray = np.ndarray
StatFn = Callable[..., float]


def _as_1d(x: pd.Series | np.ndarray | Sequence[float]) -> FloatArray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _as_2d(x: pd.DataFrame | np.ndarray) -> FloatArray:
    arr = x.to_numpy(dtype=np.float64) if isinstance(x, pd.DataFrame) else np.asarray(x, np.float64)
    return arr.reshape(arr.shape[0], -1)


def information_coefficient(
    pred: pd.Series | np.ndarray | Sequence[float],
    actual: pd.Series | np.ndarray | Sequence[float],
    method: str = "spearman",
) -> float:
    """Rank (default) or linear correlation between prediction and outcome.

    NaN pairs are dropped; returns ``nan`` if fewer than 3 finite pairs remain.
    """
    p = pd.Series(_as_1d(pred))
    a = pd.Series(_as_1d(actual))
    mask = p.notna() & a.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(p[mask].corr(a[mask], method=method))


def residualize_on(
    y: pd.Series | np.ndarray | Sequence[float],
    x_control: pd.DataFrame | np.ndarray,
    train_idx: np.ndarray | Sequence[int],
) -> FloatArray:
    """Return ``y`` minus its OLS projection on ``x_control``.

    The regression is fit on ``train_idx`` rows (where ``y`` is finite) and
    applied to every row, so the residualizer carries no look-ahead. ``x_control``
    must be finite (impute first). Rows with NaN ``y`` residualize to NaN.
    """
    from sklearn.linear_model import LinearRegression

    yv = _as_1d(y)
    xv = _as_2d(x_control)
    if xv.shape[0] != yv.shape[0]:
        raise ValueError("residualize_on: y and x_control must have equal length.")
    tr = np.asarray(train_idx, dtype=int)
    fit_rows = tr[np.isfinite(yv[tr])]
    if fit_rows.size < 2:
        raise ValueError("residualize_on: fewer than 2 finite train targets.")
    model = LinearRegression().fit(xv[fit_rows], yv[fit_rows])
    resid = yv - model.predict(xv)
    resid[~np.isfinite(yv)] = np.nan
    return resid


def incremental_ic(
    y: pd.Series | np.ndarray | Sequence[float],
    news_score: pd.Series | np.ndarray | Sequence[float],
    x_price: pd.DataFrame | np.ndarray,
    train_idx: np.ndarray | Sequence[int],
    test_idx: np.ndarray | Sequence[int],
    method: str = "spearman",
) -> float:
    """Incremental IC of a news score over the full price panel, on test rows.

    Residualizes both ``y`` and ``news_score`` on ``x_price`` (train-fit), then
    correlates the residuals on ``test_idx``. This is the mandatory guard: it
    strips everything ``x_price`` already explains and asks whether the news
    channel adds anything to the leftover.
    """
    y_res = residualize_on(y, x_price, train_idx)
    s_res = residualize_on(news_score, x_price, train_idx)
    te = np.asarray(test_idx, dtype=int)
    return information_coefficient(s_res[te], y_res[te], method=method)


def _rmse(pred: FloatArray, actual: FloatArray) -> float:
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def linear_probe(
    z: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray | Sequence[float],
    train_idx: np.ndarray | Sequence[int],
    val_idx: np.ndarray | Sequence[int],
    test_idx: np.ndarray | Sequence[int],
    task: str = "reg",
    alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
) -> dict[str, float]:
    """Fit a linear/logistic probe on ``Z``; tune on val, score on test.

    ``task="reg"`` → Ridge, reporting test RMSE and IC vs a persistence-free
    baseline. ``task="bin"`` → L2 logistic, reporting test AUC, balanced accuracy,
    and the positive base rate. Regularization is chosen on ``val_idx`` only.
    Rows with NaN target are dropped within each split.
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    zv = _as_2d(z)
    yv = _as_1d(y)
    tr = np.asarray(train_idx, dtype=int)
    va = np.asarray(val_idx, dtype=int)
    te = np.asarray(test_idx, dtype=int)

    def finite(idx: FloatArray) -> FloatArray:
        return idx[np.isfinite(yv[idx])]

    tr, va, te = finite(tr), finite(va), finite(te)
    if tr.size < 5 or te.size < 3:
        return {"n_train": float(tr.size), "n_test": float(te.size)}

    if task == "reg":
        best_alpha, best_val = alphas[0], np.inf
        for a in alphas:
            m = Ridge(alpha=a).fit(zv[tr], yv[tr])
            v = _rmse(m.predict(zv[va]), yv[va]) if va.size else np.inf
            if v < best_val:
                best_alpha, best_val = a, v
        model = Ridge(alpha=best_alpha).fit(zv[tr], yv[tr])
        pred = model.predict(zv[te])
        return {
            "alpha": float(best_alpha),
            "test_rmse": _rmse(pred, yv[te]),
            "test_ic": information_coefficient(pred, yv[te]),
            "n_train": float(tr.size),
            "n_test": float(te.size),
        }

    if task == "bin":
        yb = (yv > 0).astype(float)
        best_c, best_val = 1.0, -np.inf
        for a in alphas:
            c = 1.0 / a
            m = LogisticRegression(penalty="l2", C=c, max_iter=1000).fit(zv[tr], yb[tr])
            if va.size and len(np.unique(yb[va])) > 1:
                v = roc_auc_score(yb[va], m.predict_proba(zv[va])[:, 1])
            else:
                v = -np.inf
            if v > best_val:
                best_c, best_val = c, v
        model = LogisticRegression(penalty="l2", C=best_c, max_iter=1000).fit(zv[tr], yb[tr])
        proba = model.predict_proba(zv[te])[:, 1]
        auc = roc_auc_score(yb[te], proba) if len(np.unique(yb[te])) > 1 else float("nan")
        return {
            "C": float(best_c),
            "test_auc": float(auc),
            "test_balanced_acc": float(
                balanced_accuracy_score(yb[te], (proba > 0.5).astype(float))
            ),
            "base_rate": float(yb[te].mean()),
            "n_train": float(tr.size),
            "n_test": float(te.size),
        }

    raise ValueError(f"linear_probe: unknown task {task!r} (expected 'reg' or 'bin').")


def _block_indices(n: int, block_len: int, rng: np.random.Generator) -> FloatArray:
    """Moving-block resample of ``range(n)`` (blocks of ``block_len``, wrap-free)."""
    if block_len <= 0:
        raise ValueError("block_len must be >= 1")
    starts_hi = max(1, n - block_len + 1)
    out: list[int] = []
    while len(out) < n:
        s = int(rng.integers(0, starts_hi))
        out.extend(range(s, min(s + block_len, n)))
    return np.asarray(out[:n], dtype=int)


def block_bootstrap_ci(
    stat_fn: StatFn,
    *arrays: pd.Series | np.ndarray | Sequence[float],
    block_len: int = 10,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Moving-block bootstrap CI for ``stat_fn(*arrays)`` (arrays row-aligned)."""
    cols = [_as_1d(a) for a in arrays]
    n = cols[0].shape[0]
    if any(c.shape[0] != n for c in cols):
        raise ValueError("block_bootstrap_ci: all arrays must share length.")
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = _block_indices(n, block_len, rng)
        stats[b] = stat_fn(*[c[idx] for c in cols])
    lo, hi = np.nanquantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def block_permutation_pvalue(
    stat_fn: StatFn,
    x: pd.Series | np.ndarray | Sequence[float],
    y: pd.Series | np.ndarray | Sequence[float],
    block_len: int = 10,
    n_perm: int = 500,
    seed: int = 42,
) -> float:
    """Two-sided p-value for ``|stat_fn(x, y)|`` under a block-permutation null.

    ``y`` is split into contiguous blocks whose order is shuffled (preserving
    within-block autocorrelation) while ``x`` is held fixed, destroying only the
    cross-series alignment. Returns the add-one-smoothed exceedance rate.
    """
    xv, yv = _as_1d(x), _as_1d(y)
    if xv.shape[0] != yv.shape[0]:
        raise ValueError("block_permutation_pvalue: x and y must share length.")
    n = xv.shape[0]
    observed = abs(stat_fn(xv, yv))
    if not np.isfinite(observed):
        return float("nan")
    n_blocks = int(np.ceil(n / block_len))
    blocks = [yv[b * block_len : (b + 1) * block_len] for b in range(n_blocks)]
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        order = rng.permutation(n_blocks)
        y_perm = np.concatenate([blocks[b] for b in order])[:n]
        if abs(stat_fn(xv, y_perm)) >= observed:
            count += 1
    return (1 + count) / (n_perm + 1)
