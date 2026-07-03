"""Tests for per-fold regime features (leak-freedom is the point).

Real UMAP + HDBSCAN fits on tiny two-blob data; skipped when the libraries
are not installed, matching test_models_clustering.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("umap")
pytest.importorskip("hdbscan")

from metals.features.regimes import (  # noqa: E402
    RegimeFeatureConfig,
    build_regime_features,
    purge_days_for,
)
from metals.models.clustering import ClusteringConfig  # noqa: E402

N = 200
SWITCH = 100  # blob A rows [0, SWITCH), blob B rows [SWITCH, N)
BOUNDARY_POS = 160  # train = rows [0, 160): all of A + 60 rows of B

_SMALL = RegimeFeatureConfig(
    clustering=ClusteringConfig(
        umap_n_components=2, umap_n_neighbors=10, hdbscan_min_cluster_size=15
    ),
    target_purge_days=45,
)


def _toy_context(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=N, freq="B")
    X = rng.normal(0.0, 1.0, (N, 6))
    X[SWITCH:] += 8.0  # second, well-separated blob
    return pd.DataFrame(X, index=idx, columns=[f"f{i}" for i in range(6)])


def _boundary(ctx: pd.DataFrame) -> pd.Timestamp:
    return ctx.index[BOUNDARY_POS]


def _onehot_cols(out: pd.DataFrame) -> list[str]:
    return [c for c in out.columns if c.startswith("regime_") and c[7:].isdigit()]


def test_shape_onehots_and_model_ready_columns():
    ctx = _toy_context()
    out = build_regime_features(ctx, _boundary(ctx), config=_SMALL)

    assert out.index.equals(ctx.index)
    # Raw cluster ids are fold-local ordinals and must never be returned —
    # every column is model-ready (see the pre-registered arm-B feature list).
    assert "regime_id" not in out.columns
    onehot_cols = _onehot_cols(out)
    assert len(onehot_cols) >= 2  # both blobs discovered in train
    assert "regime_noise" in out.columns
    hot = out[onehot_cols + ["regime_noise"]]
    # Exactly one active indicator per row => assignments come from the
    # train-fit vocabulary (unseen ids would break the sum).
    np.testing.assert_allclose(hot.sum(axis=1).to_numpy(), 1.0)
    assert out["regime_confidence"].between(0.0, 1.0).all()
    # The two blobs land in different clusters (sanity of the toy setup).
    dom_a = hot.iloc[:SWITCH].mean().idxmax()
    dom_b = hot.iloc[SWITCH:].mean().idxmax()
    assert dom_a != dom_b


def test_purge_days_for_covers_the_target_lookahead():
    # h + w - 1 trading days converted to calendar at 7/5 plus holiday slack.
    assert purge_days_for(5, 20) == 44  # the pre-registered primary (<= 45)
    assert purge_days_for(20, 20) == 65  # 39 trading days span >= 53 calendar
    for h, w in [(1, 5), (5, 20), (20, 20), (60, 20)]:
        trading = h + w - 1
        # Worst case: t trading days span ceil(t * 7/5) calendar days + holidays.
        assert purge_days_for(h, w) > trading * 7 / 5


def test_post_boundary_perturbation_cannot_move_pre_boundary_features():
    """The leak regression demanded by the pre-registered design: nothing after
    the boundary may influence any feature on rows before it."""
    ctx_a = _toy_context()
    ctx_b = ctx_a.copy()
    ctx_b.iloc[BOUNDARY_POS:] += 100.0  # violent post-boundary change

    bound = _boundary(ctx_a)
    out_a = build_regime_features(ctx_a, bound, config=_SMALL)
    out_b = build_regime_features(ctx_b, bound, config=_SMALL)

    pre = ctx_a.index[ctx_a.index < bound]
    pd.testing.assert_frame_equal(out_a.loc[pre], out_b.loc[pre])


def test_target_encoding_is_purged():
    """Target values inside the purge window (or after the boundary) must not
    move ``regime_target_mean``; earlier train targets must (control)."""
    ctx = _toy_context()
    bound = _boundary(ctx)
    purge_cut = bound - pd.Timedelta(days=_SMALL.target_purge_days)
    target = pd.Series(np.linspace(1.0, 2.0, N), index=ctx.index)

    base = build_regime_features(ctx, bound, target=target, config=_SMALL)

    poisoned = target.copy()
    poisoned.loc[poisoned.index >= purge_cut] = 1e6  # purge window + val/test
    out = build_regime_features(ctx, bound, target=poisoned, config=_SMALL)
    pd.testing.assert_series_equal(out["regime_target_mean"], base["regime_target_mean"])

    control = target.copy()
    control.iloc[:10] = 1e6  # deep inside the (purged) train window
    out_c = build_regime_features(ctx, bound, target=control, config=_SMALL)
    assert not np.allclose(
        out_c["regime_target_mean"].to_numpy(), base["regime_target_mean"].to_numpy()
    )


def test_boundary_is_exclusive_for_encoding():
    ctx = _toy_context()
    bound = _boundary(ctx)
    cfg = RegimeFeatureConfig(clustering=_SMALL.clustering, target_purge_days=0)
    target = pd.Series(np.ones(N), index=ctx.index)

    base = build_regime_features(ctx, bound, target=target, config=cfg)
    spiked = target.copy()
    spiked.loc[bound] = 1e6  # exactly at the (exclusive) boundary
    out = build_regime_features(ctx, bound, target=spiked, config=cfg)
    pd.testing.assert_series_equal(out["regime_target_mean"], base["regime_target_mean"])


def test_target_encoding_falls_back_to_global_mean():
    """A cluster with no usable train target rows gets the global train mean,
    never NaN."""
    ctx = _toy_context()
    bound = _boundary(ctx)
    probe = build_regime_features(ctx, bound, config=_SMALL)
    blob_b_col = probe[_onehot_cols(probe)].iloc[SWITCH:].mean().idxmax()
    b_rows = probe[blob_b_col] == 1.0

    target = pd.Series(1.0, index=ctx.index)
    target.loc[b_rows] = np.nan  # blob B contributes nothing to the encoding
    out = build_regime_features(ctx, bound, target=target, config=_SMALL)

    assert not out["regime_target_mean"].isna().any()
    np.testing.assert_allclose(out.loc[b_rows, "regime_target_mean"].to_numpy(), 1.0)


def test_empty_train_window_raises():
    ctx = _toy_context()
    with pytest.raises(ValueError, match="no context rows before"):
        build_regime_features(ctx, ctx.index[0], config=_SMALL)
