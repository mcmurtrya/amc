"""Streaming-invariant test for the Phase 3 aggregate redesign.

The streaming aggregate processes the corpus one calendar month at a time and
concatenates per-chunk outputs. That is only correct if every calendar day lives
wholly within one month-chunk, so a day's aggregate is never split. This test
locks that invariant: aggregating a multi-month frame in one shot must equal
aggregating each month separately and concatenating.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metals.features.text_daily import aggregate_daily

SCALAR_COLS = ["timestamp_utc", "metal", "n_articles",
               "embedding_dispersion", "mean_tone_overall"]


def _toy_frame():
    ts = pd.to_datetime([
        "2022-01-05 10:00", "2022-01-05 12:00", "2022-01-20 09:00",
        "2022-02-03 11:00", "2022-02-03 15:00", "2022-02-28 23:00",
    ])
    df = pd.DataFrame({
        "timestamp_utc": ts,
        "themes_list": [
            ["ECON_GOLDPRICE"], ["ECON_CENTRALBANK"], ["ECON_INFLATION"],
            ["WB_1699_METAL_ORE_MINING"], ["ECON_GOLDPRICE"], ["SANCTIONS"],
        ],
        "tone_overall": [1.0, -2.0, 0.5, np.nan, 3.0, -1.0],
        "tone_positive": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "tone_negative": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    })
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(df), 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)  # L2-normalized, as in prod
    return df, emb


def test_monthly_chunking_matches_full_aggregate():
    df, emb = _toy_frame()

    full = (aggregate_daily(df, embeddings=emb)
            .sort_values(["timestamp_utc", "metal"]).reset_index(drop=True))

    months = df["timestamp_utc"].dt.to_period("M")
    parts = []
    for m in sorted(months.unique()):
        mask = (months == m).to_numpy()
        sub = df[mask].reset_index(drop=True)
        parts.append(aggregate_daily(sub, embeddings=emb[mask]))
    streamed = (pd.concat(parts, ignore_index=True)
                .sort_values(["timestamp_utc", "metal"]).reset_index(drop=True))

    pd.testing.assert_frame_equal(
        full[SCALAR_COLS], streamed[SCALAR_COLS], check_dtype=False, atol=1e-6,
    )


def test_dispersion_closed_form_matches_centroid_norm():
    """Streaming dispersion relies on the identity dispersion = 1 - ||mean||
    for L2-normalized embeddings; verify aggregate_daily honors it."""
    df, emb = _toy_frame()
    out = aggregate_daily(df, embeddings=emb)
    # gold on 2022-01-05 has 2 articles (ECON_GOLDPRICE + ECON_CENTRALBANK)
    row = out[(out["metal"] == "gold")
              & (out["timestamp_utc"] == pd.Timestamp("2022-01-05"))].iloc[0]
    e = emb[[0, 1]]
    expected = 1.0 - np.linalg.norm(e.mean(axis=0))
    assert row["embedding_dispersion"] == np.float32(expected).item() or \
        abs(row["embedding_dispersion"] - expected) < 1e-5
