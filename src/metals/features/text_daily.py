"""Daily aggregation of headline-level text features.

Phase 3 step 3.7. Reads from the ``headlines`` table (populated by
``metals.data.gdelt``), embeds headlines via ``metals.features.embeddings``,
and produces a single shared daily ``market`` news-state row of features:

- ``n_articles``         article count
- ``mean_embedding``     L2-normalised mean of headline embeddings
- ``embedding_dispersion`` mean cosine distance from the centroid
- ``mean_tone_overall``  average V2Tone overall score (and pos/neg variants)

The per-metal axis was collapsed: GDELT has no per-metal theme except
``ECON_GOLDPRICE``, which never occurs alone, so all four metals received
byte-identical features. We now emit one row/day labelled ``metal == "market"``
(see results/phase3_gdelt_data_assessment.md §1/§7); per-metal differentiation in
the clustering vector comes from price/COT channels in ``metals.features.context``.

Day boundaries are calendar UTC days; aggregation is independent across days.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from metals.data.db import connection

# Mapping from GDELT theme code to one or more metal labels. A theme that
# applies industry-wide (e.g. monetary policy) is mapped to every metal so
# the aggregation captures its influence regardless of which metal is the
# downstream prediction target.
METALS = ("gold", "silver", "platinum", "palladium")

# The per-metal text axis is redundant — every metal-bearing GDELT theme maps to
# all four metals and the one gold-specific theme (ECON_GOLDPRICE) never appears
# alone, so a per-(date, metal) aggregation is byte-identical across metals. Text
# is therefore collapsed to a single shared daily news-state written under this
# sentinel label. See results/phase3_gdelt_data_assessment.md §1/§7.
MARKET = "market"

THEME_TO_METALS: dict[str, tuple[str, ...]] = {
    # Metal-specific
    "ECON_GOLDPRICE": ("gold",),
    # Generic / industry-wide themes affect every metal
    "ECON_CENTRALBANK": METALS,
    "WB_1235_CENTRAL_BANKS": METALS,
    "EPU_POLICY_MONETARY_POLICY": METALS,
    "WB_444_MONETARY_POLICY": METALS,
    "ECON_INTEREST_RATES": METALS,
    "EPU_POLICY_INTEREST_RATES": METALS,
    "WB_1125_INTEREST_RATE_POLICY": METALS,
    "ECON_INFLATION": METALS,
    "WB_442_INFLATION": METALS,
    "WB_1164_COMMODITY_PRICES_SHOCKS": METALS,
    "WB_1699_METAL_ORE_MINING": METALS,
    "SANCTIONS": METALS,
    "ECON_TRADE_DISPUTE": METALS,
}


@dataclass(frozen=True)
class HeadlineRow:
    timestamp_utc: pd.Timestamp
    headline_id: str
    themes: list[str]
    tone_overall: float | None
    tone_positive: float | None
    tone_negative: float | None
    document_identifier: str


def _parse_themes_field(raw: object) -> list[str]:
    """Tolerantly parse a themes JSON blob or comma-list into a list of codes."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, list):
        return [str(t) for t in raw]
    text = str(raw).strip()
    if not text:
        return []
    try:
        loaded = json.loads(text)
        if isinstance(loaded, list):
            return [str(t) for t in loaded]
    except (ValueError, TypeError):
        pass
    return [t.strip() for t in text.split(",") if t.strip()]


def metals_for_themes(themes: Iterable[str]) -> set[str]:
    """Return the set of metals this article's themes apply to."""
    out: set[str] = set()
    for t in themes:
        for m in THEME_TO_METALS.get(t, ()):
            out.add(m)
    return out


def load_headlines(
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Pull headlines from DuckDB in a date range. Returns a long-format frame."""
    where = ["1=1"]
    params: list = []
    if start is not None:
        where.append("timestamp_utc >= ?")
        params.append(str(pd.Timestamp(start)))
    if end is not None:
        where.append("timestamp_utc <= ?")
        params.append(str(pd.Timestamp(end)))
    sql = (
        f"SELECT timestamp_utc, headline_id, source, themes, article_url, "
        f"tone_overall, tone_positive, tone_negative "
        f"FROM headlines WHERE {' AND '.join(where)} "
        f"ORDER BY timestamp_utc"
    )
    with connection() as conn:
        df = conn.execute(sql, params).fetchdf()
    if df.empty:
        return df
    df["themes_list"] = df["themes"].apply(_parse_themes_field)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    return df


def aggregate_daily(
    headlines: pd.DataFrame,
    embeddings: np.ndarray | None = None,
) -> pd.DataFrame:
    """Aggregate headlines to a single daily ``market`` news-state row.

    Returns columns:
        timestamp_utc, metal, n_articles, mean_tone_overall, mean_tone_positive,
        mean_tone_negative, mean_embedding (np.ndarray | None),
        embedding_dispersion (float | NaN).

    The per-metal axis is collapsed (see module docstring and
    results/phase3_gdelt_data_assessment.md §1/§7): every row is labelled
    ``metal == "market"``. Only articles whose themes map to >= 1 metal (i.e.
    carry >= 1 known theme) are counted, matching the prior behaviour. Caller
    aligns ``embeddings[i]`` with ``headlines.iloc[i]``; pass ``embeddings=None``
    to skip embedding-based aggregates.
    """
    cols = [
        "timestamp_utc",
        "metal",
        "n_articles",
        "mean_tone_overall",
        "mean_tone_positive",
        "mean_tone_negative",
        "mean_embedding",
        "embedding_dispersion",
    ]
    if headlines.empty:
        return pd.DataFrame(columns=cols)

    hl = headlines.reset_index(drop=True)
    # Day-floor the timestamps so groupby keys are calendar days.
    day = pd.to_datetime(hl["timestamp_utc"]).dt.floor("D")
    # Keep only articles carrying >= 1 known theme (=> >= 1 metal), then group by
    # calendar day into one shared 'market' news-state.
    if "themes_list" in hl.columns:
        themes_series = hl["themes_list"]
    elif "themes" in hl.columns:
        themes_series = hl["themes"].apply(_parse_themes_field)
    else:
        themes_series = pd.Series([[] for _ in range(len(hl))], index=hl.index)
    keep_idx = np.flatnonzero(themes_series.apply(lambda t: bool(metals_for_themes(t))).to_numpy())
    if keep_idx.size == 0:
        return pd.DataFrame(columns=cols)
    pairs = pd.DataFrame({"timestamp_utc": day.to_numpy()[keep_idx], "_row_idx": keep_idx})

    out_rows = []
    for ts, g in pairs.groupby("timestamp_utc", sort=True):
        idxs = g["_row_idx"].to_numpy()
        sub = hl.iloc[idxs]
        n = len(sub)
        # Tone — drop NaN, mean the rest
        tone_overall = (
            float(sub["tone_overall"].dropna().mean()) if "tone_overall" in sub else float("nan")
        )
        tone_pos = (
            float(sub["tone_positive"].dropna().mean()) if "tone_positive" in sub else float("nan")
        )
        tone_neg = (
            float(sub["tone_negative"].dropna().mean()) if "tone_negative" in sub else float("nan")
        )
        mean_emb = None
        dispersion = float("nan")
        if embeddings is not None and len(embeddings) > int(idxs.max()):
            E = embeddings[idxs]
            centroid = E.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            mean_emb = centroid.astype(np.float32)
            if n >= 2:
                # Cosine distance from centroid, assuming inputs are L2-normed
                # so that dot-product is cosine similarity. Dispersion is the
                # mean (1 - cos_sim).
                sims = E @ centroid
                dispersion = float(1.0 - sims.mean())
        out_rows.append(
            {
                "timestamp_utc": pd.Timestamp(ts),
                "metal": MARKET,
                "n_articles": n,
                "mean_tone_overall": tone_overall,
                "mean_tone_positive": tone_pos,
                "mean_tone_negative": tone_neg,
                "mean_embedding": mean_emb,
                "embedding_dispersion": dispersion,
            }
        )
    return pd.DataFrame(out_rows, columns=cols)


def upsert_daily(df: pd.DataFrame) -> int:
    """Persist daily aggregates into the ``daily_text_features`` table."""
    if df.empty:
        return 0
    work = df.copy()

    # Pack mean embeddings as bytes and record their dim.
    def _pack(arr):
        if arr is None:
            return None
        return np.asarray(arr, dtype=np.float32).tobytes()

    work["mean_embedding"] = work["mean_embedding"].apply(_pack)
    work["embedding_dim"] = df["mean_embedding"].apply(
        lambda a: int(len(a)) if a is not None else None
    )
    cols = [
        "timestamp_utc",
        "metal",
        "n_articles",
        "mean_embedding",
        "embedding_dim",
        "embedding_dispersion",
        "mean_tone_overall",
        "mean_tone_positive",
        "mean_tone_negative",
    ]
    with connection() as conn:
        conn.register("incoming_text_daily", work[cols])
        conn.execute(
            """
            INSERT INTO daily_text_features
                (timestamp_utc, metal, n_articles, mean_embedding, embedding_dim,
                 embedding_dispersion, mean_tone_overall, mean_tone_positive,
                 mean_tone_negative)
            SELECT timestamp_utc, metal, n_articles, mean_embedding, embedding_dim,
                   embedding_dispersion, mean_tone_overall, mean_tone_positive,
                   mean_tone_negative
            FROM incoming_text_daily
            ON CONFLICT (timestamp_utc, metal) DO UPDATE SET
                n_articles           = EXCLUDED.n_articles,
                mean_embedding       = EXCLUDED.mean_embedding,
                embedding_dim        = EXCLUDED.embedding_dim,
                embedding_dispersion = EXCLUDED.embedding_dispersion,
                mean_tone_overall    = EXCLUDED.mean_tone_overall,
                mean_tone_positive   = EXCLUDED.mean_tone_positive,
                mean_tone_negative   = EXCLUDED.mean_tone_negative
            """
        )
        conn.unregister("incoming_text_daily")
    return len(work)


def load_daily(metal: str | None = None) -> pd.DataFrame:
    """Load aggregated daily text features back out, unpacking embeddings."""
    where = ["1=1"]
    params: list = []
    if metal is not None:
        where.append("metal = ?")
        params.append(metal)
    sql = (
        f"SELECT timestamp_utc, metal, n_articles, mean_embedding, embedding_dim, "
        f"embedding_dispersion, mean_tone_overall, mean_tone_positive, "
        f"mean_tone_negative "
        f"FROM daily_text_features WHERE {' AND '.join(where)} ORDER BY timestamp_utc"
    )
    with connection() as conn:
        df = conn.execute(sql, params).fetchdf()
    if df.empty:
        return df

    def _unpack(row):
        blob = row["mean_embedding"]
        dim = row["embedding_dim"]
        if blob is None or dim is None or pd.isna(dim):
            return None
        return np.frombuffer(blob, dtype=np.float32).copy()

    df["mean_embedding"] = df.apply(_unpack, axis=1)
    return df
