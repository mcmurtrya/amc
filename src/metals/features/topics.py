"""BERTopic wrapper for Phase 3 step 3.8.

Fit one topic model on the full headline corpus, persist to disk, then apply
the fitted model to compute per-day topic-prevalence vectors that feed into
the daily contextual feature vector (step 3.9).

BERTopic and sentence-transformers are heavy dependencies, both imported
lazily so this module can be imported without them installed (tests skip
gracefully via pytest.importorskip).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from metals.data.db import connection

MODEL_DIR = (
    Path(__file__).resolve().parents[3] / "data" / "processed" / "topic_models"
)

# ---------------------------------------------------------------------------
# Themes-via-SQL topic prevalence (Phase 3 default).
#
# GDELT GKG already ships a curated, body-derived theme taxonomy, so per-day
# "topic prevalence" can be computed as a streaming DuckDB GROUP BY over the
# ``themes`` column — no embeddings, no UMAP/HDBSCAN, deterministic, and
# tractable on the full 63 M-row corpus (BERTopic over 63 M docs is not). The
# output lands in the same ``daily_topic_prevalence`` table the BERTopic path
# used, so ``load_topic_prevalence_wide`` / context / clustering are unchanged.
#
# ``TOPIC_THEMES`` is the FIXED, ordered curated theme set (mirrors
# configs/gdelt_themes.yaml). topic_id == index here, so the persisted ids are
# stable across runs and independent of any reordering of the source config.
# ---------------------------------------------------------------------------
TOPIC_THEMES: tuple[str, ...] = (
    "ECON_CENTRALBANK",
    "WB_1235_CENTRAL_BANKS",
    "EPU_POLICY_MONETARY_POLICY",
    "WB_444_MONETARY_POLICY",
    "ECON_INTEREST_RATES",
    "EPU_POLICY_INTEREST_RATES",
    "WB_1125_INTEREST_RATE_POLICY",
    "ECON_INFLATION",
    "WB_442_INFLATION",
    "ECON_GOLDPRICE",
    "WB_1164_COMMODITY_PRICES_SHOCKS",
    "WB_1699_METAL_ORE_MINING",
    "SANCTIONS",
    "ECON_TRADE_DISPUTE",
)


def theme_topic_map() -> dict[str, int]:
    """Stable ``theme code -> topic_id`` mapping (topic_id == index)."""
    return {theme: i for i, theme in enumerate(TOPIC_THEMES)}


def compute_theme_prevalence(
    conn=None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Per-day theme prevalence straight from GDELT GKG ``themes``.

    ``prevalence(theme, day) = (# articles that day tagged with theme) /
    (# articles that day)``. Multi-label, so a day's prevalences need not sum
    to 1. Returns long-format ``(timestamp_utc, topic_id, prevalence)`` ready
    for :func:`upsert_topic_prevalence`. Runs as a single streaming DuckDB
    aggregation (constant memory, no embeddings/GPU).

    Pass ``conn`` to query an existing connection (used in tests); otherwise a
    read-only connection to the canonical store is opened.
    """
    tmap = theme_topic_map()
    themes_in = ", ".join(f"'{t}'" for t in TOPIC_THEMES)

    # The same optional date filter is applied in both the per_day and exploded
    # CTEs, so its bind params appear twice, in that order.
    date_filter = ""
    date_params: list = []
    if start is not None:
        date_filter += " AND timestamp_utc >= ?"
        date_params.append(str(pd.Timestamp(start)))
    if end is not None:
        end_ts = pd.Timestamp(end)
        if end_ts == end_ts.normalize():  # bare date -> include the whole day
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        date_filter += " AND timestamp_utc <= ?"
        date_params.append(str(end_ts))

    query = f"""
        WITH per_day AS (
            SELECT date_trunc('day', timestamp_utc) AS d, count(*) AS day_total
            FROM headlines
            WHERE 1=1{date_filter}
            GROUP BY 1
        ),
        exploded AS (
            SELECT date_trunc('day', timestamp_utc) AS d,
                   unnest(from_json(themes, '["VARCHAR"]')) AS theme
            FROM headlines
            WHERE themes IS NOT NULL AND json_array_length(themes) > 0{date_filter}
        ),
        theme_day AS (
            SELECT d, theme, count(*) AS n
            FROM exploded
            WHERE theme IN ({themes_in})
            GROUP BY d, theme
        )
        SELECT td.d AS timestamp_utc, td.theme AS theme,
               td.n::DOUBLE / pd.day_total AS prevalence
        FROM theme_day td JOIN per_day pd USING (d)
        ORDER BY td.d, td.theme
    """
    params = date_params + date_params  # per_day CTE, then exploded CTE

    if conn is not None:
        df = conn.execute(query, params).fetchdf()
    else:
        from metals.data.db import connection
        with connection(read_only=True) as c:
            df = c.execute(query, params).fetchdf()
    if df.empty:
        return pd.DataFrame(columns=["timestamp_utc", "topic_id", "prevalence"])
    df["topic_id"] = df["theme"].map(tmap).astype(int)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    return df[["timestamp_utc", "topic_id", "prevalence"]]


@dataclass(frozen=True)
class TopicModelConfig:
    """Configuration capturing all BERTopic hyperparameters we expose."""

    n_topics: int | str = "auto"
    min_topic_size: int = 30
    nr_topics: int | str | None = None
    random_state: int = 42
    language: str = "english"


def fit_topic_model(
    documents: Iterable[str],
    embeddings: np.ndarray | None = None,
    config: TopicModelConfig | None = None,
):
    """Fit BERTopic on a corpus and return the fitted model.

    If ``embeddings`` is provided, BERTopic skips the sentence-transformers
    embedding step (which is the slow part of fitting). This is the
    recommended path: embed once with ``metals.features.embeddings``,
    persist, then pass the array here.
    """
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    config = config or TopicModelConfig()
    documents = list(documents)
    if not documents:
        raise ValueError("fit_topic_model: empty document list.")

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=config.random_state,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=config.min_topic_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer = CountVectorizer(
        stop_words=config.language,
        max_df=0.95,
        min_df=5,
        ngram_range=(1, 2),
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        nr_topics=config.nr_topics if config.nr_topics is not None else config.n_topics,
        calculate_probabilities=False,
        language=config.language,
        verbose=False,
    )
    topic_model.fit_transform(documents, embeddings=embeddings)
    return topic_model


def save_topic_model(model, name: str = "default") -> Path:
    """Persist a fitted BERTopic model to disk; return the path."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{name}.pkl"
    with path.open("wb") as f:
        pickle.dump(model, f)
    return path


def load_topic_model(name: str = "default"):
    """Load a previously saved BERTopic model."""
    path = MODEL_DIR / f"{name}.pkl"
    with path.open("rb") as f:
        return pickle.load(f)


def assign_topics(
    model,
    documents: Iterable[str],
    embeddings: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a fitted model to assign a topic id to each document.

    Returns a length-N array of topic ids. ``-1`` is HDBSCAN noise.
    """
    documents = list(documents)
    topics, _probs = model.transform(documents, embeddings=embeddings)
    return np.asarray(topics, dtype=int)


def topic_prevalence_per_day(
    timestamps: pd.Series,
    topic_ids: np.ndarray | pd.Series,
    n_topics: int | None = None,
    include_noise: bool = False,
) -> pd.DataFrame:
    """Return a long-format ``(timestamp_utc, topic_id, prevalence)`` frame.

    Prevalence per day is the share of that day's articles assigned to each
    topic. Noise (topic_id = -1) is excluded by default.
    """
    ts = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True)).dt.floor("D")
    tids = pd.Series(topic_ids).reset_index(drop=True).astype(int)
    df = pd.DataFrame({"timestamp_utc": ts, "topic_id": tids})
    if not include_noise:
        df = df[df["topic_id"] != -1]
    if df.empty:
        return pd.DataFrame(columns=["timestamp_utc", "topic_id", "prevalence"])

    counts = (
        df.groupby(["timestamp_utc", "topic_id"]).size()
          .rename("n").reset_index()
    )
    day_totals = counts.groupby("timestamp_utc")["n"].transform("sum")
    counts["prevalence"] = counts["n"] / day_totals
    return counts[["timestamp_utc", "topic_id", "prevalence"]]


def upsert_topic_prevalence(df: pd.DataFrame) -> int:
    """Persist (date, topic_id) prevalences into the DuckDB table."""
    if df.empty:
        return 0
    with connection() as conn:
        conn.register("incoming_topic_prev", df[["timestamp_utc", "topic_id", "prevalence"]])
        conn.execute(
            """
            INSERT INTO daily_topic_prevalence (timestamp_utc, topic_id, prevalence)
            SELECT timestamp_utc, topic_id, prevalence FROM incoming_topic_prev
            ON CONFLICT (timestamp_utc, topic_id) DO UPDATE SET
                prevalence = EXCLUDED.prevalence
            """
        )
        conn.unregister("incoming_topic_prev")
    return len(df)


def load_topic_prevalence_wide(
    n_topics: int | None = None,
) -> pd.DataFrame:
    """Pivot the long-format DuckDB table to wide ``(date x topic)`` for ML use.

    Missing (date, topic) pairs become 0.0 prevalence in the wide frame.
    """
    with connection() as conn:
        df = conn.execute(
            "SELECT timestamp_utc, topic_id, prevalence FROM daily_topic_prevalence "
            "ORDER BY timestamp_utc, topic_id"
        ).fetchdf()
    if df.empty:
        return pd.DataFrame()
    wide = df.pivot(index="timestamp_utc", columns="topic_id", values="prevalence")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.fillna(0.0)
    wide.columns = [f"topic_{c}" for c in wide.columns]
    return wide
