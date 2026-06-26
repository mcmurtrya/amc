"""Phase 3 end-to-end pipeline.

Stages, in order:

  1. ``gdelt``    — pull GDELT GKG rows for the date range, upsert to ``headlines``
  2. ``embed``    — (optional) warm the sentence-transformers embedding cache
  3. ``aggregate``— daily aggregation to a shared 'market' news-state (text features)
  4. ``topics``   — per-day topic prevalences (themes-via-SQL default; BERTopic opt-in)
  5. ``context``  — assemble the daily contextual feature vector
  6. ``cluster``  — fit UMAP + HDBSCAN, persist, write cluster assignments
  7. ``analyze``  — produce cluster summary tables to ``results/phase3_*.csv``
  8. ``label``    — LLM-label clusters (Anthropic; skipped without ANTHROPIC_API_KEY)

Each stage is independently runnable via ``--only`` or skippable from a
chosen start point via ``--resume-from``. Heavy artifacts (the BERTopic
model, the clustering pipeline pickle, embeddings cache) live under
``data/processed/`` and persist across invocations.

Run as:
    uv run python scripts/phase3_pipeline.py --start 2015-02-18 --end 2026-05-12
or chunked:
    uv run python scripts/phase3_pipeline.py --only gdelt --start 2015-02-18 --end 2018-12-31
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

STAGES = ("gdelt", "embed", "aggregate", "topics", "context", "cluster", "analyze", "label")


def _print_stage(name: str) -> None:
    print(f"\n=== {name.upper()} ===  ({datetime.now():%Y-%m-%d %H:%M:%S})")


def run_gdelt(start: str, end: str) -> None:
    _print_stage("gdelt")
    from metals.data.gdelt import refresh

    summary = refresh(start_date=start, end_date=end)
    print(f"rows_written = {summary['rows_written']}  (window {start} → {end})")


def _month_bounds(conn, start: str | None = None, end: str | None = None):
    """Return ``[(month_start, lower_iso, upper_iso), ...]`` partitioning the
    headlines corpus into calendar months, bounded by the actual data range and
    optional ``start``/``end`` (bare dates are treated as whole-day inclusive).

    Monthly ranges align to day boundaries, so every calendar day lands wholly
    inside one chunk — per-day aggregates computed per chunk therefore
    concatenate with no cross-chunk merge, and peak memory is one month of rows
    rather than the whole 63 M-row corpus.
    """
    lo, hi = conn.execute("SELECT min(timestamp_utc), max(timestamp_utc) FROM headlines").fetchone()
    if lo is None:
        return []
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    if start is not None:
        lo = max(lo, pd.Timestamp(start))
    if end is not None:
        end_ts = pd.Timestamp(end)
        if end_ts == end_ts.normalize():  # bare date -> include the whole day
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        hi = min(hi, end_ts)
    if lo > hi:
        return []
    bounds = []
    for month_start in pd.date_range(lo.normalize().replace(day=1), hi, freq="MS"):
        month_end = month_start + pd.offsets.MonthBegin(1)
        lower = max(month_start, lo)
        upper = min(month_end - pd.Timedelta(microseconds=1), hi)
        bounds.append((month_start, str(lower), str(upper)))
    return bounds


def run_embed(start: str | None = None, end: str | None = None) -> None:
    """Optional: warm the persistent on-disk embedding cache, streaming one
    month at a time so the full corpus is never materialized (the old
    whole-corpus path OOMed).

    Not required by ``aggregate`` (which encodes on the fly — for bulk
    sequential access that is faster than reading the hash-sharded cache). Run
    this only to pre-populate embeddings for other consumers, e.g. the optional
    ``--topics-method bertopic`` enrichment or a future Phase 4 model.
    """
    _print_stage("embed")
    from metals.data.db import connection
    from metals.features.embeddings import cache_embeddings

    with connection(read_only=True) as conn:
        bounds = _month_bounds(conn, start, end)
    total_rows = total_new = 0
    for month_start, lower, upper in bounds:
        with connection(read_only=True) as conn:
            urls = (
                conn.execute(
                    "SELECT article_url FROM headlines "
                    "WHERE timestamp_utc >= ? AND timestamp_utc <= ?",
                    [lower, upper],
                )
                .fetchdf()["article_url"]
                .astype(str)
                .tolist()
            )
        if not urls:
            continue
        n_new = cache_embeddings(urls)
        total_rows += len(urls)
        total_new += n_new
        print(f"  {month_start:%Y-%m}: {len(urls):>9,} urls  ({n_new:>9,} newly encoded)")
    print(f"embeddings cached. rows={total_rows:,}  newly_encoded={total_new:,}")


def _load_embeddings_for(df: pd.DataFrame) -> np.ndarray:
    """Helper: load already-cached (or encode) embeddings for a *bounded* frame
    of headlines. Never the full corpus — callers pass one month at a time."""
    from metals.features.embeddings import embed_texts

    return embed_texts(df["article_url"].astype(str).tolist())


def run_aggregate(start: str | None = None, end: str | None = None) -> None:
    """Stream the daily text-feature aggregation one month at a time.

    Each month's headlines are embedded fresh on the GPU (bounded chunk,
    ``use_cache=False`` — a fresh encode beats reading the hash-sharded cache
    for this bulk sequential access, and keeps disk flat) and reduced to a
    shared daily 'market' news-state. Because months align to day boundaries the
    per-chunk outputs are independent, so each month is upserted as it finishes
    (resumable, bounded memory). Replaces the old single ``fetchdf()`` of all
    63 M rows + the 97 GB embedding vstack.
    """
    _print_stage("aggregate")
    from metals.data.db import connection
    from metals.features.embeddings import embed_texts
    from metals.features.text_daily import (
        _parse_themes_field,
        aggregate_daily,
        upsert_daily,
    )

    cols = (
        "timestamp_utc, headline_id, source, themes, article_url, "
        "tone_overall, tone_positive, tone_negative"
    )
    with connection(read_only=True) as conn:
        bounds = _month_bounds(conn, start, end)
    if not bounds:
        print("no headlines to aggregate.")
        return

    total_rows = total_written = 0
    for month_start, lower, upper in bounds:
        with connection(read_only=True) as conn:  # read conn closed before write
            hl = conn.execute(
                f"SELECT {cols} FROM headlines "
                "WHERE timestamp_utc >= ? AND timestamp_utc <= ? ORDER BY timestamp_utc",
                [lower, upper],
            ).fetchdf()
        if hl.empty:
            continue
        hl["themes_list"] = hl["themes"].apply(_parse_themes_field)
        hl["timestamp_utc"] = pd.to_datetime(hl["timestamp_utc"])
        emb = embed_texts(hl["article_url"].astype(str).tolist(), batch_size=256, use_cache=False)
        out = aggregate_daily(hl, embeddings=emb)
        n = upsert_daily(out)  # separate write connection, no overlap with read
        total_rows += len(hl)
        total_written += n
        print(
            f"  {month_start:%Y-%m}: {len(hl):>9,} headlines  "
            f"-> {n:>4} daily 'market' rows  [{total_written:,} total]"
        )
    print(f"daily_text_features rows written = {total_written:,}  (from {total_rows:,} headlines)")


def run_topics(
    method: str = "themes",
    start: str | None = None,
    end: str | None = None,
    *,
    sample: int = 200_000,
    min_topic_size: int = 30,
    nr_topics: int | str | None = None,
) -> None:
    """Per-day topic prevalence -> ``daily_topic_prevalence``.

    ``method="themes"`` (default): deterministic, streaming SQL aggregation over
    the curated GDELT theme set — no embeddings, tractable on the full corpus.
    ``method="bertopic"``: legacy learned topics, fit on a bounded random
    ``sample`` of documents (the full-corpus fit is not scale-safe) and applied
    to that sample's days only. Optional enrichment, not the default.
    """
    _print_stage("topics")
    from metals.features.topics import upsert_topic_prevalence

    if method == "themes":
        from metals.features.topics import compute_theme_prevalence, theme_topic_map

        prev = compute_theme_prevalence(start=start, end=end)
        n = upsert_topic_prevalence(prev)
        tmap = theme_topic_map()
        out_dir = Path("results")
        out_dir.mkdir(parents=True, exist_ok=True)
        map_path = out_dir / "phase3_theme_topic_map.csv"
        pd.DataFrame({"topic_id": list(tmap.values()), "theme": list(tmap.keys())}).sort_values(
            "topic_id"
        ).to_csv(map_path, index=False)
        n_days = prev["timestamp_utc"].nunique() if not prev.empty else 0
        print(
            f"daily_topic_prevalence rows written = {n:,}  "
            f"(themes-via-SQL, {len(tmap)} themes, {n_days:,} days)"
        )
        print(f"theme->topic_id map written to {map_path}")
        return

    if method != "bertopic":
        raise ValueError(f"Unknown topics method {method!r}; use 'themes' or 'bertopic'.")

    # Legacy BERTopic path — sample-bounded so it cannot OOM on the full corpus.
    from metals.data.db import connection
    from metals.features.topics import (
        TopicModelConfig,
        assign_topics,
        fit_topic_model,
        save_topic_model,
        topic_prevalence_per_day,
    )

    where = ["themes IS NOT NULL"]
    params: list = []
    if start is not None:
        where.append("timestamp_utc >= ?")
        params.append(str(pd.Timestamp(start)))
    if end is not None:
        where.append("timestamp_utc <= ?")
        params.append(str(pd.Timestamp(end)))
    with connection(read_only=True) as conn:
        hl = conn.execute(
            "SELECT * FROM ("
            f"  SELECT timestamp_utc, article_url FROM headlines "
            f"  WHERE {' AND '.join(where)}"
            f") USING SAMPLE {int(sample)} ROWS",
            params,
        ).fetchdf()
    if hl.empty:
        print("no headlines for topic modelling.")
        return
    emb = _load_embeddings_for(hl)
    docs = hl["article_url"].astype(str).tolist()
    config = TopicModelConfig(min_topic_size=min_topic_size, nr_topics=nr_topics)
    print(f"fitting BERTopic on a {len(docs):,}-doc sample (min_topic_size={min_topic_size})...")
    model = fit_topic_model(docs, embeddings=emb, config=config)
    save_topic_model(model, name="default")
    topics = assign_topics(model, docs, embeddings=emb)
    prev = topic_prevalence_per_day(hl["timestamp_utc"], topics, include_noise=False)
    n = upsert_topic_prevalence(prev)
    print(f"daily_topic_prevalence rows written = {n:,}  (BERTopic sample, partial day coverage)")


def run_context(target_metal: str, train_until: str | None = None) -> pd.DataFrame:
    _print_stage("context")
    from metals.features.context import ContextConfig, build_context
    from metals.features.loaders import load_macro, load_prices
    from metals.features.text_daily import load_daily as load_text_daily
    from metals.features.topics import load_topic_prevalence_wide

    prices = load_prices(column="adj_close")
    macro = load_macro()
    text = load_text_daily()
    topics = load_topic_prevalence_wide()

    if prices.empty or macro.empty:
        raise RuntimeError("prices or macro empty — run Phase 1 ingestion first.")
    has_embeddings = (
        text is not None
        and not text.empty
        and "mean_embedding" in text.columns
        and text["mean_embedding"].notna().any()
    )
    if train_until is None and has_embeddings:
        print(
            "  WARNING: --train-until not set; the text-embedding PCA will be fit "
            "on the full sample (look-ahead). Pass --train-until for an honest "
            "walk-forward run."
        )
    ctx, artifacts = build_context(
        prices=prices,
        macro_wide=macro,
        text_daily=text,
        topic_prevalence=topics,
        pca_fit_until=train_until,
        config=ContextConfig(target_metal=target_metal),
    )
    print(f"context shape = {ctx.shape}  cols includes {[c for c in ctx.columns[:6]]} ...")
    return ctx


def run_cluster(
    context: pd.DataFrame, train_until: str | None, model_version: str | None = None
) -> str:
    _print_stage("cluster")
    from metals.models.clustering import (
        ClusteringConfig,
        assign_clusters,
        cluster_centroids,
        fit_clustering,
        save_pipeline,
        upsert_assignments,
        upsert_centroids,
    )

    train = context.dropna()
    if train_until:
        train = train.loc[: pd.Timestamp(train_until)]
    print(f"training rows = {len(train):,}  ({train.index.min()} → {train.index.max()})")

    cfg = ClusteringConfig()
    pipeline = fit_clustering(train, config=cfg, model_version=model_version)
    save_pipeline(pipeline)

    full = context.dropna()
    assignments = assign_clusters(pipeline, full)
    centroids = cluster_centroids(pipeline, train)
    n_a = upsert_assignments(assignments, model_version=pipeline.model_version)
    n_c = upsert_centroids(centroids, model_version=pipeline.model_version)
    print(f"cluster_assignments rows = {n_a:,}, centroids = {n_c}")
    return pipeline.model_version


def run_analyze(model_version: str, target_metal: str = "gold") -> None:
    _print_stage("analyze")
    from metals.data.db import connection
    from metals.eval.clusters import (
        cluster_summary,
        forward_returns,
    )
    from metals.features.loaders import load_prices
    from metals.features.topics import load_topic_prevalence_wide

    with connection() as conn:
        assignments = conn.execute(
            "SELECT timestamp_utc, cluster_id, confidence "
            "FROM cluster_assignments WHERE model_version = ? ORDER BY timestamp_utc",
            [model_version],
        ).fetchdf()
    if assignments.empty:
        print(f"no assignments for model_version={model_version!r}.")
        return

    prices = load_prices(column="adj_close")
    fr = forward_returns(prices, horizons=(1, 5, 20, 60))
    topics_wide = load_topic_prevalence_wide()

    summary = cluster_summary(assignments, fr, topics_wide, horizons=(1, 5, 20, 60))

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in summary.items():
        path = out_dir / f"phase3_{model_version}_{name}.csv"
        df.to_csv(path, index=False)
        print(f"wrote {path} ({len(df):,} rows)")


def _today_iso() -> str:
    return date.today().isoformat()


def _latest_model_version() -> str | None:
    """Most recent clustering ``model_version`` persisted to cluster_assignments."""
    from metals.data.db import connection

    with connection(read_only=True) as conn:
        row = conn.execute(
            "SELECT model_version FROM cluster_assignments ORDER BY timestamp_utc DESC LIMIT 1"
        ).fetchone()
    return row[0] if row else None


_LABEL_HEADLINES_PER_DAY = 50


def run_label(
    model_version: str, target_metal: str = "gold", llm_model: str = "claude-haiku-4-5-20251001"
) -> None:
    """Phase 3 step 3.12 / 3.14 LLM labelling stage. Gated on ANTHROPIC_API_KEY."""
    _print_stage("label")
    import os

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set; skipping label stage.")
        return

    from metals.data.db import connection
    from metals.eval.cluster_labeling import (
        build_cluster_context,
        label_all_clusters,
        upsert_labels,
    )
    from metals.eval.clusters import (
        cluster_forward_stats,
        dominant_topics,
        forward_returns,
    )
    from metals.features.loaders import load_prices
    from metals.features.topics import load_topic_prevalence_wide

    with connection(read_only=True) as conn:
        assignments = conn.execute(
            "SELECT timestamp_utc, cluster_id, confidence "
            "FROM cluster_assignments WHERE model_version = ? ORDER BY timestamp_utc",
            [model_version],
        ).fetchdf()
    if assignments.empty:
        print(f"no assignments for {model_version!r}; nothing to label.")
        return

    # Only a handful of example headlines per clustered day are ever shown to the
    # LLM. Bound the pull to the assignment date range with a per-day cap instead
    # of materializing the whole ~63 M-row corpus.
    lo = str(pd.Timestamp(assignments["timestamp_utc"].min()))
    hi = str(pd.Timestamp(assignments["timestamp_utc"].max()) + pd.Timedelta(days=1))
    with connection(read_only=True) as conn:
        headlines = conn.execute(
            "SELECT timestamp_utc, article_url AS headline, article_url FROM ("
            "  SELECT timestamp_utc, article_url, row_number() OVER ("
            "    PARTITION BY date_trunc('day', timestamp_utc) ORDER BY timestamp_utc"
            "  ) AS rn FROM headlines "
            "  WHERE timestamp_utc >= ? AND timestamp_utc < ?"
            f") WHERE rn <= {_LABEL_HEADLINES_PER_DAY}",
            [lo, hi],
        ).fetchdf()

    prices = load_prices(column="adj_close")
    fr = forward_returns(prices, horizons=(1, 5, 20))
    topics_wide = load_topic_prevalence_wide()
    fwd_stats = cluster_forward_stats(assignments, fr, horizons=(1, 5, 20))
    topics_per_cluster = dominant_topics(assignments, topics_wide, top_k=5)

    cluster_ids = sorted(c for c in assignments["cluster_id"].unique() if c != -1)
    contexts = [
        build_cluster_context(
            cluster_id=int(cid),
            assignments=assignments,
            headlines=headlines,
            dominant_topics=topics_per_cluster,
            forward_stats=fwd_stats,
        )
        for cid in cluster_ids
    ]
    print(f"labelling {len(contexts)} clusters via {llm_model}...")
    labels = label_all_clusters(contexts, model=llm_model)
    n = upsert_labels(labels, model_version=model_version)
    print(f"upserted {n} cluster labels.")
    for lbl in labels:
        print(f"  cluster {lbl.cluster_id:>3d} [{lbl.confidence:>6s}]  {lbl.label}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="2015-02-18", help="GDELT start date.")
    parser.add_argument("--end", default=_today_iso(), help="GDELT end date.")
    parser.add_argument(
        "--target-metal", default="gold", choices=["gold", "silver", "platinum", "palladium"]
    )
    parser.add_argument(
        "--train-until", default=None, help="ISO date; clustering trains on data up to this date."
    )
    parser.add_argument(
        "--topics-method",
        choices=["themes", "bertopic"],
        default="themes",
        help="themes = streaming SQL over GDELT themes (default); "
        "bertopic = legacy learned topics on a bounded sample.",
    )
    parser.add_argument(
        "--topics-sample",
        type=int,
        default=200_000,
        help="Sample size for --topics-method bertopic.",
    )
    parser.add_argument("--min-topic-size", type=int, default=30)
    parser.add_argument("--nr-topics", default=None, help="auto, an int, or None.")
    parser.add_argument("--only", choices=STAGES, default=None, help="Run a single stage and exit.")
    parser.add_argument(
        "--resume-from", choices=STAGES, default=None, help="Skip stages before this one."
    )
    parser.add_argument(
        "--model-version",
        default=None,
        help="Override the auto-generated clustering version label.",
    )
    parser.add_argument(
        "--llm-model",
        default="claude-haiku-4-5-20251001",
        help="Anthropic model id for the label stage.",
    )
    args = parser.parse_args()

    nr_topics: int | str | None = args.nr_topics
    if isinstance(nr_topics, str) and nr_topics.isdigit():
        nr_topics = int(nr_topics)

    if args.only is not None:
        do = {args.only}
    elif args.resume_from is not None:
        idx = STAGES.index(args.resume_from)
        do = set(STAGES[idx:])
    else:
        do = set(STAGES)

    if "gdelt" in do:
        run_gdelt(args.start, args.end)
    if "embed" in do:
        run_embed(args.start, args.end)
    if "aggregate" in do:
        run_aggregate(args.start, args.end)
    if "topics" in do:
        run_topics(
            method=args.topics_method,
            start=args.start,
            end=args.end,
            sample=args.topics_sample,
            min_topic_size=args.min_topic_size,
            nr_topics=nr_topics,
        )

    context = None
    if "context" in do:
        context = run_context(args.target_metal, train_until=args.train_until)

    model_version = args.model_version
    if "cluster" in do:
        if context is None:
            context = run_context(args.target_metal, train_until=args.train_until)
        model_version = run_cluster(context, args.train_until, model_version=model_version)

    # analyze before label, matching STAGES order.
    if "analyze" in do:
        mv = model_version or _latest_model_version()
        if mv is None:
            print("no model_version available for analyze; skipping.")
        else:
            run_analyze(mv, target_metal=args.target_metal)

    if "label" in do:
        mv = model_version or _latest_model_version()
        if mv is None:
            print("no model_version available for label stage; skipping.")
        else:
            run_label(mv, target_metal=args.target_metal, llm_model=args.llm_model)

    print("\ndone.")


if __name__ == "__main__":
    main()
