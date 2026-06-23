"""LLM-assisted cluster labeling for Phase 3 step 3.12 / 3.14.

For each cluster discovered by ``metals.models.clustering``, build a compact
context bundle (representative dates, top headlines, dominant topics, mean
forward returns per metal) and ask an LLM to produce a short label plus a
one-sentence description.

The Anthropic SDK is imported lazily and the client is fully injectable, so
the prompt builder and response parser are testable without network access.
The actual API call is a 50-line function with retries.

Costs (approximate; check current rates):

  Haiku 4.5:  ~$0.50-1 for all 12 labels in one batch
  Sonnet 4.6: ~$2-5
  Opus 4.6:   ~$10-15

Default model is Haiku — the labeling task is structured and short-form,
so the smaller model is the right choice.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import pandas as pd

from metals.data.db import connection

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
SYSTEM_PROMPT = """\
You are labeling clusters of trading days in a precious-metals research project.

For each cluster you receive:
  - representative dates and example news headlines
  - the top news topics that dominate those days
  - mean forward returns at multiple horizons for each metal

Produce a short, descriptive label and a one-sentence rationale.

Output strict JSON only, no preamble, no markdown fences:
  {"label": "...", "description": "...", "confidence": "high|medium|low"}

Label conventions: lowercase, hyphen-separated, 3-6 words.
Examples:
  "hawkish-fed-strong-usd"
  "geopolitical-flight-to-safety"
  "china-demand-pulse"
  "industrial-cyclical-rally"
  "supply-shock-palladium"
  "post-fomc-relief"
Pick label vocabulary that is descriptive of the dominant *cause*, not the
*outcome* (avoid "gold-up-week" or similar). Use "unclear" + confidence
"low" if the cluster has no obvious dominant theme.
"""


@dataclass(frozen=True)
class ClusterLabel:
    """Structured label for a single cluster."""

    cluster_id: int
    label: str
    description: str
    confidence: str   # 'high' | 'medium' | 'low'


@dataclass(frozen=True)
class ClusterContext:
    """Compact context bundle for one cluster, sized to fit in ~3K tokens."""

    cluster_id: int
    n_days: int
    representative_dates: list[pd.Timestamp]
    example_headlines: list[str]
    dominant_topics: list[tuple[str, float]]
    mean_forward_returns: dict[str, float]   # e.g. {"GC=F_fwd_5d": 0.012}


# ---------------------------------------------------------------------------
# Pure functions — testable without network
# ---------------------------------------------------------------------------

def build_labeling_prompt(ctx: ClusterContext) -> str:
    """Render a ClusterContext into the user-turn prompt body."""
    dates_str = ", ".join(d.strftime("%Y-%m-%d") for d in ctx.representative_dates[:8])
    topics_str = "\n".join(
        f"  - {name} (mean prevalence {prev:.2f})"
        for name, prev in ctx.dominant_topics[:5]
    ) or "  - (none observed)"
    headlines_str = "\n".join(f"  - {h}" for h in ctx.example_headlines[:12]) \
        or "  - (no headlines available)"
    returns_str = "\n".join(
        f"  - {k}: {v:+.2%}" for k, v in list(ctx.mean_forward_returns.items())[:8]
    ) or "  - (no forward-return data)"
    return (
        f"Cluster ID: {ctx.cluster_id}\n"
        f"Days in cluster: {ctx.n_days}\n"
        f"Representative dates: {dates_str}\n\n"
        f"Top news topics:\n{topics_str}\n\n"
        f"Example headlines:\n{headlines_str}\n\n"
        f"Mean forward returns:\n{returns_str}\n"
    )


def parse_llm_response(text: str, cluster_id: int) -> ClusterLabel:
    """Extract a ClusterLabel from the LLM's JSON output. Tolerant of fences."""
    cleaned = text.strip()
    # Strip optional ```json fences
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: try to extract the first {...} block
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not m:
            raise ValueError(f"Could not extract JSON from LLM response: {text!r}")
        obj = json.loads(m.group(0))
    label = str(obj.get("label", "unclear")).strip().lower()
    description = str(obj.get("description", "")).strip()
    confidence = str(obj.get("confidence", "low")).strip().lower()
    if confidence not in ("high", "medium", "low"):
        confidence = "low"
    return ClusterLabel(
        cluster_id=cluster_id,
        label=label or "unclear",
        description=description,
        confidence=confidence,
    )


def build_cluster_context(
    cluster_id: int,
    assignments: pd.DataFrame,
    headlines: pd.DataFrame | None,
    dominant_topics: pd.DataFrame | None,
    forward_stats: pd.DataFrame | None,
    n_representative: int = 8,
    n_headlines_per_day: int = 2,
) -> ClusterContext:
    """Assemble the compact context bundle for one cluster."""
    days_in = assignments[assignments["cluster_id"] == cluster_id]
    days_in = days_in.sort_values("confidence", ascending=False) \
        if "confidence" in days_in.columns else days_in
    rep_dates_ts = pd.to_datetime(days_in["timestamp_utc"]).dt.floor("D").head(n_representative).tolist()
    n_days = int(len(days_in))

    headline_strs: list[str] = []
    if headlines is not None and not headlines.empty:
        hl = headlines.copy()
        hl["day"] = pd.to_datetime(hl["timestamp_utc"]).dt.floor("D")
        for d in rep_dates_ts:
            day_hl = hl[hl["day"] == d]
            for _, row in day_hl.head(n_headlines_per_day).iterrows():
                headline_strs.append(
                    f"{d.strftime('%Y-%m-%d')}: {row.get('headline') or row.get('article_url') or '(no text)'}"
                )

    topics_pairs: list[tuple[str, float]] = []
    if dominant_topics is not None and not dominant_topics.empty:
        sub = dominant_topics[dominant_topics["cluster_id"] == cluster_id]
        for _, r in sub.iterrows():
            topics_pairs.append((str(r["topic_col"]), float(r["mean_prevalence"])))

    returns_map: dict[str, float] = {}
    if forward_stats is not None and not forward_stats.empty:
        sub = forward_stats[forward_stats["cluster_id"] == cluster_id]
        for _, r in sub.iterrows():
            key = f"{r['ticker']}_fwd_{int(r['horizon'])}d"
            returns_map[key] = float(r["mean"])

    return ClusterContext(
        cluster_id=cluster_id,
        n_days=n_days,
        representative_dates=rep_dates_ts,
        example_headlines=headline_strs,
        dominant_topics=topics_pairs,
        mean_forward_returns=returns_map,
    )


# ---------------------------------------------------------------------------
# LLM call — injectable client so tests don't hit the network
# ---------------------------------------------------------------------------

def _default_anthropic_caller():
    """Build a thunk that calls Anthropic via the official SDK."""
    from anthropic import Anthropic   # lazy
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _call(system: str, user: str, model: str, max_tokens: int = 400) -> str:
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # Anthropic returns a list of content blocks; the first is text.
        return "".join(b.text for b in msg.content if getattr(b, "text", None))
    return _call


def label_cluster(
    ctx: ClusterContext,
    model: str = DEFAULT_MODEL,
    caller: Callable[[str, str, str, int], str] | None = None,
    max_retries: int = 3,
    retry_delay_s: float = 2.0,
) -> ClusterLabel:
    """Label a single cluster with the LLM. Retries on transient errors."""
    caller = caller or _default_anthropic_caller()
    user_prompt = build_labeling_prompt(ctx)
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            text = caller(SYSTEM_PROMPT, user_prompt, model, 400)
            return parse_llm_response(text, ctx.cluster_id)
        except Exception as exc:
            last_err = exc
            if attempt + 1 < max_retries:
                time.sleep(retry_delay_s * (attempt + 1))
    raise RuntimeError(
        f"label_cluster: failed after {max_retries} attempts: {last_err}"
    )


def label_all_clusters(
    contexts: Iterable[ClusterContext],
    model: str = DEFAULT_MODEL,
    caller: Callable[[str, str, str, int], str] | None = None,
) -> list[ClusterLabel]:
    """Sequentially label every cluster context. Cheap enough not to parallelize
    — at $0.01-0.10 per cluster on Haiku, 12 clusters take ~10 seconds."""
    out: list[ClusterLabel] = []
    for ctx in contexts:
        out.append(label_cluster(ctx, model=model, caller=caller))
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def upsert_labels(labels: Iterable[ClusterLabel], model_version: str) -> int:
    """Update cluster_centroids with the LLM-produced labels."""
    rows = list(labels)
    if not rows:
        return 0
    df = pd.DataFrame([{
        "model_version": model_version,
        "cluster_id":    int(r.cluster_id),
        "label":         r.label,
        "label_source":  f"llm:{r.confidence}",
        "description":   r.description,
    } for r in rows])
    with connection() as conn:
        conn.register("incoming_labels", df)
        conn.execute(
            """
            UPDATE cluster_centroids AS c
            SET label        = i.label,
                label_source = i.label_source,
                description  = i.description
            FROM incoming_labels AS i
            WHERE c.model_version = i.model_version
              AND c.cluster_id    = i.cluster_id
            """
        )
        conn.unregister("incoming_labels")
    return len(rows)


def load_labels(model_version: str) -> pd.DataFrame:
    """Read labels back out for downstream use."""
    with connection() as conn:
        return conn.execute(
            "SELECT cluster_id, label, label_source, description, n_members "
            "FROM cluster_centroids WHERE model_version = ? "
            "ORDER BY cluster_id",
            [model_version],
        ).fetchdf()
