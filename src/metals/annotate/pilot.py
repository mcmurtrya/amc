"""Stage-0 runner: cost estimate (dry-run) and batch annotation.

The annotator calls Claude through the official ``anthropic`` SDK Batch API (50%
cheaper, async — no HTTP-timeout constraint). GDELT titles are public news
headlines, so sending them is ToU-clean; AMC's own data never touches this path.

``estimate_run`` is a **dry run**: it measures real input tokens from the sampled
days and models output tokens, returning a cost table across model tiers so the
~$ spend is known before any paid call. ``run_pilot`` submits the batch, polls,
parses the structured-output JSON, and caches results with provenance.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pandas as pd

from metals.annotate import schema as sch
from metals.annotate.titles import DayTitles, load_day_titles

# USD per 1M tokens (input, output), standard rate; Batch API is 50% of these.
# Sonnet 5 shows its post-intro sticker (intro $2/$10 through 2026-08-31 is lower).
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

FULL_ERA_DAYS = 1678  # title-era trading days (2019-09-22 -> 2026-06-19)

# Modelled output tokens per per-title record (compact JSON). The dry run reports
# this assumption; tune it after the first real batch returns actual usage.
PER_TITLE_OUTPUT_TOKENS = 55
DAY_OVERHEAD_OUTPUT_TOKENS = 40

MAX_OUTPUT_TOKENS = 32000


@dataclass(frozen=True)
class CostRow:
    model: str
    input_tokens: int
    output_tokens: int
    standard_usd: float
    batch_usd: float


@dataclass(frozen=True)
class CostEstimate:
    n_days: int
    n_variants: int
    per_title_output_tokens: int
    token_source: str  # "count_tokens" or "char_approx"
    sample_rows: list[CostRow]  # cost for the sampled days, per model
    full_run_rows: list[CostRow]  # extrapolated to FULL_ERA_DAYS, per model
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------
def build_params(
    titles: Sequence[str],
    *,
    model: str,
    show_date: bool = False,
    date: str | None = None,
) -> dict:
    """Message-create params for one day (a Batch request's ``params``)."""
    return {
        "model": model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "system": [
            {
                "type": "text",
                "text": sch.SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": sch.build_user_message(titles, show_date=show_date, date=date),
            }
        ],
        "output_config": {"format": {"type": "json_schema", "schema": sch.ANNOTATION_SCHEMA}},
    }


def _approx_tokens(text: str) -> int:
    # Deliberately rough (news English ~3.6 chars/token); used only when a live
    # count_tokens call is not requested. Real cost is lower once the cached
    # system prefix is served at ~0.1x.
    return max(1, round(len(text) / 3.6))


# ---------------------------------------------------------------------------
# Dry-run cost estimate
# ---------------------------------------------------------------------------
def estimate_run(
    sample: pd.DataFrame,
    *,
    model: str = sch.MODEL_DEFAULT,
    n_variants: int = 1,
    per_title_output_tokens: int = PER_TITLE_OUTPUT_TOKENS,
    use_api_count: bool = False,
    day_titles: dict[str, DayTitles] | None = None,
) -> CostEstimate:
    """Measure input tokens on the sampled days and model output tokens → cost.

    ``use_api_count=True`` uses the SDK ``count_tokens`` endpoint (accurate,
    ~free, needs a key); otherwise a char approximation (offline, conservative).
    ``day_titles`` lets a caller pass pre-loaded titles to avoid re-querying.
    """
    day_titles = day_titles or {}
    total_in = 0
    total_out = 0
    n_days = 0
    system_tokens = _approx_tokens(sch.SYSTEM_PROMPT)
    # The JSON output schema is billed as input on every structured-output request.
    schema_tokens = _approx_tokens(json.dumps(sch.ANNOTATION_SCHEMA))
    client = _client() if use_api_count else None

    for date in sample["date"].tolist():
        dt = day_titles.get(date) or load_day_titles(date)
        n_titles = len(dt.titles)
        if n_titles == 0:
            n_days += 1
            continue
        user_msg = sch.build_user_message(dt.titles)
        if client is not None:
            counted = client.messages.count_tokens(
                model=model,
                system=[{"type": "text", "text": sch.SYSTEM_PROMPT}],
                messages=[{"role": "user", "content": user_msg}],
            )
            # count_tokens takes no output_config; add the schema tokens explicitly.
            in_tok = int(counted.input_tokens) + schema_tokens
        else:
            in_tok = system_tokens + schema_tokens + _approx_tokens(user_msg)
        out_tok = n_titles * per_title_output_tokens + DAY_OVERHEAD_OUTPUT_TOKENS
        total_in += in_tok
        total_out += out_tok
        n_days += 1

    # Totals above are per single variant. The PILOT runs all `n_variants`
    # (blind + dated A/B); the FULL production run is date-blind, single-variant.
    def _rows(scale: float) -> list[CostRow]:
        out = []
        for m, (pin, pout) in PRICING.items():
            i = round(total_in * scale)
            o = round(total_out * scale)
            std = i / 1e6 * pin + o / 1e6 * pout
            out.append(CostRow(m, i, o, round(std, 2), round(std * 0.5, 2)))
        return out

    scale_full = FULL_ERA_DAYS / max(1, n_days)  # single production variant
    notes = [
        f"Output tokens modelled at {per_title_output_tokens}/title "
        f"+{DAY_OVERHEAD_OUTPUT_TOKENS}/day; refine after the first real batch.",
        "Standard rate shown; Batch API (used by run_pilot) is 50% of it.",
        f"Input includes the ~{schema_tokens}-token JSON schema billed per request.",
        "PILOT rows include all variants (blind + dated A/B); FULL RUN assumes the "
        "single date-blind production variant.",
        f"Sonnet 5 intro pricing ($2/$10 per 1M) applies through 2026-08-31 — "
        f"below the ${PRICING['claude-sonnet-5'][0]}/${PRICING['claude-sonnet-5'][1]} "
        f"sticker used here.",
    ]
    if not use_api_count:
        notes.append("Token counts are a char approximation; pass --use-api-count for exact.")

    return CostEstimate(
        n_days=n_days,
        n_variants=n_variants,
        per_title_output_tokens=per_title_output_tokens,
        token_source="count_tokens" if use_api_count else "char_approx",
        sample_rows=_rows(float(n_variants)),
        full_run_rows=_rows(scale_full),
        notes=notes,
    )


def format_estimate(est: CostEstimate) -> str:
    """One-screen cost table for the CLI."""
    lines = [
        f"Cost estimate — {est.n_days} sampled days x {est.n_variants} variant(s), "
        f"tokens via {est.token_source}",
        "",
        "PILOT (this sample):",
        f"  {'model':<18}{'in_tok':>10}{'out_tok':>10}{'standard$':>11}{'batch$':>10}",
    ]
    for r in est.sample_rows:
        lines.append(
            f"  {r.model:<18}{r.input_tokens:>10,}{r.output_tokens:>10,}"
            f"{r.standard_usd:>11.2f}{r.batch_usd:>10.2f}"
        )
    lines += ["", f"FULL RUN (extrapolated to {FULL_ERA_DAYS} days):", ""]
    lines.append(f"  {'model':<18}{'in_tok':>12}{'out_tok':>12}{'standard$':>11}{'batch$':>10}")
    for r in est.full_run_rows:
        lines.append(
            f"  {r.model:<18}{r.input_tokens:>12,}{r.output_tokens:>12,}"
            f"{r.standard_usd:>11.2f}{r.batch_usd:>10.2f}"
        )
    lines += ["", "Notes:"]
    lines += [f"  - {n}" for n in est.notes]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Live batch run
# ---------------------------------------------------------------------------
def _client():  # -> anthropic.Anthropic
    """Lazily build the SDK client, loading .env (repo convention)."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:  # pragma: no cover - dotenv optional
        pass
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        raise RuntimeError(
            "No ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN found (.env or environment)."
        )
    import anthropic

    return anthropic.Anthropic()


def _variants(date_blind_ab: bool) -> list[tuple[str, bool]]:
    # (variant_name, show_date). The date-visible arm is the leakage control.
    if date_blind_ab:
        return [("blind", False), ("dated", True)]
    return [("blind", False)]


def run_pilot(
    sample: pd.DataFrame,
    *,
    model: str = sch.MODEL_DEFAULT,
    date_blind_ab: bool = True,
    out_path: str | Path,
    poll_seconds: int = 30,
    timeout_seconds: int = 24 * 3600,
) -> pd.DataFrame:
    """Submit one Batch covering every (day, variant), poll, parse, and cache.

    Returns one row per (date, variant) with the day labels, counts, and the raw
    structured-output JSON (per-title records parsed downstream in ``checks``).
    """
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = _client()
    phash = sch.prompt_hash(model)

    # Preload + de-dupe titles once per day (shared across variants).
    dates = sample["date"].tolist()
    loaded: dict[str, DayTitles] = {d: load_day_titles(d) for d in dates}

    requests: list[Request] = []
    for date in dates:
        dt = loaded[date]
        if not dt.titles:
            continue
        for vname, show_date in _variants(date_blind_ab):
            params = build_params(dt.titles, model=model, show_date=show_date, date=date)
            requests.append(
                Request(
                    # Batch custom_id charset is [a-zA-Z0-9_-]; "|" is rejected.
                    # Dates use "-", so "__" round-trips safely.
                    custom_id=f"{date}__{vname}",
                    params=cast(MessageCreateParamsNonStreaming, params),
                )
            )

    if not requests:
        raise RuntimeError("No annotatable days in sample (all title lists empty).")

    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id
    deadline = time.monotonic() + timeout_seconds
    while True:
        b = client.messages.batches.retrieve(batch_id)
        if b.processing_status == "ended":
            break
        if time.monotonic() > deadline:
            raise TimeoutError(f"Batch {batch_id} did not finish within timeout.")
        time.sleep(poll_seconds)

    pulled_at = datetime.now(UTC).isoformat()
    stratum_by_date = dict(zip(sample["date"], sample["stratum"], strict=False))
    rows: list[dict] = []
    for result in client.messages.batches.results(batch_id):
        date, vname = result.custom_id.split("__", 1)
        row: dict = {
            "date": date,
            "variant": vname,
            "stratum": stratum_by_date.get(date),
            "model": model,
            "prompt_hash": phash,
            "task_version": sch.TASK_VERSION,
            "batch_id": batch_id,
            "pulled_at": pulled_at,
            "n_titles": len(loaded[date].titles),
            "n_raw": loaded[date].n_raw,
            "n_dropped_cap": loaded[date].n_dropped_cap,
            "ok": False,
            "gold_narrative_regime": None,
            "monetary_stance_day": None,
            "raw_json": None,
        }
        if result.result.type == "succeeded":
            msg = result.result.message
            text = next((b.text for b in msg.content if b.type == "text"), "")
            try:
                parsed = json.loads(text)
                row["ok"] = True
                row["gold_narrative_regime"] = parsed.get("gold_narrative_regime")
                row["monetary_stance_day"] = parsed.get("monetary_stance_day")
                row["raw_json"] = text
            except json.JSONDecodeError:
                row["raw_json"] = text  # keep for debugging; ok stays False
        else:
            row["raw_json"] = f"__error__:{result.result.type}"
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["date", "variant"]).reset_index(drop=True)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df
