"""Frozen annotation prompt + JSON output schema (Phase 8 §8.1, schema v2).

Everything the LLM sees is here, and it is **date-blind by design**: titles are
presented under synthetic local indices (1..N), never their GDELT ``headline_id``
(which embeds a timestamp), and the day's calendar date is withheld in the
primary variant. The date-visible variant (:func:`build_user_message` with
``show_date``) exists only to *measure* parametric leakage in the Stage-0 A/B
check (plans/phase_8_ssl_probing.md §5 trap 11).

``prompt_hash`` fingerprints the system prompt + schema + task version so cached
annotations are invalidated when any of them change, and so a pre-registration
can pin the exact instrument (§4.4).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

# Default annotator model. Overridable on the CLI; the cost note compares tiers.
MODEL_DEFAULT = "claude-opus-4-8"

# Bump when the prompt or schema changes in a way that should invalidate the
# on-disk annotation cache and any pre-registration built on it.
TASK_VERSION = "v2.0"

# Controlled event-type vocabulary (subsumes the five overlapping lens candidates
# — cb_gold_flow / trade_policy / retail_bullion_stress / macro-prints — as enum
# values). "none" means the title is not a datable occurrence.
EVENT_TYPES = [
    "cb_rate_decision",
    "cb_communication",
    "cb_gold_flow",
    "cpi_print",
    "jobs_print",
    "other_macro_print",
    "pgm_supply_disruption",
    "sanction_or_export_ban",
    "trade_policy_duty",
    "retail_bullion_stress",
    "geopolitical_escalation",
    "other",
    "none",
]

METAL_TAGS = ["gold", "silver", "platinum", "palladium", "bullion_generic"]

_PER_TITLE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "id": {"type": "integer"},
        "relevant": {"type": "boolean"},
        "metal_reads": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "metal": {"type": "string", "enum": METAL_TAGS},
                    # Direction the news READS as for this metal (supportive/adverse),
                    # never a price forecast. -2 strong bearish .. +2 strong bullish.
                    "direction": {"type": "integer", "enum": [-2, -1, 0, 1, 2]},
                },
                "required": ["metal", "direction"],
            },
        },
        "event_type": {"type": "string", "enum": EVENT_TYPES},
        "event_entity": {"type": "string"},
        "supply_demand_side": {
            "type": "string",
            "enum": ["supply", "demand", "unclear", "none"],
        },
        "framing": {"type": "string", "enum": ["anticipatory", "reaction", "neither"]},
        "monetary_stance": {
            "type": "string",
            "enum": ["hawkish", "dovish", "mixed", "none"],
        },
    },
    "required": [
        "id",
        "relevant",
        "metal_reads",
        "event_type",
        "event_entity",
        "supply_demand_side",
        "framing",
        "monetary_stance",
    ],
}

# One object per (UTC) day. The market-level day labels sit alongside the
# per-title records.
ANNOTATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "gold_narrative_regime": {
            "type": "string",
            "enum": [
                "safe_haven",
                "real_rates_usd",
                "cb_demand",
                "inflation_hedge",
                "price_action_only",
                "none",
            ],
        },
        "monetary_stance_day": {
            "type": "string",
            "enum": ["hawkish", "dovish", "mixed", "none"],
        },
        "titles": {"type": "array", "items": _PER_TITLE_SCHEMA},
    },
    "required": ["gold_narrative_regime", "monetary_stance_day", "titles"],
}

SYSTEM_PROMPT = """\
You label precious-metals news headlines for a quantitative research dataset. You
are given a numbered list of distinct news TITLES observed on a single trading day
(a de-duplicated slice of that day's metals-relevant news). You return one record
per title plus two day-level labels.

ABSOLUTE RULES
- Label ONLY from the title text provided. Do NOT use outside knowledge of what
  happened, and do NOT infer a label from an implied or realized price move. If a
  title does not support a field, use the abstaining value ("none"/[]/neither).
- You are NOT forecasting. `direction` and `monetary_stance` describe how the news
  READS for a metal or for policy, as stated in the headline — not what the price
  will do.
- Set `relevant: false` for titles that are not about a precious metal as a
  financial asset / commodity (e.g. sports "gold medal", jewellery ads, a company
  merely named "Platinum" or "Silver", "silver lining"). For `relevant: false`
  titles leave `metal_reads` empty and all other fields at their abstaining value.

PER-TITLE FIELDS
- `id`: echo the title's number.
- `metal_reads`: for each metal the title is genuinely about, {metal, direction}.
  Use "bullion_generic" when the title concerns bullion/precious metals broadly
  without singling one out. Empty if not relevant.
- `event_type`: a discrete DATABLE occurrence reported in the title (a rate
  decision, a CPI/jobs print, a mine strike / force majeure, a sanction or export
  ban, an import-duty change, a mint suspension / "sold out", an official-sector
  gold buy/sell, a geopolitical escalation). Use "none" for explainers, opinion,
  forecasts, price recaps — anything that is not a specific dated event.
- `event_entity`: the named actor/country/company at the centre of the event
  (e.g. "Nornickel", "Federal Reserve", "India"), verbatim from the title; "" if none.
- `supply_demand_side`: for a fundamental event, is the mechanism a supply
  constriction ("supply") or a demand pull ("demand")? "unclear" if the title names
  an event but not its side; "none" if there is no fundamental event.
- `framing`: "anticipatory" if the title looks forward to a not-yet-happened event
  ("ahead of", "looms", "braces", "previews"); "reaction" if it reports something
  that has happened ("after", "following", "fell on"); "neither" otherwise.
- `monetary_stance`: for monetary-policy titles, the stance conveyed —
  "hawkish"/"dovish"/"mixed"; "none" for non-monetary titles.

DAY-LEVEL FIELDS
- `gold_narrative_regime`: the dominant frame in which GOLD is being discussed
  today — "safe_haven", "real_rates_usd", "cb_demand", "inflation_hedge",
  "price_action_only" (only price moves, no narrative), or "none" (gold not discussed).
- `monetary_stance_day`: the net monetary stance across the day's monetary titles,
  or "none".
"""


def build_user_message(
    titles: Sequence[str],
    *,
    show_date: bool = False,
    date: str | None = None,
) -> str:
    """Render the numbered title list the model annotates.

    ``show_date`` is used ONLY by the Stage-0 date-blind A/B check to quantify
    parametric leakage; the primary/production variant is date-blind
    (``show_date=False``). Titles are numbered 1..N so the model never sees a
    date-encoding ``headline_id``.
    """
    lines = [f"{i + 1}. {t}" for i, t in enumerate(titles)]
    header = "Annotate the following titles.\n"
    if show_date and date is not None:
        header = f"Date: {date}\n" + header
    return header + "\n".join(lines)


def prompt_hash(model: str = MODEL_DEFAULT) -> str:
    """Stable fingerprint of (system prompt, schema, task version, model)."""
    blob = json.dumps(
        {
            "system": SYSTEM_PROMPT,
            "schema": ANNOTATION_SCHEMA,
            "task_version": TASK_VERSION,
            "model": model,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]
