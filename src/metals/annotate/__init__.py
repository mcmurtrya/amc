"""LLM-as-annotator over GDELT metals titles (Phase 8, §8.1).

The Stage-0 feasibility pilot: draw a stratified day sample, filter+dedupe each
day's metal-relevant titles, run a frozen date-blinded annotation prompt, and
score five checks (coverage, human-audit, known-event recall, date-blind A/B
drift, reproducibility) before committing to the full ~1,678-day run.

Titles only — the corpus has no article bodies (plans/phase_8_ssl_probing.md §5).
See scripts/annotate_pilot.py for the CLI driver.
"""

from __future__ import annotations

from metals.annotate.schema import (
    ANNOTATION_SCHEMA,
    MODEL_DEFAULT,
    SYSTEM_PROMPT,
    TASK_VERSION,
    build_user_message,
    prompt_hash,
)

__all__ = [
    "ANNOTATION_SCHEMA",
    "MODEL_DEFAULT",
    "SYSTEM_PROMPT",
    "TASK_VERSION",
    "build_user_message",
    "prompt_hash",
]
