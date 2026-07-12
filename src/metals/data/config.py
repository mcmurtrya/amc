"""YAML config loaders for ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_yaml(name: str) -> dict[str, Any]:
    """Load ``configs/<name>``. Accepts with or without the .yaml suffix."""
    if not name.endswith((".yaml", ".yml")):
        name = name + ".yaml"
    path = _repo_root() / "configs" / name
    with path.open("r") as f:
        return yaml.safe_load(f)


def universe() -> dict[str, Any]:
    """Return the parsed price universe config."""
    return load_yaml("universe")


def fred_series() -> dict[str, Any]:
    """Return the parsed FRED series config."""
    return load_yaml("fred_series")


def scenarios() -> dict[str, Any]:
    """Return the parsed Phase 5 scenario registry (configs/scenarios.yaml)."""
    return load_yaml("scenarios")
