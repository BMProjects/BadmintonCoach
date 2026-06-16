"""YAML config loading + validation into AppConfig."""

from __future__ import annotations

from pathlib import Path

import yaml

from .schema import AppConfig


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a YAML config preset into an AppConfig."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw)
