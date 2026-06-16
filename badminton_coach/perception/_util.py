"""Shared helpers for perception backend adapters."""

from __future__ import annotations

import importlib.util
from pathlib import Path

#: Repo root, used to locate vendored upstream submodules under third_party/.
THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party"


def module_available(name: str) -> bool:
    """True if an importable module/package `name` is installed."""
    return importlib.util.find_spec(name) is not None


def submodule_available(relative_path: str) -> bool:
    """True if a vendored upstream submodule exists at third_party/<relative_path>."""
    return (THIRD_PARTY / relative_path).exists()
