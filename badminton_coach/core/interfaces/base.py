"""Common base for all pluggable components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Component(ABC):
    """Base class for every swappable backend.

    Subclasses are constructed from their config block (a plain dict). They must
    report whether their runtime dependencies / weights are present via
    is_available() so the pipeline can fail early with a clear message instead of
    crashing mid-run.
    """

    #: registry name, set by the @register decorator
    backend_name: str = "base"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """True if this backend's dependencies and weights are usable right now."""
        raise NotImplementedError
