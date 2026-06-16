"""Shot/stroke classifier interface (L2)."""

from __future__ import annotations

from abc import abstractmethod

from ..schemas import PerceptionResult
from ..schemas.events import HitEvent, Shot
from .base import Component


class ShotClassifier(Component):
    """Classifies the stroke type of each shot (segment between two hits).

    Backends: heuristic (from the 3D trajectory shape, no weights) or BST
    (Badminton Stroke-type Transformer: pose + shuttle trajectory + player
    position, SOTA fine-grained classes).
    """

    @abstractmethod
    def classify(self, hits: list[HitEvent], perception: PerceptionResult) -> list[Shot]:
        raise NotImplementedError
