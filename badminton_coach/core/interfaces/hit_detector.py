"""Hit-frame detector interface (L2)."""

from __future__ import annotations

from abc import abstractmethod

from ..schemas import ShuttleTrajectory2D
from ..schemas.events import HitEvent
from .base import Component


class HitDetector(Component):
    """Detects shuttle strikes (hits) from the 2D trajectory.

    Backends: trajectory (direction-reversal, no weights) or a learned HitNet.
    Hit frames segment a rally into shots and give MonoTrack correct per-shot
    parabola boundaries.
    """

    @abstractmethod
    def detect(self, shuttle_2d: ShuttleTrajectory2D) -> list[HitEvent]:
        raise NotImplementedError
