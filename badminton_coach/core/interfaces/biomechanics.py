"""Biomechanics analyzer interface (L3)."""

from __future__ import annotations

from abc import abstractmethod

from ..schemas import BiomechanicsReport, PerceptionResult, PlayerProfile
from ..schemas.events import Shot
from .base import Component


class BiomechanicsAnalyzer(Component):
    """Turns pose + stroke segmentation + a player profile into per-stroke joint
    kinematics and load proxies.

    Backends: pose2d (2D-pose planar approximation, no weights); wham/opensim (3D mesh
    + inverse dynamics) plug in behind the same interface later.
    """

    @abstractmethod
    def analyze(self, shots: list[Shot], perception: PerceptionResult,
                profile: PlayerProfile | None) -> BiomechanicsReport:
        raise NotImplementedError
