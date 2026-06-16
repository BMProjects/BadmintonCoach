"""Monocular 3D shuttle reconstructor interface."""

from __future__ import annotations

from abc import abstractmethod

from ..schemas import CourtCalibration, ShuttleTrajectory2D, ShuttleTrajectory3D
from .base import Component


class Reconstructor3D(Component):
    """Lifts a 2D shuttle trajectory to 3D world coordinates.

    Backends embody the precision/latency trade-off from the reports:
    - monotrack: physics drag-model non-linear optimization (high accuracy, slow).
    - synthnet: feed-forward net trained on synthetic trajectories (low latency).
    """

    @abstractmethod
    def reconstruct(
        self,
        traj2d: ShuttleTrajectory2D,
        court: CourtCalibration,
    ) -> ShuttleTrajectory3D:
        """Reconstruct the 3D shuttle trajectory; must report reprojection error."""
        raise NotImplementedError
