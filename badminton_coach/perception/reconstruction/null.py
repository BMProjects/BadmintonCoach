"""Always-available no-op 3D reconstructor (empty trajectory, infinite error).

Lets the pipeline run with reconstruction disabled or before a real backend is
installed. Honestly reports no reconstruction rather than fabricating 3D points.
"""

from __future__ import annotations

import math

from ...core.interfaces import Reconstructor3D
from ...core.registry import register
from ...core.schemas import CourtCalibration, ShuttleTrajectory2D, ShuttleTrajectory3D


@register("reconstructor", "null")
class NullReconstructor(Reconstructor3D):
    @classmethod
    def is_available(cls) -> bool:
        return True

    def reconstruct(
        self, traj2d: ShuttleTrajectory2D, court: CourtCalibration
    ) -> ShuttleTrajectory3D:
        return ShuttleTrajectory3D(points=(), reprojection_error_px=math.inf, method="null")
