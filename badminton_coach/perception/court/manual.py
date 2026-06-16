"""Manual court calibrator — homography from 4 user-provided image corners.

Functional, dependency-free baseline. The user clicks the four doubles-court
corners once (or supplies them in config['image_corners'] as [[x,y], ...] in the
order near-left, near-right, far-right, far-left); we pair them with the BWF world
corners and solve the ground-plane homography. Always available.
"""

from __future__ import annotations

from typing import Any

from ...core.geometry import solve_homography
from ...core.geometry.court_model import court_corners_doubles
from ...core.interfaces import CourtCalibrator
from ...core.registry import register
from ...core.schemas import CourtCalibration, Frame, Point2D, Point3D


@register("court_calibrator", "manual")
class ManualCourtCalibrator(CourtCalibrator):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._corners = self.config.get("image_corners")

    @classmethod
    def is_available(cls) -> bool:
        return True

    def calibrate(self, frame: Frame) -> CourtCalibration:
        if not self._corners or len(self._corners) != 4:
            raise ValueError(
                "ManualCourtCalibrator needs config['image_corners'] = 4 [x,y] points "
                "in order near-left, near-right, far-right, far-left."
            )
        image_pts = [Point2D(float(x), float(y)) for x, y in self._corners]
        world_pts = [Point3D(x, y, 0.0) for x, y in court_corners_doubles()]
        return solve_homography(image_pts, world_pts)
