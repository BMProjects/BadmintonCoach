"""Court calibration data contract.

Holds the homography mapping the ground plane (world z=0, meters) to/from image
pixels. Image-to-world is only valid for points ON the ground (player feet,
shuttle landing point) — airborne points need the 3D reconstructor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import CameraParameters
from .geometry_types import Point2D, Point3D


@dataclass(frozen=True, slots=True)
class CourtCalibration:
    """Result of court calibration for a (segment of) video.

    homography: 3x3 matrix mapping image (px) -> world ground (m), homogeneous.
    reprojection_error_px: mean reprojection error over calibration keypoints.
    camera: optional full pinhole model (intrinsics + pose). Required for airborne
        3D shuttle reconstruction; the ground homography alone only maps z=0 points.
    """

    homography: np.ndarray
    reprojection_error_px: float
    camera: CameraParameters | None = None

    def image_to_ground(self, p: Point2D) -> Point3D:
        """Project an image point onto the world ground plane (z=0)."""
        v = self.homography @ np.array([p.x, p.y, 1.0])
        w = v / v[2]
        return Point3D(float(w[0]), float(w[1]), 0.0)

    def ground_to_image(self, p: Point3D) -> Point2D:
        """Project a world ground point (z assumed 0) back to image space."""
        h_inv = np.linalg.inv(self.homography)
        v = h_inv @ np.array([p.x, p.y, 1.0])
        w = v / v[2]
        return Point2D(float(w[0]), float(w[1]))
