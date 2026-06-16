"""Ground-plane homography estimation and reprojection error.

Solves the 3x3 homography H mapping image pixels -> world ground coords (meters)
from >=4 coplanar correspondences. Valid only for points on the ground plane.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..schemas import CourtCalibration, Point2D, Point3D


def solve_homography(
    image_points: list[Point2D],
    world_points: list[Point3D],
) -> CourtCalibration:
    """Estimate image->ground homography from corresponding points.

    image_points and world_points must be the same length (>=4) and ordered to
    correspond. world_points are taken on the ground (z ignored).
    """
    if len(image_points) != len(world_points):
        raise ValueError("image_points and world_points must have equal length")
    if len(image_points) < 4:
        raise ValueError("Homography needs at least 4 correspondences")

    src = np.array([[p.x, p.y] for p in image_points], dtype=np.float64)
    dst = np.array([[p.x, p.y] for p in world_points], dtype=np.float64)

    h, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if h is None:
        raise RuntimeError("findHomography failed to estimate a homography")

    error = _reprojection_error_px(h, src, dst)
    return CourtCalibration(homography=h, reprojection_error_px=error)


def _reprojection_error_px(h: np.ndarray, src: np.ndarray, dst_world: np.ndarray) -> float:
    """Mean pixel error after round-tripping world points back to image via H^-1."""
    h_inv = np.linalg.inv(h)
    dst_h = np.hstack([dst_world, np.ones((len(dst_world), 1))])
    back = (h_inv @ dst_h.T).T
    back = back[:, :2] / back[:, 2:3]
    return float(np.mean(np.linalg.norm(back - src, axis=1)))
