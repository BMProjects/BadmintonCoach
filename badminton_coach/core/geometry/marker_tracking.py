"""Lightweight per-frame court-marker tracking (stage 2 of calibration).

After a one-time full bootstrap, fixed-camera footage only needs cheap per-frame
correction for slow drift / micro pan-zoom: track the court markers with sparse
optical flow and re-solve the homography from the moved markers. Much cheaper than
re-detecting the court every frame.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..schemas import CourtCalibration, Point2D, Point3D
from .homography import solve_homography


def track_markers(
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
    prev_points: list[Point2D],
) -> tuple[list[Point2D], list[bool]]:
    """Track marker points from prev to cur frame via Lucas-Kanade optical flow.

    Returns the new points and a per-point success mask. Inputs are single-channel
    (grayscale) uint8 images.
    """
    pts = np.array([[p.x, p.y] for p in prev_points], dtype=np.float32).reshape(-1, 1, 2)
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, pts, None)
    new_points = [Point2D(float(x), float(y)) for x, y in nxt.reshape(-1, 2)]
    ok = [bool(s) for s in status.reshape(-1)]
    return new_points, ok


def recompute_homography(
    image_corners: list[Point2D],
    world_corners: list[Point3D],
) -> CourtCalibration:
    """Re-solve the ground homography from the (tracked) corner positions."""
    return solve_homography(image_corners, world_corners)
