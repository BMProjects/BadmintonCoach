"""Score a court calibration by how well the full BWF line model overlaps the
actual painted white lines — a backend-agnostic quality metric used to pick the
better of several calibrators (the 4-corner reprojection error is trivially ~0 and
not comparable across methods)."""

from __future__ import annotations

import cv2
import numpy as np

from ..schemas import CourtCalibration, Point3D
from . import court_model
from .court_surface import court_line_mask


def court_overlap_score(img_bgr: np.ndarray, calib: CourtCalibration, tol_px: float = 3.0) -> float:
    """Fraction of sampled court-line points landing within tol_px of a white pixel."""
    white = court_line_mask(img_bgr)
    dist = cv2.distanceTransform(255 - white, cv2.DIST_L2, 5)
    h, w = white.shape
    hits = total = 0
    for (ax, ay), (bx, by) in court_model.court_line_segments():
        for t in np.linspace(0, 1, 60):
            p = calib.ground_to_image(Point3D(ax + (bx - ax) * t, ay + (by - ay) * t, 0.0))
            x, y = int(p.x), int(p.y)
            if 0 <= x < w and 0 <= y < h:
                total += 1
                if dist[y, x] < tol_px:
                    hits += 1
    return hits / total if total else 0.0
