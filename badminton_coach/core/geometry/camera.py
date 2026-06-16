"""Camera model + single-view pose estimation.

CameraParameters is defined in core.schemas (it is a data contract); this module
re-exports it and adds the estimation helper that lives at the geometry layer.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..schemas import Point2D, Point3D
from ..schemas.camera import CameraParameters

__all__ = ["CameraParameters", "estimate_camera", "estimate_focal_from_court"]


def estimate_focal_from_court(
    image_points: list[Point2D],
    world_points: list[Point3D],
    image_size: tuple[int, int],
) -> float | None:
    """Estimate focal length (px) from the court's two orthogonal line families.

    The court ground has two perpendicular world directions (x: baselines, y: sidelines).
    Their images converge to two vanishing points v1, v2. For a camera with the principal
    point at the image centre and unit aspect, orthogonal directions satisfy the image-of-
    the-absolute-conic constraint:  (v1-c)·(v2-c) + f^2 = 0, so
        f^2 = -[(v1x-cx)(v2x-cx) + (v1y-cy)(v2y-cy)].
    Returns None when a vanishing point is near infinity (near-orthographic view) or f^2<=0
    (degenerate) — caller should fall back to a focal prior.
    """
    if len(image_points) < 4:
        return None
    src = np.array([[p.x, p.y] for p in image_points], dtype=np.float64)
    dst = np.array([[p.x, p.y] for p in world_points], dtype=np.float64)
    h_i2w, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if h_i2w is None:
        return None
    h_w2i = np.linalg.inv(h_i2w)

    def vanish(world_dir_inf: np.ndarray) -> np.ndarray | None:
        p = h_w2i @ world_dir_inf  # image of a world point at infinity (a direction)
        if abs(p[2]) < 1e-6 * (abs(p[0]) + abs(p[1])):
            return None  # vanishing point at/near infinity -> unusable
        return p[:2] / p[2]

    v1 = vanish(np.array([1.0, 0.0, 0.0]))  # baseline (world-x) direction
    v2 = vanish(np.array([0.0, 1.0, 0.0]))  # sideline (world-y) direction
    if v1 is None or v2 is None:
        return None
    w, h = image_size
    cx, cy = w / 2.0, h / 2.0
    f_sq = -((v1[0] - cx) * (v2[0] - cx) + (v1[1] - cy) * (v2[1] - cy))
    if not np.isfinite(f_sq) or f_sq <= 0:
        return None
    return float(np.sqrt(f_sq))


def estimate_camera(
    image_points: list[Point2D],
    world_points: list[Point3D],
    image_size: tuple[int, int],
    focal_px: float | None = None,
) -> CameraParameters:
    """Estimate camera intrinsics + extrinsics (PnP) from one court view.

    Focal length: if `focal_px` is given it is used; otherwise it is estimated from the
    court's two orthogonal vanishing points (estimate_focal_from_court), falling back to
    the image-width prior when that is degenerate (near-orthographic view). Pose (R, t)
    is then solved via solvePnP.

    image_size: (width, height) in pixels.
    """
    w, h = image_size
    if focal_px is not None:
        f = float(focal_px)
    else:
        f = estimate_focal_from_court(image_points, world_points, image_size) or float(w)
    k = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]], dtype=np.float64)

    obj = np.array([[p.x, p.y, p.z] for p in world_points], dtype=np.float64)
    img = np.array([[p.x, p.y] for p in image_points], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, k, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed to estimate camera pose")
    rot, _ = cv2.Rodrigues(rvec)
    return CameraParameters(intrinsic=k, rotation=rot, translation=tvec.reshape(3))
