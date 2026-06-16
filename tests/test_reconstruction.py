"""MonoTrack 3D reconstruction: recover a known drag trajectory from its 2D projection."""

from __future__ import annotations

import numpy as np
import pytest

import badminton_coach.perception  # noqa: F401  (registers backends)
from badminton_coach.core.geometry import physics, solve_homography
from badminton_coach.core.geometry.camera import estimate_camera
from badminton_coach.core.geometry.court_model import court_corners_doubles
from badminton_coach.core.registry import build
from badminton_coach.core.schemas import (
    CourtCalibration,
    Point2D,
    Point3D,
    ShuttlePoint2D,
    ShuttleTrajectory2D,
)

pytest.importorskip("scipy")


def _known_camera():
    world = [Point3D(x, y, 0.0) for x, y in court_corners_doubles()]
    obj = np.array([[p.x, p.y, p.z] for p in world])
    import cv2

    f = 1400.0
    k = np.array([[f, 0, 640], [0, f, 360], [0, 0, 1.0]])
    rvec = np.array([[1.1], [0.0], [0.0]])
    tvec = np.array([[-3.05], [-3.0], [22.0]])
    img, _ = cv2.projectPoints(obj, rvec, tvec, k, None)
    image_pts = [Point2D(float(x), float(y)) for x, y in img.reshape(-1, 2)]
    cam = estimate_camera(image_pts, world, (1280, 720), focal_px=f)
    homog = solve_homography(image_pts, world)
    return CourtCalibration(homog.homography, homog.reprojection_error_px, cam), cam


def test_monotrack_recovers_known_drag_trajectory():
    court, cam = _known_camera()
    fps = 30.0
    # Ground-truth drag trajectory: a clear from the back court arcing up and over.
    p0 = np.array([3.0, 2.0, 2.0])
    v0 = np.array([0.2, 9.0, 6.0])
    n = 15
    gt = physics.simulate(p0, v0, dt=1.0 / fps, steps=n - 1)  # (n,3)

    proj = cam.project(gt)
    points = tuple(
        ShuttlePoint2D(frame_index=i, point=Point2D(float(proj[i, 0]), float(proj[i, 1])),
                       confidence=1.0, visible=True)
        for i in range(n)
    )
    traj2d = ShuttleTrajectory2D(points=points)

    recon = build("reconstructor", {"backend": "monotrack", "fps": fps})
    out = recon.reconstruct(traj2d, court)

    assert out.method == "monotrack"
    assert len(out) == n
    # Fit must reproject onto the observed pixels well.
    assert out.reprojection_error_px < 5.0
    # And the recovered 3D should be in the right ballpark (meters).
    rec0 = np.array(out.points[0].point.as_tuple())
    assert np.linalg.norm(rec0 - p0) < 1.5
