"""Two-stage calibration: camera PnP, profile persistence, backend reuse, scene cut."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import badminton_coach.perception  # noqa: F401  (registers backends)
from badminton_coach.core.geometry import estimate_camera
from badminton_coach.core.geometry.court_model import court_corners_doubles
from badminton_coach.core.io import SceneCutDetector
from badminton_coach.core.registry import build
from badminton_coach.core.schemas import CalibrationProfile, Frame, Point2D, Point3D

CORNERS = [[300, 700], [980, 700], [880, 250], [400, 250]]


def _frame() -> Frame:
    return Frame(index=0, timestamp=0.0, image=np.zeros((720, 1280, 3), dtype=np.uint8))


def test_estimate_camera_recovers_known_pose():
    # Build a known camera, project the court corners with it, then check PnP
    # recovers a pose that reprojects those corners. (Uses the true focal; with a
    # *guessed* focal single-view PnP is only approximate — that is documented.)
    import cv2

    world_pts = [Point3D(x, y, 0.0) for x, y in court_corners_doubles()]
    obj = np.array([[p.x, p.y, p.z] for p in world_pts])
    f = 1000.0
    k = np.array([[f, 0, 640], [0, f, 360], [0, 0, 1.0]])
    rvec = np.array([[1.2], [0.0], [0.0]])  # tilt down toward the court
    tvec = np.array([[-3.0], [-2.0], [18.0]])
    img_pts, _ = cv2.projectPoints(obj, rvec, tvec, k, None)
    image_pts = [Point2D(float(x), float(y)) for x, y in img_pts.reshape(-1, 2)]

    cam = estimate_camera(image_pts, world_pts, (1280, 720), focal_px=f)
    reproj = cam.project(obj)
    observed = np.array([[p.x, p.y] for p in image_pts])
    assert np.mean(np.linalg.norm(reproj - observed, axis=1)) < 1.0


def test_calibration_profile_roundtrip(tmp_path: Path):
    image_pts = [Point2D(x, y) for x, y in CORNERS]
    world_pts = [Point3D(x, y, 0.0) for x, y in court_corners_doubles()]
    cam = estimate_camera(image_pts, world_pts, (1280, 720))
    prof = CalibrationProfile(
        source_key="m1",
        image_size=(1280, 720),
        image_corners=CORNERS,
        homography=np.eye(3),
        reprojection_error_px=0.5,
        camera=cam,
    )
    path = tmp_path / "prof.json"
    prof.save(path)
    loaded = CalibrationProfile.load(path)

    assert loaded.source_key == "m1"
    assert np.allclose(loaded.camera.intrinsic, cam.intrinsic)
    assert loaded.to_calibration().camera is not None


def test_two_stage_calibrator_persists_and_reuses(tmp_path: Path):
    path = tmp_path / "cache.json"
    cfg = {
        "backend": "two_stage",
        "image_corners": CORNERS,
        "profile_path": str(path),
        "compute_camera": True,
    }
    cal = build("court_calibrator", cfg)
    first = cal.calibrate(_frame())
    assert path.exists()  # bootstrapped and cached
    assert first.camera is not None  # camera solved for 3D
    assert first.reprojection_error_px < 1.0

    # A fresh instance must reuse the cache, not re-bootstrap.
    cal2 = build("court_calibrator", {"backend": "two_stage", "profile_path": str(path)})
    second = cal2.calibrate(_frame())
    assert np.allclose(second.homography, first.homography)


def test_line_fit_detects_court_from_lines():
    import cv2

    from badminton_coach.perception.court.line_fit import detect_court_corners_linefit

    # Draw a green court with white doubles boundary + singles + service lines.
    img = np.full((1080, 1920, 3), (40, 110, 40), dtype=np.uint8)
    nl, nr, fr, fl = (420, 1015), (1545, 1015), (1330, 388), (600, 388)
    quad = np.array([nl, nr, fr, fl], dtype=np.int32)
    cv2.fillPoly(img, [quad], (70, 130, 70))  # ensure green surface under lines
    cv2.polylines(img, [quad], True, (240, 240, 240), 3)
    # singles sidelines (inner) for validation support
    cv2.line(img, (460, 1015), (640, 388), (240, 240, 240), 2)
    cv2.line(img, (1505, 1015), (1290, 388), (240, 240, 240), 2)
    cv2.line(img, (int((420 + 1545) / 2), 1015), (int((600 + 1330) / 2), 388), (240, 240, 240), 2)

    corners = detect_court_corners_linefit(img)
    assert corners is not None
    detected = np.array([[c.x, c.y] for c in corners])
    assert np.max(np.abs(detected - np.array([nl, nr, fr, fl]))) < 25.0


def test_scene_cut_detector_flags_large_change():
    det = SceneCutDetector(threshold=0.5)
    # Use saturated colors (black vs white share H=S=0, so would look identical).
    red = Frame(0, 0.0, np.full((100, 100, 3), (0, 0, 255), dtype=np.uint8))  # BGR red
    blue = Frame(1, 0.03, np.full((100, 100, 3), (255, 0, 0), dtype=np.uint8))  # BGR blue
    assert det.is_cut(red) is False  # first frame: no reference
    assert det.is_cut(blue) is True  # different hue -> cut


def test_estimate_focal_from_court_recovers_known_focal():
    # Project the court with a known focal, then recover it from the vanishing points.
    import cv2

    from badminton_coach.core.geometry import estimate_focal_from_court

    world_pts = [Point3D(x, y, 0.0) for x, y in court_corners_doubles()]
    obj = np.array([[p.x, p.y, p.z] for p in world_pts])
    f_true = 1400.0
    k = np.array([[f_true, 0, 960], [0, f_true, 540], [0, 0, 1.0]])
    # tilt down + slight pan (yaw) so both line families converge to finite vanishing
    # points (a pure-tilt camera leaves one family parallel -> focal unrecoverable).
    rvec = np.array([[1.15], [0.12], [0.0]])
    tvec = np.array([[-3.0], [-2.5], [20.0]])
    img_pts, _ = cv2.projectPoints(obj, rvec, tvec, k, None)
    image_pts = [Point2D(float(x), float(y)) for x, y in img_pts.reshape(-1, 2)]

    f_est = estimate_focal_from_court(image_pts, world_pts, (1920, 1080))
    assert f_est is not None
    assert abs(f_est - f_true) / f_true < 0.02  # within 2%
