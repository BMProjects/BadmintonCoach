"""L2 event layer: hit detection (direction reversals) + analyze orchestration."""

from __future__ import annotations

import numpy as np

import badminton_coach.events  # noqa: F401  (registers L2 backends)
from badminton_coach.core.registry import build
from badminton_coach.core.schemas import Point2D, ShuttlePoint2D, ShuttleTrajectory2D


def _zigzag_trajectory() -> ShuttleTrajectory2D:
    """Three straight legs with two sharp direction reversals -> expect 2 hits."""
    pts = []
    fi = 0
    legs = [((100, 500), (500, 100)), ((500, 100), (100, 500)), ((100, 500), (500, 100))]
    for (x0, y0), (x1, y1) in legs:
        for t in np.linspace(0, 1, 12):
            p = Point2D(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
            pts.append(ShuttlePoint2D(fi, p, 1.0, True))
            fi += 1
    return ShuttleTrajectory2D(points=tuple(pts))


def test_trajectory_hit_detector_finds_reversals():
    det = build("hit_detector", {"backend": "trajectory", "min_gap_frames": 4})
    hits = det.detect(_zigzag_trajectory())
    # two reversals between the three legs
    assert 1 <= len(hits) <= 3
    assert all(h.frame_index >= 0 for h in hits)


def test_hit_detector_empty_on_short_track():
    det = build("hit_detector", {"backend": "trajectory"})
    short = ShuttleTrajectory2D(points=tuple(
        ShuttlePoint2D(i, Point2D(i, i), 1.0, True) for i in range(3)
    ))
    assert det.detect(short) == []


def test_shot_classifier_registered():
    clf = build("shot_classifier", {"backend": "heuristic"})
    assert clf.classify([], _perception_stub()) == []


def _perception_stub():
    from badminton_coach.core.schemas import CourtCalibration, PerceptionResult

    calib = CourtCalibration(homography=np.eye(3), reprojection_error_px=0.0)
    return PerceptionResult(
        source="x", fps=25.0, frame_count=0, court=calib, detections=(),
        player_tracks=(), poses=(), shuttle_2d=ShuttleTrajectory2D(()), shuttle_3d=None,
    )
