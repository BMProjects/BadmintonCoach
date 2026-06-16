"""L3 biomechanics: anthropometric scaling + pose2d analyzer."""

from __future__ import annotations

import badminton_coach.biomechanics  # noqa: F401  (registers backends)
from badminton_coach.core.geometry.anthropometry import segment_inertia_about_joint
from badminton_coach.core.registry import build
from badminton_coach.core.schemas import (
    BBox,
    Keypoint,
    PerceptionResult,
    PlayerProfile,
    PlayerTrack,
    Point2D,
    Pose,
    ShuttlePoint2D,
    ShuttleTrajectory2D,
    TrackedBox,
)
from badminton_coach.core.schemas.events import Shot, ShotType


def test_inertia_scales_with_mass_and_height():
    small = PlayerProfile(height_m=1.6, mass_kg=55)
    big = PlayerProfile(height_m=1.9, mass_kg=90)
    i_small = segment_inertia_about_joint(small, "elbow")
    i_big = segment_inertia_about_joint(big, "elbow")
    assert i_big > i_small > 0
    # I ~ m * L^2; doubling mass doubles I at fixed height
    base = PlayerProfile(height_m=1.8, mass_kg=70)
    heavy = PlayerProfile(height_m=1.8, mass_kg=140)
    assert abs(segment_inertia_about_joint(heavy, "elbow")
               / segment_inertia_about_joint(base, "elbow") - 2.0) < 1e-6


def _pose(frame, wrist_xy):
    # 17 COCO keypoints; right arm sh(6)/el(8)/wr(10) + leg used by the analyzer.
    pts = [(500.0, 200.0)] * 17
    pts[5], pts[6] = (520, 300), (560, 300)     # shoulders
    pts[7], pts[8] = (510, 360), (590, 360)     # elbows
    pts[9], pts[10] = (505, 420), wrist_xy      # wrists (right wrist moves)
    pts[11], pts[12] = (525, 460), (565, 460)   # hips
    pts[13], pts[14] = (520, 560), (570, 560)   # knees
    pts[15], pts[16] = (518, 650), (572, 650)   # ankles
    kps = tuple(Keypoint(Point2D(float(x), float(y)), 0.9) for x, y in pts)
    return Pose(frame, kps)


def _perception(n):
    # right wrist swings (changing elbow flexion) -> non-zero ROM + torque proxy
    poses = [_pose(f, (590 + 6 * f, 360 + 8 * f)) for f in range(n)]
    boxes = tuple(TrackedBox(f, 0, BBox(480, 280, 620, 680), 0.9) for f in range(n))
    shuttle = ShuttleTrajectory2D(
        points=tuple(ShuttlePoint2D(f, Point2D(580, 360), True) for f in range(n)))
    return PerceptionResult(
        source="x", fps=25.0, frame_count=n, court=None, detections=(),
        player_tracks=(PlayerTrack(0, boxes),), poses=tuple(poses),
        shuttle_2d=shuttle, shuttle_3d=None)


def test_biomech_analyzers_produce_joint_metrics():
    n = 12
    perception = _perception(n)
    profile = PlayerProfile(height_m=1.8, mass_kg=75, handedness="R")
    for backend in ("pose2d", "lift3d"):
        analyzer = build("biomechanics", {"backend": backend})
        report = analyzer.analyze([Shot(0, n - 1, 0, ShotType.SMASH, 0.5)], perception, profile)
        assert len(report.strokes) == 1, backend
        sb = report.strokes[0]
        assert {"shoulder", "elbow"} <= {j.name for j in sb.joints}, backend
        assert sb.effort_nm > 0, backend
        assert len(sb.kinematic_sequence) >= 2, backend
