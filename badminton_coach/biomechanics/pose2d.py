"""2D-pose biomechanics analyzer (MVP, no weights).

From the existing 2D COCO-17 poses + a PlayerProfile, per stroke compute, for the racket
side: joint flexion angles (shoulder/elbow/hip/knee), an I·alpha joint load proxy (inertia
scaled by height/weight), and the kinematic sequence (proximal->distal peak-velocity
ordering: hips->trunk->upper arm->forearm). Planar/relative approximations — the lift3d
backend gives 3D angles behind the same interface.
"""

from __future__ import annotations

import math
from typing import Any

from ..core.geometry.anthropometry import segment_inertia_about_joint
from ..core.interfaces import BiomechanicsAnalyzer
from ..core.registry import register
from ..core.schemas import BiomechanicsReport, JointMetric, PlayerProfile, StrokeBiomechanics
from ._kinematics import (
    KP_CONF,
    SIDE,
    angle_at,
    by_frame_maps,
    hitter_pose,
    hitter_tid,
    peak_angaccel,
    series_stats,
)


def _pt(pose, i):
    kp = pose.keypoints[i]
    return (kp.point.x, kp.point.y) if kp.confidence >= KP_CONF else None


def _dir(a, b):
    if a is None or b is None:
        return None
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def _unwrap(vals):
    out, prev = [], None
    for v in vals:
        if v is None:
            out.append(None)
            continue
        if prev is not None:
            while v - prev > 180:
                v -= 360
            while v - prev < -180:
                v += 360
        out.append(v)
        prev = v
    return out


def kinematic_sequence(times, seg):
    """Order segments by time-of-peak angular velocity; ok if proximal->distal."""
    ref = ["hips", "trunk", "upperarm", "forearm"]
    peaks = {}
    for name in ref:
        st = series_stats(times, _unwrap(seg[name]))
        if st is not None:
            peaks[name] = st[3]
    ordered = sorted(peaks, key=lambda k: peaks[k])
    ok = ordered == [s for s in ref if s in peaks]
    return tuple(ordered), bool(ok and len(ordered) >= 2)


@register("biomechanics", "pose2d")
class Pose2DBiomechanics(BiomechanicsAnalyzer):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)

    @classmethod
    def is_available(cls) -> bool:
        return True

    def analyze(self, shots, perception, profile):
        prof = profile or PlayerProfile(height_m=1.80, mass_kg=75.0)
        idx = SIDE.get(prof.handedness.upper(), SIDE["R"])
        fps = perception.fps or 25.0
        poses_by_frame, box_by_frame, shuttle_by_frame = by_frame_maps(perception)

        out: list[StrokeBiomechanics] = []
        for si, s in enumerate(shots, 1):
            tid = hitter_tid(s.start_frame, box_by_frame, shuttle_by_frame)
            res = self._stroke(s, si, tid, idx, prof, fps, poses_by_frame, box_by_frame)
            if res is not None:
                out.append(res)
        return BiomechanicsReport(strokes=tuple(out))

    def _stroke(self, s, si, tid, idx, prof, fps, poses_by_frame, box_by_frame):
        frames = list(range(s.start_frame, s.end_frame + 1))
        times = [f / fps for f in frames]
        flex = {"shoulder": [], "elbow": [], "hip": [], "knee": []}
        seg = {"hips": [], "trunk": [], "upperarm": [], "forearm": []}
        for f in frames:
            pose = hitter_pose(f, tid, poses_by_frame, box_by_frame)
            if pose is None:
                for d in (flex, seg):
                    for k in d:
                        d[k].append(None)
                continue
            P = {k: _pt(pose, idx[k]) for k in idx}
            flex["shoulder"].append(angle_at(P["hip"], P["sh"], P["el"]))
            flex["elbow"].append(angle_at(P["sh"], P["el"], P["wr"]))
            flex["hip"].append(angle_at(P["sh"], P["hip"], P["kn"]))
            flex["knee"].append(angle_at(P["hip"], P["kn"], P["an"]))
            seg["hips"].append(_dir(P["ohip"], P["hip"]))
            seg["trunk"].append(_dir(P["osh"], P["sh"]))
            seg["upperarm"].append(_dir(P["sh"], P["el"]))
            seg["forearm"].append(_dir(P["el"], P["wr"]))

        joints: list[JointMetric] = []
        for name in ("shoulder", "elbow", "hip", "knee"):
            st = series_stats(times, flex[name])
            if st is None:
                continue
            peak_ang, rom, peak_vel, _ = st
            torque = segment_inertia_about_joint(prof, name) * peak_angaccel(times, flex[name])
            joints.append(JointMetric(name, round(peak_ang, 1), round(rom, 1),
                                      round(peak_vel, 1), round(torque, 1)))
        if not joints:
            return None
        order, ok = kinematic_sequence(times, seg)
        effort = max((j.peak_torque_nm for j in joints), default=0.0)
        return StrokeBiomechanics(si, s.start_frame, s.end_frame, tid, tuple(joints),
                                  order, ok, round(effort, 1))
