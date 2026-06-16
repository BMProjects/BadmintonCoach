"""3D-lifted biomechanics analyzer (P2): 3D joint angles + 3D I·alpha.

Lifts the 2D COCO-17 pose to 3D per stroke, then computes 3D joint flexion angles, an
I·alpha load proxy, and a 3D kinematic sequence — removing the planar-projection error of
pose2d. Both lifters return per-frame 3D joints {name: (x,y,z)}, so the metric code is
shared.

Lifters (config `lifter`):
  - "motionbert" (default): learned monocular lifter via the MotionBERT 3D ONNX model
    (weights weights/motionbert/motionbert_3d_27.onnx, auto-downloadable). 2D COCO ->
    H36M-17 -> crop-scale norm -> 27-frame DSTformer -> root-relative 3D. Falls back to
    analytic if onnxruntime or the weights are missing.
  - "analytic": anthropometric bone-length depth recovery (no weights). Approximate.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

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

_JOINTS = ("sh", "el", "wr", "hip", "kn", "an", "ohip")

# --- analytic lifter (anthropometric depth) ---
_BONES = [("hip", "ohip", 0.11), ("hip", "sh", 0.29), ("sh", "el", 0.186),
          ("el", "wr", 0.146), ("hip", "kn", 0.245), ("kn", "an", 0.246)]
_STANCE = 0.85

# H36M-17 index of each racket-side joint in the MotionBERT output.
_H36M = {
    "R": {"sh": 14, "el": 15, "wr": 16, "hip": 1, "kn": 2, "an": 3, "ohip": 4},
    "L": {"sh": 11, "el": 12, "wr": 13, "hip": 4, "kn": 5, "an": 6, "ohip": 1},
}


def _pt2(pose, i):
    kp = pose.keypoints[i]
    return (kp.point.x, kp.point.y) if kp.confidence >= KP_CONF else None


def _analytic_frame(pose, idx, height_m):
    """Anthropometric 2D->3D for the racket-side chain -> {joint: (x,y,z)} or {}."""
    if pose is None:
        return {}
    p2 = {k: _pt2(pose, idx[k]) for k in _JOINTS}
    if p2["hip"] is None or p2["sh"] is None or p2["el"] is None:
        return {}
    ys = [k.point.y for k in pose.keypoints if k.confidence >= KP_CONF]
    s = _STANCE * height_m / max((max(ys) - min(ys)) if len(ys) >= 2 else 1.0, 1.0)
    p3 = {"hip": (p2["hip"][0] * s, p2["hip"][1] * s, 0.0)}
    for parent, child, frac in _BONES:
        if p2.get(parent) is None or p2.get(child) is None or parent not in p3:
            continue
        dx = (p2[child][0] - p2[parent][0]) * s
        dy = (p2[child][1] - p2[parent][1]) * s
        dz = math.sqrt(max(0.0, (frac * height_m) ** 2 - dx * dx - dy * dy))
        p3[child] = (p3[parent][0] + dx, p3[parent][1] + dy, p3[parent][2] + dz)
    return p3


def _coco2h36m(seq):
    """(T,17,3) COCO x,y,conf -> (T,17,3) H36M x,y,conf (standard MotionBERT mapping)."""
    h = np.zeros_like(seq)
    c = seq

    def mid(a, b):
        return (c[:, a] + c[:, b]) / 2
    h[:, 0] = mid(11, 12)                                    # pelvis
    h[:, 1], h[:, 2], h[:, 3] = c[:, 12], c[:, 14], c[:, 16]  # R leg
    h[:, 4], h[:, 5], h[:, 6] = c[:, 11], c[:, 13], c[:, 15]  # L leg
    h[:, 8] = mid(5, 6)                                      # thorax
    h[:, 7] = (h[:, 0] + h[:, 8]) / 2                        # spine
    h[:, 9] = c[:, 0]                                        # nose
    h[:, 10] = mid(1, 2)                                     # head (mid-eyes)
    h[:, 11], h[:, 12], h[:, 13] = c[:, 5], c[:, 7], c[:, 9]   # L arm
    h[:, 14], h[:, 15], h[:, 16] = c[:, 6], c[:, 8], c[:, 10]  # R arm
    return h


def _crop_scale(motion):
    """MotionBERT wild norm: scale 2D to [-1,1] by the keypoint bbox (aspect-preserving)."""
    res = np.zeros_like(motion)
    valid = motion[..., 2] > 0
    if not valid.any():
        return res
    xs, ys = motion[..., 0][valid], motion[..., 1][valid]
    scale = max(float(xs.max() - xs.min()), float(ys.max() - ys.min())) or 1.0
    res[..., 0] = (motion[..., 0] - xs.min()) / scale * 2 - 1
    res[..., 1] = (motion[..., 1] - ys.min()) / scale * 2 - 1
    res[..., 2] = motion[..., 2]
    res[motion[..., 2] <= 0] = 0
    return res


def _unit(a, b):
    if a is None or b is None:
        return None
    v = np.array(b) - np.array(a)
    n = np.linalg.norm(v)
    return (v / n) if n > 1e-9 else None


def _seq_3d(times, seg_units):
    ref = ["hips", "trunk", "upperarm", "forearm"]
    peaks = {}
    for name in ref:
        u = seg_units[name]
        ts, sp = [], []
        for i in range(1, len(u)):
            if u[i] is None or u[i - 1] is None or times[i] == times[i - 1]:
                continue
            d = float(np.clip(np.dot(u[i], u[i - 1]), -1.0, 1.0))
            sp.append(math.acos(d) / (times[i] - times[i - 1]))
            ts.append(times[i])
        if sp:
            peaks[name] = ts[int(np.argmax(sp))]
    ordered = sorted(peaks, key=lambda k: peaks[k])
    return tuple(ordered), bool(ordered == [s for s in ref if s in peaks] and len(ordered) >= 2)


@register("biomechanics", "lift3d")
class Lift3DBiomechanics(BiomechanicsAnalyzer):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._lifter = str(self.config.get("lifter", "motionbert"))
        # MotionBERT model sequence length: 27/81/243. Longer = more temporal context
        # (steadier 3D); 81 is a good default (~3s window per stroke, player stays local).
        self._seq = int(self.config.get("seq", 81))
        self._weights = self.config.get(
            "weights", f"weights/motionbert/motionbert_3d_{self._seq}.onnx")
        self._session = None
        self._seq_len = self._seq

    @classmethod
    def is_available(cls) -> bool:
        return True

    def _ensure_mb(self) -> bool:
        """Load the MotionBERT ONNX session; return False (fall back to analytic) if not set up."""
        if self._session is not None:
            return True
        try:
            from pathlib import Path

            import onnxruntime as ort
            if not Path(self._weights).exists():
                return False
            self._session = ort.InferenceSession(self._weights,
                                                 providers=["CPUExecutionProvider"])
            self._seq_len = self._session.get_inputs()[0].shape[1]
            return True
        except Exception:  # noqa: BLE001 - onnxruntime missing / load error -> analytic
            return False

    def analyze(self, shots, perception, profile):
        prof = profile or PlayerProfile(height_m=1.80, mass_kg=75.0)
        idx = SIDE.get(prof.handedness.upper(), SIDE["R"])
        fps = perception.fps or 25.0
        use_mb = self._lifter == "motionbert" and self._ensure_mb()
        poses_by_frame, box_by_frame, shuttle_by_frame = by_frame_maps(perception)

        out: list[StrokeBiomechanics] = []
        for si, s in enumerate(shots, 1):
            tid = hitter_tid(s.start_frame, box_by_frame, shuttle_by_frame)
            if use_mb:
                times, per_frame = self._mb_joints(tid, s, fps, prof.handedness,
                                                   poses_by_frame, box_by_frame)
            else:
                frames = range(s.start_frame, s.end_frame + 1)
                wposes = [hitter_pose(f, tid, poses_by_frame, box_by_frame) for f in frames]
                times = [f / fps for f in frames]
                per_frame = [_analytic_frame(p, idx, prof.height_m) for p in wposes]
            res = self._metrics(s, si, tid, prof, times, per_frame)
            if res is not None:
                out.append(res)
        return BiomechanicsReport(strokes=tuple(out))

    def _mb_joints(self, tid, s, fps, hand, poses_by_frame, box_by_frame):
        """Run MotionBERT over a seq_len context window CENTRED on the stroke (steadier 3D
        from full temporal context); return 3D only for the stroke-span frames."""
        seq = self._seq_len
        center = (s.start_frame + s.end_frame) // 2
        ctx = [center - seq // 2 + i for i in range(seq)]
        arr = np.zeros((seq, 17, 3), np.float32)
        for j, f in enumerate(ctx):
            p = hitter_pose(f, tid, poses_by_frame, box_by_frame) if f >= 0 else None
            if p is None:
                continue
            for k, kp in enumerate(p.keypoints):
                arr[j, k] = (kp.point.x, kp.point.y, kp.confidence)
        inp = _crop_scale(_coco2h36m(arr))[None].astype(np.float32)
        out3d = self._session.run(None, {self._session.get_inputs()[0].name: inp})[0][0]
        hm = _H36M.get(hand.upper(), _H36M["R"])
        times, per_frame = [], []
        for j, f in enumerate(ctx):
            if s.start_frame <= f <= s.end_frame:
                times.append(f / fps)
                per_frame.append({n: tuple(float(v) for v in out3d[j, hm[n]]) for n in _JOINTS})
        return times, per_frame

    def _metrics(self, s, si, tid, prof, times, per_frame):
        flex = {"shoulder": [], "elbow": [], "hip": [], "knee": []}
        seg = {"hips": [], "trunk": [], "upperarm": [], "forearm": []}
        for p3 in per_frame:
            g = p3.get
            flex["shoulder"].append(angle_at(g("hip"), g("sh"), g("el")))
            flex["elbow"].append(angle_at(g("sh"), g("el"), g("wr")))
            flex["hip"].append(angle_at(g("sh"), g("hip"), g("kn")))
            flex["knee"].append(angle_at(g("hip"), g("kn"), g("an")))
            seg["hips"].append(_unit(g("ohip"), g("hip")))
            seg["trunk"].append(_unit(g("hip"), g("sh")))
            seg["upperarm"].append(_unit(g("sh"), g("el")))
            seg["forearm"].append(_unit(g("el"), g("wr")))

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
        order, ok = _seq_3d(times, seg)
        effort = max((j.peak_torque_nm for j in joints), default=0.0)
        return StrokeBiomechanics(si, s.start_frame, s.end_frame, tid, tuple(joints),
                                  order, ok, round(effort, 1))
