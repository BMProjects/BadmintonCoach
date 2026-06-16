"""Shared kinematics helpers for biomechanics backends (2D and 3D).

Pure functions used by both pose2d (planar) and lift3d (3D-lifted): joint-angle math,
time-series stats, peak angular acceleration, and hitter selection. Keeping them here
avoids duplication between backends.
"""

from __future__ import annotations

import math

import numpy as np

KP_CONF = 0.3

# COCO-17 indices per racket side (+ opposite shoulder/hip for the trunk line).
SIDE = {
    "R": {"sh": 6, "el": 8, "wr": 10, "hip": 12, "kn": 14, "an": 16, "osh": 5, "ohip": 11},
    "L": {"sh": 5, "el": 7, "wr": 9, "hip": 11, "kn": 13, "an": 15, "osh": 6, "ohip": 12},
}


def angle_at(a, b, c):
    """Interior angle at b (deg) for points a-b-c (2D or 3D tuples), or None if missing."""
    if a is None or b is None or c is None:
        return None
    v1 = [ai - bi for ai, bi in zip(a, b, strict=True)]
    v2 = [ci - bi for ci, bi in zip(c, b, strict=True)]
    n1 = math.hypot(*v1) or 1e-9
    n2 = math.hypot(*v2) or 1e-9
    dot = sum(x * y for x, y in zip(v1, v2, strict=True))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (n1 * n2)))))


def series_stats(times, vals):
    """(peak |val|, range, peak |rate/s|, time-of-peak-rate) over a series, skipping None."""
    pairs = [(t, v) for t, v in zip(times, vals, strict=True) if v is not None]
    if len(pairs) < 2:
        return None
    t = np.array([p[0] for p in pairs])
    v = np.array([p[1] for p in pairs])
    dt = np.diff(t)
    rate = np.diff(v) / np.where(dt > 0, dt, 1e-9)
    peak_rate = float(np.abs(rate).max()) if len(rate) else 0.0
    peak_t = float(t[1 + int(np.argmax(np.abs(rate)))]) if len(rate) else float(t[0])
    return float(np.abs(v).max()), float(v.max() - v.min()), peak_rate, peak_t


def peak_angaccel(times, vals_deg):
    """Peak |angular acceleration| (rad/s²) of an angle series (deg, with None gaps)."""
    pairs = [(t, math.radians(v)) for t, v in zip(times, vals_deg, strict=True) if v is not None]
    if len(pairs) < 3:
        return 0.0
    t = np.array([p[0] for p in pairs])
    v = np.array([p[1] for p in pairs])
    return float(np.abs(np.gradient(np.gradient(v, t), t)).max())


def hitter_tid(frame, box_by_frame, shuttle_by_frame):
    """Track id of the player nearest the shuttle at the hit frame (the striker)."""
    boxes = box_by_frame.get(frame) or []
    if not boxes:
        return None
    sp = shuttle_by_frame.get(frame)
    if sp is None:
        return boxes[0][0]
    sx, sy = sp.point.x, sp.point.y
    return min(boxes, key=lambda tb: (tb[1].center.x - sx) ** 2 + (tb[1].center.y - sy) ** 2)[0]


def hitter_pose(f, tid, poses_by_frame, box_by_frame):
    """The pose whose centroid is nearest the hitter's box at frame f (or None)."""
    box = {t: b for t, b in box_by_frame.get(f, [])}.get(tid)
    cands = poses_by_frame.get(f, [])
    if not cands:
        return None
    if box is None:
        return cands[0]
    cx, cy = box.center.x, box.center.y

    def c(p):
        xs = [k.point.x for k in p.keypoints if k.confidence >= KP_CONF]
        ys = [k.point.y for k in p.keypoints if k.confidence >= KP_CONF]
        return (sum(xs) / len(xs), sum(ys) / len(ys)) if xs else (1e9, 1e9)
    return min(cands, key=lambda p: (c(p)[0] - cx) ** 2 + (c(p)[1] - cy) ** 2)


def by_frame_maps(perception):
    """Build (poses_by_frame, box_by_frame, shuttle_by_frame) lookups from perception."""
    poses_by_frame: dict[int, list] = {}
    for p in perception.poses:
        poses_by_frame.setdefault(p.frame_index, []).append(p)
    box_by_frame: dict[int, list] = {}
    for tr in perception.player_tracks:
        for tb in tr.boxes:
            box_by_frame.setdefault(tb.frame_index, []).append((tr.track_id, tb.bbox))
    shuttle_by_frame = {p.frame_index: p for p in perception.shuttle_2d.points}
    return poses_by_frame, box_by_frame, shuttle_by_frame
