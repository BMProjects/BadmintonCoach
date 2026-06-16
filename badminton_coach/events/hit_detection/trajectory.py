"""Trajectory-based hit detector — direction reversals in the 2D shuttle track.

A hit is where a player strikes the shuttle, reversing its travel direction. We
smooth the visible 2D track, compute frame-to-frame velocity, and flag frames where
the velocity direction reverses sharply (the shuttle turns around) — those are hits.
No weights; works directly on TrackNetV3 output and gives MonoTrack correct per-shot
segmentation. A learned HitNet can replace this behind the same interface.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...core.interfaces import HitDetector
from ...core.registry import register
from ...core.schemas import Point2D, ShuttleTrajectory2D
from ...core.schemas.events import HitEvent


@register("hit_detector", "trajectory")
class TrajectoryHitDetector(HitDetector):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.smooth = int(self.config.get("smooth_window", 3))
        self.min_gap = int(self.config.get("min_gap_frames", 6))   # min frames between hits
        self.angle_thresh = float(self.config.get("angle_thresh_deg", 60.0))

    @classmethod
    def is_available(cls) -> bool:
        return True

    def detect(self, shuttle_2d: ShuttleTrajectory2D) -> list[HitEvent]:
        pts = [p for p in shuttle_2d.points if p.visible]
        if len(pts) < 5:
            return []
        idx = np.array([p.frame_index for p in pts])
        xy = np.array([[p.point.x, p.point.y] for p in pts], dtype=np.float64)
        xy = self._smooth(xy)

        vel = np.diff(xy, axis=0)
        norm = np.linalg.norm(vel, axis=1, keepdims=True)
        unit = vel / np.clip(norm, 1e-6, None)
        cos_turn = np.sum(unit[:-1] * unit[1:], axis=1)  # between consecutive velocities
        turn_deg = np.degrees(np.arccos(np.clip(cos_turn, -1, 1)))

        # skip the first/last few frames: 'same'-mode smoothing distorts the ends
        # and yields spurious direction changes there.
        edge = max(2, self.smooth)
        n = len(xy)
        cand = [i + 1 for i, a in enumerate(turn_deg)
                if a > self.angle_thresh and edge <= i + 1 <= n - 1 - edge]
        hit_local = self._merge(cand, turn_deg)

        hits: list[HitEvent] = []
        for li in hit_local:
            fi = int(idx[li])
            hits.append(HitEvent(frame_index=fi,
                                 shuttle_image_pos=Point2D(float(xy[li, 0]), float(xy[li, 1]))))
        return hits

    def _smooth(self, xy: np.ndarray) -> np.ndarray:
        k = self.smooth
        if k <= 1 or len(xy) < k:
            return xy
        kernel = np.ones(k) / k
        out = xy.copy()
        for c in range(2):
            out[:, c] = np.convolve(xy[:, c], kernel, mode="same")
        return out

    def _merge(self, cand: list[int], turn_deg: np.ndarray) -> list[int]:
        """Keep the sharpest turn within each min_gap cluster."""
        merged: list[int] = []
        for c in sorted(cand):
            if merged and c - merged[-1] < self.min_gap:
                # keep whichever has the larger turn angle
                if turn_deg[c - 1] > turn_deg[merged[-1] - 1]:
                    merged[-1] = c
            else:
                merged.append(c)
        return merged
