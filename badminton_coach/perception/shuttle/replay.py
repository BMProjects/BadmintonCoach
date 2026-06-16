"""Replay shuttle tracker — serves a pre-recorded 2D track from a sidecar JSON.

Not a detector: a development aid that replays ground-truth / previously-annotated
shuttle positions so the downstream chain (3D reconstruction, overlays) can be
exercised without model weights. Used by the demo sample. Config:
    gt_path: JSON with {"shuttle2d": [[frame, x, y, visible], ...]}
"""

from __future__ import annotations

import json
from typing import Any

from ...core.interfaces import ShuttleTracker
from ...core.registry import register
from ...core.schemas import FrameClip, Point2D, ShuttlePoint2D, ShuttleTrajectory2D


@register("shuttle_tracker", "replay")
class ReplayShuttleTracker(ShuttleTracker):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._by_frame: dict[int, ShuttlePoint2D] = {}
        gt_path = self.config.get("gt_path")
        if gt_path:
            data = json.loads(open(gt_path, encoding="utf-8").read())
            for frame, x, y, vis in data["shuttle2d"]:
                self._by_frame[int(frame)] = ShuttlePoint2D(
                    frame_index=int(frame), point=Point2D(float(x), float(y)),
                    confidence=1.0 if vis else 0.0, visible=bool(vis),
                )

    @classmethod
    def is_available(cls) -> bool:
        return True

    def track(self, clip: FrameClip) -> ShuttleTrajectory2D:
        pts = [
            self._by_frame[i]
            for i in range(clip.start_index, clip.end_index + 1)
            if i in self._by_frame
        ]
        return ShuttleTrajectory2D(points=tuple(pts))
