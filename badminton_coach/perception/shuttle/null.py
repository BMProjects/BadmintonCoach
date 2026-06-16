"""Always-available no-op shuttle tracker (returns empty trajectory)."""

from __future__ import annotations

from ...core.interfaces import ShuttleTracker
from ...core.registry import register
from ...core.schemas import FrameClip, ShuttleTrajectory2D


@register("shuttle_tracker", "null")
class NullShuttleTracker(ShuttleTracker):
    @classmethod
    def is_available(cls) -> bool:
        return True

    def track(self, clip: FrameClip) -> ShuttleTrajectory2D:
        return ShuttleTrajectory2D(points=())
