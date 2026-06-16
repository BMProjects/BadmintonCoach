"""Shuttlecock tracker interface (heatmap-based, e.g. TrackNetV3)."""

from __future__ import annotations

from abc import abstractmethod

from ..schemas import FrameClip, ShuttleTrajectory2D
from .base import Component


class ShuttleTracker(Component):
    """Tracks the shuttlecock across a temporal window of frames.

    Heatmap regression over a multi-frame window handles the 1-2 px, motion-blurred,
    high-speed shuttle far better than box detectors. Backends should also fill
    occlusion gaps (TrackNetV3's InpaintNet) and mark filled points visible=False.
    """

    @abstractmethod
    def track(self, clip: FrameClip) -> ShuttleTrajectory2D:
        """Return per-frame 2D shuttle positions for the clip."""
        raise NotImplementedError
