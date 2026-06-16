"""Multi-object player tracker interface (ByteTrack / BoT-SORT-ReID)."""

from __future__ import annotations

from abc import abstractmethod

from ..schemas import Detection, Frame, PlayerTrack
from .base import Component


class PlayerTracker(Component):
    """Associates per-frame player detections into stable tracks across a clip.

    Backends: iou (numpy, geometry only), botsort (ReID + camera-motion compensation —
    resolves crossing ID-switches using appearance, needs the frame images).
    """

    @abstractmethod
    def track(self, detections_per_frame: list[list[Detection]],
              frames: list[Frame] | None = None) -> list[PlayerTrack]:
        """Given player detections grouped by frame (and, for appearance-based trackers,
        the aligned frame images), return stable PlayerTracks."""
        raise NotImplementedError
