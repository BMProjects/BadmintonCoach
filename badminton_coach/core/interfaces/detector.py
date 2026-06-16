"""Large-object detector interface (players, rackets, net posts, court lines)."""

from __future__ import annotations

from abc import abstractmethod

from ..schemas import Detection, Frame
from .base import Component


class Detector(Component):
    """Detects large/stable objects in a single frame.

    Backends: yolo (Ultralytics, default), rfdetr (SOTA, DINOv2).
    The shuttlecock is NOT detected here — see ShuttleTracker.
    """

    @abstractmethod
    def detect(self, frame: Frame) -> list[Detection]:
        """Return all detections for one frame."""
        raise NotImplementedError
