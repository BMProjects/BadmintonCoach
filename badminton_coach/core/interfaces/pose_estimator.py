"""Human pose estimator interface (top-down, COCO-17 output)."""

from __future__ import annotations

from abc import abstractmethod

from ..schemas import Frame, Pose
from .base import Component


class PoseEstimator(Component):
    """Estimates 2D poses for the given player boxes in a frame.

    Backends: rtmpose (SOTA, top-down), mediapipe (prototype). All backends return
    COCO-17 poses regardless of their native layout.
    """

    @abstractmethod
    def estimate(self, frame: Frame, boxes: list) -> list[Pose]:
        """Return one Pose per input box (BBox), COCO-17 ordered."""
        raise NotImplementedError
