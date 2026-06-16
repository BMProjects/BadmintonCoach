"""Court calibrator interface (keypoint detection + homography)."""

from __future__ import annotations

from abc import abstractmethod

from ..schemas import CourtCalibration, Frame
from .base import Component


class CourtCalibrator(Component):
    """Detects court keypoints and solves the ground-plane homography.

    Backends: keypoint (white-line RANSAC + BWF reference), or a learned detector
    (CourtKeyNet / TennisCourtDetector style) behind the same interface.
    """

    @abstractmethod
    def calibrate(self, frame: Frame) -> CourtCalibration:
        """Estimate the image->ground homography from a representative frame."""
        raise NotImplementedError

    def is_present(self, frame: Frame) -> bool:
        """Whether a (usable) court is visible in this specific frame.

        Used to gate the overlay/analysis per frame so no-court or incomplete-court
        frames don't show a stale 'phantom' court. Default True (no per-frame check);
        backends that can cheaply assess presence (e.g. the learned detector) override.
        """
        return True

    def present_frames(self, frames: list[Frame]) -> set[int]:
        """Indices of frames with a court visible. Default loops is_present; learned
        backends override with a batched GPU forward (much faster)."""
        return {f.index for f in frames if self.is_present(f)}
