"""Player track data contract (output of multi-object tracking + ReID)."""

from __future__ import annotations

from dataclasses import dataclass

from .geometry_types import BBox, Point2D


@dataclass(frozen=True, slots=True)
class TrackedBox:
    """A bounding box associated with a stable track id in one frame."""

    frame_index: int
    track_id: int
    bbox: BBox
    confidence: float

    @property
    def foot_image(self) -> Point2D:
        """Ground-contact point in image space, for homography projection."""
        return self.bbox.foot


@dataclass(frozen=True, slots=True)
class PlayerTrack:
    """A player's full trajectory across a clip: one TrackedBox per observed frame."""

    track_id: int
    boxes: tuple[TrackedBox, ...]
