"""Human pose data contract.

The canonical keypoint layout is COCO-17 (the layout BST and RTMPose use, and the
one both reports adopt). MediaPipe's 33-point output is remapped to COCO-17 by its
adapter so downstream code sees one consistent skeleton.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .geometry_types import Point2D


class CocoKeypoint(IntEnum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


NUM_COCO_KEYPOINTS = 17


@dataclass(frozen=True, slots=True)
class Keypoint:
    point: Point2D
    confidence: float


@dataclass(frozen=True, slots=True)
class Pose:
    """A single person's 2D pose in one frame (COCO-17 order)."""

    frame_index: int
    keypoints: tuple[Keypoint, ...]  # length == NUM_COCO_KEYPOINTS

    def keypoint(self, name: CocoKeypoint) -> Keypoint:
        return self.keypoints[int(name)]
