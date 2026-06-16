"""Data contracts passed between layers. All are frozen (immutable) dataclasses."""

from .biomech import (
    BiomechanicsReport,
    JointMetric,
    PlayerProfile,
    StrokeBiomechanics,
)
from .calibration_profile import CalibrationProfile
from .camera import CameraParameters
from .court import CourtCalibration
from .detection import Detection, ObjectClass
from .events import HitEvent, Rally, Shot, ShotType
from .frame import Frame, FrameClip
from .geometry_types import BBox, Point2D, Point3D
from .perception_result import PerceptionResult
from .pose import NUM_COCO_KEYPOINTS, CocoKeypoint, Keypoint, Pose
from .track import PlayerTrack, TrackedBox
from .trajectory import (
    ShuttlePoint2D,
    ShuttlePoint3D,
    ShuttleTrajectory2D,
    ShuttleTrajectory3D,
)

__all__ = [
    "BBox",
    "Point2D",
    "Point3D",
    "CameraParameters",
    "CalibrationProfile",
    "Frame",
    "FrameClip",
    "Detection",
    "ObjectClass",
    "Pose",
    "Keypoint",
    "CocoKeypoint",
    "NUM_COCO_KEYPOINTS",
    "PlayerTrack",
    "TrackedBox",
    "CourtCalibration",
    "ShuttlePoint2D",
    "ShuttleTrajectory2D",
    "ShuttlePoint3D",
    "ShuttleTrajectory3D",
    "PerceptionResult",
    "HitEvent",
    "Rally",
    "Shot",
    "ShotType",
    "PlayerProfile",
    "JointMetric",
    "StrokeBiomechanics",
    "BiomechanicsReport",
]
