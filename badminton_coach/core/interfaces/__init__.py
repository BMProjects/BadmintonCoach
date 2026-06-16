"""Pluggable component interfaces. One Protocol/ABC per responsibility."""

from .base import Component
from .biomechanics import BiomechanicsAnalyzer
from .court_calibrator import CourtCalibrator
from .detector import Detector
from .hit_detector import HitDetector
from .player_tracker import PlayerTracker
from .pose_estimator import PoseEstimator
from .reconstructor import Reconstructor3D
from .shot_classifier import ShotClassifier
from .shuttle_tracker import ShuttleTracker

#: Maps the config 'kind' string to its interface, for registry validation.
INTERFACE_BY_KIND: dict[str, type[Component]] = {
    "detector": Detector,
    "shuttle_tracker": ShuttleTracker,
    "pose_estimator": PoseEstimator,
    "player_tracker": PlayerTracker,
    "court_calibrator": CourtCalibrator,
    "reconstructor": Reconstructor3D,
    "hit_detector": HitDetector,
    "shot_classifier": ShotClassifier,
    "biomechanics": BiomechanicsAnalyzer,
}

__all__ = [
    "Component",
    "Detector",
    "ShuttleTracker",
    "PoseEstimator",
    "PlayerTracker",
    "CourtCalibrator",
    "Reconstructor3D",
    "HitDetector",
    "ShotClassifier",
    "BiomechanicsAnalyzer",
    "INTERFACE_BY_KIND",
]
