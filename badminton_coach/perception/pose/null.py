"""Always-available no-op pose estimator (returns no poses)."""

from __future__ import annotations

from ...core.interfaces import PoseEstimator
from ...core.registry import register
from ...core.schemas import Frame, Pose


@register("pose_estimator", "null")
class NullPoseEstimator(PoseEstimator):
    @classmethod
    def is_available(cls) -> bool:
        return True

    def estimate(self, frame: Frame, boxes: list) -> list[Pose]:
        return []
