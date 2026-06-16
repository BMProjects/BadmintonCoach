"""Always-available no-op detector.

Lets the pipeline run end-to-end (architecture smoke tests, dependency-free CI)
before real detector weights are installed. Returns no detections.
"""

from __future__ import annotations

from ...core.interfaces import Detector
from ...core.registry import register
from ...core.schemas import Detection, Frame


@register("detector", "null")
class NullDetector(Detector):
    @classmethod
    def is_available(cls) -> bool:
        return True

    def detect(self, frame: Frame) -> list[Detection]:
        return []
