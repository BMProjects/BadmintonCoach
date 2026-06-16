"""Ultralytics YOLO detector adapter (generic — works with any YOLO weights).

Registered as 'yolo'; set the weights file in config (default yolo26n.pt, NMS-free).
"""

from __future__ import annotations

from typing import Any

from ...core.interfaces import Detector
from ...core.registry import register
from ...core.schemas import BBox, Detection, Frame, ObjectClass
from .._util import module_available


@register("detector", "yolo")
class YOLODetector(Detector):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._model = None
        self._class_map: dict[int, ObjectClass] = {
            int(k): ObjectClass(v) for k, v in self.config.get("class_map", {}).items()
        }

    @classmethod
    def is_available(cls) -> bool:
        return module_available("ultralytics")

    def _ensure_model(self):
        if self._model is None:
            from ultralytics import YOLO  # lazy

            self._model = YOLO(self.config.get("weights", "yolo26n.pt"))
        return self._model

    def detect(self, frame: Frame) -> list[Detection]:
        model = self._ensure_model()
        device = self.config.get("device", "cuda")
        conf = float(self.config.get("threshold", 0.25))
        out = model.predict(frame.image, device=device, conf=conf, verbose=False)[0]
        results: list[Detection] = []
        for box in out.boxes:
            obj = self._class_map.get(int(box.cls.item()))
            if obj is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            results.append(
                Detection(frame.index, obj, BBox(x1, y1, x2, y2), float(box.conf.item()))
            )
        return results
