"""RF-DETR detector adapter (SOTA large-object detection, DINOv2 backbone).

Wraps the `rfdetr` pip package. Per the reports, RF-DETR handles players, rackets,
net posts and court lines — NOT the shuttlecock. Class id -> ObjectClass mapping
comes from config['class_map'] once a badminton-domain model is fine-tuned.
"""

from __future__ import annotations

from typing import Any

from ...core.interfaces import Detector
from ...core.registry import register
from ...core.schemas import BBox, Detection, Frame, ObjectClass
from .._util import module_available


@register("detector", "rfdetr")
class RFDETRDetector(Detector):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._model = None
        self._class_map: dict[int, ObjectClass] = {
            int(k): ObjectClass(v) for k, v in self.config.get("class_map", {}).items()
        }

    @classmethod
    def is_available(cls) -> bool:
        return module_available("rfdetr")

    def _ensure_model(self):
        if self._model is None:
            from rfdetr import RFDETRBase  # lazy: heavy import

            self._model = RFDETRBase(
                pretrain_weights=self.config.get("weights"),
                device=self.config.get("device", "cuda"),
            )
        return self._model

    def detect(self, frame: Frame) -> list[Detection]:
        model = self._ensure_model()
        threshold = float(self.config.get("threshold", 0.5))
        preds = model.predict(frame.image, threshold=threshold)  # upstream API
        results: list[Detection] = []
        for xyxy, cls_id, score in zip(preds.xyxy, preds.class_id, preds.confidence, strict=True):
            obj = self._class_map.get(int(cls_id))
            if obj is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            results.append(
                Detection(frame.index, obj, BBox(x1, y1, x2, y2), float(score))
            )
        return results
