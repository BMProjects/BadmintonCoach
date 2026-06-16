"""YOLO-pose estimator adapter (Ultralytics YOLO11-pose, native COCO-17).

A single-stage pose model that detects people and their 17 COCO keypoints in one
forward pass. Runs on the full frame, then matches each detected person to the
caller's player boxes (so only the on-court players are returned). Keeps the whole
pipeline on YOLO/torch — no onnxruntime/RTMPose dependency.
"""

from __future__ import annotations

from typing import Any

from ...core.interfaces import PoseEstimator
from ...core.registry import register
from ...core.schemas import BBox, Detection, Frame, Keypoint, ObjectClass, Point2D, Pose
from .._util import module_available


def _iou(a: BBox, b: list[float]) -> float:
    ix1, iy1 = max(a.x1, b[0]), max(a.y1, b[1])
    ix2, iy2 = min(a.x2, b[2]), min(a.y2, b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.width * a.height + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@register("pose_estimator", "yolo_pose")
class YOLOPoseEstimator(PoseEstimator):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._model = None

    @classmethod
    def is_available(cls) -> bool:
        return module_available("ultralytics")

    def _ensure(self):
        if self._model is None:
            from ultralytics import YOLO  # lazy

            self._model = YOLO(self.config.get("weights", "yolo11n-pose.pt"))
        return self._model

    def detect_and_pose(self, frame: Frame) -> tuple[list[Detection], list[Pose]]:
        """One forward -> (person Detections, Poses), index-paired. Lets the pipeline
        skip a separate detector pass (the pose model already yields boxes + keypoints,
        with exact box<->pose association — no IoU matching needed)."""
        model = self._ensure()
        out = model.predict(frame.image, device=self.config.get("device", "cuda"),
                            conf=float(self.config.get("threshold", 0.3)), verbose=False)[0]
        if out.keypoints is None or len(out.keypoints) == 0:
            return [], []
        kp_data = out.keypoints.data.cpu().numpy()        # (N, 17, 3)
        xyxy = out.boxes.xyxy.cpu().numpy()               # (N, 4)
        conf = out.boxes.conf.cpu().numpy()               # (N,)
        dets, poses = [], []
        for i in range(len(kp_data)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            dets.append(Detection(frame_index=frame.index, cls=ObjectClass.PLAYER,
                                  bbox=BBox(x1, y1, x2, y2), confidence=float(conf[i])))
            kps = tuple(Keypoint(Point2D(float(x), float(y)), float(c)) for x, y, c in kp_data[i])
            poses.append(Pose(frame.index, kps))
        return dets, poses

    def estimate(self, frame: Frame, boxes: list) -> list[Pose]:
        model = self._ensure()
        out = model.predict(frame.image, device=self.config.get("device", "cuda"),
                            conf=float(self.config.get("threshold", 0.3)), verbose=False)[0]
        if out.keypoints is None or len(out.keypoints) == 0:
            return []
        kp_data = out.keypoints.data.cpu().numpy()        # (N, 17, 3) x,y,conf
        det_boxes = out.boxes.xyxy.cpu().numpy().tolist()  # (N, 4)

        def to_pose(i: int) -> Pose:
            kps = tuple(
                Keypoint(Point2D(float(x), float(y)), float(c)) for x, y, c in kp_data[i]
            )
            return Pose(frame.index, kps)

        if not boxes:  # no player filter -> return every detected person
            return [to_pose(i) for i in range(len(kp_data))]

        poses: list[Pose] = []
        for box in boxes:  # one pose per requested player box (best IoU match)
            ious = [_iou(box, db) for db in det_boxes]
            j = max(range(len(ious)), key=lambda k: ious[k]) if ious else -1
            if j >= 0 and ious[j] > 0.1:
                poses.append(to_pose(j))
        return poses
