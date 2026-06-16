"""BoT-SORT player tracker adapter (ReID + camera-motion compensation).

Wraps `boxmot`'s BoT-SORT. Unlike the IoU baseline, it re-identifies players by
appearance (OSNet ReID) so crossing / near-net ID-switches are resolved. ReID needs the
frame images, so the pipeline passes `frames` aligned with `detections_per_frame`.

is_available() is False if boxmot isn't installed. ReID weights (osnet_x0_25_msmt17.pt)
auto-download on first use.
"""

from __future__ import annotations

from typing import Any

from ...core.interfaces import PlayerTracker
from ...core.registry import register
from ...core.schemas import BBox, Detection, PlayerTrack, TrackedBox
from .._util import module_available


@register("player_tracker", "botsort")
class BoTSORTPlayerTracker(PlayerTracker):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._tracker = None

    @classmethod
    def is_available(cls) -> bool:
        return module_available("boxmot")

    def _ensure(self):
        if self._tracker is None:
            from pathlib import Path

            import torch
            from boxmot.trackers.tracker_zoo import create_tracker, get_tracker_config

            # boxmot expects a CUDA index ("0"), not "cuda".
            device = str(self.config.get("device", "cuda"))
            if device.startswith("cuda"):
                device = device.split(":")[1] if ":" in device else "0"
            if device != "cpu" and not torch.cuda.is_available():
                device = "cpu"
            self._tracker = create_tracker(
                "botsort",
                tracker_config=get_tracker_config("botsort"),
                reid_weights=Path(self.config.get("reid_weights", "osnet_x0_25_msmt17.pt")),
                device=device,
                half=self.config.get("precision", "fp16") == "fp16" and device != "cpu",
            )
            # Tune for badminton broadcast: accept the pose model's lower-confidence boxes
            # and keep IDs alive across longer occlusions (net crossings / replays).
            for attr, key, default in (
                ("track_high_thresh", "track_high_thresh", 0.35),
                ("new_track_thresh", "new_track_thresh", 0.45),
                ("track_buffer", "track_buffer", 75),
            ):
                if hasattr(self._tracker, attr):
                    setattr(self._tracker, attr, type(default)(self.config.get(key, default)))
        return self._tracker

    def track(self, detections_per_frame: list[list[Detection]],
              frames=None) -> list[PlayerTrack]:
        import numpy as np

        if frames is None:
            raise ValueError("botsort needs frame images for ReID — pass `frames` "
                             "(use the iou tracker if frames are unavailable).")
        tracker = self._ensure()
        tracks: dict[int, list[TrackedBox]] = {}
        for dets, frame in zip(detections_per_frame, frames, strict=True):
            if dets:
                arr = np.array([[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2,
                                 d.confidence, 0] for d in dets], dtype=float)
            else:
                arr = np.empty((0, 6))
            out = np.asarray(tracker.update(arr, frame.image))  # (M,8) xyxy,id,conf,cls,idx
            for row in out:
                tid = int(row[4])
                tracks.setdefault(tid, []).append(
                    TrackedBox(frame.index, tid,
                               BBox(float(row[0]), float(row[1]), float(row[2]), float(row[3])),
                               float(row[5])))
        return [PlayerTrack(tid, tuple(b)) for tid, b in tracks.items()]
