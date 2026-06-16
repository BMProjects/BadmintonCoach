"""TrackNetV3 shuttle tracker adapter (heatmap regression).

Vendored at third_party/TrackNetV3 (qaz812345/TrackNetV3). This adapter reuses the
upstream model + helpers (get_model, to_img, predict_location) and replicates the
dataset's per-frame preprocessing for bg_mode in {'', 'concat'} — the configs the
released weights ship with. It maps heatmap peaks to our ShuttleTrajectory2D.

Provide the checkpoint via config['weights'] (the upstream .pt with a 'param_dict'
holding seq_len + bg_mode). The pipeline's io.clip_window should equal the model's
seq_len. InpaintNet occlusion repair is a documented follow-up (config hook
'inpaintnet_weights' reserved).
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from ...core.interfaces import ShuttleTracker
from ...core.registry import register
from ...core.schemas import FrameClip, Point2D, ShuttlePoint2D, ShuttleTrajectory2D
from .._util import THIRD_PARTY, module_available, submodule_available

_TN_ROOT = THIRD_PARTY / "TrackNetV3"


def _predict_location(heatmap: np.ndarray) -> tuple[int, int, int, int]:
    """Largest-contour bbox of a uint8 heatmap (inlined from upstream test.py to
    avoid its evaluation-only pycocotools dependency)."""
    import cv2

    if heatmap.max() == 0:
        return 0, 0, 0, 0
    cnts, _ = cv2.findContours(heatmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(c) for c in cnts]
    best = max(rects, key=lambda r: r[2] * r[3])
    return best


@register("shuttle_tracker", "tracknetv3")
class TrackNetV3Tracker(ShuttleTracker):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._model = None
        self._seq_len = 0
        self._bg_mode = ""

    @classmethod
    def is_available(cls) -> bool:
        return module_available("torch") and submodule_available("TrackNetV3")

    def _ensure_upstream_on_path(self) -> None:
        p = str(_TN_ROOT)
        if p not in sys.path:
            sys.path.insert(0, p)

    def _ensure_model(self):
        if self._model is not None:
            return
        import torch

        self._ensure_upstream_on_path()
        from utils.general import get_model  # type: ignore

        weights = self.config.get("weights")
        if not weights:
            raise ValueError("TrackNetV3 needs config['weights'] (upstream checkpoint .pt)")
        device = self.config.get("device", "cuda")
        ckpt = torch.load(weights, map_location=device)
        self._seq_len = int(ckpt["param_dict"]["seq_len"])
        self._bg_mode = ckpt["param_dict"].get("bg_mode", "")
        model = get_model("TrackNet", self._seq_len, self._bg_mode).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        self._model = model

    def _build_input(self, clip: FrameClip):
        """Replicate the upstream per-frame transform -> model input tensor."""
        import cv2
        from utils.general import HEIGHT, WIDTH  # type: ignore

        chw = []
        for frame in clip.frames:
            rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (WIDTH, HEIGHT)).astype(np.float32) / 255.0
            chw.append(np.moveaxis(resized, -1, 0))  # (3, H, W)
        stacked = np.concatenate(chw, axis=0)  # (seq_len*3, H, W)

        if self._bg_mode == "concat":
            median = np.median(np.stack(chw, axis=0), axis=0)  # (3, H, W)
            stacked = np.concatenate([median, stacked], axis=0)  # ((seq_len+1)*3, H, W)
        elif self._bg_mode not in ("", "concat"):
            raise NotImplementedError(
                f"bg_mode={self._bg_mode!r} not supported by this adapter (use '' or 'concat')"
            )
        return stacked

    def track(self, clip: FrameClip) -> ShuttleTrajectory2D:
        import torch

        self._ensure_model()
        if len(clip) != self._seq_len:
            raise ValueError(
                f"TrackNetV3 expects clip_window == seq_len ({self._seq_len}), got {len(clip)}. "
                f"Set io.clip_window accordingly."
            )

        from utils.general import HEIGHT, WIDTH, to_img  # type: ignore

        device = self.config.get("device", "cuda")
        x = torch.from_numpy(self._build_input(clip)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            y_pred = self._model(x)  # (1, seq_len, H, W), sigmoid heatmaps
        heatmaps = (y_pred > 0.5).detach().cpu().numpy()[0]  # (seq_len, H, W) bool

        ref = clip.frames[0].image
        w_scaler, h_scaler = ref.shape[1] / WIDTH, ref.shape[0] / HEIGHT

        points: list[ShuttlePoint2D] = []
        for local_i, frame in enumerate(clip.frames):
            x1, y1, bw, bh = _predict_location(to_img(heatmaps[local_i]))
            cx, cy = (x1 + bw / 2) * w_scaler, (y1 + bh / 2) * h_scaler
            visible = not (bw == 0 and bh == 0)
            points.append(
                ShuttlePoint2D(
                    frame_index=frame.index,
                    point=Point2D(float(cx), float(cy)),
                    confidence=1.0 if visible else 0.0,
                    visible=visible,
                )
            )
        return ShuttleTrajectory2D(points=tuple(points))
