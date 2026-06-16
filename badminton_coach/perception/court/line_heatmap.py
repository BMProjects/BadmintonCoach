"""Learned court calibrator — named line-heatmap + intersection model.

Predicts 12 named court-line heatmaps (7 horizontal + 5 vertical), fits a line per
channel and intersects each keypoint's horizontal & vertical line to recover the 22
court-line intersections (training/court_lines). This is the P1-P3 experiment winner
(efficientvit_b1: sub-pixel median, low tail under viewpoint warp). The named-channel
design resolves the court's 180° symmetry from image content, so no RANSAC identity
search is needed on low-angle amateur footage.

Config:
    weights:    path to the trained checkpoint (.pt from training/court_lines/train.py)
    world_map:  path to the index->world-xy JSON (shared with the heatmap backend)
    device, threshold (min channel confidence), decode, compute_camera, focal_px
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ...core.geometry import estimate_camera, solve_homography
from ...core.interfaces import CourtCalibrator
from ...core.registry import register
from ...core.schemas import CourtCalibration, Frame, Point2D, Point3D
from .._util import module_available

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_DEFAULT_WEIGHTS = "weights/court_lines_evit_b1.pt"
_DEFAULT_WORLD = "weights/court_kp_official_world.json"


@register("court_calibrator", "line_heatmap")
class LineHeatmapCourtCalibrator(CourtCalibrator):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._model = None
        self._world = None
        self._input = None
        self._device = None
        self._threshold = float(self.config.get("threshold", 0.3))
        self._decode_mode = str(self.config.get("decode", "robust"))
        self._compute_camera = bool(self.config.get("compute_camera", False))
        self._focal_px = self.config.get("focal_px")
        self._min_keypoints = int(self.config.get("min_keypoints", 6))
        self._max_reproj_px = float(self.config.get("max_reproj_px", 25.0))
        # Phantom rejection: line intersections can extrapolate a full court from
        # spurious pixels (geometrically self-consistent but wrong). Require the
        # reprojected BWF model to actually overlap the painted white lines. 0 = off.
        self._min_overlap = float(self.config.get("min_overlap", 0.30))
        # Incomplete-court rejection: require enough confident keypoints that lie
        # INSIDE the frame (not extrapolated off-screen). A partial/absent court yields
        # few in-frame observations -> reject rather than invent a wrong 3D mapping.
        self._min_observed = int(self.config.get("min_observed", 10))
        # 180° orientation tiebreak: the court is point-symmetric, so the named labels
        # can be flipped end-for-end. Resolve from perspective foreshortening (the
        # baseline nearer the camera spans more image pixels). Off for near-orthographic
        # views where the cue is unreliable and orientation barely matters.
        self._orient_tiebreak = bool(self.config.get("orientation_tiebreak", True))
        self._orient_margin = float(self.config.get("orientation_margin", 1.15))

    @classmethod
    def is_available(cls) -> bool:
        return module_available("torch") and module_available("timm")

    def _ensure(self):
        if self._model is not None:
            return
        import torch

        from training.court_lines.model import build_model

        weights = self.config.get("weights", _DEFAULT_WEIGHTS)
        ckpt = torch.load(weights, map_location="cpu")
        self._input = int(ckpt["input_size"])
        model = build_model(ckpt["backbone"], ckpt["n_lines"], pretrained=False,
                            input_size=self._input, out_div=ckpt["out_div"])
        model.load_state_dict(ckpt["model"])
        self._device = self.config.get("device", "cuda")
        if str(self._device).startswith("cuda") and not torch.cuda.is_available():
            self._device = "cpu"
        model.to(self._device).eval()
        self._model = model
        self._world = json.loads(
            Path(self.config.get("world_map", _DEFAULT_WORLD)).read_text()
        )

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """BGR frame -> normalized (3, n, n) float32 ready to stack into a batch."""
        import cv2

        n = self._input
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (n, n)).astype(np.float32) / 255.0
        return ((resized - _MEAN) / _STD).transpose(2, 0, 1)

    def _decode(self, heatmaps: np.ndarray, width: int, height: int):
        """Per-frame line heatmaps (12, h, w) -> (image_pts, world_pts, n_observed)."""
        from training.court_lines.decode import decode_lines

        n = self._input
        pts, conf = decode_lines(heatmaps, n, n, decode=self._decode_mode)
        if self._orient_tiebreak and self._needs_flip(pts):
            from training.court_lines.lines import SYM_PERM
            pts = {SYM_PERM[k]: v for k, v in pts.items()}
            conf = {SYM_PERM[k]: v for k, v in conf.items()}
        sx, sy = width / n, height / n
        image_pts: list[Point2D] = []
        world_pts: list[Point3D] = []
        n_observed = 0
        for i, (px, py) in pts.items():
            if self._world[i] is None or conf[i] < self._threshold:
                continue
            ix, iy = float(px * sx), float(py * sy)
            if 0 <= ix < width and 0 <= iy < height:
                n_observed += 1
            image_pts.append(Point2D(ix, iy))
            world_pts.append(Point3D(float(self._world[i][0]), float(self._world[i][1]), 0.0))
        return image_pts, world_pts, n_observed

    def _forward(self, images: list[np.ndarray]) -> np.ndarray:
        """Batched GPU forward over a list of BGR frames -> (B, 12, h, w) numpy.

        Runs under bf16 autocast on CUDA (the model was trained bf16-AMP) — roughly
        halves the GPU time vs fp32 with no accuracy loss for this task.
        """
        import torch

        self._ensure()
        batch = np.stack([self._preprocess(im) for im in images])
        tensor = torch.from_numpy(batch).to(self._device)
        amp = str(self._device).startswith("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            return self._model(tensor).float().cpu().numpy()

    def _predict(self, frame: Frame):
        """Run the model -> confident (image_pt, world_pt) correspondences."""
        heatmaps = self._forward([frame.image])[0]
        return self._decode(heatmaps, frame.width, frame.height)

    def _gate(self, image_pts, world_pts, n_observed) -> bool:
        """Cheap per-frame court-presence gate (no white-line overlap)."""
        if len(image_pts) < self._min_keypoints or n_observed < self._min_observed:
            return False
        return solve_homography(image_pts, world_pts).reprojection_error_px < self._max_reproj_px

    def present_frames(self, frames: list[Frame], batch_size: int = 32) -> set[int]:
        """Batched per-frame presence: one GPU forward per chunk instead of per frame.

        ~5x faster than looping is_present (the model forward dominates once the heavy
        white-line overlap check is excluded). Returns the set of frame indices with a
        court visible.
        """
        present: set[int] = set()
        for start in range(0, len(frames), batch_size):
            chunk = frames[start:start + batch_size]
            heatmaps = self._forward([f.image for f in chunk])
            for f, hm in zip(chunk, heatmaps, strict=True):
                if self._gate(*self._decode(hm, f.width, f.height)):
                    present.add(f.index)
        return present

    def _needs_flip(self, pts: dict[int, tuple[float, float]]) -> bool:
        """True if the far-labeled baseline appears clearly wider (more foreshortening-
        free) than the near-labeled one, i.e. the named end-labels are 180° flipped.

        Uses the doubles baseline corners: far = kpts 0 & 4, near = kpts 17 & 21.
        Returns False (trust the model) when either baseline is incomplete or the two
        widths are within the margin (ambiguous high-angle view).
        """
        if not ({0, 4} <= pts.keys() and {17, 21} <= pts.keys()):
            return False
        w_far = float(np.hypot(pts[0][0] - pts[4][0], pts[0][1] - pts[4][1]))
        w_near = float(np.hypot(pts[17][0] - pts[21][0], pts[17][1] - pts[21][1]))
        return w_far > w_near * self._orient_margin

    def _overlaps_white_lines(self, frame: Frame, calib: CourtCalibration) -> bool:
        if self._min_overlap <= 0:
            return True
        from ...core.geometry.court_eval import court_overlap_score

        return court_overlap_score(frame.image, calib) >= self._min_overlap

    def is_present(self, frame: Frame) -> bool:
        # Cheap per-frame gate: model forward + keypoint/reproj checks only. The
        # white-line overlap check (~0.5s/frame) is reserved for the one-time
        # calibrate(); per-frame batching is in present_frames().
        return self._gate(*self._predict(frame))

    def calibrate(self, frame: Frame) -> CourtCalibration:
        image_pts, world_pts, n_observed = self._predict(frame)
        if len(image_pts) < self._min_keypoints:
            raise RuntimeError(
                f"line_heatmap: only {len(image_pts)} confident keypoints "
                f"(<{self._min_keypoints})."
            )
        if n_observed < self._min_observed:
            raise RuntimeError(
                f"line_heatmap: only {n_observed} in-frame keypoints "
                f"(<{self._min_observed}) - court absent or incomplete."
            )
        calib = solve_homography(image_pts, world_pts)
        if calib.reprojection_error_px >= self._max_reproj_px:
            raise RuntimeError(
                f"line_heatmap: keypoints inconsistent "
                f"(reproj {calib.reprojection_error_px:.0f}px) - likely not a real court."
            )
        if not self._overlaps_white_lines(frame, calib):
            raise RuntimeError(
                "line_heatmap: reprojected court does not overlap painted white lines "
                f"(<{self._min_overlap:.0%}) - likely a phantom / out-of-distribution court."
            )
        camera = None
        if self._compute_camera:
            camera = estimate_camera(
                image_pts, world_pts, (frame.width, frame.height), self._focal_px
            )
        return CourtCalibration(
            homography=calib.homography,
            reprojection_error_px=calib.reprojection_error_px,
            camera=camera,
        )
