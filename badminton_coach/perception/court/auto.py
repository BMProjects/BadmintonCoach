"""Auto court calibrator — run several backends, keep whichever fits the lines best.

Best of: the traditional geometric `line_fit` (no weights, robust to subtitles / amateur /
coloured courts) and the learned `line_heatmap` (named line-heatmap + intersection, sub-pixel
on standard courts). Each is run; the one whose full BWF line model best overlaps the actual
white lines (court_overlap_score) wins. If only one succeeds it is used; if neither, raise.

Config (sub-backend blocks optional, merged with shared device/compute_camera/focal_px):
    court_calibrator:
      backend: auto
      compute_camera: true
      line_fit: {min_overlap: 0.5}
      line_heatmap: {weights: weights/court_lines_evit_b1.pt}
"""

from __future__ import annotations

from typing import Any

from ...core.geometry.court_eval import court_overlap_score
from ...core.interfaces import CourtCalibrator
from ...core.registry import get_backend, register
from ...core.schemas import CourtCalibration, Frame

_SUBS = ("line_fit", "line_heatmap")
_SHARED = ("device", "compute_camera", "focal_px")


@register("court_calibrator", "auto")
class AutoCourtCalibrator(CourtCalibrator):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        shared = {k: self.config[k] for k in _SHARED if k in self.config}
        self._subs = []
        for name in _SUBS:
            cls = get_backend("court_calibrator", name)
            if not cls.is_available():
                continue
            sub_cfg = {**shared, **(self.config.get(name) or {}), "backend": name}
            self._subs.append(cls(sub_cfg))

    @classmethod
    def is_available(cls) -> bool:
        return True  # degrades to whatever sub-backends are available

    def calibrate(self, frame: Frame) -> CourtCalibration:
        cands: list[tuple[float, CourtCalibration, str]] = []
        for sub in self._subs:
            try:
                c = sub.calibrate(frame)
            except Exception:  # noqa: BLE001 - try the other backend
                continue
            cands.append((court_overlap_score(frame.image, c), c, sub.backend_name))
        if not cands:
            raise RuntimeError("auto: no court calibrator succeeded on this frame")
        cands.sort(key=lambda x: x[0], reverse=True)
        return cands[0][1]

    def is_present(self, frame: Frame) -> bool:
        return any(sub.is_present(frame) for sub in self._subs)
