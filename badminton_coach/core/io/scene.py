"""Scene-cut detection: know when the broadcast leaves the main rally camera.

Broadcasts cut to replays / close-ups / different angles. When that happens the
cached calibration is invalid and analysis should pause (or re-bootstrap on
return). A cheap HSV-histogram correlation between consecutive frames flags cuts.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..schemas import Frame


def _hsv_hist(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


class SceneCutDetector:
    """Flags a scene cut when consecutive-frame histogram correlation drops below
    `threshold` (correlation 1.0 = identical, lower = more different)."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._prev_hist: np.ndarray | None = None

    def is_cut(self, frame: Frame) -> bool:
        hist = _hsv_hist(frame.image)
        if self._prev_hist is None:
            self._prev_hist = hist
            return False
        corr = cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_CORREL)
        self._prev_hist = hist
        return corr < self.threshold
