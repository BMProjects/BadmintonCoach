"""Stable-segment median background for robust court calibration.

Calibrating on an arbitrary first frame is fragile: it may be a replay/transition,
or players may occlude the lines. Instead:
  1. sample a frame every `every_ms` (default 200ms -> ~5fps);
  2. measure global motion between consecutive samples with sparse optical flow;
  3. find the most stable window (lowest motion -> a settled rally camera, not a cut);
  4. take the temporal MEDIAN over that window -> moving players/shuttle vanish,
     leaving a clean court (static scoreboard/subtitles remain, but those are handled
     by the green-court ROI gating in the line/keypoint detectors).
The court calibrator then runs on this clean background.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..schemas import Frame


def _motion(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    """Median displacement of tracked corner features between two frames (pixels)."""
    pts = cv2.goodFeaturesToTrack(gray_a, maxCorners=200, qualityLevel=0.01, minDistance=8)
    if pts is None or len(pts) < 10:
        return 0.0
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(gray_a, gray_b, pts, None)
    ok = status.reshape(-1) == 1
    if ok.sum() < 10:
        return 1e9
    disp = np.linalg.norm((nxt - pts).reshape(-1, 2)[ok], axis=1)
    return float(np.median(disp))


def stable_background(frames: list[Frame], fps: float, every_ms: int = 200,
                      window_s: float = 2.0, motion_thresh: float = 2.0):
    """Return (median_background_bgr, info). Falls back to a middle frame if no
    stable window is found. info = {stable, motion, start_frame, n}."""
    if not frames:
        return None, {"stable": False}
    step = max(1, round(fps * every_ms / 1000.0))
    sampled = frames[::step]
    if len(sampled) < 3:
        mid = frames[len(frames) // 2].image
        return mid, {"stable": False, "motion": None, "start_frame": frames[len(frames) // 2].index,
                     "n": 1}

    grays = [cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY) for f in sampled]
    motions = [_motion(grays[i], grays[i + 1]) for i in range(len(grays) - 1)]

    win = max(3, round(window_s / (every_ms / 1000.0)))
    win = min(win, len(sampled))
    best_i, best_motion = 0, float("inf")
    for i in range(len(sampled) - win + 1):
        m = max(motions[i:i + win - 1]) if win > 1 else 0.0
        if m < best_motion:
            best_motion, best_i = m, i

    window = sampled[best_i:best_i + win]
    bg = np.median(np.stack([f.image for f in window], axis=0), axis=0).astype(np.uint8)
    info = {"stable": best_motion <= motion_thresh, "motion": round(best_motion, 2),
            "start_frame": window[0].index, "n": len(window)}
    return bg, info
