"""Detect the badminton playing-surface region (green / blue / red mats).

Standardized courts use green, blue or red surfaces. We mask each colour, take the
dominant one, and return its largest connected region (filled + dilated) — used to
gate court-line detection so scoreboard/ad/subtitle pixels outside the court are
ignored.
"""

from __future__ import annotations

import cv2
import numpy as np

_SURFACE_HSV = {
    "green": [((28, 15, 15), (100, 255, 255))],
    "blue": [((90, 40, 30), (130, 255, 255))],
    "red": [((0, 70, 40), (10, 255, 255)), ((165, 70, 40), (180, 255, 255))],
}


def court_surface_mask(img_bgr: np.ndarray, dilate: int = 45) -> np.ndarray:
    """Largest court-colour region (green/blue/red), filled and dilated, uint8 mask."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    best_mask, best_area = None, 0
    for ranges in _SURFACE_HSV.values():
        m = np.zeros(img_bgr.shape[:2], np.uint8)
        for lo, hi in ranges:
            m |= cv2.inRange(hsv, lo, hi)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((81, 81), np.uint8))
        area = int(m.sum())
        if area > best_area:
            best_area, best_mask = area, m
    if best_mask is None:
        return np.zeros(img_bgr.shape[:2], np.uint8)
    cnts, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros(img_bgr.shape[:2], np.uint8)
    roi = np.zeros(img_bgr.shape[:2], np.uint8)
    cv2.drawContours(roi, [cv2.convexHull(max(cnts, key=cv2.contourArea))], -1, 255, -1)
    if dilate > 0:
        roi = cv2.dilate(roi, np.ones((dilate, dilate), np.uint8))
    return roi


def court_line_mask(img_bgr: np.ndarray, roi: np.ndarray | None = None,
                    gate: bool = True) -> np.ndarray:
    """Thin white court LINES via local (top-hat) thresholding inside the court surface.

    Court lines are thin, high-brightness, low-saturation strokes that are LOCALLY
    brighter than the surrounding uniform court colour. A white top-hat (image minus
    its morphological opening) keeps such thin bright structures and removes broad
    bright areas (floor glare / reflections / large highlights), which a plain
    HSV white threshold wrongly keeps. We also require low saturation (white, not a
    bright colour) and drop blob-like components (lines are elongated, small-area)."""
    if gate and roi is None:
        roi = court_surface_mask(img_bgr)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v, s = hsv[:, :, 2], hsv[:, :, 1]
    h, w = v.shape
    # kernel wider than a line but narrower than glare patches (scaled to image size)
    k = max(11, (min(h, w) // 60) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tophat = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, kernel)
    thr = max(18, int(0.6 * tophat[tophat > 0].mean()) if (tophat > 0).any() else 18)
    line = ((tophat >= thr) & (s <= 80) & (v >= 120)).astype(np.uint8) * 255
    if gate and roi is not None:
        line = cv2.bitwise_and(line, roi)
    # drop blob-like components (glare survivors): keep thin/elongated ones
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(line, 8)
    out = np.zeros_like(line)
    area_cap = 0.004 * h * w  # ~0.4% of the frame
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        fill = a / max(1, bw * bh)
        elongated = max(bw, bh) >= 3 * max(1, min(bw, bh))
        if a <= area_cap or elongated or fill < 0.4:  # thin line, or long, or sparse
            out[lbl == i] = 255
    return out
