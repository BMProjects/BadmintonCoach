"""Line-structure court calibrator — the standard sports court-registration method.

Mirrors the mature approach used by popular tennis/badminton court-detection repos
(e.g. TEXflip/tennis-court-detection, yastrebksv/TennisCourtDetector):
  1. mask the white court lines, gated to the green playing surface (drops ad-board
     and stand graphics);
  2. detect line segments (Hough), merge into horizontal / vertical court lines;
  3. take a few outer-line CANDIDATES per side; for every combination compute the
     homography from the 4 corner intersections and SCORE it by how well the FULL
     BWF court model (all lines) reprojects onto white pixels — pick the best;
  4. REFINE the 4 corners (Nelder-Mead) to maximize that overlap.

The global candidate search + full-model overlap scoring is what makes it accurate
(a single set of "extreme" lines is fragile; the score self-corrects). Calibration
is accepted only if the overlap exceeds a threshold, else it raises so the pipeline
can try another frame / fall back.
"""

from __future__ import annotations

import itertools
from typing import Any

import cv2
import numpy as np

from ...core.geometry import estimate_camera, solve_homography
from ...core.geometry.court_model import (
    COURT_LENGTH_M,
    COURT_WIDTH_DOUBLES_M,
    COURT_WIDTH_SINGLES_M,
    DOUBLES_LONG_SERVICE_FROM_BACK_M,
    SHORT_SERVICE_FROM_NET_M,
    court_corners_doubles,
)
from ...core.interfaces import CourtCalibrator
from ...core.registry import register
from ...core.schemas import CourtCalibration, Frame, Point2D, Point3D

_MIN_OVERLAP = 0.55      # accept a fit only if >= this fraction of model aligns
_DT_TOL_PX = 6.0         # a model point "hits" a court line within this distance
_WORLD4 = np.array(court_corners_doubles(), dtype=np.float64)


def _court_model_segments() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """All BWF court lines in world ground coords (meters), for overlap scoring."""
    w, ws, length = COURT_WIDTH_DOUBLES_M, COURT_WIDTH_SINGLES_M, COURT_LENGTH_M
    sx = (w - ws) / 2
    net, ss, dl = length / 2, SHORT_SERVICE_FROM_NET_M, DOUBLES_LONG_SERVICE_FROM_BACK_M
    return [
        ((0, 0), (w, 0)), ((0, length), (w, length)),               # baselines
        ((0, 0), (0, length)), ((w, 0), (w, length)),               # doubles sidelines
        ((sx, 0), (sx, length)), ((w - sx, 0), (w - sx, length)),   # singles sidelines
        ((0, net - ss), (w, net - ss)), ((0, net + ss), (w, net + ss)),  # short service
        ((0, dl), (w, dl)), ((0, length - dl), (w, length - dl)),   # doubles long service
        ((w / 2, 0), (w / 2, net - ss)), ((w / 2, net + ss), (w / 2, length)),  # centre
    ]


_MODEL = _court_model_segments()


def _court_roi(img: np.ndarray) -> np.ndarray:
    """Mask of the playing surface (green/blue/red), dilated, to gate out graphics."""
    from ...core.geometry.court_surface import court_surface_mask

    return court_surface_mask(img)


def _white_mask(img: np.ndarray) -> np.ndarray:
    # thin white court lines via local top-hat (rejects floor glare/reflections);
    # ungated here (line_fit gates to the surface ROI itself where needed)
    from ...core.geometry.court_surface import court_line_mask

    return court_line_mask(img, gate=False)


def _intersect(l1: np.ndarray, l2: np.ndarray) -> Point2D:
    p = np.cross(l1, l2)
    return Point2D(float(p[0] / p[2]), float(p[1] / p[2]))


def _hough(mask: np.ndarray):
    raw = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=60, minLineLength=180, maxLineGap=40)
    return None if raw is None else raw[:, 0]


def _merge_lines(segs, vertical: bool, merge_tol: float = 18.0):
    """Merge near-collinear segments into single fitted lines."""
    items = []
    for x1, y1, x2, y2 in segs:
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if (ang > 45) != vertical:
            continue
        span = float(np.hypot(x2 - x1, y2 - y1))
        key = (x1 + x2) / 2 if vertical else (y1 + y2) / 2
        items.append((key, [(x1, y1), (x2, y2)], span))
    items.sort(key=lambda it: it[0])

    merged = []
    for key, pts, _span in items:
        if merged and abs(key - merged[-1]["key"]) < merge_tol:
            g = merged[-1]
            g["pts"].extend(pts)
            g["keys"].append(key)
            g["key"] = float(np.mean(g["keys"]))
        else:
            merged.append({"key": key, "keys": [key], "pts": list(pts)})

    lines = []
    for g in merged:
        arr = np.array(g["pts"], dtype=np.float32)
        vx, vy, x0, y0 = cv2.fitLine(arr, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        line = np.cross([x0, y0, 1.0], [x0 + vx, y0 + vy, 1.0])
        line = line / np.linalg.norm(line[:2])
        xs, ys = arr[:, 0], arr[:, 1]
        extent = (ys.max() - ys.min()) if vertical else (xs.max() - xs.min())
        lines.append({"line": line, "key": g["key"], "extent": float(extent)})
    return lines


def _overlap_score(homog_w2i: np.ndarray, dist: np.ndarray, shape) -> float:
    """Fraction of sampled model-line points landing within _DT_TOL_PX of a white px."""
    h, w = shape
    hits = total = 0
    for (ax, ay), (bx, by) in _MODEL:
        for t in np.linspace(0, 1, 40):
            p = homog_w2i @ [ax + (bx - ax) * t, ay + (by - ay) * t, 1.0]
            if abs(p[2]) < 1e-9:
                continue
            fx, fy = p[0] / p[2], p[1] / p[2]
            if not (np.isfinite(fx) and np.isfinite(fy)):
                continue
            x, y = int(fx), int(fy)
            if 0 <= x < w and 0 <= y < h:
                total += 1
                if dist[y, x] < _DT_TOL_PX:
                    hits += 1
    return hits / total if total else 0.0


def _register_court(img: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Return (4 corner image points [near-L,near-R,far-R,far-L], overlap) or None."""
    roi = _court_roi(img)
    white = cv2.bitwise_and(_white_mask(img), roi)
    rows = np.where(roi.any(axis=1))[0]
    if len(rows) == 0:
        return None
    court_top = int(rows.min())
    dist = cv2.distanceTransform(255 - white, cv2.DIST_L2, 5)

    gated = _hough(white)
    full = _hough(_white_mask(img))
    if gated is None or full is None:
        return None
    vert = sorted((v for v in _merge_lines(gated, vertical=True) if v["extent"] > 200),
                  key=lambda v: v["key"])
    horiz = sorted((h for h in _merge_lines(full, vertical=False)
                    if h["extent"] > 250 and h["key"] > court_top - 15), key=lambda h: h["key"])
    if len(vert) < 2 or len(horiz) < 2:
        return None

    lefts, rights = vert[:2], vert[-2:]
    tops, bottoms = horiz[:3], horiz[-3:]
    best = None
    for lt, rt, tp, bt in itertools.product(lefts, rights, tops, bottoms):
        if tp["key"] >= bt["key"]:
            continue
        corners = np.array([
            [_intersect(bt["line"], lt["line"]).x, _intersect(bt["line"], lt["line"]).y],
            [_intersect(bt["line"], rt["line"]).x, _intersect(bt["line"], rt["line"]).y],
            [_intersect(tp["line"], rt["line"]).x, _intersect(tp["line"], rt["line"]).y],
            [_intersect(tp["line"], lt["line"]).x, _intersect(tp["line"], lt["line"]).y],
        ], dtype=np.float64)
        homog, _ = cv2.findHomography(_WORLD4, corners)
        if homog is None:
            continue
        s = _overlap_score(homog, dist, white.shape)
        if best is None or s > best[1]:
            best = (corners, s)
    if best is None:
        return None

    refined = _refine(best[0], dist, white.shape)
    return refined


def _refine(corners: np.ndarray, dist: np.ndarray, shape) -> tuple[np.ndarray, float]:
    from scipy.optimize import minimize

    def neg(x):
        homog, _ = cv2.findHomography(_WORLD4, x.reshape(4, 2))
        return 1.0 - _overlap_score(homog, dist, shape) if homog is not None else 1.0

    res = minimize(neg, corners.ravel(), method="Nelder-Mead",
                   options={"xatol": 0.5, "fatol": 1e-4, "maxiter": 6000})
    return res.x.reshape(4, 2), 1.0 - res.fun


def detect_court_corners_linefit(img: np.ndarray) -> list[Point2D] | None:
    """Return [near-L, near-R, far-R, far-L] court corners, or None."""
    out = _register_court(img)
    if out is None:
        return None
    corners, _ = out
    return [Point2D(float(x), float(y)) for x, y in corners]


@register("court_calibrator", "line_fit")
class LineFitCourtCalibrator(CourtCalibrator):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._min_overlap = float(self.config.get("min_overlap", _MIN_OVERLAP))
        self._compute_camera = bool(self.config.get("compute_camera", False))
        self._focal_px = self.config.get("focal_px")

    @classmethod
    def is_available(cls) -> bool:
        from .._util import module_available

        return module_available("scipy")

    def calibrate(self, frame: Frame) -> CourtCalibration:
        out = _register_court(frame.image)
        if out is None:
            raise RuntimeError("line_fit: could not find enough court lines.")
        corners, overlap = out
        if overlap < self._min_overlap:
            raise RuntimeError(f"line_fit: court fit overlap {overlap:.2f} < {self._min_overlap}.")

        image_pts = [Point2D(float(x), float(y)) for x, y in corners]
        world = [Point3D(x, y, 0.0) for x, y in court_corners_doubles()]
        calib = solve_homography(image_pts, world)
        camera = None
        if self._compute_camera:
            camera = estimate_camera(image_pts, world, (frame.width, frame.height), self._focal_px)
        return CourtCalibration(
            homography=calib.homography,
            reprojection_error_px=calib.reprojection_error_px,
            camera=camera,
        )
