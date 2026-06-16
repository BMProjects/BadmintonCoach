"""Named court lines as groups of the 22 keypoint indices, + intersection decode.

Each named line is the set of (collinear) keypoint indices lying on it. Every one of
the 22 keypoints is the intersection of exactly one horizontal and one vertical line,
so a predicted keypoint = intersection of its two predicted named lines (this is the
'line heatmap + intersection' decode, robust to off-frame corners via extrapolation).
"""

from __future__ import annotations

import cv2
import numpy as np

# Horizontal lines (constant world y), far -> near:
HORIZ = [
    [0, 1, 2, 3, 4],        # far baseline
    [5, 6],                 # far doubles long-service
    [7, 8, 9],              # far short-service
    [10, 11],               # net
    [12, 13, 14],           # near short-service
    [15, 16],               # near doubles long-service
    [17, 18, 19, 20, 21],   # near baseline
]
# Vertical lines (constant world x), left -> right:
VERT = [
    [0, 5, 7, 10, 12, 15, 17],   # left doubles sideline
    [1, 18],                     # left singles sideline
    [2, 8, 13, 19],              # centre line
    [3, 20],                     # right singles sideline
    [4, 6, 9, 11, 14, 16, 21],   # right doubles sideline
]
LINES = HORIZ + VERT          # 12 channels (0..6 horiz, 7..11 vert)
N_LINES = len(LINES)
N_KPTS = 22

# 180° court-symmetry keypoint permutation: a 180° rotation about court centre maps
# keypoint i -> 21-i (verified from official world coords). Used for flip augmentation
# to test whether the named-channel model can resolve viewing-end from image content.
SYM_PERM = [N_KPTS - 1 - i for i in range(N_KPTS)]

# keypoint index -> (horizontal channel, vertical channel)
KPT_TO_HV: dict[int, tuple[int, int]] = {}
for ci, grp in enumerate(HORIZ):
    for k in grp:
        KPT_TO_HV[k] = (ci, KPT_TO_HV.get(k, (0, 0))[1])
for ci, grp in enumerate(VERT):
    for k in grp:
        h = KPT_TO_HV.get(k, (0, 0))[0]
        KPT_TO_HV[k] = (h, len(HORIZ) + ci)


def fit_line(points: np.ndarray) -> np.ndarray | None:
    """Homogeneous line (a,b,c) through >=2 points, normalized so a^2+b^2=1."""
    if len(points) < 2:
        return None
    vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    # line direction (vx,vy) through (x0,y0) -> normal (vy,-vx)
    a, b = float(vy), float(-vx)
    c = -(a * x0 + b * y0)
    n = (a * a + b * b) ** 0.5 or 1.0
    return np.array([a / n, b / n, c / n])


def fit_line_weighted(points: np.ndarray, weights: np.ndarray) -> np.ndarray | None:
    """Sub-pixel homogeneous line via weighted total-least-squares (weighted PCA).

    points: (N,2), weights: (N,) heatmap intensities. The ridge centre is recovered
    at sub-pixel accuracy because intensity weighting locates the blurred line centroid
    rather than snapping to integer pixels. Returns (a,b,c) with a^2+b^2=1.
    """
    if len(points) < 2:
        return None
    w = weights.astype(np.float64)
    sw = w.sum()
    if sw < 1e-9:
        return None
    p = points.astype(np.float64)
    mean = (w[:, None] * p).sum(0) / sw
    d = p - mean
    cov = (w[:, None, None] * (d[:, :, None] @ d[:, None, :])).sum(0) / sw
    eigval, eigvec = np.linalg.eigh(cov)
    direction = eigvec[:, eigval.argmax()]  # principal axis
    a, b = float(direction[1]), float(-direction[0])  # normal = perp to direction
    c = -(a * mean[0] + b * mean[1])
    n = (a * a + b * b) ** 0.5 or 1.0
    return np.array([a / n, b / n, c / n])


def fit_line_ransac(points: np.ndarray, weights: np.ndarray,
                    thr: float = 2.0, iters: int = 60, seed: int = 0) -> np.ndarray | None:
    """RANSAC line fit then weighted refit on inliers — kills outlier ridge pixels.

    Sparse/noisy channels (e.g. a bleed-through from an adjacent line) drag a plain
    weighted fit and blow up far off-frame intersections; RANSAC selects the dominant
    ridge first. thr is the inlier distance in input px.
    """
    n = len(points)
    if n < 2:
        return None
    if n < 4:
        return fit_line_weighted(points, weights)
    rng = np.random.default_rng(seed)
    best, best_w = None, 0.0
    nrm = points.astype(np.float64)
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        line = fit_line(points[[i, j]])
        if line is None:
            continue
        d = np.abs(nrm @ line[:2] + line[2])
        mask = d < thr
        wsum = float(weights[mask].sum())
        if wsum > best_w:
            best_w, best = wsum, mask
    if best is None or best.sum() < 2:
        return fit_line_weighted(points, weights)
    return fit_line_weighted(points[best], weights[best])


def fit_line_robust(points: np.ndarray, weights: np.ndarray,
                    k: float = 3.0, iters: int = 2) -> np.ndarray | None:
    """Weighted fit + MAD outlier rejection refit. Keeps the full ridge body (no cost on
    clean lines) while dropping a minority second mode (e.g. an adjacent line bleeding into
    the channel) that would otherwise tilt the fit and blow up far intersections.
    """
    line = fit_line_weighted(points, weights)
    if line is None:
        return None
    p = points.astype(np.float64)
    for _ in range(iters):
        d = np.abs(p @ line[:2] + line[2])
        mad = float(np.median(d))
        mask = d < (k * mad + 1.0)
        if mask.sum() < 2 or mask.all():
            break
        line = fit_line_weighted(points[mask], weights[mask])
    return line


def intersect(l1: np.ndarray, l2: np.ndarray) -> tuple[float, float] | None:
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-9:
        return None
    return float(p[0] / p[2]), float(p[1] / p[2])


def line_endpoints(points: np.ndarray):
    """Extreme endpoints of >=2 (near-collinear) points along their principal axis."""
    line = fit_line(points)
    if line is None:
        return None
    d = np.array([-line[1], line[0]])  # direction = perpendicular to normal
    t = points @ d
    return points[t.argmin()], points[t.argmax()]
