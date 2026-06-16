"""Decode predicted named-line heatmaps into the 22 court keypoints.

Each channel is fit to a line; every keypoint is the intersection of its horizontal and
vertical named line (KPT_TO_HV). Shared by training eval and the runtime court backend.
"""

from __future__ import annotations

import numpy as np

from .lines import (
    KPT_TO_HV,
    N_KPTS,
    fit_line,
    fit_line_ransac,
    fit_line_robust,
    fit_line_weighted,
    intersect,
)


def decode_lines(line_hm: np.ndarray, iw: int, ih: int,
                 thr: float = 0.3, decode: str = "robust"):
    """line_hm: (C,Hh,Hw) -> (dict[kpt]->(x,y) in input px, dict[kpt]->confidence).

    decode='hard'     : threshold ridge -> cv2.fitLine on integer coords.
    decode='weighted' : intensity-weighted total-least-squares -> sub-pixel ridge centre.
    decode='robust'   : weighted + MAD outlier rejection refit (default, lowest tail).
    decode='ransac'   : RANSAC dominant-ridge selection -> weighted refit.
    """
    c, hh, hw = line_hm.shape
    sx, sy = iw / hw, ih / hh
    lines, ch_conf = [], []
    for ch in range(c):
        m = line_hm[ch]
        mx = float(m.max())
        ch_conf.append(mx)
        lo = 0.1 if decode in ("weighted", "ransac", "robust") else thr
        ys, xs = np.where(m > lo * mx if mx > 0 else m > 1)
        if len(xs) < 8:
            lines.append(None)
            continue
        pts = np.stack([xs * sx, ys * sy], 1)
        w = m[ys, xs]
        if decode == "robust":
            lines.append(fit_line_robust(pts, w))
        elif decode == "ransac":
            lines.append(fit_line_ransac(pts, w))
        elif decode == "weighted":
            lines.append(fit_line_weighted(pts, w))
        else:
            lines.append(fit_line(pts))

    out, conf = {}, {}
    for k in range(N_KPTS):
        hch, vch = KPT_TO_HV[k]
        if lines[hch] is None or lines[vch] is None:
            continue
        p = intersect(lines[hch], lines[vch])
        if p is not None and np.isfinite(p).all():
            out[k] = p
            conf[k] = min(ch_conf[hch], ch_conf[vch])
    return out, conf
