"""Render a multi-track event timeline aligned to the video time axis.

Rows (top -> bottom): rallies, hits, strokes (coloured by stroke type). x = time. Drawn
full-width to sit directly under the video player so the player's own scrubber doubles as
the timeline cursor. Returns an RGB uint8 image for gr.Image.
"""

from __future__ import annotations

import cv2
import numpy as np

# coarse ShotType -> BGR colour
_STROKE_COLOR = {
    "serve": (180, 180, 180), "clear": (255, 160, 0), "smash": (0, 0, 255),
    "drop": (0, 200, 255), "net": (0, 255, 255), "lift": (255, 0, 180),
    "drive": (0, 230, 0), "unknown": (110, 110, 110),
}
_BG = (28, 28, 28)
_FG = (235, 235, 235)
_LEFT = 70          # left gutter for row labels
_RIGHT = 16
_ROW_H = 34
_TOP = 40           # legend band


TIMELINE_WIDTH = 1280


def _x(t: float, dur: float, w: int) -> int:
    span = max(1, w - _LEFT - _RIGHT)
    return _LEFT + int((t / dur if dur > 0 else 0) * span)


def time_at_x(x: float, duration_s: float, width: int = TIMELINE_WIDTH) -> float:
    """Inverse of the timeline x-mapping: clicked pixel x -> time (s), for click-to-seek."""
    span = max(1, width - _LEFT - _RIGHT)
    frac = (x - _LEFT) / span
    return float(min(max(frac, 0.0), 1.0) * max(duration_s, 0.0))


def _text(img, s, x, y, color=_FG, scale=0.42):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render_timeline(analysis, duration_s: float, fps: float = 25.0,
                    width: int = 1280) -> np.ndarray:
    """analysis: MatchAnalysis (hits, shots, rallies). duration_s: video length."""
    bio = getattr(analysis, "biomechanics", None)
    effort_by_idx = {sb.shot_index: sb.effort_nm for sb in bio.strokes} if bio else {}
    rows = ["Rally", "Hit", "Stroke"] + (["Effort"] if effort_by_idx else [])
    h = _TOP + _ROW_H * len(rows) + 26
    img = np.full((h, width, 3), _BG, np.uint8)
    dur = max(duration_s, 1e-3)

    # legend (stroke colours)
    lx = _LEFT
    for name, col in _STROKE_COLOR.items():
        cv2.rectangle(img, (lx, 12), (lx + 14, 26), col, -1)
        _text(img, name, lx + 18, 24)
        lx += 22 + 9 * len(name)

    y0 = {r: _TOP + i * _ROW_H for i, r in enumerate(rows)}
    for r in rows:
        yc = y0[r] + _ROW_H // 2
        _text(img, r, 8, yc + 4, _FG, 0.45)
        cv2.line(img, (_LEFT, yc), (width - _RIGHT, yc), (60, 60, 60), 1)

    shots = list(getattr(analysis, "shots", ()) or ())
    hits = list(getattr(analysis, "hits", ()) or ())
    rallies = list(getattr(analysis, "rallies", ()) or ())

    # rallies: alternating-shade spans
    yr = y0["Rally"]
    for i, ra in enumerate(rallies):
        x0, x1 = _x(ra.start_frame / fps, dur, width), _x(ra.end_frame / fps, dur, width)
        shade = (70, 110, 70) if i % 2 == 0 else (70, 90, 130)
        cv2.rectangle(img, (x0, yr + 6), (max(x1, x0 + 2), yr + _ROW_H - 6), shade, -1)
        _text(img, f"R{i + 1}", x0 + 3, yr + _ROW_H - 11)

    # hits: ticks
    yh = y0["Hit"]
    for hev in hits:
        x = _x(hev.frame_index / fps, dur, width)
        cv2.line(img, (x, yh + 6), (x, yh + _ROW_H - 6), (0, 255, 255), 2)

    # strokes: coloured bars by type + index
    ys = y0["Stroke"]
    for j, s in enumerate(shots, 1):
        x0 = _x(s.start_frame / fps, dur, width)
        x1 = max(_x(s.end_frame / fps, dur, width), x0 + 3)
        col = _STROKE_COLOR.get(s.shot_type.value, _STROKE_COLOR["unknown"])
        cv2.rectangle(img, (x0, ys + 6), (x1, ys + _ROW_H - 6), col, -1)
        if x1 - x0 > 12:
            _text(img, str(j), x0 + 2, ys + _ROW_H - 11, (0, 0, 0))

    # effort: per-stroke load proxy, bar height/colour by magnitude
    if "Effort" in rows:
        ye = y0["Effort"]
        emax = max(effort_by_idx.values()) or 1.0
        for j, s in enumerate(shots, 1):
            e = effort_by_idx.get(j)
            if e is None:
                continue
            x0 = _x(s.start_frame / fps, dur, width)
            x1 = max(_x(s.end_frame / fps, dur, width), x0 + 3)
            frac = min(e / emax, 1.0)
            bh = int(frac * (_ROW_H - 12))
            col = (0, int(255 * (1 - frac)), 255)  # yellow->red with load
            cv2.rectangle(img, (x0, ye + _ROW_H - 6 - bh), (x1, ye + _ROW_H - 6), col, -1)

    # time ruler (5s ticks)
    yb = h - 8
    step = 5 if dur > 20 else 1
    t = 0
    while t <= dur:
        x = _x(t, dur, width)
        cv2.line(img, (x, yb - 5), (x, yb), (120, 120, 120), 1)
        _text(img, f"{t}s", x + 2, yb, (150, 150, 150), 0.4)
        t += step

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
