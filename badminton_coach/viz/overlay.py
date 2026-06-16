"""Per-frame overlay drawing (BGR uint8 in/out, OpenCV).

Each function takes the frame image plus the relevant data-contract objects and
returns a new annotated image (does not mutate the input).
"""

from __future__ import annotations

import cv2
import numpy as np

from ..core.geometry.court_model import court_corners_doubles
from ..core.schemas import (
    CourtCalibration,
    Detection,
    Point3D,
    Pose,
    ShuttleTrajectory2D,
)

# COCO-17 skeleton edges.
_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
]
_CLASS_COLOR = {
    "player": (0, 255, 0),
    "racket": (0, 200, 255),
    "net_post": (255, 0, 255),
    "court_line": (200, 200, 200),
}


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = image.copy()
    for d in detections:
        color = _CLASS_COLOR.get(d.cls.value, (255, 255, 255))
        cv2.rectangle(out, (int(d.bbox.x1), int(d.bbox.y1)),
                      (int(d.bbox.x2), int(d.bbox.y2)), color, 2)
        cv2.putText(out, f"{d.cls.value} {d.confidence:.2f}",
                    (int(d.bbox.x1), int(d.bbox.y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


def draw_poses(image: np.ndarray, poses: list[Pose], kpt_threshold: float = 0.3) -> np.ndarray:
    out = image.copy()
    # Action (pose) detection in bright yellow for high visibility (BGR).
    _YELLOW = (0, 255, 255)
    for pose in poses:
        for a, b in _SKELETON:
            ka, kb = pose.keypoints[a], pose.keypoints[b]
            if ka.confidence >= kpt_threshold and kb.confidence >= kpt_threshold:
                cv2.line(out, (int(ka.point.x), int(ka.point.y)),
                         (int(kb.point.x), int(kb.point.y)), _YELLOW, 2, cv2.LINE_AA)
        for kp in pose.keypoints:
            if kp.confidence >= kpt_threshold:
                cv2.circle(out, (int(kp.point.x), int(kp.point.y)), 4, _YELLOW, -1)
                cv2.circle(out, (int(kp.point.x), int(kp.point.y)), 4, (0, 0, 0), 1)
    return out


def draw_court(image: np.ndarray, court: CourtCalibration) -> np.ndarray:
    """Reproject the BWF doubles boundary via the homography to verify calibration."""
    out = image.copy()
    world = [Point3D(x, y, 0.0) for x, y in court_corners_doubles()]
    pts = [court.ground_to_image(p) for p in world]
    poly = np.array([[int(p.x), int(p.y)] for p in pts], dtype=np.int32)
    # Court boundary in red (BGR) per spec; pose/action overlay is yellow.
    cv2.polylines(out, [poly], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.putText(out, f"reproj err {court.reprojection_error_px:.1f}px",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return out


def draw_shuttle(
    image: np.ndarray,
    shuttle: ShuttleTrajectory2D,
    up_to_frame: int,
    trail: int = 25,
) -> np.ndarray:
    """Draw the shuttle trajectory as a connecting line over the last `trail` frames
    (~1s) so the flight path is visible, plus a marker at the current position. Breaks
    the line across invisibility gaps."""
    out = image.copy()
    recent = [p for p in shuttle.points
              if up_to_frame - trail <= p.frame_index <= up_to_frame]
    seg: list[tuple[int, int]] = []
    for p in recent:
        if p.visible:
            seg.append((int(p.point.x), int(p.point.y)))
        else:
            if len(seg) >= 2:
                cv2.polylines(out, [np.array(seg, np.int32)], False, (0, 165, 255), 2, cv2.LINE_AA)
            seg = []
    if len(seg) >= 2:
        cv2.polylines(out, [np.array(seg, np.int32)], False, (0, 165, 255), 2, cv2.LINE_AA)
    cur = next((p for p in recent if p.frame_index == up_to_frame and p.visible), None)
    if cur is not None:
        cv2.circle(out, (int(cur.point.x), int(cur.point.y)), 6, (0, 0, 255), -1)
    return out


_STROKE_LABEL_COLOR = {
    "serve": (180, 180, 180), "clear": (255, 160, 0), "smash": (0, 0, 255),
    "drop": (0, 200, 255), "net": (0, 255, 255), "lift": (255, 0, 180),
    "drive": (0, 230, 0), "unknown": (170, 170, 170),
}


def draw_event_hud(
    image: np.ndarray,
    *,
    rally: int | None,
    is_hit: bool,
    hit_pt: tuple[float, float] | None,
) -> np.ndarray:
    """Minimal temporal overlay: a small rally counter top-left, plus a BRIEF hit flash
    (ring) at the strike position only around the hit frame (not held on screen)."""
    out = image.copy()
    if rally is not None:
        cv2.putText(out, f"Rally {rally}", (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, f"Rally {rally}", (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2, cv2.LINE_AA)
    if is_hit and hit_pt is not None:
        cv2.circle(out, (int(hit_pt[0]), int(hit_pt[1])), 16, (0, 255, 255), 3, cv2.LINE_AA)
    return out


def draw_stroke_label(image: np.ndarray, text: str, box) -> np.ndarray:
    """Label the current stroke (+ optional force) next to a player's box; coloured by
    the stroke type (first token of `text`)."""
    out = image.copy()
    color = _STROKE_LABEL_COLOR.get(text.split()[0], _STROKE_LABEL_COLOR["unknown"])
    x, y = int(box.x1), max(20, int(box.y1) - 26)
    s = text.upper()
    cv2.putText(out, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return out


def draw_landings(
    image: np.ndarray,
    court: CourtCalibration,
    landings_m: list[tuple[float, float]],
    numbered: bool = True,
) -> np.ndarray:
    """Plot shot landing points (court metres) as red X markers via the homography.
    Pass only the currently-active landings for a momentary flash; numbered=False omits
    the index label (used for the transient flash)."""
    out = image.copy()
    for i, (x, y) in enumerate(landings_m, 1):
        p = court.ground_to_image(Point3D(x, y, 0.0))
        c = (int(p.x), int(p.y))
        cv2.drawMarker(out, c, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2, cv2.LINE_AA)
        if numbered:
            cv2.putText(out, str(i), (c[0] + 8, c[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return out


def draw_trunk_force(image: np.ndarray, pose: Pose, effort_norm: float,
                     thr: float = 0.3) -> np.ndarray:
    """Encode the stroke's load as the colour of the player's trunk line (mid-shoulder
    -> mid-hip): green (low) -> red (high), by `effort_norm` in [0,1]. No text."""
    ks = pose.keypoints
    if any(ks[i].confidence < thr for i in (5, 6, 11, 12)):
        return image
    out = image.copy()
    sh = ((ks[5].point.x + ks[6].point.x) / 2, (ks[5].point.y + ks[6].point.y) / 2)
    hp = ((ks[11].point.x + ks[12].point.x) / 2, (ks[11].point.y + ks[12].point.y) / 2)
    f = max(0.0, min(1.0, effort_norm))
    color = (0, int(255 * (1 - f)), int(255 * f))  # BGR: green->red
    cv2.line(out, (int(sh[0]), int(sh[1])), (int(hp[0]), int(hp[1])), color, 7, cv2.LINE_AA)
    return out
