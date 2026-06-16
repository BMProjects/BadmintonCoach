"""Per-frame speed computation from court calibration + overlay drawing.

Player speed: foot point -> court ground (homography) -> metres; speed over a short
window in m/s (and km/h). Shuttle speed: from the reconstructed 3D trajectory
(metres) between consecutive frames. Both rely on the court detection result.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..core.schemas import PerceptionResult, ShuttleTrajectory3D


def player_speeds(result: PerceptionResult, window: int = 3) -> dict[int, dict[int, float]]:
    """{frame_index: {track_id: speed_m_s}} from foot points via the homography."""
    court = result.court
    if court is None:  # no calibration -> can't map feet to metres
        return {}
    fps = result.fps or 30.0
    out: dict[int, dict[int, float]] = {}
    for tr in result.player_tracks:
        boxes = sorted(tr.boxes, key=lambda b: b.frame_index)
        world = [court.image_to_ground(b.foot_image) for b in boxes]
        for i, b in enumerate(boxes):
            j = max(0, i - window)
            dt = (boxes[i].frame_index - boxes[j].frame_index) / fps
            if dt <= 0:
                continue
            d = np.hypot(world[i].x - world[j].x, world[i].y - world[j].y)
            out.setdefault(b.frame_index, {})[tr.track_id] = d / dt
    return out


def shuttle_speeds(shuttle_3d: ShuttleTrajectory3D | None, fps: float) -> dict[int, float]:
    """{frame_index: speed_m_s} from consecutive 3D points (metres)."""
    if shuttle_3d is None or len(shuttle_3d) < 2:
        return {}
    pts = sorted(shuttle_3d.points, key=lambda p: p.frame_index)
    out: dict[int, float] = {}
    for a, b in zip(pts, pts[1:], strict=False):
        dt = (b.frame_index - a.frame_index) / fps
        if dt <= 0:
            continue
        d = np.linalg.norm(np.array(b.point.as_tuple()) - np.array(a.point.as_tuple()))
        out[b.frame_index] = d / dt
    return out


def _label(image, text, x, y, color):
    """Draw text with a dark outline for readability at (x, y)."""
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def annotate_speeds(image: np.ndarray, players: list, shuttle) -> np.ndarray:
    """Draw speeds next to each entity.

    players: list of (bbox, speed_m_s)  -> labelled in m/s above the player box.
    shuttle: (Point2D, speed_km_h) | None -> labelled in km/h beside the shuttle.
    """
    out = image.copy()
    for bbox, v in players:
        if v is None:
            continue
        x = int(bbox.x1)
        y = max(20, int(bbox.y1) - 8)
        _label(out, f"{v:.1f} m/s", x, y, (0, 255, 0))
    if shuttle is not None:
        pt, kmh = shuttle
        if kmh is not None:
            _label(out, f"{kmh:.0f} km/h", int(pt.x) + 10, int(pt.y) - 8, (0, 200, 255))
    return out
