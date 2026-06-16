"""Tactical statistics from perception + L2 events.

Aggregates a clip into a MatchStats: rally counts, stroke-type mix, per-player
court movement (distance + speed via the homography), and the court-space spread of
shuttle strikes (contact/landing zones). All court-space metrics require a calibration;
without one, movement and spatial spread are empty.
"""

from __future__ import annotations

import numpy as np

from ..core.schemas import PerceptionResult
from ..core.schemas.analysis import MatchStats, PlayerMovement
from ..core.schemas.events import HitEvent, Rally, Shot


def _smooth(xy: np.ndarray, k: int = 5) -> np.ndarray:
    """Moving-average smooth a (N,2) ground path to suppress detection jitter."""
    if len(xy) < k:
        return xy
    kernel = np.ones(k) / k
    out = xy.copy()
    for c in range(2):
        out[:, c] = np.convolve(xy[:, c], kernel, mode="same")
    return out


_MAX_PLAUSIBLE_MS = 9.0  # cap: badminton sprint ~6-7 m/s; larger = tracking teleport


def _player_movement(perception: PerceptionResult, max_players: int = 2,
                     window: int = 3) -> list[PlayerMovement]:
    """Movement for the main players (the `max_players` longest tracks). Ground path
    smoothed; per-frame steps faster than a physical cap are treated as id-switch
    teleports and dropped from distance/speed (robust to residual tracking glitches)."""
    court = perception.court
    if court is None:
        return []
    fps = perception.fps or 30.0
    main = sorted(perception.player_tracks, key=lambda t: len(t.boxes), reverse=True)[:max_players]
    out: list[PlayerMovement] = []
    for tr in sorted(main, key=lambda t: t.track_id):
        boxes = sorted(tr.boxes, key=lambda b: b.frame_index)
        if len(boxes) < 2:
            continue
        gnd = _smooth(np.array([[(g := court.image_to_ground(b.bbox.foot)).x, g.y]
                                for b in boxes], dtype=np.float64))
        dist, moving_dt, speeds = 0.0, 0.0, []
        for i in range(1, len(boxes)):
            dt = (boxes[i].frame_index - boxes[i - 1].frame_index) / fps
            if dt <= 0:
                continue
            step = float(np.linalg.norm(gnd[i] - gnd[i - 1]))
            if step / dt <= _MAX_PLAUSIBLE_MS:   # ignore teleports
                dist += step
                moving_dt += dt
        for i in range(len(boxes)):             # windowed speed, capped
            j = max(0, i - window)
            dt = (boxes[i].frame_index - boxes[j].frame_index) / fps
            sp = float(np.linalg.norm(gnd[i] - gnd[j]) / dt) if dt > 0 else 0.0
            if sp <= _MAX_PLAUSIBLE_MS:
                speeds.append(sp)
        out.append(PlayerMovement(
            track_id=tr.track_id,
            distance_m=dist,
            avg_speed_ms=dist / moving_dt if moving_dt > 0 else 0.0,
            max_speed_ms=float(np.percentile(speeds, 95)) if speeds else 0.0,
        ))
    return out


def _shot_landing(seg) -> tuple[float, float] | None:
    """Court (x,y) where a shot's 3D flight reaches the floor (z=0).

    seg: list of Point3D (metres, z up) ordered by time. Uses the descending z>0->z<=0
    crossing if the shuttle actually landed within the segment; otherwise fits z(t) and
    extrapolates to the next descending root (the *intended* landing of a returned shot).
    """
    if len(seg) < 3:
        return None
    for i in range(1, len(seg)):
        z0, z1 = seg[i - 1].z, seg[i].z
        if z0 > 0 >= z1:  # descending crossing -> interpolate
            t = z0 / (z0 - z1)
            return (seg[i - 1].x + t * (seg[i].x - seg[i - 1].x),
                    seg[i - 1].y + t * (seg[i].y - seg[i - 1].y))
    # no in-segment landing: fit parabola z(t), extrapolate to the descending root
    ts = np.arange(len(seg), dtype=float)
    a, b, c = np.polyfit(ts, [p.z for p in seg], 2)
    if abs(a) < 1e-9:
        return None
    roots = np.roots([a, b, c])
    fwd = [r.real for r in roots if abs(r.imag) < 1e-6 and r.real > ts[-1]]
    if not fwd:
        return None
    tr = min(fwd)
    if tr > ts[-1] + len(seg):  # too far to trust the extrapolation
        return None
    fx = np.polyfit(ts, [p.x for p in seg], 1)
    fy = np.polyfit(ts, [p.y for p in seg], 1)
    return (float(np.polyval(fx, tr)), float(np.polyval(fy, tr)))


def _landings(perception: PerceptionResult, shots: list[Shot]) -> list[tuple[float, float]]:
    s3d = perception.shuttle_3d
    if s3d is None:
        return []
    by_frame = {p.frame_index: p.point for p in s3d.points}
    out = []
    for s in shots:
        seg = [by_frame[f] for f in range(s.start_frame, s.end_frame + 1) if f in by_frame]
        land = _shot_landing(seg)
        if land is not None:
            out.append(land)
    return out


def compute_match_stats(
    perception: PerceptionResult,
    hits: list[HitEvent],
    shots: list[Shot],
    rallies: list[Rally],
) -> MatchStats:
    counts: dict[str, int] = {}
    for s in shots:
        counts[s.shot_type.value] = counts.get(s.shot_type.value, 0) + 1

    avg_shots = (sum(len(r.shots) for r in rallies) / len(rallies)) if rallies else 0.0

    ground_pts: list[tuple[float, float]] = []
    if perception.court is not None:
        for h in hits:
            g = perception.court.image_to_ground(h.shuttle_image_pos)
            ground_pts.append((float(g.x), float(g.y)))

    return MatchStats(
        rally_count=len(rallies),
        avg_shots_per_rally=avg_shots,
        shot_type_counts=counts,
        player_movement=tuple(_player_movement(perception)),
        hit_ground_points_m=tuple(ground_pts),
        landing_points_m=tuple(_landings(perception, shots)),
    )
