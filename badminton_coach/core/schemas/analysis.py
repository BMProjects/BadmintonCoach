"""L2 aggregate output: structured events for a clip."""

from __future__ import annotations

from dataclasses import dataclass

from .biomech import BiomechanicsReport
from .events import HitEvent, Rally, Shot
from .trajectory import ShuttleTrajectory3D


@dataclass(frozen=True, slots=True)
class PlayerMovement:
    """Court-space movement summary for one tracked player (metres / m·s⁻¹)."""

    track_id: int
    distance_m: float
    avg_speed_ms: float
    max_speed_ms: float


@dataclass(frozen=True, slots=True)
class MatchStats:
    """Tactical summary over a clip: rallies, stroke mix, movement, spatial spread."""

    rally_count: int
    avg_shots_per_rally: float
    shot_type_counts: dict[str, int]
    player_movement: tuple[PlayerMovement, ...]
    hit_ground_points_m: tuple[tuple[float, float], ...]  # court (x,y) of each strike
    landing_points_m: tuple[tuple[float, float], ...]  # court (x,y) where flight hits z=0


@dataclass(frozen=True, slots=True)
class MatchAnalysis:
    """Event-level analysis of a clip (output of the L2 layer)."""

    hits: tuple[HitEvent, ...]
    shots: tuple[Shot, ...]
    shuttle_3d: ShuttleTrajectory3D | None  # hit-segmented (per-shot parabola fits)
    rallies: tuple[Rally, ...] = ()
    stats: MatchStats | None = None
    biomechanics: BiomechanicsReport | None = None
