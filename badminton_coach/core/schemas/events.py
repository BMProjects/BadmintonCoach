"""L2 event data contracts: hits, rallies, shots.

A hit is a frame where a player strikes the shuttle (its trajectory direction
reverses). A rally is a contiguous span of play (a run of hits). A shot is the
segment between two consecutive hits, optionally classified by stroke type.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .geometry_types import Point2D


class ShotType(str, Enum):
    """BWF stroke categories (coarse set; BST provides the fine 35-class set)."""

    SERVE = "serve"
    CLEAR = "clear"        # high to back court
    SMASH = "smash"        # steep downward attack
    DROP = "drop"          # soft to front court
    NET = "net"            # net shot
    LIFT = "lift"          # defensive high to back
    DRIVE = "drive"        # flat
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class HitEvent:
    """A shuttle strike."""

    frame_index: int
    shuttle_image_pos: Point2D
    hitter_track_id: int | None = None   # which player (if resolved)
    near_side: bool | None = None        # near (bottom) vs far (top) half-court


@dataclass(frozen=True, slots=True)
class Shot:
    """The flight segment from one hit to the next, with an optional stroke label."""

    start_frame: int
    end_frame: int
    hitter_track_id: int | None
    shot_type: ShotType = ShotType.UNKNOWN
    confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class Rally:
    """A contiguous span of play."""

    start_frame: int
    end_frame: int
    hits: tuple[HitEvent, ...]
    shots: tuple[Shot, ...] = ()
