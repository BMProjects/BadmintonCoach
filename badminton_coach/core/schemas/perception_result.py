"""Aggregate L1 output: everything the perception layer produces for a clip.

This is the single hand-off contract from L1 (perception) to L2 (events). L2 code
depends only on this, never on any perception backend.
"""

from __future__ import annotations

from dataclasses import dataclass

from .court import CourtCalibration
from .detection import Detection
from .pose import Pose
from .track import PlayerTrack
from .trajectory import ShuttleTrajectory2D, ShuttleTrajectory3D


@dataclass(frozen=True, slots=True)
class PerceptionResult:
    """Full perception output for one analyzed clip/rally."""

    source: str
    fps: float
    frame_count: int

    court: CourtCalibration | None  # None if court calibration failed (e.g. amateur footage)
    detections: tuple[Detection, ...]
    player_tracks: tuple[PlayerTrack, ...]
    poses: tuple[Pose, ...]
    shuttle_2d: ShuttleTrajectory2D
    shuttle_3d: ShuttleTrajectory3D | None  # None if 3D reconstruction disabled
    # Frame indices where a court is actually visible (per-frame presence). None
    # means "not assessed" (treat court as present on all frames).
    court_frames: frozenset[int] | None = None
