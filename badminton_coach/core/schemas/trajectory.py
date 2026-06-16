"""Shuttlecock trajectory data contracts (2D tracking + 3D reconstruction)."""

from __future__ import annotations

from dataclasses import dataclass

from .geometry_types import Point2D, Point3D


@dataclass(frozen=True, slots=True)
class ShuttlePoint2D:
    """Shuttle position in one frame. visible=False marks an inpainted/missing point."""

    frame_index: int
    point: Point2D
    confidence: float
    visible: bool = True


@dataclass(frozen=True, slots=True)
class ShuttleTrajectory2D:
    """Per-frame 2D shuttle positions over a clip (TrackNetV3 output)."""

    points: tuple[ShuttlePoint2D, ...]

    def __len__(self) -> int:
        return len(self.points)


@dataclass(frozen=True, slots=True)
class ShuttlePoint3D:
    """Reconstructed shuttle world position (meters) in one frame."""

    frame_index: int
    point: Point3D


@dataclass(frozen=True, slots=True)
class ShuttleTrajectory3D:
    """Reconstructed 3D trajectory plus an honest uncertainty estimate.

    Per the reports, monocular 3D is ill-posed: end-to-end reprojection error
    runs ~28-37 px. Always surface the uncertainty rather than presenting 3D speed
    or height as exact.
    """

    points: tuple[ShuttlePoint3D, ...]
    reprojection_error_px: float
    method: str  # e.g. "monotrack" / "synthnet"

    def __len__(self) -> int:
        return len(self.points)
