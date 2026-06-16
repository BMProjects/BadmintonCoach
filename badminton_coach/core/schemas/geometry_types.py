"""Primitive geometric data contracts shared across all layers.

Coordinate conventions:
- Image coordinates: pixels, origin top-left, x right, y down (OpenCV convention).
- World coordinates: meters, origin at a court corner, x along width, y along
  length, z up (ground plane z=0). See core/geometry/court_model.py.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Point2D:
    """A point in image space (pixels)."""

    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True, slots=True)
class Point3D:
    """A point in world space (meters)."""

    x: float
    y: float
    z: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned bounding box in image space (pixels), xyxy."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> Point2D:
        return Point2D((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def foot(self) -> Point2D:
        """Bottom-center point — used as the player's ground contact for homography."""
        return Point2D((self.x1 + self.x2) / 2.0, self.y2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1
