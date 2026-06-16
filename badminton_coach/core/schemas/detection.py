"""Detection data contract for the 'large object' detector (RF-DETR / YOLO).

Shuttlecock is intentionally NOT a detection class here — it is handled by the
heatmap-based ShuttleTracker, per both technical reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .geometry_types import BBox


class ObjectClass(str, Enum):
    """Detectable large/stable objects."""

    PLAYER = "player"
    RACKET = "racket"
    NET_POST = "net_post"
    COURT_LINE = "court_line"


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected object in one frame."""

    frame_index: int
    cls: ObjectClass
    bbox: BBox
    confidence: float
