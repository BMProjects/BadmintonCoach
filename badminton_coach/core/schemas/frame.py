"""Frame and clip data contracts.

Frame.image is a BGR uint8 ndarray of shape (H, W, 3) — OpenCV native. Backends
that need RGB must convert explicitly; the color space is documented here so no
layer has to guess.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Frame:
    """A single decoded video frame.

    image: (H, W, 3) uint8, BGR.
    index: 0-based frame number within the source video.
    timestamp: seconds from the start of the video.
    """

    index: int
    timestamp: float
    image: np.ndarray

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


@dataclass(frozen=True, slots=True)
class FrameClip:
    """An ordered window of frames.

    Heatmap-based shuttle trackers (TrackNetV3) need a temporal window; pass them
    a FrameClip rather than single frames.
    """

    frames: tuple[Frame, ...]
    fps: float

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def start_index(self) -> int:
        return self.frames[0].index

    @property
    def end_index(self) -> int:
        return self.frames[-1].index
