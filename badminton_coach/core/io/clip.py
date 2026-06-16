"""Frame window helpers for trackers that need temporal context."""

from __future__ import annotations

from collections.abc import Iterator

from ..schemas import Frame, FrameClip


def sliding_clips(
    frames: list[Frame], window: int, fps: float, stride: int = 1
) -> Iterator[FrameClip]:
    """Yield overlapping FrameClips of length `window` over the frame list.

    Used to feed heatmap shuttle trackers (TrackNetV3) a temporal window. The last
    partial window is dropped so every clip has exactly `window` frames.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    for start in range(0, len(frames) - window + 1, stride):
        yield FrameClip(frames=tuple(frames[start : start + window]), fps=fps)
