"""Video I/O and frame windowing."""

from .background import stable_background
from .clip import sliding_clips
from .scene import SceneCutDetector
from .video_reader import VideoReader

__all__ = ["VideoReader", "sliding_clips", "SceneCutDetector", "stable_background"]
