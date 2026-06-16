"""Video decoding into Frame objects (OpenCV-backed, BGR uint8)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2

from ..schemas import Frame


class VideoReader:
    """Lazily decode a video file into Frame objects.

    Yields BGR uint8 frames with index + timestamp. Use as a context manager so
    the underlying capture is always released.
    """

    def __init__(self, path: str | Path, max_frames: int | None = None, stride: int = 1) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")
        self.max_frames = max_frames
        self.stride = max(1, stride)
        self._tmp: Path | None = None
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened() or self.frame_count_of(self._cap) == 0:
            # Some builds can't decode HEVC/H.265 (Safari/iPhone recordings) or
            # 10-bit HDR. Transcode to H.264 8-bit SDR via the bundled ffmpeg and retry.
            self._cap.release()
            transcoded = self._transcode_to_h264(self.path)
            if transcoded is None:
                raise RuntimeError(f"Failed to open video (unsupported codec?): {self.path}")
            self._tmp = transcoded
            self._cap = cv2.VideoCapture(str(transcoded))
            if not self._cap.isOpened():
                raise RuntimeError(f"Failed to open video after transcode: {self.path}")
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @staticmethod
    def frame_count_of(cap) -> int:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @staticmethod
    def _transcode_to_h264(src: Path) -> Path | None:
        """Transcode any codec to H.264 8-bit SDR (bt709) into a temp file."""
        try:
            import subprocess
            import tempfile

            import imageio_ffmpeg
        except ImportError:
            return None
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        out = Path(tempfile.mkdtemp()) / (src.stem + "_h264.mp4")
        cmd = [ff, "-y", "-i", str(src),
               "-vf", "scale=out_color_matrix=bt709,format=yuv420p",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(out)]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except (subprocess.CalledProcessError, OSError):
            return None
        return out if out.exists() else None

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._tmp is not None:
            import shutil

            shutil.rmtree(self._tmp.parent, ignore_errors=True)
            self._tmp = None

    def __iter__(self) -> Iterator[Frame]:
        index = 0
        emitted = 0
        while True:
            ok, image = self._cap.read()
            if not ok:
                break
            if index % self.stride == 0:
                yield Frame(index=index, timestamp=index / self.fps, image=image)
                emitted += 1
                if self.max_frames is not None and emitted >= self.max_frames:
                    break
            index += 1
