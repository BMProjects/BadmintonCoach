"""Shared fixtures: a tiny synthetic video for end-to-end pipeline tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Write a short synthetic clip; skip the test if no codec is available."""
    path = tmp_path / "rally.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (320, 240))
    if not writer.isOpened():
        pytest.skip("No video codec available to write the synthetic fixture")
    for i in range(20):
        frame = np.full((240, 320, 3), i, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path
