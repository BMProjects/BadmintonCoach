"""Persistable court calibration profile (the 'calibrate once, reuse' cache).

For fixed-camera broadcast footage the full calibration is computed once and saved
to disk; later runs load it and only do lightweight per-frame marker tracking to
catch slow drift / camera cuts. Stored as JSON (numpy arrays -> lists).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .camera import CameraParameters
from .court import CourtCalibration


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    """A saved calibration keyed by a camera/source identifier.

    source_key: stable id for this camera setup (e.g. video hash or match id).
    image_size: (width, height) the calibration was computed at.
    image_corners: the tracked court markers in image space, [[x,y], ...].
    """

    source_key: str
    image_size: tuple[int, int]
    image_corners: list[list[float]]
    homography: np.ndarray
    reprojection_error_px: float
    camera: CameraParameters | None = None

    def to_calibration(self) -> CourtCalibration:
        return CourtCalibration(
            homography=self.homography,
            reprojection_error_px=self.reprojection_error_px,
            camera=self.camera,
        )

    def to_dict(self) -> dict:
        cam = None
        if self.camera is not None:
            cam = {
                "intrinsic": self.camera.intrinsic.tolist(),
                "rotation": self.camera.rotation.tolist(),
                "translation": self.camera.translation.tolist(),
            }
        return {
            "source_key": self.source_key,
            "image_size": list(self.image_size),
            "image_corners": self.image_corners,
            "homography": self.homography.tolist(),
            "reprojection_error_px": self.reprojection_error_px,
            "camera": cam,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CalibrationProfile:
        cam = None
        if d.get("camera") is not None:
            c = d["camera"]
            cam = CameraParameters(
                intrinsic=np.array(c["intrinsic"], dtype=np.float64),
                rotation=np.array(c["rotation"], dtype=np.float64),
                translation=np.array(c["translation"], dtype=np.float64),
            )
        return cls(
            source_key=d["source_key"],
            image_size=tuple(d["image_size"]),
            image_corners=d["image_corners"],
            homography=np.array(d["homography"], dtype=np.float64),
            reprojection_error_px=float(d["reprojection_error_px"]),
            camera=cam,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CalibrationProfile:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
