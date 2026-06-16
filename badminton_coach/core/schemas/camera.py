"""Pinhole camera data contract (intrinsics + extrinsics) for 3D reprojection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CameraParameters:
    """Pinhole camera model.

    intrinsic:  3x3 K matrix.
    rotation:   3x3 world->camera rotation.
    translation: shape (3,) world->camera translation (meters).
    """

    intrinsic: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray

    def project(self, world_points: np.ndarray) -> np.ndarray:
        """Project world points (N,3) meters to image pixels (N,2)."""
        cam = (self.rotation @ world_points.T).T + self.translation
        img_h = (self.intrinsic @ cam.T).T
        return img_h[:, :2] / img_h[:, 2:3]
