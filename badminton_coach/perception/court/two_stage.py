"""Two-stage court calibrator: bootstrap once, persist, reuse.

Stage 1 (bootstrap): from the four doubles-court corners (config['image_corners'],
or — once wired — an automatic detector) solve the ground homography and,
optionally, the full camera (intrinsics + pose) for 3D reconstruction. The result
is cached to disk as a CalibrationProfile keyed by source.

Stage 2 (lightweight tracking) lives in core.geometry.marker_tracking +
core.io.SceneCutDetector and is driven by the orchestrator per frame; this backend
provides the one-shot calibrate() the CourtCalibrator interface requires, returning
the cached profile when present so fixed-camera footage is calibrated only once.

Config keys:
    image_corners: [[x,y]*4]  near-L, near-R, far-R, far-L  (bootstrap markers)
    profile_path:  optional JSON cache path; loaded if present, else written
    compute_camera: bool — also solve camera pose via PnP (needed for 3D)
    focal_px: optional assumed focal length for the camera intrinsics
    source_key: id stored in the profile
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...core.geometry import estimate_camera, solve_homography
from ...core.geometry.court_model import court_corners_doubles
from ...core.interfaces import CourtCalibrator
from ...core.registry import register
from ...core.schemas import CalibrationProfile, CourtCalibration, Frame, Point2D, Point3D


@register("court_calibrator", "two_stage")
class TwoStageCourtCalibrator(CourtCalibrator):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._corners = self.config.get("image_corners")
        self._profile_path = self.config.get("profile_path")
        self._compute_camera = bool(self.config.get("compute_camera", False))
        self._focal_px = self.config.get("focal_px")
        self._source_key = self.config.get("source_key", "default")

    @classmethod
    def is_available(cls) -> bool:
        return True

    def calibrate(self, frame: Frame) -> CourtCalibration:
        # Reuse a cached profile if one exists (fixed-camera footage: calibrate once).
        if self._profile_path and Path(self._profile_path).exists():
            return CalibrationProfile.load(self._profile_path).to_calibration()

        profile = self._bootstrap(frame)
        if self._profile_path:
            profile.save(self._profile_path)
        return profile.to_calibration()

    def _bootstrap(self, frame: Frame) -> CalibrationProfile:
        if not self._corners or len(self._corners) != 4:
            raise ValueError(
                "TwoStageCourtCalibrator needs config['image_corners'] = 4 [x,y] points "
                "(near-L, near-R, far-R, far-L) until an automatic detector is wired."
            )
        image_pts = [Point2D(float(x), float(y)) for x, y in self._corners]
        world_pts = [Point3D(x, y, 0.0) for x, y in court_corners_doubles()]

        calib = solve_homography(image_pts, world_pts)
        camera = None
        if self._compute_camera:
            camera = estimate_camera(
                image_pts, world_pts, (frame.width, frame.height), self._focal_px
            )
        return CalibrationProfile(
            source_key=self._source_key,
            image_size=(frame.width, frame.height),
            image_corners=[[p.x, p.y] for p in image_pts],
            homography=calib.homography,
            reprojection_error_px=calib.reprojection_error_px,
            camera=camera,
        )
