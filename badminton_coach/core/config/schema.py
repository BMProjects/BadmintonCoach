"""Pydantic config models.

Each module's config is an open block with a required `backend` field plus
backend-specific keys (device, precision, weights, ...). Keeping blocks open
(extra='allow') means a new backend can add its own knobs without editing this
schema — only the contract (`backend` exists) is enforced here.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModuleConfig(BaseModel):
    """One pluggable module's configuration block."""

    model_config = ConfigDict(extra="allow")

    backend: str
    device: str = "cuda"
    precision: str = "fp16"
    enabled: bool = True


class PerceptionConfig(BaseModel):
    """L1 perception layer: one block per pluggable responsibility."""

    detector: ModuleConfig
    shuttle_tracker: ModuleConfig
    pose_estimator: ModuleConfig
    player_tracker: ModuleConfig
    court_calibrator: ModuleConfig
    reconstructor: ModuleConfig

    # A generic 'person' detector also picks up umpire/line-judges/audience. When
    # true, keep only player detections whose ground point lies within the court
    # (+margin) and cap to the two largest — the actual on-court singles players.
    filter_players_to_court: bool = False
    court_margin_m: float = 1.5
    max_players: int = 2
    # When the pose backend can also detect (YOLO-pose), take boxes + keypoints from one
    # forward and skip the separate detector pass. Falls back automatically if the pose
    # backend lacks detect_and_pose. Set false to use a distinct detector (e.g. rfdetr).
    unified_perception: bool = True
    # Per-frame court-presence gating: only treat the court as visible on frames
    # where the calibrator detects it (kills 'phantom court' on no-court/incomplete
    # frames). Uses CourtCalibrator.present_frames (batched) per frame.
    court_per_frame_presence: bool = False
    # The 2D court calibration -> 3D estimation subsystem (stable_background + court
    # calibrate + per-frame presence + 3D reconstruction). Toggle off to run a fast
    # 2D-only pipeline (detection/pose/tracking/shuttle-2D) and compare efficiency.
    # The frontend exposes this as a button; can also be overridden per run().
    estimate_3d: bool = True
    # Fixed-camera reuse: path to a CalibrationProfile JSON cache. If set and present,
    # the court calibration is loaded from it (skipping stable_background + calibrate);
    # otherwise the court is calibrated once and saved there. Calibrate a whole match
    # once and reuse it across its clips. None = always recalibrate (no cache).
    court_profile_path: str | None = None


class IOConfig(BaseModel):
    clip_window: int = 8  # frames per shuttle-tracker window
    stride: int = 1
    max_frames: int | None = None  # cap for quick smoke runs


class AppConfig(BaseModel):
    """Top-level configuration loaded from a YAML preset."""

    name: str = "default"
    io: IOConfig = Field(default_factory=IOConfig)
    perception: PerceptionConfig
