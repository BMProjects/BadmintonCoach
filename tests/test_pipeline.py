"""End-to-end Phase-1 pipeline wiring with dependency-free backends.

Builds the config inline (null detector/pose/shuttle + iou tracker + manual court)
so the architecture smoke test needs no yaml file or model weights.
"""

from __future__ import annotations

from pathlib import Path

from badminton_coach.core.config import AppConfig
from badminton_coach.core.pipeline import Phase1Pipeline

BASELINE = {
    "name": "test-baseline",
    "io": {"clip_window": 8, "stride": 1, "max_frames": 30},
    "perception": {
        "detector": {"backend": "null", "device": "cpu"},
        "shuttle_tracker": {"backend": "null", "device": "cpu"},
        "pose_estimator": {"backend": "null", "device": "cpu"},
        "player_tracker": {"backend": "iou", "device": "cpu"},
        "court_calibrator": {
            "backend": "manual",
            "device": "cpu",
            "image_corners": [[100, 700], [1180, 700], [980, 200], [300, 200]],
        },
        "reconstructor": {"backend": "null", "device": "cpu", "enabled": True},
    },
}


def _config() -> AppConfig:
    return AppConfig.model_validate(BASELINE)


def test_baseline_pipeline_runs(synthetic_video: Path):
    pipeline = Phase1Pipeline.from_config(_config())
    result = pipeline.run(synthetic_video)

    assert result.frame_count > 0
    assert result.fps > 0
    # null detector/pose/shuttle: empty but well-formed outputs.
    assert result.detections == ()
    assert result.poses == ()
    assert len(result.shuttle_2d) == 0
    # manual court calibrator produced a valid homography.
    assert result.court.homography.shape == (3, 3)
    # null shuttle tracker yields no 2D points, so there is nothing to lift to 3D.
    assert result.shuttle_3d is None


def test_config_backend_swap_changes_instance():
    pipeline = Phase1Pipeline.from_config(_config())
    assert pipeline.player_tracker.backend_name == "iou"
    assert pipeline.detector.backend_name == "null"
