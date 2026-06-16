"""Registry + factory: registration, availability, and config-driven build."""

from __future__ import annotations

import pytest

import badminton_coach.perception  # noqa: F401  (registers backends)
from badminton_coach.core.interfaces import Detector
from badminton_coach.core.registry import available_backends, build, get_backend, register


def test_all_kinds_have_backends():
    reg = available_backends()
    for kind in ("detector", "shuttle_tracker", "pose_estimator", "player_tracker",
                 "court_calibrator", "reconstructor"):
        assert reg.get(kind), f"no backends registered for {kind}"


def test_build_available_baseline_backend():
    det = build("detector", {"backend": "null"})
    assert isinstance(det, Detector)
    assert det.backend_name == "null"


def test_build_unknown_backend_raises():
    with pytest.raises(KeyError):
        get_backend("detector", "does-not-exist")


def test_register_rejects_wrong_interface():
    with pytest.raises(TypeError):

        @register("detector", "bad")
        class NotADetector:  # missing Detector base
            pass


def test_register_rejects_unknown_kind():
    with pytest.raises(ValueError):

        @register("nonsense", "x")
        class _Whatever(Detector):
            @classmethod
            def is_available(cls):
                return True

            def detect(self, frame):
                return []
