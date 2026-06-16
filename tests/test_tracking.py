"""IoU+distance player tracker: coasting and distance fallback keep stable IDs."""

from __future__ import annotations

import badminton_coach.perception  # noqa: F401  (registers backends)
from badminton_coach.core.registry import build
from badminton_coach.core.schemas import BBox, Detection, ObjectClass


def _det(frame, x, y, w=40, h=80):
    return Detection(frame_index=frame, cls=ObjectClass.PLAYER,
                     bbox=BBox(x, y, x + w, y + h), confidence=0.9)


def test_two_players_stay_two_tracks_with_movement():
    # Two players drift across frames; plain IoU would fragment, coasting+distance keep 2.
    trk = build("player_tracker", {"backend": "iou"})
    frames = []
    for f in range(20):
        frames.append([_det(f, 100 + 4 * f, 600), _det(f, 800 - 4 * f, 200)])
    tracks = trk.track(frames)
    assert len(tracks) == 2
    assert all(len(t.boxes) == 20 for t in tracks)


def test_coasting_across_missed_detection_keeps_id():
    # Player A missed on frame 1; should resume the SAME track (no new id), not split.
    trk = build("player_tracker", {"backend": "iou"})
    frames = [
        [_det(0, 100, 600)],
        [],  # missed detection
        [_det(2, 108, 600)],
        [_det(3, 112, 600)],
    ]
    tracks = trk.track(frames)
    assert len(tracks) == 1
    assert len(tracks[0].boxes) == 3


def test_distinct_far_detections_make_new_track():
    # A detection on the opposite side of the frame is too far to coast onto -> new id.
    trk = build("player_tracker", {"backend": "iou", "max_age_frames": 30})
    frames = [[_det(0, 100, 600)], [_det(1, 1700, 100)]]
    tracks = trk.track(frames)
    assert len(tracks) == 2


def test_botsort_resolves_to_stable_ids_if_available():
    import numpy as np
    import pytest

    from badminton_coach.perception._util import module_available
    if not module_available("boxmot"):
        pytest.skip("boxmot not installed")
    from badminton_coach.core.schemas import Frame

    trk = build("player_tracker", {"backend": "botsort", "device": "cpu"})
    img = (np.random.rand(720, 1280, 3) * 255).astype("uint8")
    frames = [Frame(index=f, timestamp=f / 25.0, image=img) for f in range(6)]
    dets = [[_det(f, 300 + 3 * f, 400), _det(f, 900 - 3 * f, 200)] for f in range(6)]
    tracks = trk.track(dets, frames)
    assert 1 <= len(tracks) <= 2  # two players -> at most two stable IDs


def test_botsort_requires_frames():
    import pytest

    from badminton_coach.perception._util import module_available
    if not module_available("boxmot"):
        pytest.skip("boxmot not installed")
    trk = build("player_tracker", {"backend": "botsort", "device": "cpu"})
    with pytest.raises(ValueError):
        trk.track([[_det(0, 100, 600)]], frames=None)
