"""Rally segmentation + tactical stats (L3)."""

from __future__ import annotations

from badminton_coach.core.schemas import (
    PerceptionResult,
    Point2D,
    ShuttleTrajectory2D,
)
from badminton_coach.core.schemas.events import HitEvent, Shot, ShotType
from badminton_coach.events.rally import segment_rallies
from badminton_coach.events.stats import compute_match_stats


def _hit(f):
    return HitEvent(frame_index=f, shuttle_image_pos=Point2D(100.0, 100.0))


def test_segment_rallies_splits_on_long_gap():
    # Two bursts of hits separated by a > max_gap gap -> two rallies.
    hits = [_hit(0), _hit(10), _hit(20), _hit(200), _hit(212)]
    rallies = segment_rallies(hits, [], fps=25.0, max_gap_s=2.0)  # gap=50 frames
    assert len(rallies) == 2
    assert rallies[0].start_frame == 0 and rallies[0].end_frame == 20
    assert rallies[1].start_frame == 200 and rallies[1].end_frame == 212


def test_segment_rallies_drops_lone_hits():
    # A single isolated hit is not a rally.
    rallies = segment_rallies([_hit(0), _hit(500)], [], fps=25.0, max_gap_s=2.0)
    assert rallies == []


def test_match_stats_counts_shots_without_court():
    # No court -> movement/ground points empty, but stroke mix + rally stats still work.
    shots = [
        Shot(0, 10, None, ShotType.SMASH, 0.5),
        Shot(10, 20, None, ShotType.DROP, 0.4),
        Shot(20, 30, None, ShotType.SMASH, 0.6),
    ]
    hits = [_hit(0), _hit(10), _hit(20), _hit(30)]
    perception = PerceptionResult(
        source="x", fps=25.0, frame_count=31, court=None, detections=(),
        player_tracks=(), poses=(), shuttle_2d=ShuttleTrajectory2D(points=()),
        shuttle_3d=None,
    )
    rallies = segment_rallies(hits, shots, fps=25.0)
    stats = compute_match_stats(perception, hits, shots, rallies)
    assert stats.shot_type_counts == {"smash": 2, "drop": 1}
    assert stats.rally_count == 1
    assert stats.player_movement == ()
    assert stats.hit_ground_points_m == ()


def test_shot_landing_zero_crossing():
    # A descending parabola crossing z=0 -> landing at the interpolated (x,y).
    from badminton_coach.core.schemas import Point3D
    from badminton_coach.events.stats import _shot_landing

    # z: 2,1,-1 (crosses 0 between idx1 and idx2); x advances 0,1,2; y constant 3
    seg = [Point3D(0.0, 3.0, 2.0), Point3D(1.0, 3.0, 1.0), Point3D(2.0, 3.0, -1.0)]
    land = _shot_landing(seg)
    assert land is not None
    x, y = land
    assert abs(x - 1.5) < 1e-6 and abs(y - 3.0) < 1e-6


def test_shot_landing_extrapolates_returned_shot():
    # All z>0 (shuttle returned before landing) -> extrapolate parabola to z=0 ahead.
    from badminton_coach.core.schemas import Point3D
    from badminton_coach.events.stats import _shot_landing

    # descending parabola z = 10 - 0.5 t^2, all > 0 over t=0..3, crosses 0 at t~4.47
    seg = [Point3D(float(t), 0.0, 10.0 - 0.5 * t * t) for t in range(4)]
    land = _shot_landing(seg)
    assert land is not None  # extrapolated forward to z=0
    assert land[0] > 3.0  # landing x is ahead of the last observed point (t=3)
