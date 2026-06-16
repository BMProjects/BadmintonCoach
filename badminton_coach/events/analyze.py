"""L2 orchestrator: PerceptionResult -> hits, per-shot 3D, classified shots.

Key fix vs Phase-1: the 3D reconstructor is run PER SHOT (between consecutive
hits), so each call sees a single parabola. This gives MonoTrack correct segment
boundaries and dramatically lowers reprojection error (a whole-rally single-parabola
fit was ~235px).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ..core.interfaces import HitDetector, Reconstructor3D, ShotClassifier
from ..core.schemas import PerceptionResult, ShuttleTrajectory2D, ShuttleTrajectory3D
from ..core.schemas.analysis import MatchAnalysis


def analyze_events(
    perception: PerceptionResult,
    hit_detector: HitDetector,
    shot_classifier: ShotClassifier,
    reconstructor: Reconstructor3D | None = None,
    biomech_analyzer=None,
    player_profile=None,
) -> MatchAnalysis:
    hits = hit_detector.detect(perception.shuttle_2d)

    shuttle_3d = perception.shuttle_3d
    has_camera = perception.court is not None and perception.court.camera is not None
    if reconstructor is not None and has_camera and len(hits) >= 2:
        shuttle_3d = _reconstruct_per_shot(perception, hits, reconstructor)

    perception = dataclasses.replace(perception, shuttle_3d=shuttle_3d)
    shots = shot_classifier.classify(hits, perception)

    from .rally import segment_rallies
    from .stats import compute_match_stats

    rallies = segment_rallies(list(hits), list(shots), perception.fps or 30.0)
    stats = compute_match_stats(perception, list(hits), list(shots), rallies)
    biomech = None
    if biomech_analyzer is not None and shots:
        biomech = biomech_analyzer.analyze(list(shots), perception, player_profile)
    return MatchAnalysis(
        hits=tuple(hits), shots=tuple(shots), shuttle_3d=shuttle_3d,
        rallies=tuple(rallies), stats=stats, biomechanics=biomech,
    )


def _reconstruct_per_shot(perception, hits, reconstructor) -> ShuttleTrajectory3D:
    pts2d = perception.shuttle_2d.points
    all_pts, residuals = [], []
    for a, b in zip(hits, hits[1:], strict=False):
        seg = ShuttleTrajectory2D(
            points=tuple(p for p in pts2d if a.frame_index <= p.frame_index <= b.frame_index)
        )
        if len(seg) < 3:
            continue
        rec = reconstructor.reconstruct(seg, perception.court)
        all_pts.extend(rec.points)
        if np.isfinite(rec.reprojection_error_px):
            residuals.append(rec.reprojection_error_px)
    err = float(np.mean(residuals)) if residuals else float("inf")
    return ShuttleTrajectory3D(points=tuple(all_pts), reprojection_error_px=err,
                               method="monotrack-perhit")
