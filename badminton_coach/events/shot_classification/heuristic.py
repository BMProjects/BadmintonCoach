"""Heuristic shot classifier from the reconstructed 3D shuttle trajectory.

No weights: between two hits, look at the 3D flight shape — peak height, landing
depth and descent steepness — to assign a coarse BWF stroke type. Falls back to
UNKNOWN when 3D is unavailable. The SOTA fine-grained classifier is BST (see bst.py).
"""

from __future__ import annotations

from ...core.interfaces import ShotClassifier
from ...core.registry import register
from ...core.schemas import PerceptionResult, ShotType
from ...core.schemas.events import HitEvent, Shot


@register("shot_classifier", "heuristic")
class HeuristicShotClassifier(ShotClassifier):
    @classmethod
    def is_available(cls) -> bool:
        return True

    def classify(self, hits: list[HitEvent], perception: PerceptionResult) -> list[Shot]:
        shots: list[Shot] = []
        traj3d = perception.shuttle_3d
        by_frame = {p.frame_index: p.point for p in traj3d.points} if traj3d else {}
        for a, b in zip(hits, hits[1:], strict=False):
            seg = [by_frame[f] for f in range(a.frame_index, b.frame_index + 1) if f in by_frame]
            stype, conf = self._classify_segment(seg)
            shots.append(Shot(start_frame=a.frame_index, end_frame=b.frame_index,
                              hitter_track_id=a.hitter_track_id, shot_type=stype, confidence=conf))
        return shots

    @staticmethod
    def _classify_segment(seg) -> tuple[ShotType, float]:
        if len(seg) < 3:
            return ShotType.UNKNOWN, 0.0
        zs = [p.z for p in seg]
        peak = max(zs)
        z_start, z_end = seg[0].z, seg[-1].z
        descending = z_end < z_start
        if peak > 4.0:
            return (ShotType.CLEAR if descending else ShotType.LIFT), 0.5
        if descending and (z_start - z_end) > 1.5 and peak < 3.5:
            return ShotType.SMASH, 0.5
        if peak < 1.8:
            return ShotType.DRIVE, 0.4
        return ShotType.DROP, 0.4
