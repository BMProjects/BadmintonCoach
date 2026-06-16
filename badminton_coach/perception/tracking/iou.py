"""Greedy IoU + distance player tracker with track coasting.

Associates detections frame-to-frame. Two robustness features over plain IoU keep the
two singles players on stable IDs despite flicker:
  - coasting: a track stays alive for up to `max_age` frames after its last detection,
    so a missed/flickered frame doesn't spawn a new ID;
  - distance fallback: when boxes don't overlap (player moved between frames), match by
    gated centre distance (the gate scales with the frame gap) instead of failing.

Dependency-free (numpy/stdlib). Swap in BoT-SORT for doubles / near-net crossovers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...core.interfaces import PlayerTracker
from ...core.registry import register
from ...core.schemas import BBox, Detection, PlayerTrack, TrackedBox


def _iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def _diag(b: BBox) -> float:
    return (b.width ** 2 + b.height ** 2) ** 0.5 or 1.0


@dataclass
class _Track:
    tid: int
    last_box: BBox
    last_frame: int
    boxes: list[TrackedBox]


@register("player_tracker", "iou")
class IoUPlayerTracker(PlayerTracker):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.iou_threshold = float(self.config.get("iou_threshold", 0.2))
        self.max_age = int(self.config.get("max_age_frames", 30))
        # centre-distance gate as a multiple of the box diagonal, per frame of gap
        self.dist_gate = float(self.config.get("dist_gate_ratio", 1.0))
        # post-pass: chain temporally-disjoint, spatially-continuous fragments into one
        # track (bridges occlusions/replays > max_age). 0 disables.
        self.merge_gap = int(self.config.get("merge_gap_frames", 90))
        self.merge_gate = float(self.config.get("merge_dist_ratio", 2.5))

    @classmethod
    def is_available(cls) -> bool:
        return True

    def _score(self, track: _Track, det: Detection, frame_index: int) -> float:
        """Association score in [0,1]: IoU when boxes overlap, else a gated distance
        score that tolerates more movement across larger frame gaps."""
        iou = _iou(track.last_box, det.bbox)
        if iou >= self.iou_threshold:
            return iou
        gap = max(1, frame_index - track.last_frame)
        a, b = track.last_box.center, det.bbox.center
        dist = ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5
        gate = self.dist_gate * _diag(det.bbox) * gap
        if dist < gate:
            return 0.5 * (1.0 - dist / gate)  # below IoU matches, above new-track
        return 0.0

    def track(self, detections_per_frame: list[list[Detection]],
              frames=None) -> list[PlayerTrack]:
        active: list[_Track] = []
        finished: list[_Track] = []
        next_id = 0

        for frame_idx, dets in enumerate(detections_per_frame):
            fi = dets[0].frame_index if dets else frame_idx
            # retire tracks that have coasted past max_age
            still: list[_Track] = []
            for t in active:
                (still if fi - t.last_frame <= self.max_age else finished).append(t)
            active = still

            # candidate (score, det_idx, track_idx), greedy highest-first
            cands = []
            for di, det in enumerate(dets):
                for ti, t in enumerate(active):
                    s = self._score(t, det, fi)
                    if s > 0:
                        cands.append((s, di, ti))
            cands.sort(reverse=True)

            det_taken: set[int] = set()
            trk_taken: set[int] = set()
            for _s, di, ti in cands:
                if di in det_taken or ti in trk_taken:
                    continue
                det_taken.add(di)
                trk_taken.add(ti)
                t, det = active[ti], dets[di]
                t.last_box, t.last_frame = det.bbox, fi
                t.boxes.append(TrackedBox(fi, t.tid, det.bbox, det.confidence))

            for di, det in enumerate(dets):
                if di in det_taken:
                    continue
                t = _Track(next_id, det.bbox, fi,
                           [TrackedBox(fi, next_id, det.bbox, det.confidence)])
                next_id += 1
                active.append(t)

        all_tracks = self._merge_fragments(finished + active)
        all_tracks.sort(key=lambda t: t.tid)
        return [PlayerTrack(t.tid, tuple(t.boxes)) for t in all_tracks]

    def _merge_fragments(self, tracks: list[_Track]) -> list[_Track]:
        """Greedily chain fragments: a later track whose first box continues an earlier
        track's last box (small time gap + small centre distance) is appended to it."""
        if self.merge_gap <= 0:
            return tracks
        merged: list[_Track] = []
        for t in sorted(tracks, key=lambda x: x.boxes[0].frame_index):
            f0, b0 = t.boxes[0].frame_index, t.boxes[0].bbox
            best, best_d = None, None
            for m in merged:
                gap = f0 - m.last_frame
                if not 0 < gap <= self.merge_gap:
                    continue
                a, b = m.last_box.center, b0.center
                d = ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5
                if d < self.merge_gate * _diag(b0) * gap and (best_d is None or d < best_d):
                    best, best_d = m, d
            if best is not None:
                best.boxes.extend(TrackedBox(tb.frame_index, best.tid, tb.bbox, tb.confidence)
                                  for tb in t.boxes)
                best.last_box, best.last_frame = t.last_box, t.last_frame
            else:
                merged.append(t)
        return merged
