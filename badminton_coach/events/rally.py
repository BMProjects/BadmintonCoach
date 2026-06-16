"""Rally state machine: group hits into rallies (contiguous spans of play).

A rally is a run of hits with small inter-hit gaps; a gap longer than `max_gap_s`
(shuttle dead / between points) ends the rally. Single-hit groups are dropped (a
lone direction reversal isn't an exchange).
"""

from __future__ import annotations

from ..core.schemas import Rally
from ..core.schemas.events import HitEvent, Shot


def segment_rallies(
    hits: list[HitEvent],
    shots: list[Shot],
    fps: float,
    max_gap_s: float = 2.0,
    min_hits: int = 2,
) -> list[Rally]:
    """Split hits into rallies on inter-hit gaps > max_gap_s; attach each rally's shots."""
    if not hits:
        return []
    ordered = sorted(hits, key=lambda h: h.frame_index)
    max_gap = max_gap_s * (fps or 30.0)

    groups: list[list[HitEvent]] = [[ordered[0]]]
    for h in ordered[1:]:
        if h.frame_index - groups[-1][-1].frame_index > max_gap:
            groups.append([h])
        else:
            groups[-1].append(h)

    rallies: list[Rally] = []
    for g in groups:
        if len(g) < min_hits:
            continue
        start, end = g[0].frame_index, g[-1].frame_index
        rshots = tuple(s for s in shots if start <= s.start_frame < end)
        rallies.append(Rally(start_frame=start, end_frame=end, hits=tuple(g), shots=rshots))
    return rallies
