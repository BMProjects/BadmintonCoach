"""Render a PerceptionResult back onto its video as an annotated MP4.

Re-reads the source frames (the pipeline discards them) and composites the
requested overlays. Each overlay is independently toggleable so a developer can
inspect one module at a time.

Output is encoded as H.264 (yuv420p) via imageio-ffmpeg when available so it plays
in browsers / the Gradio video widget; falls back to OpenCV mp4v otherwise.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from ..core.io import VideoReader
from ..core.schemas import PerceptionResult
from . import metrics, overlay


def _event_lookups(analysis, fps, box_by_frame, shuttle_pt_by_frame):
    """Per-frame event state from a MatchAnalysis. Returns state(f) -> (rally, is_hit,
    hit_pt, [(stroke, hitter_box)]). Hits flash BRIEFLY (~0.15s) rather than persist;
    each active stroke is labelled next to the player who hit it (nearest player to the
    shuttle at the hit frame)."""
    shots = list(analysis.shots) if analysis else []
    rallies = list(analysis.rallies) if analysis else []
    effort_by_idx = {}
    if analysis is not None and analysis.biomechanics is not None:
        effort_by_idx = {sb.shot_index: sb.effort_nm for sb in analysis.biomechanics.strokes}
    effort_max = max(effort_by_idx.values(), default=0.0) or 1.0
    hold = max(1, int(0.15 * (fps or 25.0)))
    hits: dict[int, tuple[float, float]] = {}
    hit_pos: dict[int, tuple[float, float]] = {}
    for h in (analysis.hits if analysis else []):
        hit_pos[h.frame_index] = (h.shuttle_image_pos.x, h.shuttle_image_pos.y)
        for d in range(hold + 1):
            hits.setdefault(h.frame_index + d, (h.shuttle_image_pos.x, h.shuttle_image_pos.y))

    def _hitter_tid(start_f):
        boxes = box_by_frame.get(start_f) or []
        pos = hit_pos.get(start_f)
        if pos is None and start_f in shuttle_pt_by_frame:
            sp = shuttle_pt_by_frame[start_f]
            pos = (sp.point.x, sp.point.y)
        if pos is None or not boxes:
            return boxes[0][0] if boxes else None  # fall back to any tracked player
        sx, sy = pos
        return min(boxes, key=lambda tb: (tb[1].center.x - sx) ** 2
                   + (tb[1].center.y - sy) ** 2)[0]
    shot_hitter = [(s, _hitter_tid(s.start_frame)) for s in shots]

    def state(f):
        rally = None
        for i, r in enumerate(rallies, 1):
            if r.start_frame <= f <= r.end_frame:
                rally = i
        active = []
        for i, (s, tid) in enumerate(shot_hitter, 1):
            if tid is not None and s.start_frame <= f <= s.end_frame:
                box = next((b for t, b in (box_by_frame.get(f) or []) if t == tid), None)
                if box is not None:
                    e = effort_by_idx.get(i)
                    norm = (e / effort_max) if e is not None else None
                    active.append((s.shot_type.value, box, tid, norm))
        return rally, (f in hits), hits.get(f), active

    return state


def _annotated_frames(video_path, result, max_frames, show, output_fps, analysis=None):
    dets_by_frame: dict[int, list] = {}
    for d in result.detections:
        dets_by_frame.setdefault(d.frame_index, []).append(d)
    poses_by_frame: dict[int, list] = {}
    for p in result.poses:
        poses_by_frame.setdefault(p.frame_index, []).append(p)
    # player box per frame per track (to label speed next to the player)
    box_by_frame: dict[int, list] = {}
    for tr in result.player_tracks:
        for tb in tr.boxes:
            box_by_frame.setdefault(tb.frame_index, []).append((tr.track_id, tb.bbox))
    shuttle_pt_by_frame = {p.frame_index: p for p in result.shuttle_2d.points}

    p_speeds = metrics.player_speeds(result) if show["speeds"] else {}
    s_speeds = metrics.shuttle_speeds(result.shuttle_3d, result.fps) if show["speeds"] else {}
    event_state = (_event_lookups(analysis, result.fps, box_by_frame, shuttle_pt_by_frame)
                   if show.get("events") else None)
    # Timed landings: each shot's z=0 floor point, shown only briefly when it lands.
    landings_timed = []
    if show.get("events") and analysis is not None and result.court is not None \
            and result.shuttle_3d is not None:
        from ..events.stats import _shot_landing
        by_f = {p.frame_index: p.point for p in result.shuttle_3d.points}
        for s in analysis.shots:
            seg = [by_f[f] for f in range(s.start_frame, s.end_frame + 1) if f in by_f]
            land = _shot_landing(seg)
            if land is not None:
                landings_timed.append((s.end_frame, land))
    land_hold = max(1, int(1.0 * (result.fps or 25.0)))

    with VideoReader(video_path, max_frames=max_frames) as reader:
        src_fps = reader.fps
        out_fps = min(output_fps, src_fps)
        court_frames = result.court_frames
        emitted = -1
        for frame in reader:
            # timestamp-based decimation -> ~out_fps with correct playback timing
            slot = int(frame.index * out_fps / src_fps)
            if slot == emitted:
                continue
            emitted = slot
            img = frame.image
            court_here = court_frames is None or frame.index in court_frames
            if show["court"] and result.court is not None and court_here:
                img = overlay.draw_court(img, result.court)
            if show["detections"]:
                img = overlay.draw_detections(img, dets_by_frame.get(frame.index, []))
            if show["poses"]:
                img = overlay.draw_poses(img, poses_by_frame.get(frame.index, []))
            if show["shuttle"]:
                trail = max(1, int(result.fps or 25.0))  # ~1s connecting line
                img = overlay.draw_shuttle(img, result.shuttle_2d, frame.index, trail=trail)
            if show["speeds"]:
                players = [(bb, p_speeds.get(frame.index, {}).get(tid))
                           for tid, bb in box_by_frame.get(frame.index, [])]
                sp = shuttle_pt_by_frame.get(frame.index)
                ssp = s_speeds.get(frame.index)
                shuttle = (sp.point, ssp * 3.6) if (sp and sp.visible and ssp is not None) else None
                img = metrics.annotate_speeds(img, players, shuttle)
            if event_state is not None:
                rally, is_hit, hit_pt, active = event_state(frame.index)
                img = overlay.draw_event_hud(img, rally=rally, is_hit=is_hit, hit_pt=hit_pt)
                fposes = poses_by_frame.get(frame.index, [])
                for stroke, box, _tid, norm in active:
                    img = overlay.draw_stroke_label(img, stroke, box)  # stroke type next to player
                    if norm is not None and fposes:  # force as trunk-line colour
                        cx, cy = box.center.x, box.center.y
                        hp = min(fposes, key=lambda p: (p.keypoints[0].point.x - cx) ** 2
                                 + (p.keypoints[0].point.y - cy) ** 2)
                        img = overlay.draw_trunk_force(img, hp, norm)
                active_land = [xy for ef, xy in landings_timed
                               if ef <= frame.index <= ef + land_hold]
                if active_land and result.court is not None:
                    img = overlay.draw_landings(img, result.court, active_land, numbered=False)
            yield img, out_fps


def render_video(
    video_path: str | Path,
    result: PerceptionResult,
    out_path: str | Path,
    show_detections: bool = False,
    show_poses: bool = True,
    show_court: bool = True,
    show_shuttle: bool = True,
    show_speeds: bool = True,
    show_events: bool = True,
    analysis=None,
    max_frames: int | None = None,
    output_fps: float = 15.0,
) -> Path:
    """Composite overlays onto the source video; return the output path (H.264).

    Player detection boxes are off by default (the pose skeleton already marks the
    player); player speed (m/s) is labelled by each player and shuttle speed (km/h)
    by the shuttle. When `analysis` (a MatchAnalysis) is given and show_events is set,
    a temporal event HUD (rally/shot/stroke + hit flash) is drawn. Output is decimated
    to ~output_fps (default 15).
    """
    out_path = Path(out_path)
    show = {
        "court": show_court,
        "detections": show_detections,
        "poses": show_poses,
        "shuttle": show_shuttle,
        "speeds": show_speeds,
        "events": show_events and analysis is not None,
    }
    frames = _annotated_frames(video_path, result, max_frames, show, output_fps, analysis)

    try:
        import imageio.v2 as imageio  # uses bundled ffmpeg (libx264)

        first = next(frames, None)
        if first is None:
            raise RuntimeError(f"No frames to render from {video_path}")
        img, fps = first
        writer = imageio.get_writer(
            str(out_path), fps=fps, codec="libx264", pixelformat="yuv420p",
            macro_block_size=None,
        )
        writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for img, _ in frames:
            writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        writer.close()
        return out_path
    except ImportError:
        pass

    # Fallback: OpenCV mp4v (may not preview in some browsers).
    writer = None
    for img, fps in frames:
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(img)
    if writer is not None:
        writer.release()
    return out_path
