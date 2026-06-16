"""BadmintonCoach Analysis Studio — a video-editor-style Gradio app.

Pick a source clip + settings (collapsible top bar), press Process, and the layout shows
a full-width player that autoplays the ANNOTATED video (its scrubber is the timeline
cursor) above a full-width multi-track event timeline (rally / hit / stroke, colour-coded
by stroke type along the time axis). Stats + backends fold into an accordion below.

Run:  python -m apps.dev_console      (needs the [ui] extra: gradio, pillow)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr

from badminton_coach.core.config import load_config
from badminton_coach.core.pipeline import Phase1Pipeline
from badminton_coach.core.registry import available_backends
from badminton_coach.viz import render_video
from badminton_coach.viz.report import build_player_report
from badminton_coach.viz.timeline import render_timeline, time_at_x

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def _list_configs() -> list[str]:
    return sorted(str(p) for p in CONFIG_DIR.glob("*.yaml"))


def _backend_table() -> str:
    lines = ["Registered backends:"]
    for kind, names in available_backends().items():
        lines.append(f"  {kind:18s}: {', '.join(names)}")
    return "\n".join(lines)


def analyze(video_path, config_path, max_frames, estimate_3d, classifier, biomech_backend,
            mb_seq, tracker, height_m, weight_kg, handedness, show_pose, show_court,
            show_shuttle, show_speeds, show_events):
    if not video_path:
        return None, None, "Please upload a video.", 1.0, "_Upload a clip._"
    biomech_backend = biomech_backend or "lift3d"
    import dataclasses
    import time

    import badminton_coach.biomechanics  # noqa: F401  (register L3 backends)
    import badminton_coach.events  # noqa: F401  (register L2 backends)
    from badminton_coach.core.registry import build
    from badminton_coach.core.schemas import PlayerProfile
    from badminton_coach.events.analyze import analyze_events

    cfg = load_config(config_path)
    if max_frames and max_frames > 0:
        io = cfg.io.model_copy(update={"max_frames": int(max_frames)})
        cfg = cfg.model_copy(update={"io": io})
    if tracker:
        pt = cfg.perception.player_tracker.model_copy(update={"backend": tracker})
        cfg = cfg.model_copy(update={"perception": cfg.perception.model_copy(
            update={"player_tracker": pt})})

    t0 = time.perf_counter()
    result = Phase1Pipeline.from_config(cfg).run(video_path, estimate_3d=estimate_3d)
    t_l1 = time.perf_counter() - t0

    # L2/L3: hits + per-shot 3D + stroke classification + rallies/stats + biomechanics.
    # Skipped when 3D estimation is off (no court). bst = SOTA transformer (default).
    ma = None
    clf = classifier or "bst"
    profile = PlayerProfile(height_m=float(height_m), mass_kg=float(weight_kg),
                            handedness=handedness or "R")
    t_l2 = 0.0
    if estimate_3d:
        t0 = time.perf_counter()
        rec = build("reconstructor", cfg.perception.reconstructor.model_dump())
        biomech = build("biomechanics", {"backend": biomech_backend, "seq": int(mb_seq)})
        ma = analyze_events(result, build("hit_detector", {"backend": "trajectory"}),
                            build("shot_classifier", {"backend": clf}), rec,
                            biomech, profile)
        if ma.shuttle_3d is not None:
            result = dataclasses.replace(result, shuttle_3d=ma.shuttle_3d)
        t_l2 = time.perf_counter() - t0

    t0 = time.perf_counter()
    out_path = Path(tempfile.mkdtemp()) / "annotated.mp4"
    render_video(
        video_path, result, out_path,
        show_detections=False, show_poses=show_pose, show_court=show_court and estimate_3d,
        show_shuttle=show_shuttle, show_speeds=show_speeds and estimate_3d,
        show_events=show_events, analysis=ma,
        max_frames=int(max_frames) if max_frames else None,
    )
    t_render = time.perf_counter() - t0

    if not estimate_3d:
        court_str = "OFF (3D estimation disabled — fast 2D-only pipeline)"
    elif result.court is not None:
        court_str = f"{result.court.reprojection_error_px:.2f} px"
    else:
        court_str = "NOT DETECTED (amateur/oblique footage? court overlay+3D+speeds disabled)"
    summary = "\n".join([
        f"config           : {cfg.name}   3D estimation: {'ON' if estimate_3d else 'OFF'}",
        f"timing           : L1 {t_l1:.2f}s | L2 {t_l2:.2f}s | render {t_render:.2f}s",
        f"fps / frames     : {result.fps:.2f} / {result.frame_count}",
        f"court            : {court_str}",
        f"player tracks    : {len(result.player_tracks)} | poses {len(result.poses)}",
        f"shuttle 2D / 3D  : {len(result.shuttle_2d)} / "
        f"{(len(ma.shuttle_3d) if ma and ma.shuttle_3d else 0)} pts",
        f"hits (L2)        : {len(ma.hits) if ma else '-'}",
        f"shots ({clf})    : {', '.join(s.shot_type.value for s in ma.shots) if ma else '-'}",
    ])
    if ma and ma.stats is not None:
        st = ma.stats
        mix = ", ".join(f"{k}:{v}" for k, v in sorted(st.shot_type_counts.items()))
        summary += "\n" + "\n".join([
            "--- tactics (L3) ---",
            f"rallies          : {st.rally_count} | avg {st.avg_shots_per_rally:.1f} shots/rally",
            f"shot mix         : {mix or '-'}",
            *[f"player {m.track_id} move  : {m.distance_m:.1f} m | avg {m.avg_speed_ms:.1f} "
              f"| max {m.max_speed_ms:.1f} m/s" for m in st.player_movement],
            f"landings (z=0)   : {len(st.landing_points_m)} pts  "
            + " ".join(f"({x:.1f},{y:.1f})" for x, y in st.landing_points_m),
        ])
    if ma and ma.biomechanics is not None and ma.biomechanics.strokes:
        rows = [f"--- biomechanics ({biomech_backend}, h={height_m}m {weight_kg}kg "
                f"{handedness}) ---"]
        for sb in ma.biomechanics.strokes:
            jt = " ".join(f"{j.name[:2]}{j.peak_angle_deg:.0f}d/{j.peak_torque_nm:.0f}Nm"
                          for j in sb.joints)
            seq = "->".join(s[:4] for s in sb.kinematic_sequence)
            rows.append(f"  shot{sb.shot_index:<2d} effort {sb.effort_nm:5.0f}Nm  "
                        f"seq {seq} {'OK' if sb.sequence_ok else '!'}  {jt}")
        summary += "\n" + "\n".join(rows)
    fps = result.fps or 25.0
    duration_s = result.frame_count / fps if result.frame_count else 1.0
    timeline = render_timeline(ma, duration_s, fps=fps) if ma else None
    report = build_player_report(ma, profile, biomech_backend)
    return str(out_path), timeline, summary, duration_s, report


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="BadmintonCoach Studio") as demo:
        gr.Markdown("## BadmintonCoach — Analysis Studio")

        # Input + settings live in a collapsible left sidebar; the main area is the
        # editor (video + timeline) and the player report.
        with gr.Sidebar(label="Input & settings", open=True):
            video_in = gr.Video(label="Source clip", height=150)
            config_in = gr.Dropdown(_list_configs(), label="Config preset",
                                    value=(_list_configs() or [None])[0])
            classifier = gr.Dropdown(["bst", "heuristic"], value="bst",
                                     label="Stroke classifier")
            biomech_backend = gr.Dropdown(["lift3d", "pose2d"], value="lift3d",
                                          label="Biomechanics")
            mb_seq = gr.Dropdown(["27", "81", "243"], value="81", label="MotionBERT seq")
            tracker = gr.Dropdown(["iou", "botsort"], value="iou", label="Player tracker",
                                  info="botsort = ReID, fixes crossings")
            max_frames = gr.Slider(0, 600, value=0, step=30, label="Max frames (0 = all)")
            with gr.Row():
                height_m = gr.Number(1.80, label="Height (m)", minimum=1.2, maximum=2.2,
                                     step=0.01)
                weight_kg = gr.Number(73, label="Weight (kg)", minimum=30, maximum=150,
                                      step=1)
            handedness = gr.Dropdown(["R", "L"], value="R", label="Racket hand")
            estimate_3d = gr.Checkbox(True, label="3D estimation")
            with gr.Row():
                show_pose = gr.Checkbox(True, label="Poses")
                show_court = gr.Checkbox(True, label="Court")
                show_shuttle = gr.Checkbox(True, label="Shuttle")
            with gr.Row():
                show_speeds = gr.Checkbox(True, label="Speeds")
                show_events = gr.Checkbox(True, label="Events")
            run = gr.Button("Process", variant="primary")

        # Editor layout in tabs: Studio (player + timeline) and Player Report.
        with gr.Tab("Studio"):
            video_out = gr.Video(label="Processed video", autoplay=True, height=460,
                                 elem_id="bc_video")
            timeline_out = gr.Image(label="Event timeline — click to seek",
                                    show_label=True, height=190, interactive=False)
            with gr.Accordion("Summary & backends", open=False):
                summary_out = gr.Textbox(label="Summary", lines=14)
                gr.Textbox(_backend_table(), label="Backends", lines=8)
        with gr.Tab("Player Report"):
            report_out = gr.Markdown("_Process a clip to generate the report._")

        duration_state = gr.State(1.0)
        seek_t = gr.Number(visible=False)

        run.click(
            analyze,
            inputs=[video_in, config_in, max_frames, estimate_3d, classifier,
                    biomech_backend, mb_seq, tracker, height_m, weight_kg, handedness,
                    show_pose, show_court, show_shuttle, show_speeds, show_events],
            outputs=[video_out, timeline_out, summary_out, duration_state, report_out],
        )

        # Click on the timeline -> map pixel x to time -> seek the <video> via JS.
        def _seek(duration, evt: gr.SelectData):
            return time_at_x(evt.index[0], duration)

        timeline_out.select(_seek, inputs=[duration_state], outputs=[seek_t]).then(
            None, inputs=[seek_t], outputs=None,
            js="(t) => { const v = document.querySelector('#bc_video video');"
               " if (v && !isNaN(t)) { v.currentTime = t; v.play(); } }",
        )
    return demo


def main() -> None:
    # Bind to all interfaces so the console is reachable from other machines on the
    # network. Fixed port (default 7860); Ctrl+C exits cleanly and frees the port.
    import os

    host = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo = build_ui()
    try:
        demo.launch(server_name=host, server_port=port, theme=gr.themes.Base())
    except OSError:
        print(f"\nPort {port} is already in use — a previous console is still running.\n"
              f"Stop it first:  pkill -f apps.dev_console   (or Ctrl+C in its terminal)\n"
              f"or pick another port:  GRADIO_SERVER_PORT=7870 python -m apps.dev_console")
    except KeyboardInterrupt:
        pass
    finally:
        demo.close()  # release the port on exit


if __name__ == "__main__":
    main()
