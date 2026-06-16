"""Court annotation tool — click 4 corners to label amateur/phone footage.

Workflow to collect court training data (and instantly calibrate fixed-camera
phone videos):
  1. upload a video, pick a frame;
  2. click the 4 doubles-court corners in order: near-Left, near-Right, far-Right,
     far-Left;
  3. the tool projects the full BWF court (and the 22 keypoints) from those corners
     so you can verify the fit;
  4. "Label video" auto-generates a 22-keypoint COCO dataset for the clip (fixed
     camera -> every sampled frame), to merge into training;
     "Copy manual corners" prints the corners for configs (manual/two_stage backend)
     for immediate calibration with no retraining.

Run:  python -m apps.annotate_court   (binds 0.0.0.0:7861)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from badminton_coach.core.geometry.court_model import court_corners_doubles, court_line_segments

CORNER_NAMES = ["near-Left", "near-Right", "far-Right", "far-Left"]
_WORLD_MAP = "weights/court_kp_official_world.json"


def _first_frame(video, frame_idx):
    if not video:
        return None, [], "Upload a video."
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None, [], "Could not read frame."
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), [], f"Click corner 1/4: {CORNER_NAMES[0]}"


def _homography(corners):
    world4 = np.array(court_corners_doubles(), np.float32)
    return cv2.getPerspectiveTransform(world4, np.array(corners, np.float32))


def _draw(frame_rgb, corners):
    vis = frame_rgb.copy()
    for i, (x, y) in enumerate(corners):
        cv2.circle(vis, (int(x), int(y)), 6, (255, 0, 0), -1)
        cv2.putText(vis, CORNER_NAMES[i], (int(x) + 6, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    if len(corners) == 4:
        h = _homography(corners)
        for (ax, ay), (bx, by) in court_line_segments():
            a = h @ [ax, ay, 1]
            b = h @ [bx, by, 1]
            a, b = a[:2] / a[2], b[:2] / b[2]
            cv2.line(vis, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 255, 0), 2)
    return vis


def on_click(frame_rgb, corners, evt: gr.SelectData):
    if frame_rgb is None:
        return None, corners, "Load a frame first."
    corners = (corners or []) + [list(evt.index)]
    corners = corners[:4]
    vis = _draw(frame_rgb, corners)
    if len(corners) < 4:
        msg = f"Click corner {len(corners)+1}/4: {CORNER_NAMES[len(corners)]}"
    else:
        msg = "4 corners set. Verify the green court overlay, then Label / Copy corners."
    return vis, corners, msg


def label_video(video, corners, every):
    if not video or len(corners or []) != 4:
        return "Need a video and 4 corners."
    from scripts.label_from_corners import project_22

    world_map = json.loads(Path(_WORLD_MAP).read_text())
    out = Path("data/amateur_court/train")
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video)
    images, anns, fi, saved = [], [], 0, 0
    kp = project_22(np.array(corners), world_map)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % int(every) == 0:
            fn = f"{Path(video).stem}_{fi:05d}.jpg"
            cv2.imwrite(str(out / fn), frame)
            xs, ys = kp[0::3], kp[1::3]
            images.append({"id": saved, "file_name": fn,
                           "width": frame.shape[1], "height": frame.shape[0]})
            anns.append({"id": saved, "image_id": saved, "category_id": 1, "iscrowd": 0,
                         "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                         "num_keypoints": len(world_map), "keypoints": kp})
            saved += 1
        fi += 1
    cap.release()
    names = [str(i) for i in range(len(world_map))]
    (out / "_annotations.coco.json").write_text(json.dumps(
        {"images": images, "annotations": anns,
         "categories": [{"id": 1, "name": "court", "keypoints": names}]}))
    return f"Labeled {saved} frames -> {out}. Merge into training, then retrain."


def copy_corners(corners):
    if len(corners or []) != 4:
        return "Set 4 corners first."
    pts = [[round(x, 1), round(y, 1)] for x, y in corners]
    return f"For manual/two_stage backend:\n  image_corners: {pts}"


def build_ui():
    with gr.Blocks(title="Court Annotation") as demo:
        gr.Markdown("# Court Annotation — click 4 doubles corners (near-L, near-R, far-R, far-L)")
        corners_state = gr.State([])
        with gr.Row():
            with gr.Column():
                video = gr.Video(label="Video")
                frame_idx = gr.Slider(0, 600, value=0, step=1, label="Frame")
                load = gr.Button("Load frame")
                every = gr.Slider(1, 60, value=15, step=1, label="Label every N frames")
                with gr.Row():
                    label_btn = gr.Button("Label video", variant="primary")
                    copy_btn = gr.Button("Copy manual corners")
            with gr.Column():
                canvas = gr.Image(label="Click 4 corners", type="numpy", interactive=True)
                status = gr.Textbox(label="Status")
        load.click(_first_frame, [video, frame_idx], [canvas, corners_state, status])
        canvas.select(on_click, [canvas, corners_state], [canvas, corners_state, status])
        label_btn.click(label_video, [video, corners_state, every], [status])
        copy_btn.click(copy_corners, [corners_state], [status])
    return demo


def main() -> None:
    host = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7861"))
    demo = build_ui()
    try:
        demo.launch(server_name=host, server_port=port)
    except OSError:
        print(f"\nPort {port} in use. Stop the running tool: pkill -f apps.annotate_court")
    except KeyboardInterrupt:
        pass
    finally:
        demo.close()


if __name__ == "__main__":
    main()
