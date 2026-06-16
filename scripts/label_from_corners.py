"""Auto-label a video's court keypoints from 4 clicked corners (collection engine).

The court has a fixed metric geometry and the 22 keypoints' world coordinates are
known (weights/court_kp_official_world.json). So 4 corner clicks fully determine the
homography -> we project all 22 world keypoints into the image. For a fixed-camera
video this labels every sampled frame consistently, turning a few clicks into a
22-keypoint training set for amateur/phone footage.

Usage:
  python -m scripts.label_from_corners --video clip.mp4 \
      --corners "nLx,nLy nRx,nRy fRx,fRy fLx,fLy" \
      --out data/amateur_court/train --every 15
(corner order: near-Left, near-Right, far-Right, far-Left = doubles outline)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from badminton_coach.core.geometry.court_model import court_corners_doubles


def project_22(image_corners: np.ndarray, world_map: list) -> list[float]:
    """Project the 22 world keypoints into the image via the 4-corner homography."""
    world4 = np.array(court_corners_doubles(), np.float32)
    h = cv2.getPerspectiveTransform(world4, image_corners.astype(np.float32))
    kp: list[float] = []
    for wx, wy in world_map:
        v = h @ np.array([wx, wy, 1.0])
        kp += [float(v[0] / v[2]), float(v[1] / v[2]), 2]
    return kp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--corners", required=True, help="'nLx,nLy nRx,nRy fRx,fRy fLx,fLy'")
    ap.add_argument("--out", default="data/amateur_court/train")
    ap.add_argument("--every", type=int, default=15)
    ap.add_argument("--world", default="weights/court_kp_official_world.json")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max", type=int, default=80)
    args = ap.parse_args()

    corners = np.array([[float(v) for v in pair.split(",")] for pair in args.corners.split()])
    assert corners.shape == (4, 2), "need 4 'x,y' corners"
    world_map = json.loads(Path(args.world).read_text())
    kp_names = [str(i) for i in range(len(world_map))]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    images, annotations = [], []
    fi = saved = 0
    while saved < args.max:
        ok, frame = cap.read()
        if not ok:
            break
        if fi >= args.start and (fi - args.start) % args.every == 0:
            kp = project_22(corners, world_map)
            fname = f"{Path(args.video).stem}_{fi:05d}.jpg"
            cv2.imwrite(str(out_dir / fname), frame)
            xs, ys = kp[0::3], kp[1::3]
            images.append({"id": saved, "file_name": fname,
                           "width": frame.shape[1], "height": frame.shape[0]})
            annotations.append({"id": saved, "image_id": saved, "category_id": 1,
                                "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                                "iscrowd": 0, "num_keypoints": len(world_map), "keypoints": kp})
            saved += 1
        fi += 1
    cap.release()
    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "court", "keypoints": kp_names}]}
    (out_dir / "_annotations.coco.json").write_text(json.dumps(coco))
    print(f"labeled {saved} frames -> {out_dir}")


if __name__ == "__main__":
    main()
