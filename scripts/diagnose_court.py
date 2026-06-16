"""Court-detection diagnostic on still images.

For each image: run the line_heatmap calibrator, then draw on the output
  - detected white lines (court_line_mask)        -> cyan tint
  - matched BWF court model (court_line_segments)  -> green, reprojected via homography
  - decoded keypoints                              -> yellow dots
  - reproj error + white-line overlap score        -> text

Usage:
    uv run python -m scripts.diagnose_court assets --backend line_heatmap
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import badminton_coach.perception  # noqa: F401 register backends
from badminton_coach.core.geometry import court_model, solve_homography
from badminton_coach.core.geometry.court_eval import court_overlap_score
from badminton_coach.core.geometry.court_surface import court_line_mask
from badminton_coach.core.registry import build
from badminton_coach.core.schemas import Frame, Point3D

_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _draw(img, calib, kpts, accepted):
    out = img.copy()
    # detected white lines -> cyan tint
    mask = court_line_mask(img)
    out[mask > 0] = (0.4 * out[mask > 0] + 0.6 * np.array([255, 255, 0])).astype(np.uint8)
    # matched BWF court model: green if accepted, red if rejected (phantom/wrong)
    color = (0, 255, 0) if accepted else (0, 0, 255)
    if calib is not None:
        for (ax, ay), (bx, by) in court_model.court_line_segments():
            a = calib.ground_to_image(Point3D(ax, ay, 0.0))
            b = calib.ground_to_image(Point3D(bx, by, 0.0))
            cv2.line(out, (int(a.x), int(a.y)), (int(b.x), int(b.y)), color, 2, cv2.LINE_AA)
    # decoded keypoints -> yellow
    for x, y in kpts:
        cv2.circle(out, (int(x), int(y)), 5, (0, 255, 255), -1)
        cv2.circle(out, (int(x), int(y)), 5, (0, 0, 0), 1)
    return out


def _label(out, lines):
    for i, t in enumerate(lines):
        y = 28 + 26 * i
        cv2.putText(out, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="image file or directory")
    ap.add_argument("--backend", default="line_heatmap")
    ap.add_argument("--weights", default="weights/court_lines_evit_b1.pt")
    ap.add_argument("--out", default="assets/court_diag")
    args = ap.parse_args()

    cal = build("court_calibrator", {
        "backend": args.backend, "device": "cuda", "weights": args.weights,
        "world_map": "weights/court_kp_official_world.json", "compute_camera": False,
    })
    src = Path(args.src)
    files = [src] if src.is_file() else sorted(
        p for p in src.iterdir() if p.suffix.lower() in _EXT
    )
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        frame = Frame(0, 0.0, img)
        calib, kpts, accepted, note = None, [], False, ""
        try:
            ip, wp, _n = cal._predict(frame)
            kpts = [(p.x, p.y) for p in ip]
            try:
                calib = cal.calibrate(frame)
                accepted = True
            except Exception as e:  # noqa: BLE001 - show the rejected raw match
                note = f"REJECTED: {str(e)[:64]}"
                if len(ip) >= 4:
                    calib = solve_homography(ip, wp)
        except Exception as e:  # noqa: BLE001
            note = f"NO PREDICTION: {str(e)[:60]}"
        out = _draw(img, calib, kpts, accepted)
        info = [f"{f.name}"]
        ov = f"{court_overlap_score(img, calib):.2f}" if calib is not None else "-"
        if accepted:
            info.append(f"OK  reproj {calib.reprojection_error_px:.1f}px  "
                        f"overlap {ov}  kpts {len(kpts)}")
        else:
            info.append(note)
            if calib is not None:
                info.append(f"(red=rejected match) reproj {calib.reprojection_error_px:.0f}px "
                            f"overlap {ov} kpts {len(kpts)}")
        _label(out, info)
        dst = outdir / f"{f.stem}_diag.png"
        cv2.imwrite(str(dst), out)
        print(f"{f.name:42s} -> {dst}  | {info[-1]}")


if __name__ == "__main__":
    main()
