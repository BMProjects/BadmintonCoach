"""Validate the BST stroke classifier against ShuttleSet ground truth (yolo_pose path).

ShuttleSet ships per-stroke labels (type + frame_num) but no raw video/poses, so we run
our own perception (yolo detect + yolo_pose + tracknetv3 + court) on a downloaded match,
segment per GT stroke, classify with BST, and score top-1 vs the labels.

Strokes start deep into the match (~frame 11.7k) and the file is 80 min, so we extract a
frame-accurate play segment with ffmpeg and validate on a bounded sample of strokes.

    uv run python -m scripts.validate_bst_shuttleset \
        --video data/shuttleset_videos/1_Kento_MOMOTA_..._Finals.f136.mp4 \
        --max-strokes 120 --max-span 6000
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import imageio_ffmpeg

import badminton_coach.events  # noqa: F401
import badminton_coach.perception  # noqa: F401
from badminton_coach.core.config import load_config
from badminton_coach.core.pipeline import Phase1Pipeline
from badminton_coach.core.registry import build
from badminton_coach.core.schemas import Point2D
from badminton_coach.core.schemas.events import HitEvent
from badminton_coach.events.shot_classification.bst import _TYPE_MAP

_SET_DIR = Path("third_party/BST/ShuttleSet/set")
_TYPES17 = ['放小球', '擋小球', '殺球', '點扣', '挑球', '防守回挑', '長球', '平球',
            '後場抽平球', '切球', '過渡切球', '推球', '撲球', '防守回抽', '勾球',
            '發短球', '發長球']
_T2I = {t: i for i, t in enumerate(_TYPES17)}


def _match_name(video: Path) -> str:
    return re.sub(r"^\d+_", "", video.stem).split(".f")[0]


def _strokes(match: str):
    """All (frame_num, type_idx, coarse_label) strokes for a match, sorted by frame."""
    out = []
    for csv_path in sorted((_SET_DIR / match).glob("set*.csv")):
        for row in csv.DictReader(csv_path.open()):
            t = (row.get("type") or "").strip()
            fn = row.get("frame_num")
            if t in _T2I and fn:
                out.append((int(float(fn)), _T2I[t], _TYPE_MAP[_T2I[t]].value))
    return sorted(out)


def _extract_segment(video: Path, f0: int, f1: int) -> Path:
    """Frame-accurate [f0, f1] segment via ffmpeg select filter; frames re-index from 0."""
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    out = Path(tempfile.mkdtemp()) / "seg.mp4"
    cmd = [ff, "-y", "-i", str(video), "-vf", f"select=between(n\\,{f0}\\,{f1})",
           "-vsync", "0", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(out)]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--max-strokes", type=int, default=120)
    ap.add_argument("--max-span", type=int, default=6000, help="max frame span to extract")
    ap.add_argument("--prior-correction", action="store_true")
    ap.add_argument("--det-threshold", type=float, default=0.3,
                    help="player detector confidence (lower = better coverage)")
    args = ap.parse_args()

    video = Path(args.video)
    match = _match_name(video)
    strokes = _strokes(match)
    if not strokes:
        raise SystemExit(f"no annotations for {match} under {_SET_DIR}")

    f0 = strokes[0][0]
    sample = [s for s in strokes if s[0] - f0 <= args.max_span][: args.max_strokes]
    f1 = sample[-1][0] + 60
    print(f"match: {match}\nstrokes total {len(strokes)}, sampling {len(sample)} "
          f"in frames [{f0}, {f1}] ({(f1 - f0) / 25 / 60:.1f} min)")

    seg = _extract_segment(video, f0, f1)
    cfg = load_config("configs/singles.yaml")
    det = cfg.perception.detector.model_copy(update={"threshold": args.det_threshold})
    cfg = cfg.model_copy(update={"perception": cfg.perception.model_copy(update={"detector": det})})
    result = Phase1Pipeline.from_config(cfg).run(str(seg))
    court_str = f"OK {result.court.reprojection_error_px:.1f}px" if result.court else "NONE"
    print(f"court: {court_str} | poses {len(result.poses)} | shuttle {len(result.shuttle_2d)}")
    if result.court is None:
        raise SystemExit("court calibration failed on segment -> BST needs court; abort")

    hits = [HitEvent(frame_index=fn - f0, shuttle_image_pos=Point2D(0.0, 0.0))
            for fn, _, _ in sample]
    gt_type = [ti for _, ti, _ in sample]       # fine 17-class type index
    gt_coarse = [lab for _, _, lab in sample]   # coarse ShotType value

    clf = build("shot_classifier", {"backend": "bst", "prior_correction": args.prior_correction,
                                    "device": "cuda"})
    shots = clf.classify(hits, result)
    raw_cls = clf.last_cls  # 35-class argmax per stroke (None if unknown / <3 frames)

    n = min(len(shots), len(gt_type))
    decided = [j for j in range(n) if raw_cls[j] is not None and raw_cls[j] < 34]
    coarse_ok = sum(shots[j].shot_type.value == gt_coarse[j] for j in decided)
    fine_ok = sum(raw_cls[j] % 17 == gt_type[j] for j in decided)  # 17-type, side-agnostic
    print(f"\nprior_correction={args.prior_correction}  det_threshold={args.det_threshold}")
    print(f"strokes {n} | decided {len(decided)} | unknown {n - len(decided)} "
          f"({(n - len(decided)) / n:.0%})")
    if decided:
        nd = len(decided)
        print(f"top-1 coarse (7-class) = {coarse_ok}/{nd} = {coarse_ok / nd:.1%}")
        print(f"top-1 fine  (17-type)  = {fine_ok}/{nd} = {fine_ok / nd:.1%}")
    print(f"GT   coarse mix: {dict(Counter(gt_coarse[:n]))}")
    print(f"pred coarse mix: {dict(Counter(s.shot_type.value for s in shots[:n]))}")


if __name__ == "__main__":
    main()
