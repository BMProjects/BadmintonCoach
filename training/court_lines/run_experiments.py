"""Sweep backbones for the line-heatmap + intersection experiment, print a table.

Each backbone is trained from ImageNet-pretrained weights on the viewpoint-warped
court data and scored by intersection-decoded keypoint PCK on the warped val set.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from .train import train_and_eval

BACKBONES = ["convnextv2_nano", "efficientvit_b1", "repvit_m1", "hrnet_w18"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=BACKBONES)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out-div", type=int, default=4)
    ap.add_argument("--hard", action="store_true")
    ap.add_argument("--aa", action="store_true")
    ap.add_argument("--flip180", type=float, default=0.0)
    ap.add_argument("--decode", default="weighted")
    ap.add_argument("--out", default="training/court_lines/results.json")
    args = ap.parse_args()

    results = {}
    for bb in args.backbones:
        print(f"\n=== {bb} ===")
        t0 = time.time()
        try:
            m = train_and_eval(bb, epochs=args.epochs, batch=args.batch,
                               out_div=args.out_div, hard=args.hard, decode=args.decode,
                               aa=args.aa, flip180=args.flip180)
            m["minutes"] = round((time.time() - t0) / 60, 1)
            results[bb] = m
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {e}")
            results[bb] = {"error": str(e)}
        gc.collect()
        torch.cuda.empty_cache()
        Path(args.out).write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 78)
    cols = ["params_M", "PCK@0.05", "PCK@0.02", "med_px", "cov", "min"]
    print(f"{'backbone':<18}" + "".join(f"{c:>10}" for c in cols))
    print("-" * 78)
    for bb, m in results.items():
        if "error" in m:
            print(f"{bb:<18}  ERROR: {m['error'][:50]}")
            continue
        vals = [m["params_M"], m["PCK@0.05"], m["PCK@0.02"],
                m["median_px"], m["coverage"], m["minutes"]]
        print(f"{bb:<18}" + "".join(f"{v:>10.2f}" for v in vals))


if __name__ == "__main__":
    main()
