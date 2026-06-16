"""Train a line-heatmap model and evaluate via line-intersection keypoints.

Eval decodes each predicted named-line channel to a line, intersects each keypoint's
horizontal+vertical lines -> derived keypoint, and scores PCK / median px against GT
on the (perspective-warped) viewpoint test set.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import CourtLineDataset
from .decode import decode_lines
from .lines import N_KPTS
from .model import build_model, count_params


def _decode_keypoints(line_hm: np.ndarray, iw: int, ih: int,
                      thr: float = 0.3, decode: str = "weighted"):
    return decode_lines(line_hm, iw, ih, thr=thr, decode=decode)[0]


@torch.no_grad()
def evaluate(model, loader, device, input_size, decode="weighted"):
    model.eval()
    ih, iw = input_size
    diag = float(np.hypot(iw, ih))
    errs, cov, tot = [], 0, 0
    amp = device == "cuda"
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(img).float()
        hm = out.cpu().numpy()
        gt = batch["kp_input"].numpy()
        vis = batch["vis"].numpy()
        for b in range(len(img)):
            der = _decode_keypoints(hm[b], iw, ih, decode=decode)
            for k in range(N_KPTS):
                if vis[b, k] <= 0:
                    continue
                tot += 1
                if k in der:
                    cov += 1
                    errs.append(float(np.hypot(der[k][0] - gt[b, k, 0], der[k][1] - gt[b, k, 1])))
    errs = np.array(errs) if errs else np.array([1e9])
    return {
        "coverage": cov / tot if tot else 0.0,
        "median_px": float(np.median(errs)),
        "mean_px": float(np.mean(errs)),
        "PCK@0.02": float((errs < 0.02 * diag).mean()),
        "PCK@0.05": float((errs < 0.05 * diag).mean()),
    }


def train_and_eval(backbone, data="data/court_kp_official", val_split="valid",
                   epochs=30, batch=16, lr=1e-3, input=384, sigma=2.5,
                   workers=8, device="cuda", out_div=4, hard=False,
                   decode="weighted", aa=False, flip180=0.0, save=None):
    torch.backends.cudnn.benchmark = True
    isz = (input, input)
    hsz = (input // out_div, input // out_div)
    dev = device if torch.cuda.is_available() else "cpu"
    tr = CourtLineDataset(data, "train", isz, hsz, sigma, augment=True,
                          perspective=True, hard=hard, aa=aa, flip180=flip180)
    va = CourtLineDataset(data, val_split, isz, hsz, sigma, augment=True,
                          perspective=True, hard=hard, aa=aa, flip180=flip180)
    tl = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=workers,
                    pin_memory=True, persistent_workers=workers > 0, drop_last=True)
    vl = DataLoader(va, batch_size=batch, num_workers=workers, pin_memory=True,
                    persistent_workers=workers > 0)
    from .lines import N_LINES
    model = build_model(backbone, N_LINES, pretrained=True,
                        input_size=input, out_div=out_div).to(dev)
    params = count_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    amp = dev == "cuda"
    best = None
    for ep in range(epochs):
        model.train()
        losses = []
        for batch_ in tl:
            img = batch_["image"].to(dev, non_blocking=True)
            tgt = batch_["lines"].to(dev, non_blocking=True)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = ((model(img) - tgt) ** 2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sch.step()
        if ep == epochs - 1 or (ep + 1) % 10 == 0:
            m = evaluate(model, vl, dev, isz, decode=decode)
            print(f"  [{backbone}] ep{ep+1:02d} loss {np.mean(losses):.5f} | "
                  f"PCK@0.05 {m['PCK@0.05']:.3f} med {m['median_px']:.1f}px "
                  f"cov {m['coverage']:.2f}")
            best = m
    best["params_M"] = round(params, 1)
    if save:
        from pathlib import Path
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "backbone": backbone,
            "n_lines": N_LINES,
            "input_size": input,
            "out_div": out_div,
            "decode": decode,
            "metrics": best,
        }, save)
        print(f"  saved -> {save}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="convnextv2_nano")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--input", type=int, default=384)
    ap.add_argument("--out-div", type=int, default=2)
    ap.add_argument("--hard", action="store_true")
    ap.add_argument("--aa", action="store_true")
    ap.add_argument("--decode", default="robust")
    ap.add_argument("--save")
    args = ap.parse_args()
    m = train_and_eval(args.backbone, epochs=args.epochs, batch=args.batch,
                       input=args.input, out_div=args.out_div, hard=args.hard,
                       aa=args.aa, decode=args.decode, save=args.save)
    print(args.backbone, m)


if __name__ == "__main__":
    main()
