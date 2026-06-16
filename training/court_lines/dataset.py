"""Dataset producing named-line heatmaps from the 22-keypoint annotations.

Reuses the court_kp image cache / affine / perspective augmentation. For each image
we transform the 22 keypoints, render each named line (a thin Gaussian stroke through
its member points), and also return GT keypoints for evaluation. Perspective warp is
ON by default here — this is the 'viewpoint-transformed' dataset.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .lines import LINES, N_KPTS, SYM_PERM, line_endpoints

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_SHIFT = 4
_SF = 1 << _SHIFT


class CourtLineDataset(Dataset):
    def __init__(self, root, split, input_size=(384, 384), heatmap_size=(96, 96),
                 sigma=2.5, augment=False, perspective=False, cache=True, hard=False,
                 aa=False, flip180=0.0):
        self.dir = Path(root) / split
        coco = json.loads((self.dir / "_annotations.coco.json").read_text())
        cat = next(c for c in coco["categories"] if c.get("keypoints"))
        imgs = {im["id"]: im for im in coco["images"]}
        self.samples = [
            (str(self.dir / imgs[a["image_id"]]["file_name"]), a["keypoints"])
            for a in coco["annotations"]
            if a.get("category_id") == cat["id"] and a.get("keypoints")
        ]
        self.input_size, self.heatmap_size = input_size, heatmap_size
        self.sigma, self.augment, self.perspective = sigma, augment, perspective
        self.hard, self.aa, self.flip180 = hard, aa, flip180
        self._cache = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p, _ in self.samples] \
            if cache else None

    def __len__(self):
        return len(self.samples)

    def _warp(self, img, xy):
        h, w = img.shape[:2]
        rot, scale, trans, pj = (12, (0.7, 1.3), 0.10, 0.22) if self.hard \
            else (7, (0.85, 1.15), 0.06, 0.13)
        m = cv2.getRotationMatrix2D((w / 2, h / 2), np.random.uniform(-rot, rot),
                                    np.random.uniform(*scale))
        m[0, 2] += np.random.uniform(-trans, trans) * w
        m[1, 2] += np.random.uniform(-trans, trans) * h
        img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        xy = np.hstack([xy, np.ones((len(xy), 1))]) @ m.T
        if self.perspective:
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            jit = np.random.uniform(-1, 1, src.shape).astype(np.float32)
            dst = src + jit * (pj * np.array([w, h], np.float32))
            hmat = cv2.getPerspectiveTransform(src, dst)
            img = cv2.warpPerspective(img, hmat, (w, h), borderMode=cv2.BORDER_REPLICATE)
            p = np.hstack([xy, np.ones((len(xy), 1))]) @ hmat.T
            xy = p[:, :2] / p[:, 2:3]
        return img, xy

    def __getitem__(self, idx):
        path, kp = self.samples[idx]
        img = self._cache[idx] if self._cache is not None \
            else cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]
        kp = np.array(kp, np.float32).reshape(-1, 3)
        xy, vis = kp[:, :2].copy(), kp[:, 2].copy()
        if self.flip180 and np.random.rand() < self.flip180:
            img = cv2.rotate(img, cv2.ROTATE_180)
            xy[:, 0] = (w0 - 1) - xy[:, 0]
            xy[:, 1] = (h0 - 1) - xy[:, 1]
            xy, vis = xy[SYM_PERM], vis[SYM_PERM]
        if self.augment:
            img, xy = self._warp(img, xy)
            if np.random.rand() < 0.5:
                g = np.random.uniform(0.7, 1.3)
                img = np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8)

        ih, iw = self.input_size
        img_r = cv2.resize(img, (iw, ih))
        xy[:, 0] *= iw / w0
        xy[:, 1] *= ih / h0

        hh, hw = self.heatmap_size
        sx, sy = hw / iw, hh / ih
        heat = np.zeros((len(LINES), hh, hw), np.float32)
        for ci, grp in enumerate(LINES):
            pts = np.array([[xy[k, 0] * sx, xy[k, 1] * sy] for k in grp if vis[k] > 0], np.float32)
            if len(pts) < 2:
                continue
            ends = line_endpoints(pts)
            if ends is None:
                continue
            a, b = ends
            if any(math.isnan(v) for v in (*a, *b)):
                continue
            canvas = np.zeros((hh, hw), np.float32)
            if self.aa:
                pa = (int(round(a[0] * _SF)), int(round(a[1] * _SF)))
                pb = (int(round(b[0] * _SF)), int(round(b[1] * _SF)))
                cv2.line(canvas, pa, pb, 1.0, 1, cv2.LINE_AA, _SHIFT)
            else:
                cv2.line(canvas, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 1.0, 1)
            heat[ci] = cv2.GaussianBlur(canvas, (0, 0), self.sigma)
            if heat[ci].max() > 0:
                heat[ci] /= heat[ci].max()

        img_t = (img_r.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return {
            "image": torch.from_numpy(img_t.transpose(2, 0, 1)),
            "lines": torch.from_numpy(heat),
            "kp_input": torch.from_numpy(xy.astype(np.float32)),  # GT kpts in input px
            "vis": torch.from_numpy(vis),
        }

    @staticmethod
    def num_kpts() -> int:
        return N_KPTS
