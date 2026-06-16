"""Line-heatmap model: timm backbone (features_only) + multi-scale fusion neck + head.

The neck projects every backbone scale to a common dim, upsamples all to input/4 and
sums them (lightweight FPN/U-Net-style fusion) — the high-res fusion that thin-line
localization needs. Backbone is swappable for the comparison experiment.
"""

from __future__ import annotations

import timm
import torch.nn.functional as F
from torch import nn


class FusionNeck(nn.Module):
    def __init__(self, in_chs: list[int], out_size: int, c: int = 128):
        super().__init__()
        self.lat = nn.ModuleList(nn.Conv2d(ic, c, 1) for ic in in_chs)
        self.fuse = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        self.out_size = out_size

    def forward(self, feats):
        s = None
        for lat, f in zip(self.lat, feats, strict=True):
            u = F.interpolate(lat(f), size=(self.out_size, self.out_size),
                              mode="bilinear", align_corners=False)
            s = u if s is None else s + u
        return self.fuse(s)


class LineHeatmapNet(nn.Module):
    def __init__(self, backbone: str, n_lines: int, pretrained: bool = True,
                 input_size: int = 384, out_div: int = 4):
        super().__init__()
        self.body = timm.create_model(backbone, pretrained=pretrained, features_only=True)
        in_chs = [i["num_chs"] for i in self.body.feature_info]
        self.neck = FusionNeck(in_chs, out_size=input_size // out_div)
        self.head = nn.Conv2d(128, n_lines, 1)

    def forward(self, x):
        return self.head(self.neck(self.body(x)))


def build_model(backbone: str, n_lines: int, pretrained: bool = True,
                input_size: int = 384, out_div: int = 4):
    return LineHeatmapNet(backbone, n_lines, pretrained, input_size, out_div)


def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6
