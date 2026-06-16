"""Generate a synthetic broadcast-style badminton clip for the dev console.

Renders a perspective BWF court + two moving players + a multi-shot shuttle rally,
using a known camera so the court corners and the shuttle's 2D track are exact.
Writes:
  assets/sample_rally.mp4
  assets/sample_rally.gt.json   {image_corners, focal_px, shuttle2d:[[frame,x,y,vis]...]}

This lets the demo exercise the *weight-free* parts of the pipeline end-to-end:
two-stage court calibration (+ camera/3D), shuttle 2D (via the replay tracker),
MonoTrack 3D reconstruction, and all overlays. Detection/pose need real footage +
weights and are not exercised by this synthetic clip.

Run:  python -m scripts.make_sample
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from badminton_coach.core.geometry import physics
from badminton_coach.core.geometry.court_model import (
    COURT_LENGTH_M,
    COURT_WIDTH_DOUBLES_M,
    COURT_WIDTH_SINGLES_M,
    SHORT_SERVICE_FROM_NET_M,
    court_corners_doubles,
)

W, H, FPS = 1280, 720, 30
FOCAL = 1400.0
ASSETS = Path(__file__).resolve().parents[1] / "assets"


def camera_matrix() -> np.ndarray:
    k = np.array([[FOCAL, 0, W / 2], [0, FOCAL, H / 2], [0, 0, 1.0]])
    rvec = np.array([[1.1], [0.0], [0.0]])
    tvec = np.array([[-COURT_WIDTH_DOUBLES_M / 2], [-3.0], [22.0]])
    rot, _ = cv2.Rodrigues(rvec)
    return k @ np.hstack([rot, tvec])


def project(p: np.ndarray, world: np.ndarray) -> np.ndarray:
    h = np.hstack([world, np.ones((len(world), 1))])
    q = (p @ h.T).T
    return q[:, :2] / q[:, 2:3]


def court_segments() -> list[tuple[np.ndarray, np.ndarray]]:
    w, ws, length = COURT_WIDTH_DOUBLES_M, COURT_WIDTH_SINGLES_M, COURT_LENGTH_M
    sx = (w - ws) / 2
    net, sl = length / 2, SHORT_SERVICE_FROM_NET_M
    segs = [
        ((0, 0), (w, 0)), ((0, length), (w, length)),            # baselines
        ((0, 0), (0, length)), ((w, 0), (w, length)),            # doubles sidelines
        ((sx, 0), (sx, length)), ((w - sx, 0), (w - sx, length)),  # singles sidelines
        ((0, net), (w, net)),                                    # net
        ((0, net - sl), (w, net - sl)), ((0, net + sl), (w, net + sl)),  # service lines
    ]
    return [(np.array([a[0], a[1], 0.0]), np.array([b[0], b[1], 0.0])) for a, b in segs]


def make_rally(p_mat: np.ndarray):
    """Three chained shots; returns per-frame (image, shuttle2d-or-None)."""
    shots = [
        (np.array([3.0, 2.0, 2.0]), np.array([0.2, 9.0, 6.0]), 24),
        (np.array([3.2, 11.0, 2.2]), np.array([-0.2, -9.5, 6.2]), 24),
        (np.array([2.8, 2.5, 2.0]), np.array([0.1, 9.2, 5.8]), 24),
    ]
    segs = court_segments()
    seg_px = [(project(p_mat, a[None])[0], project(p_mat, b[None])[0]) for a, b in segs]

    frames, shuttle2d, fidx = [], [], 0
    for si, (p0, v0, n) in enumerate(shots):
        traj = physics.simulate(p0, v0, dt=1.0 / FPS, steps=n - 1)
        proj = project(p_mat, traj)
        for i in range(n):
            img = np.full((H, W, 3), (40, 110, 40), np.uint8)  # court green
            for a_px, b_px in seg_px:
                cv2.line(img, tuple(a_px.astype(int)), tuple(b_px.astype(int)), (240, 240, 240), 2)
            _draw_players(img, p_mat, fidx)
            sx, sy = float(proj[i, 0]), float(proj[i, 1])
            cv2.circle(img, (int(sx), int(sy)), 5, (255, 255, 255), -1)
            cv2.circle(img, (int(sx), int(sy)), 6, (0, 0, 0), 1)
            frames.append(img)
            shuttle2d.append([fidx, sx, sy, True])
            fidx += 1
        if si < len(shots) - 1:  # 1 invisible 'hit' frame between shots -> segment break
            img = np.full((H, W, 3), (40, 110, 40), np.uint8)
            for a_px, b_px in seg_px:
                cv2.line(img, tuple(a_px.astype(int)), tuple(b_px.astype(int)), (240, 240, 240), 2)
            _draw_players(img, p_mat, fidx)
            frames.append(img)
            shuttle2d.append([fidx, 0.0, 0.0, False])
            fidx += 1
    return frames, shuttle2d


def _draw_players(img: np.ndarray, p_mat: np.ndarray, fidx: int) -> None:
    phase = fidx * 0.15
    near = np.array([[COURT_WIDTH_DOUBLES_M / 2 + 1.5 * np.sin(phase), 3.0, 0.0]])
    far = np.array([[COURT_WIDTH_DOUBLES_M / 2 - 1.5 * np.sin(phase + 1), 10.5, 0.0]])
    for ground, color, scale in [(near, (60, 60, 220), 1.0), (far, (220, 120, 60), 0.7)]:
        c = project(p_mat, ground)[0].astype(int)
        cv2.ellipse(img, tuple(c), (int(22 * scale), int(60 * scale)), 0, 0, 360, color, -1)


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    p_mat = camera_matrix()
    frames, shuttle2d = make_rally(p_mat)

    video_path = ASSETS / "sample_rally.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for img in frames:
        writer.write(img)
    writer.release()

    world_corners = np.array([[x, y, 0.0] for x, y in court_corners_doubles()])
    image_corners = project(p_mat, world_corners).tolist()
    gt = {"image_corners": image_corners, "focal_px": FOCAL, "shuttle2d": shuttle2d}
    (ASSETS / "sample_rally.gt.json").write_text(json.dumps(gt, indent=2), encoding="utf-8")

    print(f"wrote {video_path} ({len(frames)} frames) + sample_rally.gt.json")
    print(f"image_corners (near-L, near-R, far-R, far-L): {np.round(image_corners,1).tolist()}")


if __name__ == "__main__":
    main()
