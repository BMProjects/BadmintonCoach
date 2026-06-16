"""BST (Badminton Stroke-type Transformer) shot classifier — SOTA, real wiring.

Vendored at third_party/BST (Va6lue/BST-...). We construct the upstream BST model
(in_dim=72, seq_len=30, 35 classes, JnB_bone), load the pretrained ShuttleSet
weights, and feed per-shot windows built EXACTLY as the upstream data pipeline:
  - joints: 2 players x COCO-17, normalized by each player's bbox (TemPose style);
  - pos: each player's court-ground position, normalized by court borders -> [0,1];
  - shuttle: 2D shuttle projected to the court ground, normalized -> [0,1];
  - bones from COCO pairs concatenated to joints -> 36 points x 2 = in_dim 72;
  - seq_len 30 via the upstream make_seq_len_same (stride subsample + pad).
Output class (Top_/Bottom_ + 17 stroke types) is mapped to our coarse ShotType.

Needs the submodule + weights/bst/bst_0_JnB_bone.pt (see docs/TRAINING.md).
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from ...core.geometry.court_model import COURT_LENGTH_M, COURT_WIDTH_DOUBLES_M
from ...core.interfaces import ShotClassifier
from ...core.registry import register
from ...core.schemas import BBox, PerceptionResult, ShotType
from ...core.schemas.events import HitEvent, Shot
from ...perception._util import THIRD_PARTY, module_available, submodule_available

_BST_ROOT = THIRD_PARTY / "BST" / "stroke_classification"
_SEQ_LEN = 30

# 17 upstream stroke types (order per get_stroke_types) -> coarse ShotType.
_TYPE_MAP = [
    ShotType.NET, ShotType.NET, ShotType.SMASH, ShotType.SMASH, ShotType.LIFT, ShotType.LIFT,
    ShotType.CLEAR, ShotType.DRIVE, ShotType.DRIVE, ShotType.DROP, ShotType.DROP, ShotType.DRIVE,
    ShotType.NET, ShotType.DRIVE, ShotType.NET, ShotType.SERVE, ShotType.SERVE,
]

# ShuttleSet 17-class training prior (frequency over 35,008 bundled stroke labels, in
# _TYPE_MAP order). 'net' types dominate (~34% combined), so the released weights lean to
# net on ambiguous/out-of-distribution input. Optional prior correction (config
# prior_correction=True) divides the posterior by this prior to counter that lean.
_CLASS_PRIOR = [
    0.1797, 0.1034, 0.0739, 0.0471, 0.1523, 0.0086, 0.0835, 0.0200, 0.0135,
    0.0612, 0.0387, 0.0836, 0.0146, 0.0116, 0.0392, 0.0586, 0.0107,
]


def _make_zeropos(out_dim: int):
    """nn.Module returning zeros of shape (*pos.shape[:-1], out_dim) — disables the
    position-fusion branch so the model matches the released (no-pos) weights."""
    import torch
    from torch import nn

    class _ZeroPos(nn.Module):
        def forward(self, pos):
            return torch.zeros(*pos.shape[:-1], out_dim, device=pos.device, dtype=pos.dtype)

    return _ZeroPos()


def _pose_bbox(pose) -> BBox:
    xs = [k.point.x for k in pose.keypoints if k.confidence > 0]
    ys = [k.point.y for k in pose.keypoints if k.confidence > 0]
    if not xs:
        return BBox(0, 0, 1, 1)
    return BBox(min(xs), min(ys), max(xs), max(ys))


@register("shot_classifier", "bst")
class BSTShotClassifier(ShotClassifier):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._model = None
        self._device = self.config.get("device", "cuda")
        # Divide the posterior by the ShuttleSet class prior to counter the net-lean
        # (heuristic de-bias; assumes a ~uniform test prior). Off by default.
        self._prior_correction = bool(self.config.get("prior_correction", False))
        self.last_cls: list[int | None] = []

    @classmethod
    def is_available(cls) -> bool:
        return module_available("torch") and submodule_available("BST")

    def _ensure(self):
        if self._model is not None:
            return
        import torch

        # BST and TrackNetV3 both expose top-level 'model'/'utils' packages; isolate
        # BST's import then restore so the shuttle tracker keeps working.
        collide = ("model", "utils", "test", "dataset", "predict")
        saved = {n: sys.modules.pop(n) for n in list(sys.modules) if n.split(".")[0] in collide}
        sys.path.insert(0, str(_BST_ROOT))
        try:
            from model.bst import BST
            from preparing_data.shuttleset_dataset import (
                create_bones,
                get_bone_pairs,
                make_seq_len_same,
            )
            self._create_bones = create_bones
            self._get_pairs = get_bone_pairs
            self._make_seq = make_seq_len_same
            model = BST(in_dim=72, seq_len=_SEQ_LEN, n_class=35, depth_tem=2, depth_inter=1)
            weights = self.config.get("weights", "weights/bst/bst_0_JnB_bone.pt")
            sd = torch.load(weights, map_location="cpu", weights_only=True)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            # Released weights predate the 'mlp_positions' (pose-position fusion) branch.
            # Replace it with a zero module so forward's `JnB*pos_impact + JnB == JnB`,
            # i.e. exactly the no-position-fusion model these weights were trained as.
            assert set(missing) == {
                "mlp_positions.mlp.0.weight", "mlp_positions.mlp.0.bias",
                "mlp_positions.mlp.3.weight", "mlp_positions.mlp.3.bias",
            } and not unexpected, f"unexpected BST arch mismatch: miss={missing} unexp={unexpected}"
            model.mlp_positions = _make_zeropos(72)
        finally:
            sys.path.remove(str(_BST_ROOT))
            for n in list(sys.modules):
                if n.split(".")[0] in collide:
                    sys.modules.pop(n, None)
            sys.modules.update(saved)
        if str(self._device).startswith("cuda") and not torch.cuda.is_available():
            self._device = "cpu"
        self._model = model.to(self._device).eval()

    @staticmethod
    def _video_size(perception: PerceptionResult) -> tuple[float, float]:
        """Video (width, height) for shuttle normalization: from the camera principal
        point (image centre x2) when a camera is solved, else inferred from the largest
        observed image coordinate (poses + shuttle)."""
        court = perception.court
        if court is not None and court.camera is not None:
            k = court.camera.intrinsic
            return float(2 * k[0, 2]), float(2 * k[1, 2])
        xs = [kp.point.x for p in perception.poses for kp in p.keypoints]
        ys = [kp.point.y for p in perception.poses for kp in p.keypoints]
        xs += [p.point.x for p in perception.shuttle_2d.points]
        ys += [p.point.y for p in perception.shuttle_2d.points]
        return (max(xs) if xs else 1920.0), (max(ys) if ys else 1080.0)

    def classify(self, hits: list[HitEvent], perception: PerceptionResult) -> list[Shot]:
        if perception.court is None:  # BST needs court-normalized coords
            return [Shot(h.frame_index, h.frame_index, h.hitter_track_id, ShotType.UNKNOWN, 0.0)
                    for h in hits]
        self._ensure()
        vw, vh = self._video_size(perception)
        tracks_by_frame = self._players_by_frame(perception)
        poses_by_frame = self._poses_by_frame(perception)
        shuttle_by_frame = {p.frame_index: p for p in perception.shuttle_2d.points}
        ctx = (tracks_by_frame, poses_by_frame, shuttle_by_frame, vw, vh)

        # Upstream segments each stroke as a window CENTRED on the hit frame: from the
        # previous hit (or fn-0.5s) to the next hit (+0.25s), clamped to +/-1.5s. This
        # captures the backswing -> swing -> follow-through that defines the stroke, NOT
        # the post-hit flight. One Shot per hit (= per stroke).
        fps = perception.fps or 25.0
        t, limit = max(1, round(fps * 0.5)), max(1, round(fps * 1.5))
        eps = t // 2
        shots: list[Shot] = []
        self.last_cls = []  # raw 35-class argmax per hit (None if unknown) — for fine eval
        for i, h in enumerate(hits):
            fn = h.frame_index
            prev = hits[i - 1].frame_index if i > 0 else fn - t
            nxt = hits[i + 1].frame_index if i < len(hits) - 1 else fn + t
            start = max(prev, fn - limit)
            end = min(nxt + eps, fn + limit + eps)
            stype, conf, cls = self._classify_window(start, end, ctx)
            self.last_cls.append(cls)
            shots.append(Shot(fn, int(nxt if i < len(hits) - 1 else end),
                              h.hitter_track_id, stype, conf))
        return shots

    def _classify_window(self, start: int, end: int, ctx) -> tuple[ShotType, float, int | None]:
        tracks_by_frame, poses_by_frame, shuttle_by_frame, vw, vh = ctx
        wnorm, lnorm = COURT_WIDTH_DOUBLES_M, COURT_LENGTH_M
        joints, pos, shuttle = [], [], []
        for f in range(start, end + 1):
            players = tracks_by_frame.get(f)
            if not players or len(players) < 2:
                continue
            fposes = poses_by_frame.get(f, {})
            jt, ps = [], []
            for (box, foot_world) in players:
                pose = fposes.get(id(box))
                jt.append(self._norm_joints(pose, box))
                ps.append([foot_world[0] / wnorm, foot_world[1] / lnorm])
            joints.append(jt)
            pos.append(ps)
            sp = shuttle_by_frame.get(f)
            shuttle.append([sp.point.x / vw, sp.point.y / vh] if sp and sp.visible else [0.0, 0.0])
        if len(joints) < 3:
            return ShotType.UNKNOWN, 0.0, None
        return self._infer(np.array(joints, np.float32), np.array(pos, np.float32),
                           np.array(shuttle, np.float32))

    def _infer(self, joints, pos, shuttle):
        import torch

        joints, pos, shuttle, vlen = self._make_seq(_SEQ_LEN, joints, pos, shuttle)
        bones = self._create_bones(joints, self._get_pairs("coco"))
        human_pose = np.concatenate((joints, bones), axis=-2)         # (t,2,36,2)
        hp = torch.from_numpy(human_pose).float().reshape(1, _SEQ_LEN, 2, -1)
        sh = torch.from_numpy(shuttle).float().unsqueeze(0)            # (1,t,2)
        ps = torch.from_numpy(pos).float().unsqueeze(0)               # (1,t,2,2)
        vl = torch.tensor([vlen])
        d = self._device
        with torch.no_grad():
            logits = self._model(hp.to(d), sh.to(d), ps.to(d), vl.to(d))
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        if self._prior_correction:
            prior = np.array(_CLASS_PRIOR * 2 + [float(np.mean(_CLASS_PRIOR))])  # 35 classes
            probs = probs / prior
            probs = probs / probs.sum()
        cls = int(probs.argmax())
        conf = float(probs[cls])
        if cls >= 34:
            return ShotType.UNKNOWN, conf, cls
        return _TYPE_MAP[cls % 17], conf, cls

    def _norm_joints(self, pose, box: BBox) -> np.ndarray:
        diag = float(np.hypot(box.width, box.height)) or 1.0
        out = np.zeros((17, 2), np.float32)
        if pose is None:
            return out
        for i, k in enumerate(pose.keypoints):
            if k.confidence > 0:
                out[i, 0] = (k.point.x - box.x1) / diag
                out[i, 1] = (k.point.y - box.y1) / diag
        return out

    def _players_by_frame(self, perception):
        out: dict[int, list] = {}
        for tr in perception.player_tracks:
            for tb in tr.boxes:
                g = perception.court.image_to_ground(tb.bbox.foot)
                out.setdefault(tb.frame_index, []).append((tb.bbox, (g.x, g.y)))
        # order Top (far, smaller image-y foot) first to match BST Top/Bottom
        for f in out:
            out[f].sort(key=lambda bf: bf[0].foot.y)
        return out

    def _poses_by_frame(self, perception):
        """Associate each pose to the player bbox it overlaps (by foot proximity)."""
        out: dict[int, dict] = {}
        tbf = {}
        for tr in perception.player_tracks:
            for tb in tr.boxes:
                tbf.setdefault(tb.frame_index, []).append(tb.bbox)
        for pose in perception.poses:
            boxes = tbf.get(pose.frame_index, [])
            if not boxes:
                continue
            pb = _pose_bbox(pose).foot
            best = min(boxes, key=lambda bx: abs(bx.foot.x - pb.x) + abs(bx.foot.y - pb.y))
            out.setdefault(pose.frame_index, {})[id(best)] = pose
        return out
