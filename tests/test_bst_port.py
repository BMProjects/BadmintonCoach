"""Validate our BST input pipeline against the upstream ShuttleSet source.

The full BST accuracy can't be reproduced locally (the bundled ShuttleSet has labels +
homography but no raw videos / poses, which the model needs). What we CAN validate is
that our preprocessing reproduces the upstream normalization exactly — so any residual
prediction bias is a domain gap (pose estimator / footage), not a port bug.

We extract the pure numpy functions `normalize_joints` / `normalize_shuttlecock` directly
from the upstream source via AST (its module top-imports mmpose, so we can't import it)
and compare to our implementation.
"""

from __future__ import annotations

import ast

import numpy as np
import pytest

from badminton_coach.perception._util import THIRD_PARTY

_UP = (THIRD_PARTY / "BST" / "stroke_classification" / "preparing_data"
       / "prepare_train_on_shuttleset.py")


def _load_upstream(name: str):
    if not _UP.exists():
        pytest.skip("BST submodule not present")
    tree = ast.parse(_UP.read_text())
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)
    if fn is None:
        pytest.skip(f"{name} not found in upstream")
    ns: dict = {"np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<upstream>", "exec"), ns)
    return ns[name]


def test_norm_joints_matches_upstream():
    from badminton_coach.core.schemas import BBox, Keypoint, Point2D, Pose
    from badminton_coach.events.shot_classification.bst import BSTShotClassifier

    rng = np.random.default_rng(0)
    box = BBox(100.0, 200.0, 180.0, 360.0)  # x1,y1,x2,y2
    coords = rng.uniform([110, 210], [170, 350], size=(17, 2))  # all inside, nonzero
    pose = Pose(frame_index=0,
                keypoints=[Keypoint(point=Point2D(float(x), float(y)), confidence=0.9)
                           for x, y in coords])

    mine = BSTShotClassifier({})._norm_joints(pose, box)

    normalize_joints = _load_upstream("normalize_joints")
    up = normalize_joints(coords[None].astype(float),
                          np.array([[box.x1, box.y1, box.x2, box.y2]], dtype=float),
                          center_align=False)[0]
    assert np.allclose(mine, up, atol=1e-5)


def test_shuttle_norm_matches_upstream_formula():
    # Upstream normalize_shuttlecock divides raw 2D image coords by (v_width, v_height);
    # our bst.py does sp.x/vw, sp.y/vh. Verify against the extracted source.
    normalize_shuttlecock = _load_upstream("normalize_shuttlecock")
    arr = np.array([[960.0, 540.0], [480.0, 270.0]])
    up = normalize_shuttlecock(arr, 1920.0, 1080.0)
    mine = np.stack([arr[:, 0] / 1920.0, arr[:, 1] / 1080.0], axis=-1)
    assert np.allclose(mine, up, atol=1e-9)
