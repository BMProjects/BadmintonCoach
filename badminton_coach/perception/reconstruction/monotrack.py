"""MonoTrack 3D reconstructor (physics drag-model optimization).

This is a faithful port of MonoTrack's per-hit reconstruction method
(third_party/monotrack .../ai_badminton/rally_reconstructor.py :: reconstruct_one_hit)
onto our data contracts. For each contiguous visible run of the 2D track we fit the
launch state (position, velocity) and a drag coefficient so the forward-simulated
drag-ODE trajectory reprojects onto the observed pixels with least error.

Requires the calibration to carry a camera (CourtCalibration.camera) — the ground
homography alone cannot place an airborne shuttle. Proper per-hit segmentation
comes from L2 (HitNet/BST); here we segment on visibility gaps and reserve a
config['hits'] hook. fps is taken from config['fps'] (default 30).

Honest about uncertainty: reports the measured mean reprojection error (px).
Monocular 3D is ill-posed (MonoTrack end-to-end ~28-37 px).
"""

from __future__ import annotations

import numpy as np

from ...core.geometry import physics
from ...core.interfaces import Reconstructor3D
from ...core.registry import register
from ...core.schemas import (
    CourtCalibration,
    Point3D,
    ShuttlePoint3D,
    ShuttleTrajectory2D,
    ShuttleTrajectory3D,
)
from .._util import module_available

_GRAVITY = np.array([0.0, 0.0, -physics.GRAVITY_M_S2])
# Court bounds (meters) for the out-of-bounds penalty, from the BWF model.
_X_MAX, _Y_MAX, _Z_MAX = 6.10, 13.40, 6.0
_SUBSTEPS = 20


@register("reconstructor", "monotrack")
class MonoTrackReconstructor(Reconstructor3D):
    @classmethod
    def is_available(cls) -> bool:
        return module_available("scipy")

    def reconstruct(
        self, traj2d: ShuttleTrajectory2D, court: CourtCalibration
    ) -> ShuttleTrajectory3D:
        if court.camera is None:
            raise ValueError(
                "monotrack reconstructor needs CourtCalibration.camera (use the "
                "two_stage calibrator with compute_camera: true)."
            )
        fps = float(self.config.get("fps", 30.0))
        cam_proj = self._camera_projection(court)

        all_points: list[ShuttlePoint3D] = []
        residuals: list[float] = []
        for run in self._visible_runs(traj2d):
            pts3d, run_resid = self._fit_segment(run, court, cam_proj, fps)
            all_points.extend(pts3d)
            residuals.extend(run_resid)

        error = float(np.mean(residuals)) if residuals else float("inf")
        return ShuttleTrajectory3D(
            points=tuple(all_points), reprojection_error_px=error, method="monotrack"
        )

    @staticmethod
    def _camera_projection(court: CourtCalibration) -> np.ndarray:
        cam = court.camera
        rt = np.hstack([cam.rotation, cam.translation.reshape(3, 1)])  # 3x4
        return cam.intrinsic @ rt

    @staticmethod
    def _visible_runs(traj2d: ShuttleTrajectory2D) -> list[list]:
        runs, cur = [], []
        for p in traj2d.points:
            if p.visible:
                cur.append(p)
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        return [r for r in runs if len(r) >= 3]

    def _fit_segment(self, run, court, cam_proj, fps):
        import scipy.optimize

        observed = np.array([[p.point.x, p.point.y] for p in run])
        n = len(run)
        s_ground = court.image_to_ground(run[0].point)
        e_ground = court.image_to_ground(run[-1].point)
        s2d = np.array([s_ground.x, s_ground.y])
        e2d = np.array([e_ground.x, e_ground.y])
        td = max((n - 1) / fps, 1e-3)

        x0 = np.array([s2d[0], s2d[1], 1.7])
        v0 = np.array([(e2d[0] - s2d[0]) / td, (e2d[1] - s2d[1]) / td,
                       physics.GRAVITY_M_S2 * td / 2.0])
        init = np.concatenate([x0, v0, [0.2]])
        bounds = [(0, _X_MAX), (0, _Y_MAX), (0.1, _Z_MAX)] + [(-150, 150)] * 3 + [(0.0, 0.4)]

        dt = 1.0 / (fps * _SUBSTEPS)

        def simulate(p):
            x = np.array(p[:3], float)
            v = np.array(p[3:6], float)
            c = float(p[6])
            out = [x.copy()]
            for t in range(1, _SUBSTEPS * (n - 1) + 1):
                v = v + dt * (_GRAVITY - c * np.linalg.norm(v) * v)
                x = x + dt * v
                if t % _SUBSTEPS == 0:
                    out.append(x.copy())
            return np.array(out)  # (n, 3)

        def loss(p):
            traj = simulate(p)
            proj = self._project(cam_proj, traj)
            return float(np.mean(np.linalg.norm(proj - observed, axis=1) ** 2))

        res = scipy.optimize.minimize(loss, init, bounds=bounds, method="SLSQP")
        traj3d = simulate(res.x)
        proj = self._project(cam_proj, traj3d)
        resid = list(np.linalg.norm(proj - observed, axis=1))
        points = [
            ShuttlePoint3D(
                frame_index=run[i].frame_index,
                point=Point3D(float(traj3d[i, 0]), float(traj3d[i, 1]), float(traj3d[i, 2])),
            )
            for i in range(n)
        ]
        return points, resid

    @staticmethod
    def _project(cam_proj: np.ndarray, world: np.ndarray) -> np.ndarray:
        h = np.hstack([world, np.ones((len(world), 1))])  # (n,4)
        q = (cam_proj @ h.T).T  # (n,3)
        return q[:, :2] / q[:, 2:3]
