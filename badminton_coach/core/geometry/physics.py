"""Shuttlecock aerodynamic model — Cooke (2002) drag equations of motion.

The shuttle is dominated by quadratic air drag, not pure projectile motion. The
3D reconstructor (MonoTrack-style) integrates these equations and fits the launch
state so the reprojected 3D path matches the observed 2D track.

ODE (per the reports):
    a = g  -  (rho * Cd * A / (2 m)) * |v| * v
with g pointing down. Cd varies with speed (wind-tunnel ~0.48, literature
0.48-0.74); we expose it as a parameter rather than hardcoding a single value.
"""

from __future__ import annotations

import numpy as np

# Physical constants / typical shuttle parameters (SI units).
GRAVITY_M_S2: float = 9.81
AIR_DENSITY_KG_M3: float = 1.225
SHUTTLE_MASS_KG: float = 0.005  # ~5 g
SHUTTLE_AREA_M2: float = 0.00342  # ~ pi * (0.033)^2, skirt cross-section
DRAG_COEFFICIENT: float = 0.60  # mid-range of reported 0.48-0.74


def acceleration(velocity: np.ndarray, cd: float = DRAG_COEFFICIENT) -> np.ndarray:
    """Instantaneous acceleration (m/s^2) for a shuttle with velocity `velocity`.

    velocity: shape (3,) world-frame velocity [vx, vy, vz], meters/second.
    Returns shape (3,) acceleration including gravity and quadratic drag.
    """
    g = np.array([0.0, 0.0, -GRAVITY_M_S2])
    speed = float(np.linalg.norm(velocity))
    if speed == 0.0:
        return g
    drag_coeff = AIR_DENSITY_KG_M3 * cd * SHUTTLE_AREA_M2 / (2.0 * SHUTTLE_MASS_KG)
    drag = -drag_coeff * speed * velocity
    return g + drag


def simulate(
    p0: np.ndarray,
    v0: np.ndarray,
    dt: float,
    steps: int,
    cd: float = DRAG_COEFFICIENT,
) -> np.ndarray:
    """Forward-integrate the drag ODE (semi-implicit Euler).

    p0, v0: shape (3,) initial world position (m) and velocity (m/s).
    Returns positions of shape (steps + 1, 3).
    """
    positions = np.empty((steps + 1, 3), dtype=np.float64)
    p = p0.astype(np.float64).copy()
    v = v0.astype(np.float64).copy()
    positions[0] = p
    for i in range(1, steps + 1):
        v = v + acceleration(v, cd) * dt
        p = p + v * dt
        positions[i] = p
    return positions
