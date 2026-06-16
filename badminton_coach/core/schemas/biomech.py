"""Biomechanics data contracts (L3).

A PlayerProfile (height/weight/handedness/sex) scales an anthropometric model so joint
kinematics from pose can be turned into approximate joint loads. The MVP works from 2D
pose (planar-angle approximation) — a 3D upgrade (WHAM/OpenSim) plugs in behind the same
schema later.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class PlayerProfile:
    """Per-player anthropometrics for biomechanics scaling."""

    height_m: float
    mass_kg: float
    handedness: str = "R"   # "R" | "L" (racket arm)
    sex: str = "M"          # "M" | "F"


@dataclass(frozen=True, slots=True)
class JointMetric:
    """Per-joint kinematics + load over one stroke (planar/relative approximations)."""

    name: str
    peak_angle_deg: float          # max flexion angle in the window
    rom_deg: float                 # range of motion (max-min)
    peak_ang_vel_dps: float        # peak angular velocity (deg/s)
    peak_torque_nm: float          # I*alpha proxy of the distal segment about the joint


@dataclass(frozen=True, slots=True)
class StrokeBiomechanics:
    """Biomechanics summary for one stroke."""

    shot_index: int
    start_frame: int
    end_frame: int
    track_id: int | None
    joints: tuple[JointMetric, ...]
    kinematic_sequence: tuple[str, ...]   # segments ordered by time-of-peak angular velocity
    sequence_ok: bool                     # proximal->distal (hips->trunk->arm->forearm)?
    effort_nm: float                      # overall load proxy = max joint peak torque


@dataclass(frozen=True, slots=True)
class BiomechanicsReport:
    """Per-stroke biomechanics over a clip."""

    strokes: tuple[StrokeBiomechanics, ...] = field(default_factory=tuple)
