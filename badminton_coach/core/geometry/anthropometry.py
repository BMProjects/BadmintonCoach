"""Anthropometric scaling (Winter / de Leva) — segment masses, lengths, inertias.

Scales a generic body-segment table to a player's height/weight so pose-derived joint
kinematics can be turned into approximate joint loads (I*alpha). Fractions are standard
(Winter, "Biomechanics and Motor Control of Human Movement"): segment mass as a fraction
of body mass, segment length as a fraction of stature, radius of gyration as a fraction
of segment length (about the proximal joint).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _Seg:
    mass_frac: float   # of total body mass
    len_frac: float    # of stature (height)
    k_prox: float      # radius of gyration about the proximal joint (frac of seg length)


# Distal segment hanging off each joint (what that joint must swing): used for I*alpha.
# Values are standard cadaver-derived means (Winter, Dempster); coarse but adequate for
# a *relative* torque proxy from planar pose.
_JOINT_DISTAL_SEGMENT = {
    "wrist": _Seg(0.006, 0.108, 0.587),                 # hand
    "elbow": _Seg(0.022, 0.146, 0.827),                 # forearm + hand
    "shoulder": _Seg(0.050, 0.332, 0.645),              # whole arm
    "knee": _Seg(0.061, 0.246, 0.735),                  # shank + foot
    "hip": _Seg(0.161, 0.285, 0.560),                   # whole leg (thigh+shank+foot)
}


def segment_inertia_about_joint(profile, joint: str) -> float:
    """Moment of inertia (kg·m²) of the segment distal to `joint`, about that joint:
    I = m · (k·L)². Returns 0 for unknown joints."""
    seg = _JOINT_DISTAL_SEGMENT.get(joint)
    if seg is None:
        return 0.0
    m = seg.mass_frac * profile.mass_kg
    length = seg.len_frac * profile.height_m
    return m * (seg.k_prox * length) ** 2


def known_joints() -> tuple[str, ...]:
    return tuple(_JOINT_DISTAL_SEGMENT)
