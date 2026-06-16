"""Geometry: homography round-trip and shuttle physics sanity."""

from __future__ import annotations

import numpy as np

from badminton_coach.core.geometry import physics, solve_homography
from badminton_coach.core.geometry.court_model import COURT_LENGTH_M, court_corners_doubles
from badminton_coach.core.schemas import Point2D, Point3D


def test_homography_roundtrip_recovers_ground_point():
    # Arrange: a plausible trapezoid of image corners <-> BWF doubles corners.
    image = [Point2D(300, 700), Point2D(980, 700), Point2D(880, 250), Point2D(400, 250)]
    world = [Point3D(x, y, 0.0) for x, y in court_corners_doubles()]

    # Act
    calib = solve_homography(image, world)
    back = calib.ground_to_image(calib.image_to_ground(image[0]))

    # Assert: round-trip returns to the same pixel, error tiny.
    assert calib.reprojection_error_px < 1.0
    assert abs(back.x - image[0].x) < 1.0
    assert abs(back.y - image[0].y) < 1.0


def test_net_line_is_court_midline():
    from badminton_coach.core.geometry.court_model import net_line_y

    assert net_line_y() == COURT_LENGTH_M / 2.0


def test_shuttle_falls_under_gravity_and_drag():
    # A shuttle launched horizontally must lose height and slow down.
    p0 = np.array([3.0, 6.7, 3.0])
    v0 = np.array([0.0, 20.0, 0.0])
    traj = physics.simulate(p0, v0, dt=0.01, steps=100)

    assert traj.shape == (101, 3)
    assert traj[-1, 2] < p0[2]  # dropped in height
    # Horizontal distance covered is less than the drag-free 20 m/s * 1 s.
    assert traj[-1, 1] - p0[1] < 20.0
