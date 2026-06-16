"""BWF standard badminton court geometry (the world-coordinate prior).

World frame: meters, origin at one court corner, x along the court WIDTH (doubles,
6.10 m), y along the court LENGTH (13.40 m), z up. Ground plane is z=0.

All dimensions are BWF regulation values (meters):
- length 13.40, doubles width 6.10, singles width 5.18
- net height at posts 1.55, at centre 1.524
- short service line 1.98 from net, doubles long service line 0.76 from back line
- line width 0.04
These are the known world points used to solve the ground-plane homography.
"""

from __future__ import annotations

COURT_LENGTH_M: float = 13.40
COURT_WIDTH_DOUBLES_M: float = 6.10
COURT_WIDTH_SINGLES_M: float = 5.18
NET_HEIGHT_POST_M: float = 1.55
NET_HEIGHT_CENTRE_M: float = 1.524
SHORT_SERVICE_FROM_NET_M: float = 1.98
DOUBLES_LONG_SERVICE_FROM_BACK_M: float = 0.76
LINE_WIDTH_M: float = 0.04


def court_corners_doubles() -> list[tuple[float, float]]:
    """The four outer doubles-court corners in world ground coords (x, y), meters.

    Order: near-left, near-right, far-right, far-left (counter-clockwise from the
    origin corner). Use these as the canonical 4+ correspondence points for the
    homography when the full doubles boundary is visible.
    """
    w = COURT_WIDTH_DOUBLES_M
    h = COURT_LENGTH_M
    return [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]


def net_line_y() -> float:
    """World y-coordinate of the net line (court centre)."""
    return COURT_LENGTH_M / 2.0


def court_line_segments() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """All painted BWF court lines as world ground segments ((x0,y0),(x1,y1)), meters.

    Includes baselines, doubles + singles sidelines, short-service lines, doubles
    long-service lines and the two centre-line halves. Used for line-fitting /
    overlap scoring and homography refinement.
    """
    w, ws, length = COURT_WIDTH_DOUBLES_M, COURT_WIDTH_SINGLES_M, COURT_LENGTH_M
    sx = (w - ws) / 2
    net, ss, dl = length / 2, SHORT_SERVICE_FROM_NET_M, DOUBLES_LONG_SERVICE_FROM_BACK_M
    return [
        ((0, 0), (w, 0)), ((0, length), (w, length)),               # baselines
        ((0, 0), (0, length)), ((w, 0), (w, length)),               # doubles sidelines
        ((sx, 0), (sx, length)), ((w - sx, 0), (w - sx, length)),   # singles sidelines
        ((0, net - ss), (w, net - ss)), ((0, net + ss), (w, net + ss)),  # short service
        ((0, dl), (w, dl)), ((0, length - dl), (w, length - dl)),   # doubles long service
        ((w / 2, 0), (w / 2, net - ss)), ((w / 2, net + ss), (w / 2, length)),  # centre
    ]
