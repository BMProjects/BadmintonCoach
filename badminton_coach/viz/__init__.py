"""Visualization layer: overlays + annotated-video rendering.

Depends only on data contracts (core.schemas), never on perception backends, so it
can render the output of any backend combination.
"""

from . import metrics
from .overlay import draw_court, draw_detections, draw_poses, draw_shuttle
from .render import render_video

__all__ = ["draw_detections", "draw_poses", "draw_court", "draw_shuttle", "render_video", "metrics"]
