"""L2 event layer: hit detection, rally segmentation, shot classification.

Importing this package self-registers its backends.
"""

from . import hit_detection, shot_classification  # noqa: F401

__all__ = ["hit_detection", "shot_classification"]
