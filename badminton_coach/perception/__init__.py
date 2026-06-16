"""L1 perception layer. Importing this package self-registers all backends.

Each submodule registers its backends with the core registry on import, so the
pipeline can build them by name from config.
"""

from . import court, detection, pose, reconstruction, shuttle, tracking  # noqa: F401

__all__ = ["court", "detection", "pose", "reconstruction", "shuttle", "tracking"]
