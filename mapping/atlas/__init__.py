"""Atlas — Model Internal Behavior Library (MIBL) extraction system.

Turns an open-weight transformer into an indexable, queryable on-disk data
library of its static weights, observed firing behavior, and inferred
component relationships (fire-together and signal-flow), organized into the
storage / metadata / index / graph / summary layers described in the project
plan.

Top-level entry point: ``atlas.cli.main`` (installed as the ``atlas`` console
script). The library handle is ``atlas.manifest.Library``; per-stage logic lives
in ``atlas.stages``.
"""

from __future__ import annotations

__version__ = "1.0.0"

from .config import ExtractionConfig
from .manifest import Library

__all__ = ["ExtractionConfig", "Library", "__version__"]
