"""Stage registry and the resumable-pipeline contract.

Each stage is a callable ``run(library, backend, **kwargs)`` that reads prior
artifacts, validates prerequisites, writes temporary outputs, and commits them
atomically via the ``Library`` handle. Stages declare which artifacts they
require so the CLI can fail fast with a clear message instead of a deep
traceback when run out of order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Stage:
    name: str
    order: int
    run: Callable
    requires: list[str]
    produces: list[str]
    version_scope: str  # "v1" | "v2" | "v3"


#Populated by register() at import time.
REGISTRY: dict[str, Stage] = {}


def register(name, order, requires, produces, version_scope="v1"):
    def deco(fn):
        REGISTRY[name] = Stage(name, order, fn, requires, produces, version_scope)
        return fn
    return deco


def ordered_stages() -> list[Stage]:
    return sorted(REGISTRY.values(), key=lambda s: s.order)


def load_all() -> None:
    """Import every stage module so it registers itself."""
    from . import (  # noqa: F401
        stage00_init, stage01_scan_tensors, stage02_build_units,
        stage03_static_stats, stage04_static_graph, stage05_calibration,
        stage06_capture_activations, stage07_probes, stage08_build_graphs,
        stage09_cluster, stage10_score_clusters, stage11_interventions,
        stage12_export_summaries, stage_quality, stage_dashboard_summaries,
        stage13_build_indexes,
    )


def check_prerequisites(library, stage: Stage) -> list[str]:
    """Return a list of missing required artifacts for ``stage``."""
    missing = []
    for rel in stage.requires:
        if not library.path(rel).exists():
            missing.append(rel)
    return missing
