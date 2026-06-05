"""Stable identifier generation for the Model Internal Behavior Library (MIBL).

Every high-level object in the library must be traceable downward, and that
traceability depends on identifiers that do not change across runs or model
families. Display names (e.g. "model.layers.0.mlp.down_proj") change between
architectures; the IDs minted here are derived deterministically from stable
inputs so that the same logical object always resolves to the same ID.

The format is human-readable on purpose. An ID like

    L12.mlp_neuron.000345

tells you the layer, the unit type, and the local index at a glance, while
still being unique within a model. Cross-model uniqueness is provided by the
``model_id`` (a short content hash) that prefixes the library, not every row.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _short_hash(payload: str, length: int = 12) -> str:
    """Return a short, stable hex digest for ``payload``."""
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:length]


def model_id(name: str, architecture: str, param_count: int) -> str:
    """Mint a stable model ID from identity-defining fields.

    Two checkpoints that share name, architecture, and parameter count are
    treated as the same model identity. The extraction *run* (which may differ
    in calibration corpus, thresholds, etc.) is tracked separately via the run
    log, not by the model ID.
    """
    payload = json.dumps(
        {"name": name, "architecture": architecture, "param_count": int(param_count)},
        sort_keys=True,
    )
    return f"m_{_short_hash(payload, 10)}"


def run_id(model_id_: str, config: dict[str, Any]) -> str:
    """Mint a stable extraction-run ID from the model ID and config."""
    payload = json.dumps({"model": model_id_, "config": config}, sort_keys=True)
    return f"run_{_short_hash(payload, 10)}"


def tensor_id(tensor_name: str) -> str:
    """ID for a raw checkpoint tensor, derived from its canonical name."""
    return f"t.{_short_hash(tensor_name, 12)}"


def logical_tensor_id(logical_name: str, layer_id: int) -> str:
    """ID for a logical tensor slice (e.g. a single head's slice of a fused QKV)."""
    return f"lt.{_short_hash(f'{layer_id}:{logical_name}', 12)}"


def unit_id(layer_id: int, unit_type: str, local_index: int) -> str:
    """Human-readable, stable unit ID.

    Example: ``L09.attn_head.0007``. The width of the index is generous so that
    string sorting matches numeric ordering for typical model sizes.
    """
    return f"L{layer_id:02d}.{unit_type}.{local_index:06d}"


def cluster_id(level: str, index: int) -> str:
    """Stable cluster ID namespaced by hierarchy level.

    Levels: ``local``, ``xlayer`` (cross-layer), ``family``.
    """
    prefix = {"local": "cl", "xlayer": "cx", "family": "cf"}.get(level, "c")
    return f"{prefix}.{index:06d}"


def probe_id(probe_kind: str, index: int) -> str:
    """Stable ID for an internal probe direction."""
    return f"p.{probe_kind}.{index:06d}"


def histogram_id(entity_type: str, entity_id: str, metric_name: str) -> str:
    """Stable ID for a precomputed histogram keyed by entity and metric."""
    return f"h.{_short_hash(f'{entity_type}:{entity_id}:{metric_name}', 14)}"
