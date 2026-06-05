"""Configuration and path conventions for the MIBL extraction system.

The save-location conventions here mirror the ``/rescaling`` pipeline: large
artifacts live under ``/home/sfo/data/models`` rather than inside the source
repository, and the analysed model defaults to the same Llama 3 8B family that
``/rescaling`` operates on. Activation capture requires forward hooks, so the
default backend is the HuggingFace/PyTorch checkpoint rather than the GGUF used
by Ollama; the path still resolves under the shared data directory.

Every value here is overridable from the CLI. Nothing in the library code
hard-codes a path outside of these defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


#Shared data root, matching the /rescaling save convention.
DATA_ROOT = Path(os.environ.get("ATLAS_DATA_ROOT", "/home/sfo/data/models"))

#Default library output directory (the MIBL "atlas_library/").
DEFAULT_LIBRARY_DIR = DATA_ROOT / "atlas_library"

#Default model + tokenizer. /rescaling references the Ollama `llama3:8b` GGUF;
#activation capture needs an HF-format checkpoint, expected under the same root.
DEFAULT_MODEL_PATH = DATA_ROOT / "Meta-Llama-3-8B"
DEFAULT_TOKENIZER_PATH = DEFAULT_MODEL_PATH

#GGUF reference kept for provenance / static-only fallbacks. This is the same
#model /rescaling scales; resolved lazily via Ollama only when needed.
REFERENCE_OLLAMA_MODEL = "llama3:8b"


@dataclass
class ExtractionConfig:
    """All knobs that define an extraction *run*.

    These fields feed the run ID, so changing any of them produces a distinct,
    independently tracked run within the same model identity.
    """

    #--- model / backend ---
    model_path: str = str(DEFAULT_MODEL_PATH)
    tokenizer_path: str = str(DEFAULT_TOKENIZER_PATH)
    backend: str = "hf"  # "hf" (full hooks) — GGUF static-only reserved for later
    dtype: str = "bfloat16"
    device: str = "auto"

    #--- calibration ---
    target_tokens: int = 500_000
    max_sequence_length: int = 512
    calibration_seed: int = 17

    #--- activation capture ---
    batch_size: int = 4
    #Per-unit number of strongest activation events to retain.
    top_events_per_unit: int = 16
    #Number of histogram bins for per-unit activation distributions.
    activation_hist_bins: int = 32
    #Activation thresholds are layer/component relative: a unit is "active" when
    #its value exceeds this quantile of its own observed distribution.
    active_quantile: float = 0.90

    #--- sketches ---
    #Random-projection signature dimensionality for similarity search.
    signature_dim: int = 256
    #Length (in bits) of the active-position bitset sample. Kept modest because
    #every unit holds one packed bitset for the whole capture pass.
    bitset_positions: int = 2048
    #Reservoir size per unit for approximate activation quantiles/histograms.
    reservoir_size: int = 128
    sketch_seed: int = 101

    #--- graphs ---
    #Keep only the top-k strongest edges per source unit (sparsity rule).
    edges_top_k: int = 32
    #Minimum combined score to retain an edge.
    edge_min_score: float = 0.05
    #Evidence-type weights when computing the combined score.
    evidence_weights: dict[str, float] = field(
        default_factory=lambda: {
            "activation": 1.0,
            "lagged": 1.0,
            "static_alignment": 0.5,
            "attention_routing": 0.75,
            "probe_response": 0.5,
            "causal": 2.0,
        }
    )

    #--- clustering ---
    local_resolution: float = 1.0
    xlayer_resolution: float = 1.0
    clustering_algorithm: str = "leiden_or_agglomerative"
    min_cluster_size: int = 3

    #--- unit selection ---
    include_mlp_neurons: bool = True
    include_attention_heads: bool = True
    include_residual_directions: bool = False  # v3
    #Optionally subsample MLP neurons per layer to bound cost on huge models.
    max_mlp_neurons_per_layer: int = 0  # 0 = all

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractionConfig":
        known = {f for f in cls().to_dict()}
        return cls(**{k: v for k, v in data.items() if k in known})


#--------------------------------------------------------------------------
#On-disk layout. These relative paths are the library *contract*: downstream
#pipelines depend on them, so they live in one place.
#--------------------------------------------------------------------------

LAYOUT = {
    "manifest": [
        "manifest/library.json",
        "manifest/model.json",
        "manifest/architecture.json",
        "manifest/tokenizer.json",
        "manifest/extraction_config.json",
        "manifest/artifact_versions.json",
        "manifest/run_log.jsonl",
    ],
    "catalog": [
        "catalog/tensor_index.parquet",
        "catalog/logical_tensor_index.parquet",
        "catalog/unit_index.parquet",
        "catalog/unit_weight_refs.parquet",
        "catalog/artifact_index.parquet",
    ],
    "tensors": [
        "tensors/tensor_stats.parquet",
        "tensors/tensor_histograms.parquet",
        "tensors/row_col_stats.parquet",
        "tensors/singular_summaries.parquet",
    ],
    "units": [
        "units/unit_static_stats.parquet",
        "units/unit_direction_sketches.zarr",
        "units/unit_role_scores.parquet",
    ],
    "calibration": [
        "calibration/corpus_index.parquet",
        "calibration/token_index.parquet",
        "calibration/token_group_stats.parquet",
        "calibration/calibration_splits.parquet",
    ],
    "activations": [
        "activations/activation_stats.parquet",
        "activations/activation_histograms.parquet",
        "activations/activation_top_events.parquet",
        "activations/activation_sketches.zarr",
        "activations/active_bitsets.zarr",
        "activations/activation_quantile_sketches.zarr",
    ],
    "probes": [
        "probes/probe_index.parquet",
        "probes/probe_vectors.zarr",
        "probes/probe_responses.parquet",
        "probes/probe_response_sketches.zarr",
    ],
    "graphs": [
        "graphs/unit_edges_activation.parquet",
        "graphs/unit_edges_static_alignment.parquet",
        "graphs/unit_edges_lagged.parquet",
        "graphs/unit_edges_attention_routing.parquet",
        "graphs/unit_edges_probe_response.parquet",
        "graphs/unit_edges_combined.parquet",
        "graphs/cluster_edges.parquet",
        "graphs/graph_partitions.parquet",
    ],
    "clusters": [
        "clusters/cluster_index.parquet",
        "clusters/cluster_membership.parquet",
        "clusters/cluster_stats.parquet",
        "clusters/cluster_hierarchy.parquet",
        "clusters/cluster_centroids.zarr",
        "clusters/cluster_exemplars.parquet",
    ],
    "summaries": [
        "summaries/histogram_index.parquet",
        "summaries/histogram_bins.parquet",
        "summaries/quantile_summaries.parquet",
        "summaries/topk_index.parquet",
        "summaries/materialized_cluster_views.parquet",
        "summaries/baseline_comparisons.parquet",
    ],
    "indexes": [
        "indexes/atlas.duckdb",
        "indexes/vector_units",
        "indexes/vector_clusters",
        "indexes/graph_adjacency",
        "indexes/bitmap_indexes",
    ],
    "exports": [
        "exports/graphml",
        "exports/parquet_views",
        "exports/report_inputs",
    ],
}


def all_subdirs() -> list[str]:
    """Return the set of directories that must exist in a library."""
    dirs = set()
    for paths in LAYOUT.values():
        for p in paths:
            dirs.add(str(Path(p).parent))
    #Directory-style artifacts (zarr stores, index dirs) are dirs themselves.
    for p in LAYOUT["indexes"] + LAYOUT["exports"]:
        if not p.endswith((".duckdb",)):
            dirs.add(p)
    return sorted(dirs)
