"""Stage 1 — scan tensors.

Builds the tensor catalog (catalog/tensor_index.parquet) and the model-wide
statistical baselines (tensors/tensor_stats.parquet, tensor_histograms.parquet,
row_col_stats.parquet, singular_summaries.parquet). Every later object references
these tensor IDs rather than copying raw weights.
"""

from __future__ import annotations

import re

import numpy as np

from .. import ids
from ..manifest import Library
from ..model_backend import ModelBackend, LLAMA_LAYER_TENSORS
from ..sketches import fixed_histogram
from . import register

_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def _classify(name: str) -> tuple[int, str, str, str]:
    """Return (layer_id, component_type, architecture_role, logical_name)."""
    m = _LAYER_RE.search(name)
    layer_id = int(m.group(1)) if m else -1
    for suffix, (component, role) in LLAMA_LAYER_TENSORS.items():
        if name.endswith(suffix):
            return layer_id, component, role, role
    if "embed" in name:
        return -1, "embedding", "embedding", "embedding"
    if "lm_head" in name:
        return -1, "embedding", "lm_head", "lm_head"
    if name.endswith("norm.weight"):
        return layer_id, "norm", "final_norm", "final_norm"
    return layer_id, "other", "other", name.split(".")[-1]


@register("scan-tensors", 1, requires=["manifest/library.json"],
          produces=["catalog/tensor_index.parquet", "tensors/tensor_stats.parquet"])
def run(library: Library, backend: ModelBackend, hist_bins: int = 64,
        stat_sample: int = 1_000_000, svd_dim: int = 256, **kwargs):
    model_id = library.model_id()

    index_rows = []
    stat_rows = []
    hist_rows = []
    rowcol_rows = []
    sing_rows = []
    rng = np.random.default_rng(0)

    for name, shape, dtype in backend.iter_named_tensors():
        layer_id, component, role, logical = _classify(name)
        tid = ids.tensor_id(name)
        param_count = int(np.prod(shape)) if shape else 0

        index_rows.append({
            "tensor_id": tid, "model_id": model_id, "tensor_name": name,
            "logical_name": logical, "layer_id": layer_id,
            "component_type": component, "architecture_role": role,
            "shape": list(shape), "dtype": dtype, "parameter_count": param_count,
            "checkpoint_file": "", "checkpoint_offset": -1, "hash": tid,
        })

        arr = backend.get_tensor(name)
        flat = arr.ravel()
        #Subsample huge tensors for distribution stats to keep this fast; full
        #tensors (e.g. the 128k x 4096 embedding) would make SVD/quantiles crawl.
        sample = flat if flat.size <= stat_sample else flat[
            rng.integers(0, flat.size, stat_sample)]
        qs = np.quantile(sample, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).tolist()
        stat_rows.append({
            "tensor_id": tid,
            "mean": float(sample.mean()), "std": float(sample.std()),
            "min": float(flat.min()), "max": float(flat.max()),
            "abs_max": float(np.abs(flat).max()),
            "quantiles": qs,
            "skewness": _skew(sample), "kurtosis": _kurt(sample),
            "row_norm_summary": _norm_summary(arr, axis=1),
            "col_norm_summary": _norm_summary(arr, axis=0),
        })

        lo, hi = float(sample.min()), float(sample.max())
        bl, br, ct = fixed_histogram(sample, hist_bins, lo, hi)
        hist_rows.append({"tensor_id": tid, "bin_left": bl, "bin_right": br, "count": ct})

        if arr.ndim == 2:
            _append_axis_stats(rowcol_rows, tid, arr, axis="row")
            _append_axis_stats(rowcol_rows, tid, arr, axis="col")
            sing_rows.append(_singular_summary(tid, arr, svd_dim, rng))

    library.commit_table("catalog/tensor_index.parquet", index_rows, "catalog", "scan-tensors")
    library.commit_table("tensors/tensor_stats.parquet", stat_rows, "tensors", "scan-tensors")
    library.commit_table("tensors/tensor_histograms.parquet", hist_rows, "tensors", "scan-tensors")
    library.commit_table("tensors/row_col_stats.parquet", rowcol_rows, "tensors", "scan-tensors")
    library.commit_table("tensors/singular_summaries.parquet", sing_rows, "tensors", "scan-tensors")

    library.log("scan-tensors", "tensor catalog built", tensors=len(index_rows))
    library.update_artifact_versions("scan-tensors")
    return {"tensors": len(index_rows)}


def _skew(x: np.ndarray) -> float:
    s = x.std()
    return float(((x - x.mean()) ** 3).mean() / (s ** 3)) if s else 0.0


def _kurt(x: np.ndarray) -> float:
    s = x.std()
    return float(((x - x.mean()) ** 4).mean() / (s ** 4) - 3.0) if s else 0.0


def _norm_summary(arr: np.ndarray, axis: int) -> list[float]:
    if arr.ndim != 2:
        return []
    norms = np.linalg.norm(arr, axis=axis)
    return np.quantile(norms, [0.0, 0.25, 0.5, 0.75, 1.0]).tolist()


def _append_axis_stats(rows, tid, arr, axis, sample=64):
    a = 1 if axis == "row" else 0
    n = arr.shape[0 if axis == "row" else 1]
    idxs = np.linspace(0, n - 1, min(sample, n)).astype(int)
    for i in idxs:
        vec = arr[i] if axis == "row" else arr[:, i]
        rows.append({
            "tensor_id": tid, "axis": axis, "index": int(i),
            "norm": float(np.linalg.norm(vec)),
            "mean": float(vec.mean()), "std": float(vec.std()),
        })


def _singular_summary(tid, arr, max_dim, rng):
    #Bound cost hard: SVD on a random square submatrix (<= max_dim) so even the
    #huge embedding/projection tensors summarize in milliseconds, not minutes.
    a = arr
    if a.shape[0] > max_dim:
        a = a[rng.integers(0, a.shape[0], max_dim), :]
    if a.shape[1] > max_dim:
        a = a[:, rng.integers(0, a.shape[1], max_dim)]
    try:
        sv = np.linalg.svd(a, compute_uv=False)
    except np.linalg.LinAlgError:
        sv = np.array([0.0])
    energy = (sv ** 2)
    total = float(energy.sum()) or 1.0
    eff_rank = float((energy.sum() ** 2) / (energy ** 2).sum()) if (energy ** 2).sum() else 0.0
    return {
        "tensor_id": tid,
        "singular_values": sv[:32].tolist(),
        "effective_rank": eff_rank,
        "energy_top1": float(energy[:1].sum() / total),
        "energy_top8": float(energy[:8].sum() / total),
    }
