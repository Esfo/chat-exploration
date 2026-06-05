"""Stage 3 — compute static per-unit geometry.

For each unit, summarize the read side (what residual directions can activate it)
and the write side (what it writes back). Also build the per-unit direction
sketches (read/write) used for fast similarity search and the static alignment
graph. No text is run through the model in this stage.
"""

from __future__ import annotations

import numpy as np

from ..manifest import Library
from ..model_backend import ModelBackend
from ..parallel import thread_map
from ..sketches import random_projection_matrix
from ..storage import read_parquet, write_zarr_array
from . import register


@register("static-analysis", 3,
          requires=["catalog/unit_index.parquet", "catalog/unit_weight_refs.parquet"],
          produces=["units/unit_static_stats.parquet", "units/unit_direction_sketches.zarr"])
def run(library: Library, backend: ModelBackend, **kwargs):
    config = library.config()
    arch = backend.architecture()

    units = read_parquet(library.path("catalog/unit_index.parquet")).to_pylist()
    refs = read_parquet(library.path("catalog/unit_weight_refs.parquet")).to_pylist()
    refs_by_unit: dict[str, list[dict]] = {}
    for r in refs:
        refs_by_unit.setdefault(r["unit_id"], []).append(r)

    #Shared projection so read/write sketches live in the same space (hidden dim).
    proj = random_projection_matrix(arch.hidden_size, config.signature_dim, config.sketch_seed)

    #Pre-load every referenced tensor once, single-threaded, so the per-unit
    #workers below read from a fully-populated, read-only cache (thread-safe).
    tensor_cache: dict[str, np.ndarray] = {}
    for u in units:
        for r in refs_by_unit.get(u["unit_id"], []):
            name = _tensor_name_for_unit(u, r)
            if name and name not in tensor_cache:
                tensor_cache[name] = backend.get_tensor(name)

    def process(u):
        """Per-unit static geometry + direction sketches. Runs on a worker."""
        uid = u["unit_id"]
        read_vec, write_vec = _unit_vectors(backend, u, refs_by_unit.get(uid, []), tensor_cache)
        read_norm = float(np.linalg.norm(read_vec)) if read_vec is not None else 0.0
        write_norm = float(np.linalg.norm(write_vec)) if write_vec is not None else 0.0
        stat_row = {
            "unit_id": uid,
            "read_norm": read_norm, "write_norm": write_norm,
            "read_mean": _safe(read_vec, np.mean), "read_std": _safe(read_vec, np.std),
            "write_mean": _safe(write_vec, np.mean), "write_std": _safe(write_vec, np.std),
            "read_kurtosis": _kurt(read_vec), "write_kurtosis": _kurt(write_vec),
            "weight_tail_ratio": _tail_ratio(write_vec),
            "static_outlier_score": _outlier_score(read_vec, write_vec),
        }
        return (uid, stat_row,
                _sketch(read_vec, proj, config.signature_dim),
                _sketch(write_vec, proj, config.signature_dim))

    results = thread_map(process, units)

    stat_rows, unit_ids, read_sketches, write_sketches = [], [], [], []
    for uid, stat_row, read_sk, write_sk in results:
        unit_ids.append(uid)
        stat_rows.append(stat_row)
        read_sketches.append(read_sk)
        write_sketches.append(write_sk)

    library.commit_table("units/unit_static_stats.parquet", stat_rows, "units", "static-analysis")
    write_zarr_array(
        {
            "unit_ids": np.array(unit_ids, dtype=object).astype("U64"),
            "read_sketch": np.vstack(read_sketches) if read_sketches else np.zeros((0, config.signature_dim), np.float32),
            "write_sketch": np.vstack(write_sketches) if write_sketches else np.zeros((0, config.signature_dim), np.float32),
        },
        library.path("units/unit_direction_sketches.zarr"),
    )
    library.register_artifact("units/unit_direction_sketches.zarr", "units", "static-analysis",
                              row_count=len(unit_ids), fmt="zarr")
    library.log("static-analysis", "static unit stats computed", units=len(stat_rows))
    library.update_artifact_versions("static-analysis")
    return {"units": len(stat_rows)}


def _unit_vectors(backend, unit, refs, cache):
    """Return (read_vec, write_vec) in hidden-space for a unit."""
    read_vec = write_vec = None
    for r in refs:
        name = _tensor_name_for_unit(unit, r)
        if name is None:
            continue
        if name not in cache:
            cache[name] = backend.get_tensor(name)
        arr = cache[name]
        sl = slice(r["start_index"], r["end_index"])
        if r["slice_type"] in ("read", "qk"):
            #Read direction in hidden space.
            vec = arr[sl, :] if r["dimension"] == 0 else arr[:, sl]
            read_vec = vec.mean(axis=0) if vec.ndim == 2 else vec
        elif r["slice_type"] in ("write", "ov"):
            vec = arr[:, sl] if r["dimension"] == 1 else arr[sl, :]
            write_vec = vec.mean(axis=1) if vec.ndim == 2 else vec
    return read_vec, write_vec


def _tensor_name_for_unit(unit, ref):
    """Reconstruct the state-dict tensor name for a unit's ref by role."""
    layer = unit["layer_id"]
    interp = ref["interpretation"]
    mapping = {
        "gate_row": f"model.layers.{layer}.mlp.gate_proj.weight",
        "down_col": f"model.layers.{layer}.mlp.down_proj.weight",
        "q_head_rows": f"model.layers.{layer}.self_attn.q_proj.weight",
        "o_head_cols": f"model.layers.{layer}.self_attn.o_proj.weight",
    }
    return mapping.get(interp)


def _sketch(vec, proj, dim):
    if vec is None:
        return np.zeros(dim, dtype=np.float32)
    v = np.asarray(vec, dtype=np.float32)
    if v.shape[0] != proj.shape[0]:
        #Length mismatch (e.g. head_dim read slice): pad/truncate to hidden dim.
        out = np.zeros(proj.shape[0], dtype=np.float32)
        out[: min(len(v), proj.shape[0])] = v[: proj.shape[0]]
        v = out
    s = v @ proj
    n = np.linalg.norm(s)
    return (s / n).astype(np.float32) if n else s.astype(np.float32)


def _safe(v, fn):
    return float(fn(v)) if v is not None and len(v) else 0.0


def _kurt(v):
    if v is None or len(v) == 0:
        return 0.0
    s = np.std(v)
    return float(((v - np.mean(v)) ** 4).mean() / (s ** 4) - 3.0) if s else 0.0


def _tail_ratio(v):
    if v is None or len(v) == 0:
        return 0.0
    a = np.abs(v)
    thr = np.quantile(a, 0.99)
    return float(a[a >= thr].sum() / a.sum()) if a.sum() else 0.0


def _outlier_score(read_vec, write_vec):
    parts = [x for x in (read_vec, write_vec) if x is not None and len(x)]
    if not parts:
        return 0.0
    return float(max(np.abs(p).max() / (np.abs(p).mean() + 1e-9) for p in parts))
