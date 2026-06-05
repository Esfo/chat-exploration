"""Stage 3 — compute static per-unit geometry (vectorized per layer).

For each unit we summarize the read side (what residual directions can activate
it) and the write side (what it writes back), plus direction sketches for fast
similarity search and the static alignment graph. No text is run through the
model here.

The key to doing this for hundreds of thousands of units in seconds rather than
hours is to never loop over units. Within one decoder layer the units *are* rows
or slices of the weight tensors:

  * MLP neuron j: read direction = row j of ``gate_proj`` ([inter, hidden]);
    write direction = column j of ``down_proj`` ([hidden, inter]).
  * Attention head h: read direction = the head's ``q_proj`` rows, averaged;
    write direction = the head's ``o_proj`` columns, averaged.

So every metric (norms, mean/std/kurtosis, tail ratio, outlier score) and the
random-projection sketches are computed for a whole layer's units at once with
NumPy matrix ops, which also actually uses all the cores via BLAS.
"""

from __future__ import annotations

import numpy as np

from .. import ids
from ..manifest import Library
from ..model_backend import ModelBackend
from ..sketches import random_projection_matrix
from ..storage import write_zarr_array
from . import register


@register("static-analysis", 3,
          requires=["catalog/unit_index.parquet", "catalog/unit_weight_refs.parquet"],
          produces=["units/unit_static_stats.parquet", "units/unit_direction_sketches.zarr"])
def run(library: Library, backend: ModelBackend, **kwargs):
    config = library.config()
    arch = backend.architecture()
    proj = random_projection_matrix(arch.hidden_size, config.signature_dim, config.sketch_seed)

    stat_rows: list[dict] = []
    unit_ids: list[str] = []
    read_sketches: list[np.ndarray] = []
    write_sketches: list[np.ndarray] = []

    n_mlp = arch.intermediate_size
    if config.max_mlp_neurons_per_layer:
        n_mlp = min(n_mlp, config.max_mlp_neurons_per_layer)
    hd, n_heads = arch.head_dim, arch.num_attention_heads

    for layer in range(arch.num_layers):
        if config.include_mlp_neurons:
            gate = backend.get_tensor(f"model.layers.{layer}.mlp.gate_proj.weight")
            down = backend.get_tensor(f"model.layers.{layer}.mlp.down_proj.weight")
            #read rows = gate_proj rows; write rows = down_proj columns (transpose).
            read_mat = np.ascontiguousarray(gate[:n_mlp])            # [n_mlp, hidden]
            write_mat = np.ascontiguousarray(down[:, :n_mlp].T)      # [n_mlp, hidden]
            _emit_group(layer, "mlp_neuron", n_mlp, read_mat, write_mat, proj,
                        config.signature_dim, stat_rows, unit_ids,
                        read_sketches, write_sketches)
            print(f"[static] layer {layer} mlp done ({n_mlp} neurons)", flush=True)

        if config.include_attention_heads:
            q = backend.get_tensor(f"model.layers.{layer}.self_attn.q_proj.weight")
            o = backend.get_tensor(f"model.layers.{layer}.self_attn.o_proj.weight")
            #q_proj rows grouped by head -> mean over the head's hd rows.
            read_mat = q[: n_heads * hd].reshape(n_heads, hd, arch.hidden_size).mean(axis=1)
            #o_proj columns grouped by head -> mean over the head's hd columns.
            write_mat = o[:, : n_heads * hd].reshape(arch.hidden_size, n_heads, hd).mean(axis=2).T
            _emit_group(layer, "attn_head", n_heads,
                        np.ascontiguousarray(read_mat), np.ascontiguousarray(write_mat),
                        proj, config.signature_dim, stat_rows, unit_ids,
                        read_sketches, write_sketches)

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


def _emit_group(layer, unit_type, n_units, read_mat, write_mat, proj, sig_dim,
                stat_rows, unit_ids, read_sketches, write_sketches):
    """Compute every per-unit metric for one (layer, unit_type) group at once."""
    rn = _row_norm(read_mat)
    wn = _row_norm(write_mat)
    r_mean, r_std, r_kurt = _row_moments(read_mat)
    w_mean, w_std, w_kurt = _row_moments(write_mat)
    tail = _row_tail_ratio(write_mat)
    outlier = np.maximum(_row_outlier(read_mat), _row_outlier(write_mat))
    read_sk = _row_sketch(read_mat, proj)
    write_sk = _row_sketch(write_mat, proj)

    for j in range(n_units):
        uid = ids.unit_id(layer, unit_type, j)
        unit_ids.append(uid)
        stat_rows.append({
            "unit_id": uid,
            "read_norm": float(rn[j]), "write_norm": float(wn[j]),
            "read_mean": float(r_mean[j]), "read_std": float(r_std[j]),
            "write_mean": float(w_mean[j]), "write_std": float(w_std[j]),
            "read_kurtosis": float(r_kurt[j]), "write_kurtosis": float(w_kurt[j]),
            "weight_tail_ratio": float(tail[j]),
            "static_outlier_score": float(outlier[j]),
        })
        read_sketches.append(read_sk[j])
        write_sketches.append(write_sk[j])


#--- vectorized row-wise metrics ------------------------------------------

def _row_norm(m):
    return np.sqrt((m.astype(np.float64) ** 2).sum(axis=1))


def _row_moments(m):
    m = m.astype(np.float64)
    mean = m.mean(axis=1)
    centered = m - mean[:, None]
    var = (centered ** 2).mean(axis=1)
    std = np.sqrt(var)
    safe = np.where(std == 0, 1.0, std)
    kurt = (centered ** 4).mean(axis=1) / (safe ** 4) - 3.0
    return mean, std, kurt


def _row_tail_ratio(m):
    a = np.abs(m.astype(np.float64))
    q99 = np.quantile(a, 0.99, axis=1, keepdims=True)
    total = a.sum(axis=1)
    tail = np.where(a >= q99, a, 0.0).sum(axis=1)
    return np.where(total == 0, 0.0, tail / np.where(total == 0, 1.0, total))


def _row_outlier(m):
    a = np.abs(m.astype(np.float64))
    mean = a.mean(axis=1)
    return a.max(axis=1) / (mean + 1e-9)


def _row_sketch(m, proj):
    s = m.astype(np.float32) @ proj  # [n_units, sig_dim]
    norms = np.linalg.norm(s, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (s / norms).astype(np.float32)
