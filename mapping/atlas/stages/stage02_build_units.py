"""Stage 2 — build the unit registry.

Maps architecture components to stable unit IDs and exact tensor-slice
references. V1 units: MLP neurons (one per intermediate dimension) and attention
heads (one per head). Each unit records its read/write tensors and the slices
that define it, so a downstream drilling pipeline can recover the exact
checkpoint weights from a unit ID.
"""

from __future__ import annotations

from .. import ids
from ..manifest import Library
from ..model_backend import ModelBackend
from . import register


@register("build-units", 2, requires=["catalog/tensor_index.parquet"],
          produces=["catalog/unit_index.parquet", "catalog/unit_weight_refs.parquet"])
def run(library: Library, backend: ModelBackend, **kwargs):
    model_id = library.model_id()
    config = library.config()
    arch = backend.architecture()

    unit_rows = []
    ref_rows = []

    for layer_id in range(arch.num_layers):
        gate_tid = _layer_tid(layer_id, "mlp.gate_proj.weight")
        up_tid = _layer_tid(layer_id, "mlp.up_proj.weight")
        down_tid = _layer_tid(layer_id, "mlp.down_proj.weight")
        q_tid = _layer_tid(layer_id, "self_attn.q_proj.weight")
        o_tid = _layer_tid(layer_id, "self_attn.o_proj.weight")

        #--- MLP neurons -------------------------------------------------
        if config.include_mlp_neurons:
            n_neurons = arch.intermediate_size
            if config.max_mlp_neurons_per_layer:
                n_neurons = min(n_neurons, config.max_mlp_neurons_per_layer)
            for j in range(n_neurons):
                uid = ids.unit_id(layer_id, "mlp_neuron", j)
                unit_rows.append({
                    "unit_id": uid, "model_id": model_id, "layer_id": layer_id,
                    "unit_type": "mlp_neuron", "component_name": f"mlp.neuron.{j}",
                    "local_index": j, "read_tensor_id": gate_tid,
                    "write_tensor_id": down_tid,
                    "parent_tensor_ids": [gate_tid, up_tid, down_tid],
                    "hidden_size": arch.hidden_size, "unit_dim": 1,
                    "is_clusterable": True,
                })
                #Read side: row j of gate_proj (in: hidden -> out: intermediate).
                ref_rows.append(_ref(uid, gate_tid, "read", 0, j, j + 1, "gate_row"))
                #Write side: column j of down_proj (in: intermediate -> out: hidden).
                ref_rows.append(_ref(uid, down_tid, "write", 1, j, j + 1, "down_col"))

        #--- Attention heads --------------------------------------------
        if config.include_attention_heads:
            hd = arch.head_dim
            for h in range(arch.num_attention_heads):
                uid = ids.unit_id(layer_id, "attn_head", h)
                unit_rows.append({
                    "unit_id": uid, "model_id": model_id, "layer_id": layer_id,
                    "unit_type": "attn_head", "component_name": f"attn.head.{h}",
                    "local_index": h, "read_tensor_id": q_tid,
                    "write_tensor_id": o_tid,
                    "parent_tensor_ids": [q_tid, o_tid],
                    "hidden_size": arch.hidden_size, "unit_dim": hd,
                    "is_clusterable": True,
                })
                #QK read slice: rows [h*hd, (h+1)*hd) of q_proj.
                ref_rows.append(_ref(uid, q_tid, "qk", 0, h * hd, (h + 1) * hd, "q_head_rows"))
                #OV write slice: cols [h*hd, (h+1)*hd) of o_proj.
                ref_rows.append(_ref(uid, o_tid, "ov", 1, h * hd, (h + 1) * hd, "o_head_cols"))

    library.commit_table("catalog/unit_index.parquet", unit_rows, "catalog", "build-units")
    library.commit_table("catalog/unit_weight_refs.parquet", ref_rows, "catalog", "build-units")
    library.log("build-units", "unit registry built", units=len(unit_rows))
    library.update_artifact_versions("build-units")
    return {"units": len(unit_rows)}


def _layer_tid(layer_id: int, suffix: str) -> str:
    return ids.tensor_id(f"model.layers.{layer_id}.{suffix}")


def _ref(unit_id, tensor_id, slice_type, dimension, start, end, interp):
    return {
        "unit_id": unit_id, "tensor_id": tensor_id, "slice_type": slice_type,
        "dimension": dimension, "start_index": start, "end_index": end,
        "stride": 1, "interpretation": interp,
    }
