"""Stage 8 — build the unit graphs.

Produces the separate evidence graphs the plan insists on keeping distinct, then
a combined graph that preserves every evidence column plus a weighted combined
score:

  * activation graph     -> units that fire together (signature cosine + bitset)
  * lagged graph         -> earlier unit predicts later unit (forward window)
  * attention routing     -> attention-head source feeding later units
  * static alignment      -> reused from stage 4 (weight geometry)
  * probe response        -> contract table (populated by the V2 probe engine)
  * combined              -> all evidence columns + combined_score + confidence

All graphs are kept sparse via top-k-per-source within a forward/neighbor layer
window, satisfying the "no dense unit-by-unit matrix" rule.
"""

from __future__ import annotations

import numpy as np

from ..manifest import Library
from ..model_backend import ModelBackend
from ..sketches import ActiveBitset
from ..storage import read_parquet, read_zarr_group, read_zarr_str
from . import register


@register("build-graphs", 8,
          requires=["activations/activation_sketches.zarr", "catalog/unit_index.parquet"],
          produces=["graphs/unit_edges_activation.parquet",
                    "graphs/unit_edges_combined.parquet"])
def run(library: Library, backend: ModelBackend, layer_window: int = 3,
        per_layer_cap: int = 2048, **kwargs):
    config = library.config()
    units = read_parquet(library.path("catalog/unit_index.parquet")).to_pylist()
    layer_of = {u["unit_id"]: u["layer_id"] for u in units}
    type_of = {u["unit_id"]: u["unit_type"] for u in units}

    g = read_zarr_group(library.path("activations/activation_sketches.zarr"))
    uids = read_zarr_str(g, "unit_ids")
    sig = np.asarray(g["signature"][:])
    bitsets = _load_bitsets(library)

    pos = {uid: i for i, uid in enumerate(uids)}
    by_layer: dict[int, list[int]] = {}
    for uid in uids:
        by_layer.setdefault(layer_of[uid], []).append(pos[uid])
    rng = np.random.default_rng(config.sketch_seed)
    for l, idxs in by_layer.items():
        if len(idxs) > per_layer_cap:
            by_layer[l] = list(rng.choice(idxs, per_layer_cap, replace=False))

    activation_edges, lagged_edges, routing_edges = [], [], []
    layers = sorted(by_layer)

    for src_layer in layers:
        src_idx = np.array(by_layer[src_layer])
        if not len(src_idx):
            continue
        src_sig = sig[src_idx]
        #Neighborhood window includes same and nearby layers (co-firing is local).
        tgt_idx = [i for tl in layers
                   if abs(tl - src_layer) <= layer_window
                   for i in by_layer[tl]]
        tgt_idx = np.array(tgt_idx)
        sims = src_sig @ sig[tgt_idx].T  # cosine (signatures are unit-norm)
        k = min(config.edges_top_k, sims.shape[1])
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]

        for si in range(sims.shape[0]):
            su = uids[src_idx[si]]
            for tj in top[si]:
                tu = uids[tgt_idx[tj]]
                if su == tu:
                    continue
                score = float(sims[si, tj])
                if score < config.edge_min_score:
                    continue
                tl = layer_of[tu]
                overlap = _overlap(bitsets, src_idx[si], tgt_idx[tj])
                act_score = 0.5 * score + 0.5 * overlap
                activation_edges.append(_edge(su, tu, src_layer, tl, "activation_score", act_score))
                if tl > src_layer:
                    lagged_edges.append(_edge(su, tu, src_layer, tl, "lagged_score", score))
                    if type_of[su] == "attn_head":
                        routing_edges.append(_edge(su, tu, src_layer, tl, "attention_routing_score", score))

    library.commit_table("graphs/unit_edges_activation.parquet", activation_edges, "graphs", "build-graphs")
    library.commit_table("graphs/unit_edges_lagged.parquet", lagged_edges, "graphs", "build-graphs")
    library.commit_table("graphs/unit_edges_attention_routing.parquet", routing_edges, "graphs", "build-graphs")
    #Probe-response unit-unit edges are produced by the V2 live probe engine;
    #commit the empty contract table so downstream joins always find it.
    library.commit_table("graphs/unit_edges_probe_response.parquet", [], "graphs", "build-graphs")

    combined = _combine(library, config, layer_of)
    library.commit_table("graphs/unit_edges_combined.parquet", combined, "graphs", "build-graphs")

    library.log("build-graphs", "unit graphs built",
                activation=len(activation_edges), lagged=len(lagged_edges),
                routing=len(routing_edges), combined=len(combined))
    library.update_artifact_versions("build-graphs")
    return {"activation": len(activation_edges), "combined": len(combined)}


def _edge(su, tu, sl, tl, field, score):
    return {"source_unit_id": su, "target_unit_id": tu,
            "source_layer": sl, "target_layer": tl, field: score}


def _load_bitsets(library):
    path = library.path("activations/active_bitsets.zarr")
    if not path.exists():
        return None
    g = read_zarr_group(path)
    return {"uids": read_zarr_str(g, "unit_ids"), "bits": np.asarray(g["bitset"][:])}


def _overlap(bitsets, i, j):
    if bitsets is None:
        return 0.0
    a = np.unpackbits(bitsets["bits"][i])
    b = np.unpackbits(bitsets["bits"][j])
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def _combine(library, config, layer_of):
    """Merge all evidence edge tables into the combined graph by (src, tgt)."""
    files = {
        "activation_score": "graphs/unit_edges_activation.parquet",
        "lagged_score": "graphs/unit_edges_lagged.parquet",
        "static_alignment_score": "graphs/unit_edges_static_alignment.parquet",
        "attention_routing_score": "graphs/unit_edges_attention_routing.parquet",
        "probe_response_score": "graphs/unit_edges_probe_response.parquet",
    }
    merged: dict[tuple, dict] = {}
    for field, rel in files.items():
        p = library.path(rel)
        if not p.exists():
            continue
        for row in read_parquet(p).to_pylist():
            key = (row["source_unit_id"], row["target_unit_id"])
            rec = merged.setdefault(key, {
                "source_unit_id": key[0], "target_unit_id": key[1],
                "source_layer": row["source_layer"], "target_layer": row["target_layer"],
                "activation_score": 0.0, "lagged_score": 0.0,
                "static_alignment_score": 0.0, "attention_routing_score": 0.0,
                "probe_response_score": 0.0, "causal_score": 0.0,
            })
            rec[field] = float(row.get(field, 0.0) or 0.0)

    w = config.evidence_weights
    out = []
    for rec in merged.values():
        combined = sum(w.get(name, 0.0) * rec[f"{name}_score"]
                       for name in ("activation", "lagged", "static_alignment",
                                    "attention_routing", "probe_response", "causal"))
        rec["combined_score"] = float(combined)
        #Confidence rises with the number of independent evidence types present.
        present = sum(1 for name in ("activation", "lagged", "static_alignment",
                                     "attention_routing", "probe_response", "causal")
                      if rec[f"{name}_score"] > 0)
        rec["edge_confidence"] = float(present / 6.0)
        out.append(rec)
    return out
