"""Stage 4 — static alignment graph (possible-flow graph).

If unit A writes in a direction that unit B reads, and A is in an earlier layer
than B, then A gets a directed *candidate* edge to B. This is not proof of real
signal flow; it means the architecture permits the relationship. We compute it
from the read/write direction sketches as a top-k cosine search within a forward
layer window, keeping the graph sparse per the performance rules.
"""

from __future__ import annotations

import numpy as np

from ..manifest import Library
from ..model_backend import ModelBackend
from ..storage import read_parquet, read_zarr_group, read_zarr_str
from . import register


@register("build-static-graph", 4,
          requires=["units/unit_direction_sketches.zarr", "catalog/unit_index.parquet"],
          produces=["graphs/unit_edges_static_alignment.parquet"])
def run(library: Library, backend: ModelBackend, layer_window: int = 4,
        per_layer_cap: int = 2048, **kwargs):
    config = library.config()
    units = read_parquet(library.path("catalog/unit_index.parquet")).to_pylist()
    g = read_zarr_group(library.path("units/unit_direction_sketches.zarr"))
    uids = read_zarr_str(g, "unit_ids")
    read_sk = np.asarray(g["read_sketch"][:])
    write_sk = np.asarray(g["write_sketch"][:])

    layer_of = {u["unit_id"]: u["layer_id"] for u in units}
    pos = {uid: i for i, uid in enumerate(uids)}

    #Group unit indices by layer, capping per-layer to bound the matmuls.
    by_layer: dict[int, list[int]] = {}
    for uid in uids:
        by_layer.setdefault(layer_of[uid], []).append(pos[uid])
    rng = np.random.default_rng(config.sketch_seed)
    for l, idxs in by_layer.items():
        if len(idxs) > per_layer_cap:
            by_layer[l] = list(rng.choice(idxs, per_layer_cap, replace=False))

    edges = []
    layers = sorted(by_layer)
    for src_layer in layers:
        src_idx = np.array(by_layer[src_layer])
        src_w = write_sk[src_idx]
        tgt_idx = [i for tl in layers
                   if src_layer < tl <= src_layer + layer_window
                   for i in by_layer[tl]]
        if not len(src_idx) or not tgt_idx:
            continue
        tgt_idx = np.array(tgt_idx)
        tgt_r = read_sk[tgt_idx]
        #Cosine = dot of normalized sketches.
        sims = src_w @ tgt_r.T  # [n_src, n_tgt]
        k = min(config.edges_top_k, sims.shape[1])
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        for si in range(sims.shape[0]):
            su = uids[src_idx[si]]
            for tj in top[si]:
                score = float(sims[si, tj])
                if score < config.edge_min_score:
                    continue
                tu = uids[tgt_idx[tj]]
                edges.append({
                    "source_unit_id": su, "target_unit_id": tu,
                    "source_layer": src_layer, "target_layer": layer_of[tu],
                    "static_alignment_score": score,
                })

    library.commit_table("graphs/unit_edges_static_alignment.parquet", edges,
                         "graphs", "build-static-graph")
    library.log("build-static-graph", "static alignment graph built", edges=len(edges))
    library.update_artifact_versions("build-static-graph")
    return {"edges": len(edges)}
