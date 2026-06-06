"""Stage 9 — fire-together clustering at multiple levels.

Three levels, each kept indexed so analysis can move from fine units to broad
regions, per the plan:

  1. local clusters   -> units within one layer & component type that co-fire
     (connected components of the activation graph, restricted to same
     layer+type so MLP neurons, heads, etc. are not mixed prematurely).
  2. cross-layer clusters -> local clusters linked by forward (lagged) edges
     across depth, revealing structure that persists or relays through the model.
  3. family clusters  -> cross-layer clusters grouped by coarse structural role
     (dominant unit type + layer band). Statistical role scoring is refined in
     stage 10.

Clustering uses union-find on the sparse graphs — no dense pairwise matrix — so
it scales with the number of edges, not the square of the unit count.
"""

from __future__ import annotations

import numpy as np

from .. import ids
from ..manifest import Library
from ..model_backend import ModelBackend
from ..storage import read_parquet, read_zarr_group, read_zarr_str, write_zarr_array
from . import register


class UnionFind:
    def __init__(self):
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def groups(self) -> dict:
        out: dict = {}
        for node in list(self.parent):
            out.setdefault(self.find(node), []).append(node)
        return out


@register("cluster", 9,
          requires=["graphs/unit_edges_activation.parquet", "catalog/unit_index.parquet"],
          produces=["clusters/cluster_index.parquet", "clusters/cluster_membership.parquet"])
def run(library: Library, backend: ModelBackend, **kwargs):
    config = library.config()
    units = read_parquet(library.path("catalog/unit_index.parquet")).to_pylist()
    layer_of = {u["unit_id"]: u["layer_id"] for u in units}
    type_of = {u["unit_id"]: u["unit_type"] for u in units}

    act_edges = read_parquet(library.path("graphs/unit_edges_activation.parquet")).to_pylist()
    lag_path = library.path("graphs/unit_edges_lagged.parquet")
    lag_edges = read_parquet(lag_path).to_pylist() if lag_path.exists() else []

    index_rows, membership_rows, hierarchy_rows, exemplar_rows = [], [], [], []

    #--- level 1: local clusters ----------------------------------------
    #(Issue 6) Mutual-kNN: only union two units when each is in the other's
    #top-k activation neighbours, so weak one-directional bridge edges cannot
    #merge unrelated units into one giant component.
    same_group = (lambda s, t: layer_of[s] == layer_of[t]
                  and type_of[s] == type_of[t])
    directed = set()
    for e in act_edges:
        s, t = e["source_unit_id"], e["target_unit_id"]
        if same_group(s, t) and e.get("activation_score", 1.0) >= config.cluster_edge_min_score:
            directed.add((s, t))

    uf = UnionFind()
    for u in units:
        uf.find(u["unit_id"])  # ensure singletons exist
    for (s, t) in directed:
        if config.mutual_knn and (t, s) not in directed:
            continue  # keep only mutual neighbours
        uf.union(s, t)
    algo = "mutual_knn_components" if config.mutual_knn else "union_find_components"

    #Population per (layer, unit_type) to flag suspiciously giant components.
    pop: dict[tuple, int] = {}
    for u in units:
        pop[(u["layer_id"], u["unit_type"])] = pop.get((u["layer_id"], u["unit_type"]), 0) + 1

    local_of_unit: dict[str, str] = {}
    local_clusters: dict[str, list[str]] = {}
    for i, (_, members) in enumerate(sorted(uf.groups().items())):
        if len(members) < config.min_cluster_size:
            continue
        cid = ids.cluster_id("local", i)
        layers = [layer_of[m] for m in members]
        types = [type_of[m] for m in members]
        dom_type = max(set(types), key=types.count)
        giant = len(members) > config.giant_component_fraction * pop.get(
            (layers[0], dom_type), len(members))
        index_rows.append(_cluster_row(cid, "local", "co_firing", "", layers,
                                       dom_type, len(members), config.local_resolution,
                                       algo=algo, giant=giant))
        for rank, m in enumerate(members):
            membership_rows.append(_member(cid, m, "unit", 1.0, rank))
            local_of_unit[m] = cid
        local_clusters[cid] = members

    #--- centroids + exemplars from activation signatures ---------------
    centroid_ids, centroid_vecs = _centroids_and_exemplars(
        library, local_clusters, exemplar_rows)

    #--- level 2: cross-layer clusters ----------------------------------
    uf2 = UnionFind()
    for cid in local_clusters:
        uf2.find(cid)
    for e in lag_edges:
        s, t = e["source_unit_id"], e["target_unit_id"]
        cs, ct = local_of_unit.get(s), local_of_unit.get(t)
        if cs and ct and cs != ct:
            uf2.union(cs, ct)

    xlayer_of_local: dict[str, str] = {}
    xlayer_clusters: dict[str, list[str]] = {}
    for i, (_, members) in enumerate(sorted(uf2.groups().items())):
        cid = ids.cluster_id("xlayer", i)
        member_layers, member_types = [], []
        for lc in members:
            member_layers += [layer_of[m] for m in local_clusters[lc]]
            member_types += [type_of[m] for m in local_clusters[lc]]
            xlayer_of_local[lc] = cid
        dom_type = max(set(member_types), key=member_types.count) if member_types else ""
        index_rows.append(_cluster_row(cid, "xlayer", "signal_flow", "", member_layers,
                                       dom_type, len(member_layers), config.xlayer_resolution))
        for rank, lc in enumerate(members):
            membership_rows.append(_member(cid, lc, "cluster", 1.0, rank))
            hierarchy_rows.append({"child_cluster_id": lc, "parent_cluster_id": cid,
                                   "child_level": "local", "parent_level": "xlayer"})
        xlayer_clusters[cid] = members

    #--- level 3: family clusters (coarse structural grouping) ----------
    families: dict[tuple, list[str]] = {}
    for xid, members in xlayer_clusters.items():
        all_layers = [layer_of[m] for lc in members for m in local_clusters[lc]]
        all_types = [type_of[m] for lc in members for m in local_clusters[lc]]
        dom_type = max(set(all_types), key=all_types.count) if all_types else ""
        band = int(np.mean(all_layers) // 4) if all_layers else 0
        families.setdefault((dom_type, band), []).append(xid)

    for i, (key, members) in enumerate(sorted(families.items())):
        cid = ids.cluster_id("family", i)
        dom_type, band = key
        member_layers = [layer_of[m] for xid in members
                         for lc in xlayer_clusters[xid] for m in local_clusters[lc]]
        index_rows.append(_cluster_row(cid, "family", f"{dom_type}_band{band}",
                                       "", member_layers, dom_type,
                                       len(member_layers), 0.0))
        for rank, xid in enumerate(members):
            membership_rows.append(_member(cid, xid, "cluster", 1.0, rank))
            hierarchy_rows.append({"child_cluster_id": xid, "parent_cluster_id": cid,
                                   "child_level": "xlayer", "parent_level": "family"})
            #Backfill parent on the xlayer index row.
            for r in index_rows:
                if r["cluster_id"] == xid:
                    r["parent_cluster_id"] = cid

    #Backfill local parents (xlayer).
    for r in index_rows:
        if r["cluster_level"] == "local":
            r["parent_cluster_id"] = xlayer_of_local.get(r["cluster_id"], "")

    library.commit_table("clusters/cluster_index.parquet", index_rows, "clusters", "cluster")
    library.commit_table("clusters/cluster_membership.parquet", membership_rows, "clusters", "cluster")
    library.commit_table("clusters/cluster_hierarchy.parquet", hierarchy_rows, "clusters", "cluster")
    library.commit_table("clusters/cluster_exemplars.parquet", exemplar_rows, "clusters", "cluster")

    if centroid_ids:
        write_zarr_array({
            "cluster_ids": np.array(centroid_ids, dtype=object).astype("U64"),
            "centroid": np.vstack(centroid_vecs),
        }, library.path("clusters/cluster_centroids.zarr"))
        library.register_artifact("clusters/cluster_centroids.zarr", "clusters", "cluster",
                                  row_count=len(centroid_ids), fmt="zarr")

    library.log("cluster", "clustering complete",
                local=len(local_clusters), xlayer=len(xlayer_clusters),
                family=len(families))
    library.update_artifact_versions("cluster")
    return {"local": len(local_clusters), "xlayer": len(xlayer_clusters),
            "family": len(families)}


def _cluster_row(cid, level, ctype, parent, layers, dom_type, member_count, resolution,
                 algo="union_find_components", giant=False):
    return {
        "cluster_id": cid, "cluster_level": level, "cluster_type": ctype,
        "parent_cluster_id": parent,
        "layer_min": int(min(layers)) if layers else -1,
        "layer_max": int(max(layers)) if layers else -1,
        "dominant_layer": int(np.bincount(layers).argmax()) if layers else -1,
        "dominant_unit_type": dom_type, "member_count": int(member_count),
        "clustering_algorithm": algo, "resolution": float(resolution),
        "giant_component_warning": bool(giant),
    }


def _member(cid, member_id, member_type, weight, rank):
    return {"cluster_id": cid, "member_id": member_id, "member_type": member_type,
            "membership_weight": float(weight), "membership_rank": int(rank)}


def _centroids_and_exemplars(library, local_clusters, exemplar_rows):
    path = library.path("activations/activation_sketches.zarr")
    if not path.exists():
        return [], []
    g = read_zarr_group(path)
    uids = read_zarr_str(g, "unit_ids")
    sig = np.asarray(g["signature"][:])
    pos = {u: i for i, u in enumerate(uids)}

    centroid_ids, centroid_vecs = [], []
    for cid, members in local_clusters.items():
        filtered = [m for m in members if m in pos]
        if not filtered:
            continue
        vecs = sig[[pos[m] for m in filtered]]
        centroid = vecs.mean(axis=0)
        n = np.linalg.norm(centroid)
        centroid = centroid / n if n else centroid
        centroid_ids.append(cid)
        centroid_vecs.append(centroid.astype(np.float32))
        #Exemplars = members closest to the centroid (rows align with `filtered`).
        central = vecs @ centroid
        order = np.argsort(-central)[:8]
        for rank, j in enumerate(order):
            exemplar_rows.append({
                "cluster_id": cid, "unit_id": filtered[j],
                "exemplar_rank": rank, "centrality": float(central[j]),
            })
    return centroid_ids, centroid_vecs
