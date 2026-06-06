"""Thin convenience layer over a built library for downstream pipelines.

This is the "query/index layer" entry point the plan describes: it lets a later
process answer the canonical questions — "all units in this cluster", "all
clusters that feed into this cluster", "histogram bins for a category", "the
exact tensor slices for a unit" — without touching the extraction runtime or
reloading the model. Everything here reads the committed Parquet/Zarr/DuckDB
artifacts only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .manifest import Library
from .storage import read_parquet, duckdb_connect


class AtlasQuery:
    def __init__(self, library_dir: str | Path):
        self.library = Library(library_dir)

    #--- relational --------------------------------------------------------
    def duckdb(self):
        """Open the DuckDB connection with all artifact views registered."""
        return duckdb_connect(self.library.path("indexes/atlas.duckdb"))

    #--- cluster traversal -------------------------------------------------
    def units_in_cluster(self, cluster_id: str) -> list[str]:
        members = read_parquet(self.library.path("clusters/cluster_membership.parquet")).to_pylist()
        children = {m["member_id"] for m in members
                    if m["cluster_id"] == cluster_id and m["member_type"] == "unit"}
        if children:
            return sorted(children)
        #Recurse through sub-clusters for xlayer/family clusters.
        by_cluster = {}
        for m in members:
            by_cluster.setdefault(m["cluster_id"], []).append(m)
        out, stack = [], [cluster_id]
        while stack:
            cid = stack.pop()
            for m in by_cluster.get(cid, []):
                if m["member_type"] == "unit":
                    out.append(m["member_id"])
                else:
                    stack.append(m["member_id"])
        return sorted(out)

    def upstream_clusters(self, cluster_id: str) -> list[tuple[str, float]]:
        edges = read_parquet(self.library.path("graphs/cluster_edges.parquet")).to_pylist()
        return sorted([(e["source_cluster_id"], e["sum_combined_score"])
                       for e in edges if e["target_cluster_id"] == cluster_id],
                      key=lambda x: -x[1])

    def downstream_clusters(self, cluster_id: str) -> list[tuple[str, float]]:
        edges = read_parquet(self.library.path("graphs/cluster_edges.parquet")).to_pylist()
        return sorted([(e["target_cluster_id"], e["sum_combined_score"])
                       for e in edges if e["source_cluster_id"] == cluster_id],
                      key=lambda x: -x[1])

    #--- drilling: unit -> exact tensor slices ----------------------------
    def tensor_slices_for_unit(self, unit_id: str) -> list[dict]:
        refs = read_parquet(self.library.path("catalog/unit_weight_refs.parquet")).to_pylist()
        return [r for r in refs if r["unit_id"] == unit_id]

    #--- similarity search -------------------------------------------------
    def similar_units(self, unit_id: str, k: int = 10) -> list[tuple[str, float]]:
        vdir = self.library.path("indexes/vector_units")
        uids = np.load(vdir / "unit_ids.npy", allow_pickle=True)
        sig = np.load(vdir / "signatures.npy")
        pos = {u: i for i, u in enumerate(uids.tolist())}
        if unit_id not in pos:
            return []
        q = sig[pos[unit_id]]
        sims = sig @ q
        order = np.argsort(-sims)[: k + 1]
        return [(str(uids[i]), float(sims[i])) for i in order if str(uids[i]) != unit_id][:k]
