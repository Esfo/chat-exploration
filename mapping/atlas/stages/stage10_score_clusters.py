"""Stage 10 — cluster statistics, role scores, and cluster-level edges.

Aggregates unit-level evidence to the cluster level and computes the source /
sink / relay / routing role scores the plan calls for, storing them as numeric
fields so downstream pipelines can histogram, threshold, or rank them. Also
builds the cluster-to-cluster edge graph from the combined unit edges.

Role scores (from cluster in/out edge strength):
  * source = out / (in + out)        -> writes signals others use
  * sink   = in  / (in + out)        -> collects/resolves signals
  * relay  = 2*min(in,out)/(in+out)  -> passes/transforms signals through

Coherence:
  * fire_coherence       -> mean alignment of member firing signatures to centroid
  * structural_coherence -> same for static read/write direction sketches
"""

from __future__ import annotations

import numpy as np

from ..manifest import Library
from ..model_backend import ModelBackend
from ..storage import read_parquet, read_zarr_group, read_zarr_str
from . import register


@register("score-clusters", 10,
          requires=["clusters/cluster_index.parquet", "graphs/unit_edges_combined.parquet"],
          produces=["clusters/cluster_stats.parquet", "graphs/cluster_edges.parquet"])
def run(library: Library, backend: ModelBackend, **kwargs):
    clusters = read_parquet(library.path("clusters/cluster_index.parquet")).to_pylist()
    membership = read_parquet(library.path("clusters/cluster_membership.parquet")).to_pylist()

    #unit -> local cluster (members of local clusters are units).
    unit_to_local = {m["member_id"]: m["cluster_id"]
                     for m in membership if m["member_type"] == "unit"}
    #child cluster -> parent cluster (for rolling local stats up the hierarchy).
    members_of = {}
    for m in membership:
        members_of.setdefault(m["cluster_id"], []).append(m["member_id"])

    #--- cluster edges from combined unit edges -------------------------
    combined = read_parquet(library.path("graphs/unit_edges_combined.parquet")).to_pylist()
    routing = _routing_pairs(library)
    cl_edges: dict[tuple, dict] = {}
    out_strength: dict[str, float] = {}
    in_strength: dict[str, float] = {}
    routing_strength: dict[str, float] = {}
    for e in combined:
        cs = unit_to_local.get(e["source_unit_id"])
        ct = unit_to_local.get(e["target_unit_id"])
        if not cs or not ct or cs == ct:
            continue
        score = float(e["combined_score"])
        rec = cl_edges.setdefault((cs, ct), {"edge_count": 0, "sum": 0.0})
        rec["edge_count"] += 1
        rec["sum"] += score
        out_strength[cs] = out_strength.get(cs, 0.0) + score
        in_strength[ct] = in_strength.get(ct, 0.0) + score
        if (e["source_unit_id"], e["target_unit_id"]) in routing:
            routing_strength[cs] = routing_strength.get(cs, 0.0) + score

    edge_rows = [{
        "source_cluster_id": s, "target_cluster_id": t,
        "edge_count": v["edge_count"], "mean_combined_score": v["sum"] / v["edge_count"],
        "sum_combined_score": v["sum"],
    } for (s, t), v in cl_edges.items()]
    library.commit_table("graphs/cluster_edges.parquet", edge_rows, "graphs", "score-clusters")

    #--- per-unit metrics for aggregation -------------------------------
    unit_metrics = _unit_metrics(library)
    fire_coh = _coherence(library, "activations/activation_sketches.zarr", "signature", members_of)
    struct_coh = _coherence(library, "units/unit_direction_sketches.zarr", "write_sketch", members_of)

    #--- stats per cluster (all levels) ---------------------------------
    stat_rows = []
    for c in clusters:
        cid = c["cluster_id"]
        units = _resolve_units(cid, members_of)
        m = _aggregate(units, unit_metrics)
        o = out_strength.get(cid, _rollup(cid, members_of, out_strength))
        i = in_strength.get(cid, _rollup(cid, members_of, in_strength))
        r = routing_strength.get(cid, _rollup(cid, members_of, routing_strength))
        total = o + i
        stat_rows.append({
            "cluster_id": cid, "member_count": len(units),
            "mean_activation_rate": m["activation_rate"],
            "mean_specificity": m["specificity"],
            "mean_read_norm": m["read_norm"], "mean_write_norm": m["write_norm"],
            "fire_coherence": float(fire_coh.get(cid, 0.0)),
            "structural_coherence": float(struct_coh.get(cid, 0.0)),
            "source_score": float(o / total) if total else 0.0,
            "sink_score": float(i / total) if total else 0.0,
            "relay_score": float(2 * min(o, i) / total) if total else 0.0,
            "routing_score": float(r / o) if o else 0.0,
            "causal_confidence": 0.0,
        })

    library.commit_table("clusters/cluster_stats.parquet", stat_rows, "clusters", "score-clusters")
    library.log("score-clusters", "cluster stats computed",
                clusters=len(stat_rows), cluster_edges=len(edge_rows))
    library.update_artifact_versions("score-clusters")
    return {"clusters": len(stat_rows), "cluster_edges": len(edge_rows)}


def _routing_pairs(library):
    p = library.path("graphs/unit_edges_attention_routing.parquet")
    if not p.exists():
        return set()
    return {(r["source_unit_id"], r["target_unit_id"])
            for r in read_parquet(p).to_pylist()}


def _unit_metrics(library):
    metrics: dict[str, dict] = {}
    ap = library.path("activations/activation_stats.parquet")
    if ap.exists():
        for r in read_parquet(ap).to_pylist():
            metrics.setdefault(r["unit_id"], {}).update(
                activation_rate=r["activation_rate"], specificity=r["specificity_score"])
    sp = library.path("units/unit_static_stats.parquet")
    if sp.exists():
        for r in read_parquet(sp).to_pylist():
            metrics.setdefault(r["unit_id"], {}).update(
                read_norm=r["read_norm"], write_norm=r["write_norm"])
    return metrics


def _resolve_units(cid, members_of):
    """Recursively resolve a cluster to its underlying unit IDs."""
    out = []
    stack = list(members_of.get(cid, []))
    while stack:
        m = stack.pop()
        if m.startswith("L") and "." in m:  # unit IDs look like L09.mlp_neuron.000123
            out.append(m)
        else:
            stack.extend(members_of.get(m, []))
    return out


def _aggregate(units, metrics):
    keys = ["activation_rate", "specificity", "read_norm", "write_norm"]
    acc = {k: [] for k in keys}
    for u in units:
        m = metrics.get(u, {})
        for k in keys:
            if k in m:
                acc[k].append(m[k])
    return {k: float(np.mean(v)) if v else 0.0 for k, v in acc.items()}


def _rollup(cid, members_of, strength):
    return sum(_rollup(c, members_of, strength) if c in members_of else strength.get(c, 0.0)
               for c in members_of.get(cid, []))


def _coherence(library, rel, array_name, members_of):
    path = library.path(rel)
    if not path.exists():
        return {}
    g = read_zarr_group(path)
    id_key = "unit_ids" if "unit_ids" in list(g.array_keys()) else "cluster_ids"
    uids = read_zarr_str(g, id_key)
    vecs = np.asarray(g[array_name][:])
    pos = {u: i for i, u in enumerate(uids)}
    out = {}
    for cid, members in members_of.items():
        idx = [pos[m] for m in members if m in pos]
        if len(idx) < 2:
            continue
        v = vecs[idx]
        centroid = v.mean(axis=0)
        n = np.linalg.norm(centroid)
        if n == 0:
            continue
        centroid /= n
        out[cid] = float((v @ centroid).mean())
    return out
