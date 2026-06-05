"""Stage 12 — export fast summary structures.

This is what makes large-scale analysis cheap after extraction. We precompute the
summaries the plan describes so a downstream tool answers small questions by
hitting these tables instead of scanning raw data:

  * histograms by meaningful category (layer, unit type, cluster) with a
    whole-model baseline reference attached to every bin,
  * quantile summaries per (entity, metric),
  * top-k indexes (top downstream clusters per cluster, top activation events
    per unit),
  * materialized cluster views (cluster stats + upstream/downstream counts),
  * baseline comparisons (cluster metric vs model baseline, with z-score/ratio).

The guiding idea from the plan: a histogram of *all* weights is too blunt; a
histogram of a metric *inside a structural category* is meaningful. So we build
the categories first, then export distributions within them.
"""

from __future__ import annotations

import numpy as np

from .. import ids
from ..manifest import Library
from ..model_backend import ModelBackend
from ..sketches import fixed_histogram
from ..storage import read_parquet
from . import register

_METRICS = ["write_norm", "read_norm", "activation_rate", "specificity_score"]
_QPOINTS = [0.05, 0.25, 0.5, 0.75, 0.95]
_BINS = 24


@register("export-summaries", 12,
          requires=["clusters/cluster_stats.parquet", "catalog/unit_index.parquet"],
          produces=["summaries/histogram_bins.parquet",
                    "summaries/materialized_cluster_views.parquet"])
def run(library: Library, backend: ModelBackend, **kwargs):
    units = read_parquet(library.path("catalog/unit_index.parquet")).to_pylist()
    unit_vals = _unit_value_table(library, units)  # unit_id -> {metric: value, layer, type}

    #Group unit IDs by entity for histogram building.
    by_layer, by_type = {}, {}
    for u in units:
        by_layer.setdefault(("layer", str(u["layer_id"])), []).append(u["unit_id"])
        by_type.setdefault(("unit_type", u["unit_type"]), []).append(u["unit_id"])

    membership = read_parquet(library.path("clusters/cluster_membership.parquet")).to_pylist()
    by_cluster = {}
    for m in membership:
        if m["member_type"] == "unit":
            by_cluster.setdefault(("cluster", m["cluster_id"]), []).append(m["member_id"])

    entities = {**by_layer, **by_type, **by_cluster}

    #Global ranges + baseline distribution per metric.
    global_vals = {met: np.array([unit_vals[u][met] for u in unit_vals
                                  if met in unit_vals[u]], dtype=float)
                   for met in _METRICS}

    hist_index, hist_bins, quant_rows, baseline_rows = [], [], [], []

    for (etype, eid), member_ids in entities.items():
        for met in _METRICS:
            vals = np.array([unit_vals[u][met] for u in member_ids
                             if u in unit_vals and met in unit_vals[u]], dtype=float)
            if vals.size == 0:
                continue
            g = global_vals[met]
            lo, hi = float(g.min()), float(g.max())
            hid = ids.histogram_id(etype, eid, met)
            bl, br, ct = fixed_histogram(vals, _BINS, lo, hi)
            hist_index.append({
                "histogram_id": hid, "entity_type": etype, "entity_id": eid,
                "metric_name": met, "bin_count": _BINS,
                "baseline_entity_type": "model", "baseline_entity_id": "all",
            })
            total = max(vals.size, 1)
            for l, r, c in zip(bl, br, ct):
                hist_bins.append({
                    "histogram_id": hid, "entity_type": etype, "entity_id": eid,
                    "metric_name": met, "bin_left": l, "bin_right": r,
                    "count": int(c), "density": c / total,
                    "baseline_entity_type": "model", "baseline_entity_id": "all",
                })
            quant_rows.append({
                "entity_type": etype, "entity_id": eid, "metric_name": met,
                "quantile_points": _QPOINTS,
                "quantile_values": np.quantile(vals, _QPOINTS).tolist(),
            })
            #Baseline comparison: entity mean vs model mean.
            ev, bv = float(vals.mean()), float(g.mean())
            bstd = float(g.std()) or 1.0
            baseline_rows.append({
                "entity_type": etype, "entity_id": eid, "metric_name": met,
                "entity_value": ev, "baseline_entity_type": "model",
                "baseline_entity_id": "all", "baseline_value": bv,
                "z_score": (ev - bv) / bstd, "ratio": ev / bv if bv else 0.0,
            })

    library.commit_table("summaries/histogram_index.parquet", hist_index, "summaries", "export-summaries")
    library.commit_table("summaries/histogram_bins.parquet", hist_bins, "summaries", "export-summaries")
    library.commit_table("summaries/quantile_summaries.parquet", quant_rows, "summaries", "export-summaries")
    library.commit_table("summaries/baseline_comparisons.parquet", baseline_rows, "summaries", "export-summaries")

    topk = _build_topk(library)
    library.commit_table("summaries/topk_index.parquet", topk, "summaries", "export-summaries")

    views = _materialized_views(library)
    library.commit_table("summaries/materialized_cluster_views.parquet", views, "summaries", "export-summaries")

    library.log("export-summaries", "summaries exported",
                histograms=len(hist_index), topk=len(topk), views=len(views))
    library.update_artifact_versions("export-summaries")
    return {"histograms": len(hist_index), "views": len(views)}


def _unit_value_table(library, units):
    table = {u["unit_id"]: {"layer": u["layer_id"], "type": u["unit_type"]} for u in units}
    sp = library.path("units/unit_static_stats.parquet")
    if sp.exists():
        for r in read_parquet(sp).to_pylist():
            table.setdefault(r["unit_id"], {}).update(
                write_norm=r["write_norm"], read_norm=r["read_norm"])
    ap = library.path("activations/activation_stats.parquet")
    if ap.exists():
        for r in read_parquet(ap).to_pylist():
            table.setdefault(r["unit_id"], {}).update(
                activation_rate=r["activation_rate"],
                specificity_score=r["specificity_score"])
    return table


def _build_topk(library):
    rows = []
    #Top downstream clusters per cluster (from cluster_edges).
    ce = library.path("graphs/cluster_edges.parquet")
    if ce.exists():
        by_src = {}
        for e in read_parquet(ce).to_pylist():
            by_src.setdefault(e["source_cluster_id"], []).append(e)
        for src, edges in by_src.items():
            edges.sort(key=lambda e: e["sum_combined_score"], reverse=True)
            for rank, e in enumerate(edges[:16]):
                rows.append({
                    "entity_type": "cluster", "entity_id": src,
                    "relation": "downstream_cluster", "rank": rank,
                    "target_id": e["target_cluster_id"],
                    "score": float(e["sum_combined_score"]),
                })
    #Top activation events per unit (from activation_top_events).
    te = library.path("activations/activation_top_events.parquet")
    if te.exists():
        for e in read_parquet(te).to_pylist():
            if e["event_rank"] < 8:
                rows.append({
                    "entity_type": "unit", "entity_id": e["unit_id"],
                    "relation": "top_activation", "rank": e["event_rank"],
                    "target_id": f"seq{e['sequence_id']}:pos{e['token_position']}",
                    "score": float(e["activation_value"]),
                })
    return rows


def _materialized_views(library):
    idx = {c["cluster_id"]: c for c in read_parquet(library.path("clusters/cluster_index.parquet")).to_pylist()}
    stats = {s["cluster_id"]: s for s in read_parquet(library.path("clusters/cluster_stats.parquet")).to_pylist()}
    up, down = {}, {}
    ce = library.path("graphs/cluster_edges.parquet")
    if ce.exists():
        for e in read_parquet(ce).to_pylist():
            down.setdefault(e["source_cluster_id"], set()).add(e["target_cluster_id"])
            up.setdefault(e["target_cluster_id"], set()).add(e["source_cluster_id"])
    rows = []
    for cid, c in idx.items():
        s = stats.get(cid, {})
        rows.append({
            "cluster_id": cid, "cluster_level": c["cluster_level"],
            "layer_min": c["layer_min"], "layer_max": c["layer_max"],
            "dominant_unit_type": c["dominant_unit_type"],
            "member_count": c["member_count"],
            "mean_activation_rate": s.get("mean_activation_rate", 0.0),
            "mean_write_norm": s.get("mean_write_norm", 0.0),
            "source_score": s.get("source_score", 0.0),
            "sink_score": s.get("sink_score", 0.0),
            "relay_score": s.get("relay_score", 0.0),
            "upstream_cluster_count": len(up.get(cid, ())),
            "downstream_cluster_count": len(down.get(cid, ())),
        })
    return rows
