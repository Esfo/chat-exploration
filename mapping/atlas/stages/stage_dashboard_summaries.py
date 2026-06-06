"""Stage C — export fast dashboard summary artifacts (Tech Doc 2, sections 19-20).

The dashboard must load pages instantly, not act like an analysis notebook. This
stage precomputes the compact summary tables the fixed dashboards read, so
Streamlit never scans millions of edge rows on a page load:

  * library_health     -> per-artifact size/row-count/status (Library Health)
  * layer_summary      -> per-layer aggregates + role scores (Layer Dynamics)
  * cluster_quality    -> cluster coherence percentiles + giant flags (Mechanics)
  * layer_flow_matrix  -> per-evidence layer->layer flow (Signal Flow)
  * evidence_profiles  -> per-edge evidence composition, sampled (Evidence Strength)

Everything is computed in DuckDB straight off the committed Parquet (no pandas,
no model), mirroring the quality-report stage.
"""

from __future__ import annotations

from ..manifest import Library
from ..model_backend import ModelBackend
from . import register

#Evidence columns on unit_edges_combined, in display order.
_EVIDENCE = ["activation", "lagged", "static_alignment", "attention_routing",
             "probe_response", "causal"]
#Cap for the per-edge evidence sample so the dashboard scatter stays fast.
_EVIDENCE_SAMPLE = 200_000


@register("export-dashboard-summaries", 12.7,
          requires=["clusters/cluster_stats.parquet",
                    "graphs/unit_edges_combined.parquet"],
          produces=["summaries/library_health.parquet",
                    "summaries/layer_summary.parquet",
                    "summaries/cluster_quality.parquet",
                    "summaries/layer_flow_matrix.parquet",
                    "summaries/evidence_profiles.parquet"])
def run(library: Library, backend: ModelBackend, **kwargs):
    import duckdb

    con = duckdb.connect()

    def p(rel):
        return str(library.path(rel))

    def has(rel):
        return library.path(rel).exists()

    def commit_arrow(rel, sql, params=None):
        rel_obj = con.execute(sql, params or [])
        to_arrow = getattr(rel_obj, "to_arrow_table", None) or rel_obj.fetch_arrow_table
        tbl = to_arrow()
        library.commit_table(rel, tbl, "summaries", "export-dashboard-summaries")
        return tbl.num_rows

    counts = {}

    #--- library_health: scan the artifact index + file sizes ------------
    counts["library_health"] = _library_health(library)

    #--- layer_summary ---------------------------------------------------
    if has("catalog/unit_index.parquet"):
        astats = (f"LEFT JOIN read_parquet('{p('activations/activation_stats.parquet')}') a USING (unit_id)"
                  if has("activations/activation_stats.parquet") else "")
        sstats = (f"LEFT JOIN read_parquet('{p('units/unit_static_stats.parquet')}') s USING (unit_id)"
                  if has("units/unit_static_stats.parquet") else "")
        #Per-layer cluster aggregates keyed by each cluster's dominant layer.
        clus_join = ""
        if has("clusters/cluster_index.parquet") and has("clusters/cluster_stats.parquet"):
            clus_join = f"""
            LEFT JOIN (
              SELECT ci.dominant_layer AS layer_id,
                     count(*) AS cluster_count,
                     avg(cs.source_score) AS mean_source_score,
                     avg(cs.sink_score) AS mean_sink_score,
                     avg(cs.relay_score) AS mean_relay_score
              FROM read_parquet('{p('clusters/cluster_index.parquet')}') ci
              JOIN read_parquet('{p('clusters/cluster_stats.parquet')}') cs USING (cluster_id)
              WHERE ci.dominant_layer >= 0
              GROUP BY ci.dominant_layer
            ) c ON c.layer_id = u.layer_id"""
        counts["layer_summary"] = commit_arrow(
            "summaries/layer_summary.parquet",
            f"""
            SELECT u.layer_id AS layer_id,
                   count(*) AS unit_count,
                   {"avg(a.activation_rate)" if astats else "NULL"} AS mean_activation_rate,
                   {"median(a.activation_rate)" if astats else "NULL"} AS median_activation_rate,
                   {"quantile_cont(a.activation_rate,0.9)" if astats else "NULL"} AS p90_activation_rate,
                   {"avg(a.specificity_score)" if astats else "NULL"} AS mean_specificity,
                   {"avg(a.burstiness_score)" if astats else "NULL"} AS mean_burstiness,
                   {"avg(s.read_norm)" if sstats else "NULL"} AS mean_read_norm,
                   {"avg(s.write_norm)" if sstats else "NULL"} AS mean_write_norm,
                   {"avg(s.weight_tail_ratio)" if sstats else "NULL"} AS mean_weight_tail_ratio,
                   {"any_value(c.cluster_count)" if clus_join else "NULL"} AS cluster_count,
                   {"any_value(c.mean_source_score)" if clus_join else "NULL"} AS mean_source_score,
                   {"any_value(c.mean_sink_score)" if clus_join else "NULL"} AS mean_sink_score,
                   {"any_value(c.mean_relay_score)" if clus_join else "NULL"} AS mean_relay_score
            FROM read_parquet('{p('catalog/unit_index.parquet')}') u
            {astats} {sstats} {clus_join}
            GROUP BY u.layer_id ORDER BY u.layer_id
            """)

    #--- cluster_quality -------------------------------------------------
    if has("clusters/cluster_stats.parquet") and has("clusters/cluster_index.parquet"):
        counts["cluster_quality"] = commit_arrow(
            "summaries/cluster_quality.parquet",
            f"""
            SELECT cs.cluster_id,
                   ci.cluster_level,
                   ci.dominant_unit_type,
                   ci.dominant_layer,
                   ci.member_count,
                   cs.fire_coherence,
                   cs.structural_coherence,
                   cs.mean_activation_rate,
                   cs.mean_specificity,
                   cs.mean_read_norm,
                   cs.mean_write_norm,
                   cs.source_score, cs.sink_score, cs.relay_score, cs.routing_score,
                   percent_rank() OVER (ORDER BY cs.fire_coherence) AS fire_coherence_percentile,
                   percent_rank() OVER (ORDER BY cs.structural_coherence) AS structural_coherence_percentile,
                   COALESCE(ci.giant_component_warning, FALSE) AS giant_component_warning
            FROM read_parquet('{p('clusters/cluster_stats.parquet')}') cs
            JOIN read_parquet('{p('clusters/cluster_index.parquet')}') ci USING (cluster_id)
            """)

    #--- layer_flow_matrix (one block per evidence type) -----------------
    if has("graphs/unit_edges_combined.parquet"):
        ec = p("graphs/unit_edges_combined.parquet")
        blocks = []
        for ev in ["combined"] + _EVIDENCE:
            col = f"{ev}_score"
            blocks.append(
                f"""SELECT source_layer, target_layer, '{ev}' AS evidence_type,
                        count(*) FILTER (WHERE {col} > 0) AS edge_count,
                        sum({col}) AS total_score,
                        avg(edge_confidence) AS mean_confidence
                    FROM read_parquet('{ec}') GROUP BY source_layer, target_layer""")
        counts["layer_flow_matrix"] = commit_arrow(
            "summaries/layer_flow_matrix.parquet",
            " UNION ALL ".join(blocks) + " ORDER BY evidence_type, source_layer, target_layer")

        #--- evidence_profiles (sampled per-edge composition) ------------
        s = "+".join(f"{e}_score" for e in _EVIDENCE)
        support = "+".join(f"CAST({e}_score>0 AS INT)" for e in _EVIDENCE)
        greatest = "greatest(" + ",".join(f"{e}_score" for e in _EVIDENCE) + ")"
        #entropy over the normalised evidence mix
        ent_terms = "+".join(
            f"CASE WHEN {e}_score>0 AND tot>0 THEN ({e}_score/tot)*ln({e}_score/tot) ELSE 0 END"
            for e in _EVIDENCE)
        dom = "CASE " + " ".join(
            f"WHEN {e}_score = mx THEN '{e}'" for e in _EVIDENCE) + " ELSE 'none' END"
        counts["evidence_profiles"] = commit_arrow(
            "summaries/evidence_profiles.parquet",
            f"""
            WITH base AS (
              SELECT source_unit_id, target_unit_id, source_layer, target_layer,
                     {", ".join(f'{e}_score' for e in _EVIDENCE)},
                     combined_score, edge_confidence,
                     ({s}) AS tot, ({greatest}) AS mx, ({support}) AS support_count
              FROM read_parquet('{ec}') USING SAMPLE {_EVIDENCE_SAMPLE} ROWS (reservoir, 42)
            )
            SELECT *, {dom} AS dominant_evidence_type,
                   CASE WHEN tot>0 THEN -({ent_terms}) ELSE 0 END AS evidence_entropy,
                   abs(static_alignment_score - activation_score) AS disagreement_score,
                   TRUE AS sampled
            FROM base
            """)

    library.log("export-dashboard-summaries", "dashboard summaries exported", **counts)
    library.update_artifact_versions("export-dashboard-summaries")
    return counts


def _library_health(library: Library) -> int:
    """Per-artifact size / row-count / presence, read from the artifact index."""
    from ..storage import read_parquet
    idx_path = library.path("catalog/artifact_index.parquet")
    rows = []

    def size_of(rel):
        p = library.path(rel)
        if not p.exists():
            return 0
        if p.is_file():
            return p.stat().st_size
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

    if idx_path.exists():
        for r in read_parquet(idx_path).to_pylist():
            rel = r["artifact_path"]
            rows.append({
                "artifact_path": rel,
                "artifact_class": r.get("artifact_class", ""),
                "stage": r.get("stage", ""),
                "format": r.get("format", ""),
                "row_count": int(r.get("row_count", -1) or -1),
                "file_size_bytes": int(size_of(rel)),
                "present": library.path(rel).exists(),
            })
    library.commit_table("summaries/library_health.parquet", rows,
                         "summaries", "export-dashboard-summaries")
    return len(rows)
