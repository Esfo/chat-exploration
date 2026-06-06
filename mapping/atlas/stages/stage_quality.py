"""Extraction-quality report (Tech Doc 2, Stage E / Page: Extraction Quality).

Turns extraction trust into first-class, queryable artifacts so the dashboard can
tell — before anyone interprets a cluster — whether the data is sound:

  * activation_quality : per-layer activation coverage and bitset saturation
  * graph_quality      : per-layer graph participation (Issue 7 coverage)
  * extraction_warnings: explicit, severity-tagged problems (saturated bitsets,
    sampled graphs, giant clusters, missing stats, low top-event coverage)

Runs after clustering and before index building, reading committed Parquet
directly (no model, no DuckDB views, and no pandas dependency).
"""

from __future__ import annotations

from ..manifest import Library
from ..model_backend import ModelBackend
from . import register


@register("quality-report", 12.5,
          requires=["activations/activation_stats.parquet"],
          produces=["summaries/activation_quality.parquet",
                    "summaries/graph_quality.parquet",
                    "summaries/extraction_warnings.parquet"])
def run(library: Library, backend: ModelBackend, **kwargs):
    import duckdb

    con = duckdb.connect()

    def path(rel):
        return str(library.path(rel))

    def exists(rel):
        return library.path(rel).exists()

    def rows(sql, params):
        """Run a query and return a list of dicts (no pandas)."""
        cur = con.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    warnings: list[dict] = []

    def warn(severity, kind, scope, detail, value=0.0):
        warnings.append({"severity": severity, "kind": kind, "scope": scope,
                         "detail": detail, "value": float(value)})

    #--- activation quality, per layer ----------------------------------
    act_rows = []
    if exists("activations/activation_stats.parquet") and exists("catalog/unit_index.parquet"):
        act_rows = rows(
            """
            SELECT u.layer_id AS layer_id,
                   count(*) AS units,
                   count(a.unit_id) AS units_with_stats,
                   avg(a.activation_rate) AS mean_activation_rate,
                   median(a.activation_rate) AS median_activation_rate,
                   median(a.bitset_density) AS median_bitset_density,
                   max(a.bitset_density) AS max_bitset_density
            FROM read_parquet(?) u
            LEFT JOIN read_parquet(?) a USING (unit_id)
            GROUP BY u.layer_id ORDER BY u.layer_id
            """,
            [path("catalog/unit_index.parquet"), path("activations/activation_stats.parquet")],
        )

        if exists("activations/activation_top_events.parquet"):
            cov = rows(
                """
                SELECT
                  (SELECT count(DISTINCT unit_id) FROM read_parquet(?)) AS with_events,
                  (SELECT count(*) FROM read_parquet(?)) AS total
                """,
                [path("activations/activation_top_events.parquet"),
                 path("catalog/unit_index.parquet")],
            )[0]
            frac = float(cov["with_events"]) / max(float(cov["total"]), 1.0)
            if frac < 0.5:
                warn("warn", "low_top_event_coverage", "activations",
                     f"only {frac:.0%} of units have any top events", frac)

        for r in act_rows:
            d = float(r.get("median_bitset_density") or 0.0)
            lid = int(r["layer_id"])
            if d >= 0.5:
                warn("error", "saturated_bitset", f"layer {lid}",
                     f"median bitset density {d:.2f} >= 0.50; co-firing unreliable", d)
            elif d >= 0.25:
                warn("warn", "high_bitset_density", f"layer {lid}",
                     f"median bitset density {d:.2f} >= 0.25", d)
            missing = int((r.get("units") or 0) - (r.get("units_with_stats") or 0))
            if missing > 0:
                warn("warn", "units_missing_stats", f"layer {lid}",
                     f"{missing} units have no activation stats", missing)
    library.commit_table("summaries/activation_quality.parquet", act_rows,
                         "summaries", "quality-report")

    #--- graph quality, per layer ---------------------------------------
    graph_rows = []
    if exists("graphs/graph_participation.parquet"):
        graph_rows = rows(
            """
            SELECT layer_id, unit_type,
                   count(*) AS units,
                   sum(CASE WHEN included_in_graph THEN 1 ELSE 0 END) AS included,
                   avg(CASE WHEN included_in_graph THEN 1.0 ELSE 0.0 END) AS coverage
            FROM read_parquet(?) GROUP BY layer_id, unit_type ORDER BY layer_id, unit_type
            """,
            [path("graphs/graph_participation.parquet")],
        )
        for r in graph_rows:
            c = float(r.get("coverage") or 1.0)
            if c < 0.999:
                warn("warn", "graph_coverage_loss",
                     f"layer {int(r['layer_id'])}/{r['unit_type']}",
                     f"only {c:.0%} of units in the graph", c)
    if exists("graphs/graph_meta.parquet"):
        meta = rows("SELECT * FROM read_parquet(?)", [path("graphs/graph_meta.parquet")])
        if meta and bool(meta[0].get("graph_sampled")):
            warn("warn", "graph_sampled", "graphs",
                 f"graph used a per-layer cap of {int(meta[0]['per_layer_cap'])}",
                 float(meta[0]["per_layer_cap"]))
    library.commit_table("summaries/graph_quality.parquet", graph_rows,
                         "summaries", "quality-report")

    #--- giant clusters --------------------------------------------------
    if exists("clusters/cluster_index.parquet"):
        giants = int(rows(
            "SELECT count(*) n FROM read_parquet(?) WHERE giant_component_warning",
            [path("clusters/cluster_index.parquet")])[0]["n"])
        if giants > 0:
            warn("warn", "giant_clusters", "clusters",
                 f"{giants} local clusters look like over-merged giants", giants)

    library.commit_table("summaries/extraction_warnings.parquet", warnings,
                         "summaries", "quality-report")
    library.log("quality-report", "extraction quality computed",
                layers=len(act_rows), warnings=len(warnings))
    library.update_artifact_versions("quality-report")
    return {"layers": len(act_rows), "warnings": len(warnings)}
