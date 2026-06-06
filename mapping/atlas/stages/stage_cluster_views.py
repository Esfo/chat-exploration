"""Stage D — cluster embeddings + materialized drilldown views (Tech Doc 2 s7-8).

The plan calls cluster visualization "the most important product work". This
stage turns clusters into a navigable space and precomputes everything the
Cluster Landscape and Cluster Drilldown pages need, so those pages render
instantly instead of joining millions of rows live:

  * cluster_embedding_2d      -> 2D behavior-space coordinates per cluster
  * cluster_nearest_neighbors -> top-k nearest clusters in feature space
  * cluster_member_view       -> per (cluster, member unit) with member stats
  * cluster_edge_neighborhoods-> top-k upstream/downstream cluster edges
  * cluster_metric_baselines  -> cluster metrics vs layer/type/global baselines

The embedding uses the cluster activation-signature centroid (when available)
plus standardized behavioral/structural features, reduced to 2D with PCA by
default (deterministic, no extra dependency) or UMAP if installed.
"""

from __future__ import annotations

import numpy as np

from ..manifest import Library
from ..model_backend import ModelBackend
from ..storage import read_parquet, read_zarr_group, read_zarr_str
from . import register

_SCALAR_FEATURES = ["source_score", "sink_score", "relay_score", "routing_score",
                    "fire_coherence", "structural_coherence", "mean_activation_rate",
                    "mean_specificity", "mean_read_norm", "mean_write_norm"]
_KNN = 10


@register("export-cluster-views", 12.8,
          requires=["clusters/cluster_stats.parquet", "clusters/cluster_index.parquet"],
          produces=["summaries/cluster_embedding_2d.parquet",
                    "summaries/cluster_member_view.parquet",
                    "summaries/cluster_edge_neighborhoods.parquet"])
def run(library: Library, backend: ModelBackend, **kwargs):
    config = library.config()
    stats = read_parquet(library.path("clusters/cluster_stats.parquet")).to_pylist()
    index = {r["cluster_id"]: r for r in
             read_parquet(library.path("clusters/cluster_index.parquet")).to_pylist()}
    if not stats:
        library.log("export-cluster-views", "no clusters; nothing to do")
        return {"clusters": 0}

    cids = [s["cluster_id"] for s in stats]
    cpos = {c: i for i, c in enumerate(cids)}

    #--- feature matrix -------------------------------------------------
    scal = np.array([[float(s.get(f) or 0.0) for f in _SCALAR_FEATURES] for s in stats])
    logm = np.log10(np.array([max(int(index.get(c, {}).get("member_count", 1)), 1)
                              for c in cids]))[:, None]
    feats = [_standardize(scal), _standardize(logm)]

    centroids = _load_centroids(library, cpos)
    if centroids is not None:
        #Reduce the 256-d signature centroid to a few PCs and include it so the
        #map reflects behavioral co-firing, not only scalar summaries.
        feats.append(_pca(_standardize(centroids), min(8, centroids.shape[1])))
    X = np.hstack(feats)

    #--- 2D embedding ----------------------------------------------------
    method, xy = _embed_2d(X, config)
    run_id = library.run_id() if hasattr(library, "run_id") else ""
    emb_rows = [{"cluster_id": cids[i], "x": float(xy[i, 0]), "y": float(xy[i, 1]),
                 "embedding_method": method, "embedding_run_id": str(run_id)}
                for i in range(len(cids))]
    library.commit_table("summaries/cluster_embedding_2d.parquet", emb_rows,
                         "summaries", "export-cluster-views")

    #--- nearest neighbours in feature space (chunked) ------------------
    nn_rows = _nearest(cids, X, _KNN)
    library.commit_table("summaries/cluster_nearest_neighbors.parquet", nn_rows,
                         "summaries", "export-cluster-views")

    #--- materialized drilldown views via DuckDB ------------------------
    _duckdb_views(library)

    library.log("export-cluster-views", "cluster views exported",
                clusters=len(cids), embedding=method, neighbors=len(nn_rows))
    library.update_artifact_versions("export-cluster-views")
    return {"clusters": len(cids), "embedding": method}


#--------------------------------------------------------------------------
def _standardize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, np.float64)
    mu = m.mean(axis=0, keepdims=True)
    sd = m.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (m - mu) / sd


def _pca(m: np.ndarray, k: int) -> np.ndarray:
    m = m - m.mean(axis=0, keepdims=True)
    #SVD-based PCA; robust and dependency-free.
    u, s, _ = np.linalg.svd(m, full_matrices=False)
    return u[:, :k] * s[:k]


def _embed_2d(X: np.ndarray, config):
    if X.shape[0] <= 2:
        return "trivial", np.zeros((X.shape[0], 2))
    if not getattr(config, "embedding_disable_umap", False):
        try:
            import umap  # noqa: F401
            reducer = umap.UMAP(n_components=2, random_state=config.sketch_seed)
            return "umap", reducer.fit_transform(X)
        except Exception:  # noqa: BLE001 — umap optional
            pass
    return "pca", _pca(X, 2)


def _load_centroids(library: Library, cpos: dict):
    path = library.path("clusters/cluster_centroids.zarr")
    if not path.exists():
        return None
    g = read_zarr_group(path)
    ids = read_zarr_str(g, "cluster_ids")
    vecs = np.asarray(g["centroid"][:])
    out = np.zeros((len(cpos), vecs.shape[1]))
    for i, cid in enumerate(ids):
        j = cpos.get(cid)
        if j is not None:
            out[j] = vecs[i]
    return out


def _nearest(cids, X, k):
    rows = []
    n = X.shape[0]
    k = min(k, n - 1)
    if k <= 0:
        return rows
    #Cosine similarity in feature space, chunked to bound memory.
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    chunk = 1024
    for c0 in range(0, n, chunk):
        sims = Xn[c0:c0 + chunk] @ Xn.T  # [chunk, n]
        for r in range(sims.shape[0]):
            i = c0 + r
            sims[r, i] = -np.inf  # exclude self
            top = np.argpartition(-sims[r], k - 1)[:k]
            top = top[np.argsort(-sims[r, top])]
            for rank, j in enumerate(top):
                rows.append({"cluster_id": cids[i], "neighbor_cluster_id": cids[j],
                             "rank": rank, "similarity": float(sims[r, j])})
    return rows


def _duckdb_views(library: Library):
    import duckdb
    con = duckdb.connect()

    def p(rel):
        return str(library.path(rel))

    def has(rel):
        return library.path(rel).exists()

    def commit(rel, sql):
        rel_obj = con.execute(sql)
        to_arrow = getattr(rel_obj, "to_arrow_table", None) or rel_obj.fetch_arrow_table
        library.commit_table(rel, to_arrow(), "summaries", "export-cluster-views")

    #Per (cluster, member unit) with member stats joined in.
    if has("clusters/cluster_membership.parquet") and has("catalog/unit_index.parquet"):
        astats = (f"LEFT JOIN read_parquet('{p('activations/activation_stats.parquet')}') a USING (unit_id)"
                  if has("activations/activation_stats.parquet") else "")
        sstats = (f"LEFT JOIN read_parquet('{p('units/unit_static_stats.parquet')}') s USING (unit_id)"
                  if has("units/unit_static_stats.parquet") else "")
        commit("summaries/cluster_member_view.parquet", f"""
            SELECT m.cluster_id, m.member_id AS unit_id, m.membership_rank,
                   u.layer_id, u.unit_type,
                   {"a.activation_rate" if astats else "NULL"} AS activation_rate,
                   {"a.specificity_score" if astats else "NULL"} AS specificity_score,
                   {"s.read_norm" if sstats else "NULL"} AS read_norm,
                   {"s.write_norm" if sstats else "NULL"} AS write_norm
            FROM read_parquet('{p('clusters/cluster_membership.parquet')}') m
            JOIN read_parquet('{p('catalog/unit_index.parquet')}') u ON u.unit_id = m.member_id
            {astats} {sstats}
            WHERE m.member_type = 'unit'
        """)

    #Top-k upstream/downstream cluster edges per cluster.
    if has("graphs/cluster_edges.parquet"):
        ce = p("graphs/cluster_edges.parquet")
        commit("summaries/cluster_edge_neighborhoods.parquet", f"""
            WITH out_e AS (
              SELECT source_cluster_id AS cluster_id, target_cluster_id AS other_cluster_id,
                     'downstream' AS direction, edge_count, mean_combined_score, sum_combined_score,
                     row_number() OVER (PARTITION BY source_cluster_id ORDER BY sum_combined_score DESC) AS rank
              FROM read_parquet('{ce}')),
            in_e AS (
              SELECT target_cluster_id AS cluster_id, source_cluster_id AS other_cluster_id,
                     'upstream' AS direction, edge_count, mean_combined_score, sum_combined_score,
                     row_number() OVER (PARTITION BY target_cluster_id ORDER BY sum_combined_score DESC) AS rank
              FROM read_parquet('{ce}'))
            SELECT * FROM out_e WHERE rank <= 10
            UNION ALL SELECT * FROM in_e WHERE rank <= 10
        """)

    #Per-cluster metrics vs layer / type / global baselines.
    if has("clusters/cluster_stats.parquet") and has("clusters/cluster_index.parquet"):
        commit("summaries/cluster_metric_baselines.parquet", f"""
            WITH c AS (
              SELECT cs.cluster_id, ci.dominant_layer, ci.dominant_unit_type,
                     cs.mean_activation_rate, cs.mean_specificity,
                     cs.mean_read_norm, cs.mean_write_norm
              FROM read_parquet('{p('clusters/cluster_stats.parquet')}') cs
              JOIN read_parquet('{p('clusters/cluster_index.parquet')}') ci USING (cluster_id))
            SELECT c.*,
                   avg(mean_activation_rate) OVER () AS global_activation_rate,
                   avg(mean_specificity) OVER () AS global_specificity,
                   avg(mean_activation_rate) OVER (PARTITION BY dominant_layer) AS layer_activation_rate,
                   avg(mean_specificity) OVER (PARTITION BY dominant_unit_type) AS type_specificity
            FROM c
        """)
