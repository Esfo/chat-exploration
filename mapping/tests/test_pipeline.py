"""End-to-end pipeline test over the fake backend.

Runs the full V1 stage sequence (plus the static probe stage) against the tiny
deterministic FakeBackend, then asserts that every expected artifact exists, that
committed Parquet tables match their declared schemas, and that the query layer
can traverse clusters and recover tensor slices.
"""

from __future__ import annotations

import pyarrow.parquet as pq

from atlas import stages
from atlas.manifest import Library
from atlas.query import AtlasQuery
from atlas.schemas import SCHEMA_REGISTRY
from atlas.storage import validate_against_schema


STAGE_SEQUENCE = [
    "init", "scan-tensors", "build-units", "static-analysis",
    "build-static-graph", "build-calibration", "capture-activations",
    "run-probes", "build-graphs", "cluster", "score-clusters",
    "plan-interventions", "run-interventions", "export-summaries",
    "quality-report", "export-dashboard-summaries", "export-cluster-views",
    "build-indexes",
]


def test_full_pipeline(tmp_path, fake_config, fake_backend):
    stages.load_all()
    lib_dir = tmp_path / "atlas_library"
    library = Library(lib_dir)

    for name in STAGE_SEQUENCE:
        stage = stages.REGISTRY[name]
        missing = stages.check_prerequisites(library, stage)
        assert not missing, f"{name} missing prerequisites: {missing}"
        if name == "init":
            stage.run(library, fake_backend, config=fake_config)
        else:
            stage.run(library, fake_backend)

    #Manifests exist and are coherent.
    assert library.exists()
    assert library.model_id().startswith("m_")
    assert library.run_id().startswith("run_")

    #Core artifacts exist.
    for rel in [
        "catalog/tensor_index.parquet", "catalog/unit_index.parquet",
        "units/unit_static_stats.parquet", "calibration/token_index.parquet",
        "activations/activation_stats.parquet", "activations/activation_sketches.zarr",
        "graphs/unit_edges_combined.parquet", "clusters/cluster_index.parquet",
        "clusters/cluster_stats.parquet", "summaries/histogram_bins.parquet",
        "summaries/materialized_cluster_views.parquet", "indexes/atlas.duckdb",
    ]:
        assert library.path(rel).exists(), f"missing artifact: {rel}"

    #Every committed Parquet matches its declared schema.
    for rel, schema in SCHEMA_REGISTRY.items():
        p = library.path(rel)
        if not p.exists() or p.is_dir():
            continue
        table = pq.read_table(str(p))
        problems = validate_against_schema(table, schema)
        assert not problems, f"{rel}: {problems}"

    #There is real structure: units, clusters, and edges were produced.
    units = pq.read_table(str(library.path("catalog/unit_index.parquet")))
    assert units.num_rows == 3 * (32 + 4)  # layers * (mlp neurons + heads)
    clusters = pq.read_table(str(library.path("clusters/cluster_index.parquet")))
    assert clusters.num_rows > 0
    combined = pq.read_table(str(library.path("graphs/unit_edges_combined.parquet")))
    assert combined.num_rows > 0


def test_query_layer(tmp_path, fake_config, fake_backend):
    stages.load_all()
    lib_dir = tmp_path / "atlas_library"
    library = Library(lib_dir)
    for name in STAGE_SEQUENCE:
        stage = stages.REGISTRY[name]
        if name == "init":
            stage.run(library, fake_backend, config=fake_config)
        else:
            stage.run(library, fake_backend)

    q = AtlasQuery(lib_dir)
    clusters = pq.read_table(str(library.path("clusters/cluster_index.parquet"))).to_pylist()
    local = [c for c in clusters if c["cluster_level"] == "local"][0]

    #Cluster -> units traceability.
    members = q.units_in_cluster(local["cluster_id"])
    assert members, "cluster has no resolvable units"

    #Unit -> exact tensor slices traceability.
    refs = q.tensor_slices_for_unit(members[0])
    assert refs and all("tensor_id" in r for r in refs)

    #Similarity search returns neighbors.
    sims = q.similar_units(members[0], k=5)
    assert isinstance(sims, list)
