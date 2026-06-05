"""Stage 13 — build query indexes.

Creates the access layers downstream pipelines should use first:

  * DuckDB views over every Parquet artifact (the relational query layer),
  * a vector index over unit firing signatures (flat NumPy by default; FAISS if
    available) for "find similar units" queries,
  * a graph adjacency export (combined edges partitioned by source layer) for
    fast upstream/downstream neighborhood traversal,
  * a bitmap index copy of active-position bitsets.

These are treated as rebuildable indexes, not the source of truth — they can be
regenerated from the Parquet/Zarr layers at any time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..manifest import Library
from ..model_backend import ModelBackend
from ..storage import duckdb_connect, read_parquet, read_zarr_group, read_zarr_str, write_parquet
from . import register


@register("build-indexes", 13,
          requires=["catalog/unit_index.parquet"],
          produces=["indexes/atlas.duckdb"])
def run(library: Library, backend: ModelBackend, **kwargs):
    n_views = _build_duckdb(library)
    n_vec = _build_vector_index(library)
    n_adj = _build_graph_adjacency(library)
    _copy_bitmaps(library)

    library.log("build-indexes", "query indexes built",
                duckdb_views=n_views, vectors=n_vec, adjacency_partitions=n_adj)
    library.update_artifact_versions("build-indexes")
    return {"duckdb_views": n_views, "vectors": n_vec}


def _build_duckdb(library: Library) -> int:
    con = duckdb_connect(library.path("indexes/atlas.duckdb"))
    count = 0
    #Create one view per committed Parquet artifact, named by its file stem.
    for parquet in sorted(library.root.rglob("*.parquet")):
        if "indexes/" in str(parquet):
            continue
        view = parquet.stem
        rel = parquet.relative_to(library.root).as_posix()
        try:
            con.execute(
                f'CREATE OR REPLACE VIEW "{view}" AS '
                f"SELECT * FROM read_parquet('{library.root / rel}')"
            )
            count += 1
        except Exception as exc:  # noqa: BLE001 — surface but keep going
            library.log("build-indexes", f"view failed for {rel}: {exc}")
    con.close()
    return count


def _build_vector_index(library: Library) -> int:
    path = library.path("activations/activation_sketches.zarr")
    if not path.exists():
        return 0
    g = read_zarr_group(path)
    uids = np.array(read_zarr_str(g, "unit_ids"))
    sig = np.asarray(g["signature"][:]).astype(np.float32)
    out = library.path("indexes/vector_units")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "unit_ids.npy", uids)
    np.save(out / "signatures.npy", sig)
    try:
        import faiss  # optional acceleration
        index = faiss.IndexFlatIP(sig.shape[1])
        index.add(sig)
        faiss.write_index(index, str(out / "faiss.index"))
    except Exception:  # noqa: BLE001 — flat numpy fallback is fine
        pass
    library.register_artifact("indexes/vector_units", "indexes", "build-indexes",
                              row_count=len(uids), fmt="vector")
    return len(uids)


def _build_graph_adjacency(library: Library) -> int:
    p = library.path("graphs/unit_edges_combined.parquet")
    if not p.exists():
        return 0
    edges = read_parquet(p)
    out = library.path("indexes/graph_adjacency/edges_by_source_layer.parquet")
    #Partitioned by source_layer so neighborhood queries scan one partition.
    write_parquet(edges, out, partition_cols=["source_layer"])
    library.register_artifact("indexes/graph_adjacency", "indexes", "build-indexes",
                              partition_keys=["source_layer"], fmt="parquet")
    layers = set(edges.column("source_layer").to_pylist())
    return len(layers)


def _copy_bitmaps(library: Library) -> None:
    src = library.path("activations/active_bitsets.zarr")
    if not src.exists():
        return
    import shutil
    dst = library.path("indexes/bitmap_indexes/active_bitsets.zarr")
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    library.register_artifact("indexes/bitmap_indexes", "indexes", "build-indexes", fmt="zarr")
