"""Stage 7 — internal probe engine.

A probe is a residual-space direction presented to the network to ask "which
units respond to this kind of internal signal?". The full engine (running probe
directions through the live model and reading responses from hooks) is V2 scope.

This V1-compatible implementation builds the probe *catalog* and a static
response approximation: probe directions are derived from random residual vectors
and the principal components of the unit write-direction sketches, and responses
are the cosine alignment between a probe and each unit's read-direction sketch.
That yields the probe tables and the probe-response graph contract so downstream
pipelines and later stages can depend on them now; the response numbers get
sharper when the live-model engine lands.
"""

from __future__ import annotations

import numpy as np

from .. import ids
from ..manifest import Library
from ..model_backend import ModelBackend
from ..storage import read_zarr_group, read_zarr_str
from . import register


@register("run-probes", 7,
          requires=["units/unit_direction_sketches.zarr"],
          produces=["probes/probe_index.parquet", "probes/probe_responses.parquet"],
          version_scope="v2")
def run(library: Library, backend: ModelBackend, n_random: int = 16,
        n_pca: int = 16, top_responses: int = 64, **kwargs):
    config = library.config()
    g = read_zarr_group(library.path("units/unit_direction_sketches.zarr"))
    uids = read_zarr_str(g, "unit_ids")
    read_sk = np.asarray(g["read_sketch"][:])
    write_sk = np.asarray(g["write_sketch"][:])
    dim = read_sk.shape[1] if read_sk.size else config.signature_dim

    rng = np.random.default_rng(config.sketch_seed + 7)
    probes = []  # (probe_id, kind, vector)

    for i in range(n_random):
        v = rng.standard_normal(dim).astype(np.float32)
        probes.append((ids.probe_id("random", i), "random_residual", _norm(v)))

    #PCA probes from write-direction sketches (common internal variation).
    if write_sk.shape[0] > n_pca:
        comps = _top_components(write_sk, n_pca)
        for i, c in enumerate(comps):
            probes.append((ids.probe_id("pca", i), "residual_pca", _norm(c)))

    index_rows, response_rows = [], []
    for pid, kind, vec in probes:
        index_rows.append({
            "probe_id": pid, "probe_kind": kind, "layer_id": -1,
            "source_ref": "", "norm": float(np.linalg.norm(vec)),
        })
        if read_sk.size:
            resp = read_sk @ vec  # cosine-like response per unit
            k = min(top_responses, len(resp))
            top = np.argpartition(-np.abs(resp), k - 1)[:k]
            top = top[np.argsort(-np.abs(resp[top]))]
            for rank, j in enumerate(top):
                response_rows.append({
                    "probe_id": pid, "unit_id": uids[j],
                    "response": float(resp[j]), "response_rank": rank,
                })

    library.commit_table("probes/probe_index.parquet", index_rows, "probes", "run-probes")
    library.commit_table("probes/probe_responses.parquet", response_rows, "probes", "run-probes")
    library.log("run-probes", "probe catalog + static responses built",
                probes=len(probes), responses=len(response_rows))
    library.update_artifact_versions("run-probes")
    return {"probes": len(probes), "responses": len(response_rows)}


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def _top_components(x, k):
    xc = x - x.mean(axis=0, keepdims=True)
    #SVD on the (units x dim) matrix; right singular vectors are the components.
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    return vt[:k]
