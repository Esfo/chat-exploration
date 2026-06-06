"""Stage 8 — build the unit graphs.

Produces the separate evidence graphs the plan insists on keeping distinct, then
a combined graph that preserves every evidence column plus a weighted combined
score:

  * activation graph     -> units that fire together (signature cosine + bitset)
  * lagged graph         -> earlier unit predicts later unit (forward window)
  * attention routing     -> attention-head source feeding later units
  * static alignment      -> reused from stage 4 (weight geometry)
  * probe response        -> contract table (populated by the V2 probe engine)
  * combined              -> all evidence columns + combined_score + confidence

All graphs are kept sparse via top-k-per-source within a forward/neighbor layer
window, satisfying the "no dense unit-by-unit matrix" rule.
"""

from __future__ import annotations

import numpy as np

from ..manifest import Library
from ..model_backend import ModelBackend
from ..parallel import resolve_workers
from ..progress import Progress
from ..sketches import ActiveBitset
from ..storage import read_parquet, read_zarr_group, read_zarr_str
from . import register

#Per-byte popcount lookup, so bitset overlap is a vectorized table lookup
#instead of an unpackbits allocation per edge.
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)

#Read-only arrays shared with worker processes via fork (copy-on-write), so the
#multi-hundred-MB signature/bitset tables are never pickled per task.
_SHARED: dict = {}


def _layer_edges(src_layer):
    """Compute activation edges for one source layer as columnar arrays.

    Returns (source_pos, target_pos, score, activation_score) — unit *positions*
    (mapped to ids by the parent) plus scores, all NumPy so it pickles cheaply.
    """
    S = _SHARED
    sig, bits, by_layer, layers = S["sig"], S["bits"], S["by_layer"], S["layers"]
    layer_window, top_k = S["layer_window"], S["top_k"]
    min_score, chunk = S["min_score"], S["chunk"]

    src_idx = np.array(by_layer[src_layer])
    if not len(src_idx):
        z = np.empty(0, np.int64)
        return z, z, np.empty(0, np.float64), np.empty(0, np.float64)
    tgt_idx = np.array([i for tl in layers
                        if abs(tl - src_layer) <= layer_window
                        for i in by_layer[tl]])
    tgt_sig = sig[tgt_idx]
    sp_l, tp_l, sc_l, act_l = [], [], [], []
    for c0 in range(0, len(src_idx), chunk):
        cidx = src_idx[c0:c0 + chunk]
        sims = sig[cidx] @ tgt_sig.T  # cosine (signatures are unit-norm)
        k = min(top_k, sims.shape[1])
        #Top-k without negating the whole [chunk x targets] block (saves a large
        #copy per chunk): partition for the k largest, then keep that slice.
        part = np.argpartition(sims, sims.shape[1] - k, axis=1)[:, -k:]
        sc = np.take_along_axis(sims, part, axis=1).ravel()
        sp = np.repeat(cidx, k)
        tp = tgt_idx[part].ravel()
        if bits is not None:
            inter = _POPCOUNT[bits[sp] & bits[tp]].sum(axis=1)
            union = _POPCOUNT[bits[sp] | bits[tp]].sum(axis=1)
            ov = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
        else:
            ov = np.zeros(sp.shape[0])
        keep = (sc >= min_score) & (sp != tp)
        sp, tp, sc, ov = sp[keep], tp[keep], sc[keep], ov[keep]
        sp_l.append(sp); tp_l.append(tp); sc_l.append(sc); act_l.append(0.5 * sc + 0.5 * ov)
    cat = lambda parts, dt: np.concatenate(parts) if parts else np.empty(0, dt)
    return (cat(sp_l, np.int64), cat(tp_l, np.int64),
            cat(sc_l, np.float64), cat(act_l, np.float64))


@register("build-graphs", 8,
          requires=["activations/activation_sketches.zarr", "catalog/unit_index.parquet"],
          produces=["graphs/unit_edges_activation.parquet",
                    "graphs/unit_edges_combined.parquet",
                    "graphs/graph_participation.parquet",
                    "graphs/graph_meta.parquet"])
def run(library: Library, backend: ModelBackend, layer_window: int | None = None,
        per_layer_cap: int | None = None, **kwargs):
    config = library.config()
    layer_window = layer_window if layer_window is not None else config.graph_layer_window
    per_layer_cap = per_layer_cap if per_layer_cap is not None else config.graph_per_layer_cap
    units = read_parquet(library.path("catalog/unit_index.parquet")).to_pylist()
    layer_of = {u["unit_id"]: u["layer_id"] for u in units}
    type_of = {u["unit_id"]: u["unit_type"] for u in units}

    g = read_zarr_group(library.path("activations/activation_sketches.zarr"))
    uids = read_zarr_str(g, "unit_ids")
    sig = np.asarray(g["signature"][:])
    bitsets = _load_bitsets(library)

    pos = {uid: i for i, uid in enumerate(uids)}
    by_layer: dict[int, list[int]] = {}
    for uid in uids:
        by_layer.setdefault(layer_of[uid], []).append(pos[uid])

    #Issue 7: only subsample if an explicit cap is set; record what was included.
    sampled = bool(per_layer_cap and per_layer_cap > 0)
    if sampled:
        rng = np.random.default_rng(config.sketch_seed)
        for l, idxs in by_layer.items():
            if len(idxs) > per_layer_cap:
                by_layer[l] = list(rng.choice(idxs, per_layer_cap, replace=False))
    included = {uids[i] for idxs in by_layer.values() for i in idxs}

    layers = sorted(by_layer)
    chunk = max(1, config.graph_source_chunk)
    bits = bitsets["bits"] if bitsets is not None else None
    min_score = config.edge_min_score

    #Position-indexed lookups so the hot loop stays in NumPy, never per-edge dicts.
    uids_arr = np.asarray(uids, dtype=object)
    layer_pos = np.array([layer_of[u] for u in uids], dtype=np.int32)
    is_attn = np.array([type_of[u] == "attn_head" for u in uids], dtype=bool)

    total_sources = sum(len(v) for v in by_layer.values())
    print(f"[build-graphs] {total_sources} source units across {len(layers)} layers, "
          f"window +/-{layer_window}, top_k={config.edges_top_k}, chunk={chunk}"
          f"{' (sampled)' if sampled else ' (full coverage)'}", flush=True)
    prog = Progress("build-graphs", total_sources, every=chunk, step_label="batch")

    #Accumulate edges as columnar arrays, one task per source layer. Layers are
    #independent, so we fan them out across processes; on Linux fork shares the
    #big sig/bits arrays copy-on-write (no pickling), and each task returns only
    #compact NumPy arrays.
    _SHARED.update(sig=sig, bits=bits, by_layer=by_layer, layers=layers,
                   layer_window=layer_window, top_k=config.edges_top_k,
                   min_score=min_score, chunk=chunk)
    #Only fan out for real workloads; small/test runs stay serial (no fork cost).
    workers = 1 if total_sources < 20000 else min(resolve_workers(), 8, len(layers))
    src_parts, tgt_parts, score_parts, act_parts = [], [], [], []

    def _collect(res, src_layer):
        sp_, tp_, sc_, act_ = res
        src_parts.append(sp_); tgt_parts.append(tp_)
        score_parts.append(sc_); act_parts.append(act_)
        prog.tick(len(by_layer[src_layer]), extra=f"L{src_layer} done")

    if workers <= 1:
        for src_layer in layers:
            _collect(_layer_edges(src_layer), src_layer)
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        ctx = mp.get_context("fork")
        print(f"[build-graphs] fanning {len(layers)} layers across {workers} processes", flush=True)
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = {ex.submit(_layer_edges, l): l for l in layers}
            from concurrent.futures import as_completed
            for fut in as_completed(futs):
                _collect(fut.result(), futs[fut])
    prog.done()

    print("[build-graphs] assembling edge tables…", flush=True)
    sp = np.concatenate(src_parts) if src_parts else np.empty(0, np.int64)
    tp = np.concatenate(tgt_parts) if tgt_parts else np.empty(0, np.int64)
    sc = np.concatenate(score_parts) if score_parts else np.empty(0, np.float64)
    act = np.concatenate(act_parts) if act_parts else np.empty(0, np.float64)
    sl, tl = layer_pos[sp], layer_pos[tp]
    activation_edges = _edge_table(uids_arr, sp, tp, sl, tl, "activation_score", act)
    lag = tl > sl
    lagged_edges = _edge_table(uids_arr, sp[lag], tp[lag], sl[lag], tl[lag],
                               "lagged_score", sc[lag])
    rt = lag & is_attn[sp]
    routing_edges = _edge_table(uids_arr, sp[rt], tp[rt], sl[rt], tl[rt],
                                "attention_routing_score", sc[rt])
    print(f"[build-graphs] {activation_edges.num_rows:,} activation / "
          f"{lagged_edges.num_rows:,} lagged / {routing_edges.num_rows:,} routing edges; "
          f"merging combined graph…", flush=True)

    #Coverage records so the dashboard can show graph honesty (Issue 7).
    participation = [{"unit_id": u["unit_id"], "layer_id": u["layer_id"],
                      "unit_type": u["unit_type"],
                      "included_in_graph": u["unit_id"] in included} for u in units]
    library.commit_table("graphs/graph_participation.parquet", participation,
                         "graphs", "build-graphs")
    library.commit_table("graphs/graph_meta.parquet", [{
        "graph_sampled": sampled,
        "per_layer_cap": int(per_layer_cap or 0),
        "layer_window": int(layer_window),
        "candidate_generation_method": ("random_per_layer_cap" if sampled
                                        else "chunked_exact_topk"),
        "total_units": len(units),
        "included_units": len(included),
    }], "graphs", "build-graphs")

    library.commit_table("graphs/unit_edges_activation.parquet", activation_edges, "graphs", "build-graphs")
    library.commit_table("graphs/unit_edges_lagged.parquet", lagged_edges, "graphs", "build-graphs")
    library.commit_table("graphs/unit_edges_attention_routing.parquet", routing_edges, "graphs", "build-graphs")
    #Probe-response unit-unit edges are produced by the V2 live probe engine;
    #commit the empty contract table so downstream joins always find it.
    library.commit_table("graphs/unit_edges_probe_response.parquet", [], "graphs", "build-graphs")

    combined = _combine(library, config)
    library.commit_table("graphs/unit_edges_combined.parquet", combined, "graphs", "build-graphs")

    library.log("build-graphs", "unit graphs built",
                activation=activation_edges.num_rows, lagged=lagged_edges.num_rows,
                routing=routing_edges.num_rows, combined=combined.num_rows)
    library.update_artifact_versions("build-graphs")
    return {"activation": activation_edges.num_rows, "combined": combined.num_rows}


def _edge(su, tu, sl, tl, field, score):
    return {"source_unit_id": su, "target_unit_id": tu,
            "source_layer": sl, "target_layer": tl, field: score}


def _edge_table(uids_arr, sp, tp, sl, tl, field, score):
    """Build an edge table straight from columnar arrays (no per-edge dicts)."""
    import pyarrow as pa
    return pa.table({
        "source_unit_id": pa.array(uids_arr[sp], type=pa.string()),
        "target_unit_id": pa.array(uids_arr[tp], type=pa.string()),
        "source_layer": pa.array(np.asarray(sl, np.int32)),
        "target_layer": pa.array(np.asarray(tl, np.int32)),
        field: pa.array(np.asarray(score, np.float64)),
    })


def _load_bitsets(library):
    path = library.path("activations/active_bitsets.zarr")
    if not path.exists():
        return None
    g = read_zarr_group(path)
    return {"uids": read_zarr_str(g, "unit_ids"), "bits": np.asarray(g["bitset"][:])}


def _combine(library, config):
    """Merge all evidence edge tables into the combined graph by (src, tgt).

    Done in DuckDB via a single grouped UNION ALL so it scales to the tens of
    millions of edges that full graph coverage (Issue 7) produces, rather than a
    Python dict keyed by every (source, target) pair.
    """
    import duckdb

    specs = [
        ("activation", "graphs/unit_edges_activation.parquet", "activation_score"),
        ("lagged", "graphs/unit_edges_lagged.parquet", "lagged_score"),
        ("static_alignment", "graphs/unit_edges_static_alignment.parquet", "static_alignment_score"),
        ("attention_routing", "graphs/unit_edges_attention_routing.parquet", "attention_routing_score"),
        ("probe_response", "graphs/unit_edges_probe_response.parquet", "probe_response_score"),
    ]
    parts = []
    for name, rel, field in specs:
        p = library.path(rel)
        if not p.exists():
            continue
        parts.append(
            f"SELECT source_unit_id, target_unit_id, source_layer, target_layer, "
            f"'{name}' AS ev, {field} AS score FROM read_parquet('{p}')")
    names = ("activation", "lagged", "static_alignment",
             "attention_routing", "probe_response", "causal")
    if not parts:
        from ..schemas import UNIT_EDGES_COMBINED
        return UNIT_EDGES_COMBINED.empty_table()

    w = config.evidence_weights
    union = " UNION ALL ".join(parts)

    def csum(n):  # summed score for one evidence type
        return f"sum(CASE WHEN ev='{n}' THEN score ELSE 0 END)"

    score_cols = ", ".join(f"{csum(n)} AS {n}_score" for n in names)
    combined_expr = " + ".join(f"{w.get(n, 0.0)} * {csum(n)}" for n in names)
    present_expr = ("count(DISTINCT CASE WHEN score > 0 THEN ev END) / 6.0")

    con = duckdb.connect()
    return con.execute(
        f"""
        SELECT source_unit_id, target_unit_id,
               CAST(max(source_layer) AS INTEGER) AS source_layer,
               CAST(max(target_layer) AS INTEGER) AS target_layer,
               {score_cols},
               {combined_expr} AS combined_score,
               {present_expr} AS edge_confidence
        FROM ({union})
        GROUP BY source_unit_id, target_unit_id
        """
    ).fetch_arrow_table()
