"""Stage 6 — capture activations (the empirical core).

Runs the calibration corpus through the model with forward hooks, accumulating
compressed firing summaries per unit and discarding the raw activation tensors.
Produces activation stats, histograms, top events, firing-signature sketches, and
active-position bitsets — the basis for fire-together clustering.
"""

from __future__ import annotations

import time

import numpy as np

from ..capture_accum import GroupAccumulator
from ..manifest import Library
from ..model_backend import ModelBackend
from ..progress import Progress
from ..sketches import fixed_histogram
from ..storage import read_parquet, write_zarr_array
from . import register

_QPOINTS = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]


@register("capture-activations", 6,
          requires=["calibration/token_index.parquet", "catalog/unit_index.parquet"],
          produces=["activations/activation_stats.parquet",
                    "activations/activation_sketches.zarr"])
def run(library: Library, backend: ModelBackend, batch_size: int | None = None,
        hist_bins: int | None = None, **kwargs):
    config = library.config()
    batch_size = batch_size or config.batch_size
    hist_bins = hist_bins or config.activation_hist_bins
    run_id = library.run_id()
    arch = backend.architecture()

    units = read_parquet(library.path("catalog/unit_index.parquet")).to_pylist()
    tokens = read_parquet(library.path("calibration/token_index.parquet")).to_pylist()

    #Reconstruct sequences from the token index.
    sequences: dict[int, list[dict]] = {}
    for row in tokens:
        sequences.setdefault(row["sequence_id"], []).append(row)
    for s in sequences.values():
        s.sort(key=lambda r: r["token_position"])

    #Map units -> array position within each (layer, unit_type) group.
    groups = _build_groups(units, arch, config)

    backend.load()
    tok = backend.tokenizer
    pad_id = tok.pad_token_id or 0
    seq_ids = sorted(sequences)

    total = len(seq_ids)
    n_batches = (total + batch_size - 1) // batch_size
    total_tokens = sum(len(s) for s in sequences.values())
    n_units = sum(len(g["uids"]) for g in groups.values())
    print(f"[capture-activations] {total} sequences / {total_tokens} tokens / "
          f"{n_units} units in {n_batches} batches (batch_size={batch_size}). "
          f"This is the heavy stage; forward passes use all cores via BLAS.",
          flush=True)
    prog = Progress("capture-activations", n_batches)
    tokens_seen = 0

    for start in range(0, total, batch_size):
        batch_seq_ids = seq_ids[start:start + batch_size]
        input_ids, attn_mask, meta = _build_batch(sequences, batch_seq_ids, pad_id)
        batch_tokens = sum(m[3] for m in meta)

        def on_layer(layer_id, captured, _meta=meta):
            _accumulate_layer(groups, layer_id, captured, _meta)

        #The backend owns torch conversion/device placement (see ModelBackend).
        backend.capture(input_ids, attn_mask, on_layer)

        tokens_seen += batch_tokens
        elapsed = time.time() - prog.start
        tok_rate = tokens_seen / elapsed if elapsed else 0.0
        prog.tick(extra=f"{tokens_seen}/{total_tokens} tokens  ({tok_rate:.0f} tok/s)")

    prog.done(extra=f"{tokens_seen} tokens")
    print("[capture-activations] writing compressed summaries to disk…", flush=True)
    _write_outputs(library, groups, run_id, hist_bins)
    library.log("capture-activations", "activation capture complete",
                sequences=len(seq_ids))
    library.update_artifact_versions("capture-activations")
    return {"sequences": len(seq_ids)}


def _build_groups(units, arch, config):
    """Create a GroupAccumulator per (layer, unit_type) and unit-id ordering."""
    groups = {}
    order: dict[tuple, list[str]] = {}
    for u in units:
        key = (u["layer_id"], u["unit_type"])
        order.setdefault(key, []).append(u["unit_id"])
    for key, uids in order.items():
        layer_id, unit_type = key
        n = len(uids)
        groups[key] = {
            "uids": uids,
            "index": {uid: i for i, uid in enumerate(uids)},
            "acc": GroupAccumulator(
                num_units=n, sig_dim=config.signature_dim,
                n_bits=config.bitset_positions, top_k=config.top_events_per_unit,
                reservoir=config.reservoir_size, active_quantile=config.active_quantile,
                seed=config.sketch_seed + layer_id,
            ),
        }
    return groups


def _build_batch(sequences, batch_seq_ids, pad_id):
    maxlen = max(len(sequences[s]) for s in batch_seq_ids)
    input_ids = np.full((len(batch_seq_ids), maxlen), pad_id, dtype=np.int64)
    attn = np.zeros((len(batch_seq_ids), maxlen), dtype=np.int64)
    meta = []  # per (row, col): (seq_id, position, token_id) for valid tokens
    for r, s in enumerate(batch_seq_ids):
        rows = sequences[s]
        for c, tokrow in enumerate(rows):
            input_ids[r, c] = tokrow["token_id"]
            attn[r, c] = 1
        meta.append((s, [tr["token_position"] for tr in rows],
                     [tr["token_id"] for tr in rows], len(rows)))
    return input_ids, attn, meta


def _accumulate_layer(groups, layer_id, captured, meta):
    """Flatten valid tokens for this layer and feed each unit-type accumulator."""
    for unit_type, key_name in (("mlp_neuron", "mlp"), ("attn_head", "attn_head_norm")):
        key = (layer_id, unit_type)
        if key not in groups or key_name not in captured:
            continue
        arr = captured[key_name]  # [B, T, num_units]
        flat, seqs, poss, toks = _flatten_valid(arr, meta)
        if flat.shape[0]:
            groups[key]["acc"].update(flat, seqs, poss, toks)


def _flatten_valid(arr, meta):
    rows_vals, seqs, poss, toks = [], [], [], []
    for r, (seq_id, positions, token_ids, length) in enumerate(meta):
        rows_vals.append(arr[r, :length, :])
        seqs.extend([seq_id] * length)
        poss.extend(positions)
        toks.extend(token_ids)
    if not rows_vals:
        return np.zeros((0, arr.shape[-1])), np.array([]), np.array([]), np.array([])
    return (np.concatenate(rows_vals, axis=0),
            np.array(seqs, np.int64), np.array(poss, np.int64), np.array(toks, np.int64))


def _write_outputs(library, groups, run_id, hist_bins):
    stat_rows, hist_rows, event_rows = [], [], []
    all_uids, all_sigs, all_bitsets, all_quants = [], [], [], []

    #Resolve token text/corpus from the token index for top events.
    tok_lookup = _token_lookup(library)

    for (layer_id, unit_type), g in groups.items():
        acc = g["acc"]
        uids = g["uids"]
        mean = acc.mean(); std = acc.std(); arate = acc.activation_rate()
        spec = acc.specificity(); burst = acc.burstiness()
        quants = acc.quantiles(_QPOINTS)
        sig = acc.signature_normalized()

        for i, uid in enumerate(uids):
            stat_rows.append({
                "unit_id": uid, "calibration_run_id": run_id,
                "observed_tokens": int(acc.count),
                "active_count": int(acc.active_count[i]),
                "activation_rate": float(arate[i]),
                "mean": float(mean[i]), "std": float(std[i]),
                "min": float(acc.vmin[i]) if np.isfinite(acc.vmin[i]) else 0.0,
                "max": float(acc.vmax[i]) if np.isfinite(acc.vmax[i]) else 0.0,
                "quantiles": quants[i].tolist(),
                "skewness": 0.0, "kurtosis": 0.0,
                "sparsity_score": float(1.0 - arate[i]),
                "specificity_score": float(spec[i]),
                "burstiness_score": float(burst[i]),
            })
            #Histogram from reservoir samples (approximate distribution).
            res = acc.reservoir[i, : max(acc.reservoir_fill, 1)]
            bl, br, ct = fixed_histogram(res, hist_bins, float(res.min()), float(res.max()))
            hist_rows.append({"unit_id": uid, "bin_left": bl, "bin_right": br, "count": ct})

            for rank in range(acc.top_k):
                val = acc.top_vals[i, rank]
                if not np.isfinite(val) or acc.top_seq[i, rank] < 0:
                    continue
                seq = int(acc.top_seq[i, rank]); pos = int(acc.top_pos[i, rank])
                corpus_item = tok_lookup.get((seq, pos), (-1, ""))
                event_rows.append({
                    "unit_id": uid, "event_rank": rank,
                    "corpus_item_id": corpus_item[0], "sequence_id": seq,
                    "token_position": pos, "token_id": int(acc.top_tok[i, rank]),
                    "token_text": corpus_item[1], "activation_value": float(val),
                })

            all_uids.append(uid)
            all_sigs.append(sig[i])
            all_bitsets.append(acc.bitset[i])
            all_quants.append(quants[i])

    library.commit_table("activations/activation_stats.parquet", stat_rows, "activations", "capture-activations")
    library.commit_table("activations/activation_histograms.parquet", hist_rows, "activations", "capture-activations")
    library.commit_table("activations/activation_top_events.parquet", event_rows, "activations", "capture-activations")

    uid_arr = np.array(all_uids, dtype=object).astype("U64")
    write_zarr_array({
        "unit_ids": uid_arr,
        "signature": np.vstack(all_sigs) if all_sigs else np.zeros((0, 1), np.float32),
    }, library.path("activations/activation_sketches.zarr"))
    write_zarr_array({
        "unit_ids": uid_arr,
        "bitset": np.vstack(all_bitsets) if all_bitsets else np.zeros((0, 1), np.uint8),
    }, library.path("activations/active_bitsets.zarr"))
    write_zarr_array({
        "unit_ids": uid_arr,
        "quantile_points": np.array(_QPOINTS, np.float32),
        "quantile_values": np.vstack(all_quants) if all_quants else np.zeros((0, len(_QPOINTS)), np.float32),
    }, library.path("activations/activation_quantile_sketches.zarr"))

    for rel in ("activations/activation_sketches.zarr", "activations/active_bitsets.zarr",
                "activations/activation_quantile_sketches.zarr"):
        library.register_artifact(rel, "activations", "capture-activations",
                                  row_count=len(all_uids), fmt="zarr")


def _token_lookup(library):
    lookup = {}
    for row in read_parquet(library.path("calibration/token_index.parquet")).to_pylist():
        lookup[(row["sequence_id"], row["token_position"])] = (
            row["corpus_item_id"], row["token_text"])
    return lookup
