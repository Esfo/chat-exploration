"""Vectorized per-unit accumulators for activation capture.

The naive approach — a Python object per unit updated per token — is far too slow
for a model with hundreds of thousands of units and hundreds of thousands of
token positions. Instead we hold, for each (layer, unit_type) group, a small set
of NumPy arrays shaped ``[num_units, ...]`` and update them with batched matrix
operations. Memory stays bounded by the unit count, never by the token count.

What we keep per unit, all updated in a single forward pass over the corpus:

  * running count / sum / sumsq / min / max  -> mean, std, min, max
  * a firing *signature* (acts.T @ R) for co-firing similarity, where R is a
    shared per-token random projection so two units that fire on the same tokens
    get similar signatures,
  * a reservoir of values for approximate quantiles and the histogram,
  * a compressed active-position bitset for co-firing overlap,
  * the top-k strongest activation events (approximated as one candidate per
    batch per unit, which fills k over many batches).

Active is defined per batch by a layer+component-relative quantile threshold, as
the plan requires (thresholds must not be a single global constant).
"""

from __future__ import annotations

import numpy as np


class GroupAccumulator:
    """Accumulators for one (layer, unit_type) group of units."""

    def __init__(self, num_units: int, sig_dim: int, n_bits: int,
                 top_k: int, reservoir: int, active_quantile: float, seed: int):
        self.num_units = num_units
        self.sig_dim = sig_dim
        self.n_bits = n_bits
        self.top_k = top_k
        self.reservoir_size = reservoir
        self.active_quantile = active_quantile
        self.rng = np.random.default_rng(seed)

        self.count = 0
        self.sum = np.zeros(num_units, np.float64)
        self.sumsq = np.zeros(num_units, np.float64)
        self.vmin = np.full(num_units, np.inf)
        self.vmax = np.full(num_units, -np.inf)
        self.active_count = np.zeros(num_units, np.int64)
        self.signature = np.zeros((num_units, sig_dim), np.float64)

        #Packed active bitset: ceil(n_bits/8) bytes per unit.
        self.bitset = np.zeros((num_units, (n_bits + 7) // 8), np.uint8)

        #Reservoir of sampled raw values for quantiles/histogram.
        self.reservoir = np.zeros((num_units, reservoir), np.float64)
        self.reservoir_fill = 0

        #Top-k events (descending). Parallel metadata arrays.
        self.top_vals = np.full((num_units, top_k), -np.inf)
        self.top_seq = np.full((num_units, top_k), -1, np.int64)
        self.top_pos = np.full((num_units, top_k), -1, np.int64)
        self.top_tok = np.full((num_units, top_k), -1, np.int64)

    def update(self, acts: np.ndarray, seq_ids: np.ndarray, positions: np.ndarray,
               token_ids: np.ndarray) -> None:
        """Update from a batch. ``acts`` is [N_tokens, num_units]."""
        n = acts.shape[0]
        if n == 0:
            return
        self.count += n
        self.sum += acts.sum(axis=0)
        self.sumsq += (acts.astype(np.float64) ** 2).sum(axis=0)
        self.vmin = np.minimum(self.vmin, acts.min(axis=0))
        self.vmax = np.maximum(self.vmax, acts.max(axis=0))

        #Signature: shared random projection over the tokens in this batch.
        R = self.rng.standard_normal((n, self.sig_dim)).astype(np.float64)
        self.signature += acts.T.astype(np.float64) @ R

        #Active mask via layer+component-relative quantile threshold.
        thr = np.quantile(acts, self.active_quantile)
        active = acts > thr  # [N, num_units]
        self.active_count += active.sum(axis=0)

        #Bitset marking: hash each token to a bit, OR into active units' bytes.
        bit_idx = (np.abs(self._hash_tokens(seq_ids, positions)) % self.n_bits)
        byte_idx = bit_idx // 8
        bit_mask = (1 << (bit_idx % 8)).astype(np.uint8)
        for t in range(n):
            au = active[t]
            if au.any():
                self.bitset[au, byte_idx[t]] |= bit_mask[t]

        #Reservoir fill (simple prefix fill then random replacement).
        self._update_reservoir(acts)

        #Top-k: one candidate per batch per unit (the batch-max token per unit).
        bmax_pos = acts.argmax(axis=0)
        bmax_val = acts[bmax_pos, np.arange(self.num_units)]
        self._insert_top(bmax_val, seq_ids[bmax_pos], positions[bmax_pos], token_ids[bmax_pos])

    def _hash_tokens(self, seq_ids, positions):
        #Cheap reproducible per-token hash for bit placement.
        return (seq_ids.astype(np.int64) * 1000003 + positions.astype(np.int64) * 9176)

    def _update_reservoir(self, acts):
        n = acts.shape[0]
        if self.reservoir_fill < self.reservoir_size:
            take = min(n, self.reservoir_size - self.reservoir_fill)
            self.reservoir[:, self.reservoir_fill:self.reservoir_fill + take] = acts[:take].T
            self.reservoir_fill += take
        else:
            #Replace a random slot per unit with a random token from the batch.
            slot = self.rng.integers(0, self.reservoir_size, self.num_units)
            pick = self.rng.integers(0, n, self.num_units)
            self.reservoir[np.arange(self.num_units), slot] = acts[pick, np.arange(self.num_units)]

    def _insert_top(self, vals, seqs, poss, toks):
        #Replace the current smallest kept value where the new value is larger.
        worst = self.top_vals[:, -1]
        better = vals > worst
        if not better.any():
            return
        idx = np.where(better)[0]
        self.top_vals[idx, -1] = vals[idx]
        self.top_seq[idx, -1] = seqs[idx]
        self.top_pos[idx, -1] = poss[idx]
        self.top_tok[idx, -1] = toks[idx]
        order = np.argsort(-self.top_vals[idx], axis=1)
        rows = idx[:, None]
        self.top_vals[idx] = self.top_vals[rows, order]
        self.top_seq[idx] = self.top_seq[rows, order]
        self.top_pos[idx] = self.top_pos[rows, order]
        self.top_tok[idx] = self.top_tok[rows, order]

    #--- finalization -----------------------------------------------------
    def mean(self):
        return self.sum / max(self.count, 1)

    def std(self):
        m = self.mean()
        var = np.maximum(self.sumsq / max(self.count, 1) - m * m, 0.0)
        return np.sqrt(var)

    def activation_rate(self):
        return self.active_count / max(self.count, 1)

    def signature_normalized(self):
        norms = np.linalg.norm(self.signature, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (self.signature / norms).astype(np.float32)

    def quantiles(self, points):
        r = self.reservoir[:, : max(self.reservoir_fill, 1)]
        return np.quantile(r, points, axis=1).T  # [num_units, len(points)]

    def specificity(self):
        """High when a unit fires rarely but sharply: max / (mean+eps)."""
        return self.vmax / (np.abs(self.mean()) + 1e-6)

    def burstiness(self):
        """Coefficient-of-variation style burstiness: std / (mean+eps)."""
        return self.std() / (np.abs(self.mean()) + 1e-6)
