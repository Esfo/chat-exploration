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

Active is defined per *unit* by that unit's own quantile threshold (not one
global constant across units of different scale), and the random projection used
for the firing signature is derived deterministically from each token's global
identity and shared across all groups — so signatures are reproducible across
reruns and comparable across layers.
"""

from __future__ import annotations

import numpy as np

#splitmix64 constants — a fast, well-mixed integer hash used to turn a token's
#global id into deterministic Gaussian projection weights.
_C1 = np.uint64(0x9E3779B97F4A7C15)
_C2 = np.uint64(0xBF58476D1CE4E5B9)
_C3 = np.uint64(0x94D049BB133111EB)
_C4 = np.uint64(0xD1B54A32D192ED03)
_S30, _S27, _S31, _S11 = (np.uint64(30), np.uint64(27), np.uint64(31), np.uint64(11))
_TWO53 = 1.0 / 9007199254740992.0


def _splitmix64(z: np.ndarray) -> np.ndarray:
    """Vectorized splitmix64 over a uint64 array (overflow wraps mod 2**64)."""
    with np.errstate(over="ignore"):
        z = z + _C1
        z = (z ^ (z >> _S30)) * _C2
        z = (z ^ (z >> _S27)) * _C3
        return z ^ (z >> _S31)


def _u01(h: np.ndarray) -> np.ndarray:
    """uint64 hash -> float64 uniform in [0, 1)."""
    return (h >> _S11).astype(np.float64) * _TWO53


class GroupAccumulator:
    """Accumulators for one (layer, unit_type) group of units."""

    def __init__(self, num_units: int, sig_dim: int, n_bits: int,
                 top_k: int, reservoir: int, active_quantile: float, seed: int,
                 top_m: int = 8, sig_seed: int = 0):
        self.num_units = num_units
        self.sig_dim = sig_dim
        self.n_bits = n_bits
        self.top_k = top_k
        self.top_m = max(1, top_m)
        self.reservoir_size = reservoir
        self.active_quantile = active_quantile
        self.sig_seed = np.uint64(sig_seed & 0xFFFFFFFFFFFFFFFF)
        self.rng = np.random.default_rng(seed)

        self.count = 0
        self.sum = np.zeros(num_units, np.float64)
        self.sumsq = np.zeros(num_units, np.float64)
        self.vmin = np.full(num_units, np.inf)
        self.vmax = np.full(num_units, -np.inf)
        self.active_count = np.zeros(num_units, np.int64)
        self.signature = np.zeros((num_units, sig_dim), np.float64)

        #Per-unit active threshold, averaged over batches (Issue 2: thresholds
        #must be per-unit, not one global constant across differently-scaled units).
        self.thr_sum = np.zeros(num_units, np.float64)
        self.thr_batches = 0

        #Packed active bitset: ceil(n_bits/8) bytes per unit.
        self.bitset = np.zeros((num_units, (n_bits + 7) // 8), np.uint8)

        #Reservoir of sampled raw values for quantiles/histogram (Algorithm R).
        self.reservoir = np.zeros((num_units, reservoir), np.float64)
        self.reservoir_fill = 0
        self.seen = 0  # tokens seen, for reservoir replacement probability

        #Top-k events (descending). Parallel metadata arrays.
        self.top_vals = np.full((num_units, top_k), -np.inf)
        self.top_seq = np.full((num_units, top_k), -1, np.int64)
        self.top_pos = np.full((num_units, top_k), -1, np.int64)
        self.top_tok = np.full((num_units, top_k), -1, np.int64)

    def update(self, acts: np.ndarray, seq_ids: np.ndarray, positions: np.ndarray,
               token_ids: np.ndarray, token_ord: np.ndarray | None = None) -> None:
        """Update from a batch. ``acts`` is [N_tokens, num_units].

        ``token_ord`` is an optional dense global token index. When provided (and
        ``n_bits`` >= total tokens) each token maps to its own bit, so the active
        bitset is an *exact* set of fired positions and overlap is exact Jaccard
        with no hash-collision saturation (Issue 1). Without it we fall back to a
        reproducible per-token hash.
        """
        n = acts.shape[0]
        if n == 0:
            return
        acts = acts.astype(np.float64, copy=False)
        self.count += n
        self.sum += acts.sum(axis=0)
        self.sumsq += (acts ** 2).sum(axis=0)
        self.vmin = np.minimum(self.vmin, acts.min(axis=0))
        self.vmax = np.maximum(self.vmax, acts.max(axis=0))

        #Signature: deterministic per-token projection shared across all groups,
        #so it is reproducible and comparable across layers (Issue 3).
        R = self._token_projection(seq_ids, positions)
        self.signature += acts.T @ R

        #Active mask via a *per-unit* quantile threshold within this batch.
        thr = np.quantile(acts, self.active_quantile, axis=0)  # [num_units]
        active = acts > thr[None, :]  # [N, num_units]
        self.active_count += active.sum(axis=0)
        self.thr_sum += thr
        self.thr_batches += 1

        #Bitset marking: map each token to a bit, OR into active units' bytes.
        if token_ord is not None:
            bit_idx = (token_ord.astype(np.int64) % self.n_bits)
        else:
            bit_idx = (np.abs(self._hash_tokens(seq_ids, positions)) % self.n_bits)
        byte_idx = bit_idx // 8
        bit_mask = (1 << (bit_idx % 8)).astype(np.uint8)
        for t in range(n):
            au = active[t]
            if au.any():
                self.bitset[au, byte_idx[t]] |= bit_mask[t]

        self._update_reservoir(acts)

        #Top-k: keep the top-m candidates per unit this batch (Issue 4), merged
        #into the running global top-k. m>1 avoids missing true events when the
        #strongest tokens for a unit cluster in one batch.
        m = min(self.top_m, n)
        part = np.argpartition(-acts, m - 1, axis=0)[:m]  # [m, num_units]
        cols = np.arange(self.num_units)
        for r in range(m):
            rows = part[r]
            self._insert_top(acts[rows, cols], seq_ids[rows], positions[rows],
                             token_ids[rows])

    def _token_projection(self, seq_ids, positions):
        """Deterministic standard-normal projection [N, sig_dim] from token ids."""
        g = (seq_ids.astype(np.uint64) * np.uint64(1000003)
             + positions.astype(np.uint64)) ^ self.sig_seed
        d = np.arange(self.sig_dim, dtype=np.uint64)
        keys = g[:, None] * np.uint64(self.sig_dim) + d[None, :]
        u1 = np.clip(_u01(_splitmix64(keys)), 1e-12, 1.0)
        u2 = _u01(_splitmix64(keys ^ _C4))
        #Box-Muller: two uniforms -> standard normal, fully vectorized.
        return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)

    def _hash_tokens(self, seq_ids, positions):
        #Cheap reproducible per-token hash for bit placement.
        return (seq_ids.astype(np.int64) * 1000003 + positions.astype(np.int64) * 9176)

    def _update_reservoir(self, acts):
        """True reservoir sampling (Algorithm R) at token granularity.

        The same keep/replace decision is applied to every unit, so each unit's
        reservoir is a uniform sample of its values over the *whole* run — unlike
        the previous logic, which replaced a slot every batch and biased quantiles.
        """
        n = acts.shape[0]
        for j in range(n):
            self.seen += 1
            if self.reservoir_fill < self.reservoir_size:
                self.reservoir[:, self.reservoir_fill] = acts[j]
                self.reservoir_fill += 1
            else:
                r = int(self.rng.integers(0, self.seen))
                if r < self.reservoir_size:
                    self.reservoir[:, r] = acts[j]

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

    def active_threshold(self):
        """Per-unit active threshold (mean of the per-batch quantile thresholds)."""
        return self.thr_sum / max(self.thr_batches, 1)

    def bitset_density(self):
        """Fraction of bits set per unit. Near 1.0 means the bitset is saturated
        and co-firing overlap is no longer trustworthy (Issue 1)."""
        pc = np.unpackbits(self.bitset, axis=1).sum(axis=1).astype(np.float64)
        return pc / max(self.n_bits, 1)

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
