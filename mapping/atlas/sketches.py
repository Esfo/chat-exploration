"""Compressed summaries: the reason the library can scale.

The plan's large-scale performance rules forbid dense token-by-unit activation
matrices and dense unit-by-unit similarity matrices. Instead we keep, per unit:

  * a streaming statistics accumulator (mean/var/min/max/skew/kurtosis without
    holding the data),
  * a random-projection *signature* for approximate cosine similarity,
  * a compressed *active-position bitset* for co-firing overlap,
  * an approximate *quantile sketch* for percentile queries,
  * a fixed-size histogram.

All of these update incrementally as activations stream through, so memory stays
bounded regardless of how many token positions are observed.
"""

from __future__ import annotations

import numpy as np


class StreamingMoments:
    """Online mean/variance/skew/kurtosis via running central moments.

    Uses the standard incremental formulas so a unit's distribution can be
    summarized over hundreds of thousands of token positions without storing
    them. Tracks min/max as well.
    """

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.M3 = 0.0
        self.M4 = 0.0
        self.min = np.inf
        self.max = -np.inf

    def update_batch(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).ravel()
        for v in x:
            self.update(float(v))

    def update(self, x: float) -> None:
        n1 = self.n
        self.n += 1
        delta = x - self.mean
        delta_n = delta / self.n
        delta_n2 = delta_n * delta_n
        term1 = delta * delta_n * n1
        self.mean += delta_n
        self.M4 += (term1 * delta_n2 * (self.n * self.n - 3 * self.n + 3)
                    + 6 * delta_n2 * self.M2 - 4 * delta_n * self.M3)
        self.M3 += term1 * delta_n * (self.n - 2) - 3 * delta_n * self.M2
        self.M2 += term1
        if x < self.min:
            self.min = x
        if x > self.max:
            self.max = x

    @property
    def variance(self) -> float:
        return self.M2 / self.n if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance))

    @property
    def skewness(self) -> float:
        if self.n < 2 or self.M2 == 0:
            return 0.0
        return float(np.sqrt(self.n) * self.M3 / (self.M2 ** 1.5))

    @property
    def kurtosis(self) -> float:
        if self.n < 2 or self.M2 == 0:
            return 0.0
        return float(self.n * self.M4 / (self.M2 * self.M2) - 3.0)

    def as_dict(self) -> dict[str, float]:
        return {
            "n": self.n,
            "mean": self.mean,
            "std": self.std,
            "min": float(self.min) if self.n else 0.0,
            "max": float(self.max) if self.n else 0.0,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
        }


def random_projection_matrix(in_dim: int, out_dim: int, seed: int) -> np.ndarray:
    """Deterministic Gaussian random-projection matrix (Johnson-Lindenstrauss).

    Cosine similarity is approximately preserved after projecting to ``out_dim``,
    which is how units get comparable fixed-length signatures regardless of the
    model's hidden size.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((in_dim, out_dim)).astype(np.float32) / np.sqrt(out_dim)


class SignatureAccumulator:
    """Accumulates a unit's activation signature via random projection.

    We project the per-token contribution and sum, yielding a vector whose
    direction reflects which token positions the unit fired on. Cosine similarity
    between two units' signatures approximates how often they co-fire.
    """

    def __init__(self, out_dim: int):
        self.vec = np.zeros(out_dim, dtype=np.float64)

    def add(self, projected_row: np.ndarray) -> None:
        self.vec += projected_row

    def finalize(self) -> np.ndarray:
        norm = np.linalg.norm(self.vec)
        if norm == 0:
            return self.vec.astype(np.float32)
        return (self.vec / norm).astype(np.float32)


def simhash(vec: np.ndarray) -> np.ndarray:
    """Sign-bit (SimHash) reduction of a real vector to a packed bit array."""
    bits = (np.asarray(vec) > 0).astype(np.uint8)
    return np.packbits(bits)


class ActiveBitset:
    """Compressed record of which sampled token positions a unit was active on.

    We hash each active (sequence, position) into a fixed-width bit array. Jaccard
    / overlap between two units' bitsets estimates shared active positions — the
    central measurement for fire-together clustering — in O(bits) space.
    """

    def __init__(self, n_bits: int, seed: int):
        self.n_bits = n_bits
        self.seed = seed
        self.bits = np.zeros(n_bits, dtype=np.uint8)

    def mark(self, sequence_id: int, position: int) -> None:
        h = (hash((self.seed, sequence_id, position)) & 0x7FFFFFFF) % self.n_bits
        self.bits[h] = 1

    @property
    def popcount(self) -> int:
        return int(self.bits.sum())

    def packed(self) -> np.ndarray:
        return np.packbits(self.bits)

    @staticmethod
    def overlap(a: "ActiveBitset", b: "ActiveBitset") -> float:
        """Jaccard overlap of two bitsets in [0, 1]."""
        inter = int(np.logical_and(a.bits, b.bits).sum())
        union = int(np.logical_or(a.bits, b.bits).sum())
        return inter / union if union else 0.0


class QuantileSketch:
    """Tiny reservoir-style approximate quantile sketch.

    Keeps a bounded random sample of values; quantiles are read off the sorted
    sample. Adequate for the approximate percentile queries the summary layer
    serves, without the dependency weight of t-digest.
    """

    def __init__(self, capacity: int = 2048, seed: int = 0):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.samples: list[float] = []
        self.count = 0

    def update_batch(self, x: np.ndarray) -> None:
        for v in np.asarray(x, dtype=np.float64).ravel():
            self.update(float(v))

    def update(self, x: float) -> None:
        self.count += 1
        if len(self.samples) < self.capacity:
            self.samples.append(x)
        else:
            j = int(self.rng.integers(0, self.count))
            if j < self.capacity:
                self.samples[j] = x

    def quantiles(self, points: list[float]) -> list[float]:
        if not self.samples:
            return [0.0 for _ in points]
        arr = np.sort(np.array(self.samples))
        return [float(np.quantile(arr, p)) for p in points]


def fixed_histogram(values: np.ndarray, bins: int, lo: float, hi: float):
    """Return (bin_left, bin_right, counts) for a fixed-range histogram."""
    if hi <= lo:
        hi = lo + 1e-6
    counts, edges = np.histogram(np.asarray(values, dtype=np.float64), bins=bins, range=(lo, hi))
    return edges[:-1].tolist(), edges[1:].tolist(), counts.astype(np.int64).tolist()
