"""Correctness guarantees for the capture accumulator (Tech Doc 2, Issues 2-5)."""

from __future__ import annotations

import numpy as np

from atlas.capture_accum import GroupAccumulator


def _acc(num_units=5, **kw):
    return GroupAccumulator(num_units=num_units, sig_dim=16, n_bits=64, top_k=4,
                            reservoir=8, active_quantile=0.9, seed=1, top_m=4,
                            sig_seed=123, **kw)


def test_signature_is_deterministic_and_batch_order_independent():
    """Issue 3: signatures must not depend on batch order or rerun RNG state."""
    rng = np.random.default_rng(0)
    n_units = 5
    # Two "tokens" with distinct global ids, fed in opposite orders.
    acts = rng.random((6, n_units))
    seqs = np.arange(6, dtype=np.int64)
    pos = np.zeros(6, dtype=np.int64)
    toks = np.arange(6, dtype=np.int64)

    a = _acc(n_units)
    a.update(acts[:3], seqs[:3], pos[:3], toks[:3])
    a.update(acts[3:], seqs[3:], pos[3:], toks[3:])

    b = _acc(n_units)  # reversed batch order
    b.update(acts[3:], seqs[3:], pos[3:], toks[3:])
    b.update(acts[:3], seqs[:3], pos[:3], toks[:3])

    assert np.allclose(a.signature, b.signature, atol=1e-9)
    # And a fresh run reproduces it exactly.
    c = _acc(n_units)
    c.update(acts, seqs, pos, toks)
    assert np.allclose(a.signature, c.signature, atol=1e-9)


def test_per_unit_threshold_tracks_unit_scale():
    """Issue 2: a high-scale unit should get a higher active threshold than a
    low-scale unit (a single global constant could not do this)."""
    n = 200
    acts = np.zeros((n, 2))
    acts[:, 0] = np.linspace(0, 1, n)      # low scale
    acts[:, 1] = np.linspace(0, 100, n)    # high scale
    seqs = np.arange(n, dtype=np.int64)
    pos = np.zeros(n, dtype=np.int64)
    a = _acc(2)
    a.update(acts, seqs, pos, seqs)
    thr = a.active_threshold()
    assert thr[1] > 10 * thr[0]  # threshold scales with the unit's own range


def test_reservoir_is_unbiased_sample():
    """Issue 5: reservoir mean should approximate the true mean of a stream."""
    rng = np.random.default_rng(3)
    a = GroupAccumulator(num_units=1, sig_dim=4, n_bits=16, top_k=2, reservoir=200,
                         active_quantile=0.9, seed=7, top_m=2, sig_seed=1)
    stream = rng.normal(5.0, 2.0, 5000)
    a.update(stream[:, None], np.arange(5000, dtype=np.int64),
             np.zeros(5000, np.int64), np.arange(5000, dtype=np.int64))
    res = a.reservoir[0, : a.reservoir_fill]
    assert abs(res.mean() - 5.0) < 0.5  # close to the true mean, not biased


def test_bitset_density_reported():
    a = _acc(3)
    n = 50
    acts = np.random.default_rng(0).random((n, 3))
    a.update(acts, np.arange(n, dtype=np.int64), np.zeros(n, np.int64),
             np.arange(n, dtype=np.int64))
    dens = a.bitset_density()
    assert dens.shape == (3,)
    assert np.all((dens >= 0) & (dens <= 1))
