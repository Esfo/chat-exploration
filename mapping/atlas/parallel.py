"""Parallelism helpers: thread configuration and shared-memory thread pools.

The heaviest work in the pipeline is the model forward passes during activation
capture. Those are already parallel *inside* PyTorch — every matmul fans out over
CPU cores via the BLAS/OpenMP backend — so the single most effective speedup is
simply making sure torch and the BLAS libraries are allowed to use every core.
``configure_threads`` does that.

For the CPU-bound NumPy stages (tensor scanning, static unit geometry) the work
is embarrassingly parallel across tensors/units, and NumPy releases the GIL
during its heavy array ops, so a *thread* pool gives real speedup while sharing
the one in-memory model — no extra RAM, unlike process pools that would need a
full copy of the multi-gigabyte model per worker.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")

#Process-wide default worker count, set by configure_threads() so that
#thread_map() honors the CLI --threads choice without threading it through every
#stage signature.
_DEFAULT_WORKERS = 0


def resolve_workers(n: int = 0) -> int:
    """Return a concrete worker/thread count.

    ``n > 0`` is used as-is; ``n == 0`` falls back to the process default set by
    configure_threads (and finally to all cores).
    """
    if n and n > 0:
        return n
    if _DEFAULT_WORKERS and _DEFAULT_WORKERS > 0:
        return _DEFAULT_WORKERS
    return os.cpu_count() or 1


def configure_threads(n: int = 0) -> int:
    """Allow torch + BLAS to use ``n`` cores (0 -> all). Returns the count.

    Sets the BLAS env vars (only if the caller hasn't already, so an explicit
    shell setting wins) and torch's intra-op thread count. For BLAS the env vars
    are most reliable when exported before NumPy is imported — the shell driver
    does that — but torch's runtime setting always takes effect here.
    """
    global _DEFAULT_WORKERS
    threads = resolve_workers(n)
    _DEFAULT_WORKERS = threads
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, str(threads))
    try:
        import torch
        torch.set_num_threads(threads)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(max(1, threads // 2))
            except RuntimeError:
                pass  # already set once this process; harmless
    except Exception:  # noqa: BLE001 — torch optional for pure-data tooling
        pass
    return threads


def thread_map(fn: Callable[[T], R], items: Iterable[T],
               max_workers: int = 0) -> list[R]:
    """Apply ``fn`` across ``items`` on a thread pool, preserving input order.

    Falls back to a plain serial map for trivial sizes so small inputs (and the
    test backend) avoid pool overhead.
    """
    items = list(items)
    workers = resolve_workers(max_workers)
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))
