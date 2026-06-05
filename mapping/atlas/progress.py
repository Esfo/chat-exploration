"""Small, thread-safe progress reporter for long-running stages.

Stages that iterate over many items (tensors, layers, capture batches) use this
to print uniform, informative lines: count, percent, elapsed, ETA, and an
optional throughput/detail suffix. It is safe to ``tick`` from worker threads.
"""

from __future__ import annotations

import threading
import time


def fmt_duration(seconds: float) -> str:
    s = int(max(seconds, 0))
    if s < 60:
        return f"{s}s"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{m}m{sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


class Progress:
    """Thread-safe counter that prints a formatted progress line on each tick."""

    def __init__(self, label: str, total: int, every: int = 1):
        self.label = label
        self.total = max(total, 1)
        self.every = max(every, 1)
        self.n = 0
        self.start = time.time()
        self._lock = threading.Lock()

    def tick(self, inc: int = 1, extra: str = "") -> None:
        """Increment the counter; print only on the configured interval."""
        with self._lock:
            self.n += inc
            if self.n % self.every == 0 or self.n >= self.total:
                self._emit(extra)

    def done(self, extra: str = "") -> None:
        with self._lock:
            elapsed = time.time() - self.start
            msg = f"[{self.label}] complete: {self.n}/{self.total} in {fmt_duration(elapsed)}"
            if extra:
                msg += f"  {extra}"
            print(msg, flush=True)

    def _emit(self, extra: str = "") -> None:
        elapsed = time.time() - self.start
        pct = 100.0 * self.n / self.total
        rate = self.n / elapsed if elapsed else 0.0
        eta = (self.total - self.n) / rate if rate else 0.0
        msg = (f"[{self.label}] {self.n}/{self.total} ({pct:4.0f}%)  "
               f"elapsed {fmt_duration(elapsed)}  ETA {fmt_duration(eta)}")
        if extra:
            msg += f"  {extra}"
        print(msg, flush=True)
