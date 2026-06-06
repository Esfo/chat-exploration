"""Small, thread-safe progress reporter for long-running stages.

Stages that iterate over many items (tensors, layers, capture batches) use this
to print uniform lines: count, percent, the time the last step took, total
elapsed, and an optional throughput/detail suffix. It is safe to ``tick`` from
worker threads.
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

    def __init__(self, label: str, total: int, every: int = 1, step_label: str = "step"):
        self.label = label
        self.total = max(total, 1)
        self.every = max(every, 1)
        self.step_label = step_label
        self.n = 0
        self.start = time.time()
        self._last = self.start  # time of the previous printed line
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
        now = time.time()
        step = now - self._last  # time taken since the previous printed line
        self._last = now
        elapsed = now - self.start
        pct = 100.0 * self.n / self.total
        msg = (f"[{self.label}] {self.n}/{self.total} ({pct:4.0f}%)  "
               f"{self.step_label} {fmt_duration(step)}  elapsed {fmt_duration(elapsed)}")
        if extra:
            msg += f"  {extra}"
        print(msg, flush=True)
