"""Optional, thread-safe per-stage render timing for local diagnostics only.

Stage totals are summed worker time across parallel strips, not wall time; with
four strip workers their sum may exceed the job's elapsed time. Passing no
timer keeps every rendering function on its original, untimed behavior.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import threading
import time

STAGES = ("geometry", "tones", "hsl", "curve", "reduction", "quantize", "histogram")


class StageTimer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._totals: dict[str, float] = {}

    def add(self, stage: str, seconds: float) -> None:
        with self._lock:
            self._totals[stage] = self._totals.get(stage, 0.0) + seconds

    @contextmanager
    def stage(self, stage: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(stage, time.perf_counter() - start)

    def milliseconds(self) -> dict[str, float]:
        with self._lock:
            return {name: round(value * 1000, 2) for name, value in self._totals.items()}


def timed(timer: StageTimer | None, stage: str):
    """Context manager that is a no-op when timing is not requested."""
    return nullcontext() if timer is None else timer.stage(stage)
