"""Latest-only local coaching intent; time is injectable for deterministic tests."""
from __future__ import annotations

import time


class AutoCoachSchedule:
    def __init__(self, *, clock=time.monotonic, idle_seconds=1.2, interval_seconds=10.0):
        self.clock = clock
        self.idle_seconds = idle_seconds
        self.interval_seconds = interval_seconds
        self.pending = None
        self.changed_at = 0.0
        self.last_started = None

    def offer(self, binding):
        self.pending = binding
        self.changed_at = self.clock()

    def cancel(self):
        self.pending = None

    def released(self):
        if self.pending is not None:
            self.changed_at = self.clock()

    def take(self, binding, *, blocked=False):
        if self.pending is None:
            return False
        if self.pending != binding:
            self.cancel()
            return False
        now = self.clock()
        if blocked or now - self.changed_at < self.idle_seconds:
            return False
        if self.last_started is not None and now - self.last_started < self.interval_seconds:
            return False
        self.started()
        return True

    def started(self):
        self.cancel()
        self.last_started = self.clock()
