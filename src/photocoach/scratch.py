"""Worker-local reusable scratch arrays for one bounded render.

A ScratchPool lives only for a single render call; each strip worker thread
gets its own Scratch, so buffers are reused across that worker's strips and are
released when the render returns or is cancelled. This is temporary process
memory, never part of the shared 512 MiB retained linear-cache budget.
"""
from __future__ import annotations

import threading

import numpy as np


class Scratch:
    """Named flat buffers; get() returns a C-contiguous view of the requested shape.

    Callers must treat returned arrays as uninitialized and must not retain
    them beyond the current strip: the next get() with the same name reuses
    the memory.
    """

    def __init__(self) -> None:
        self._buffers: dict[tuple[str, np.dtype], np.ndarray] = {}

    def get(self, name: str, shape, dtype) -> np.ndarray:
        dtype = np.dtype(dtype)
        count = 1
        for dimension in shape:
            count *= int(dimension)
        key = (name, dtype)
        buffer = self._buffers.get(key)
        if buffer is None or buffer.size < count:
            buffer = np.empty(count, dtype)
            self._buffers[key] = buffer
        return buffer[:count].reshape(shape)

    @property
    def nbytes(self) -> int:
        return sum(buffer.nbytes for buffer in self._buffers.values())


class ScratchPool:
    """One Scratch per worker thread for the lifetime of one render."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._scratch: dict[int, Scratch] = {}

    def for_thread(self) -> Scratch:
        ident = threading.get_ident()
        with self._lock:
            scratch = self._scratch.get(ident)
            if scratch is None:
                scratch = self._scratch[ident] = Scratch()
            return scratch

    @property
    def nbytes(self) -> int:
        with self._lock:
            return sum(scratch.nbytes for scratch in self._scratch.values())
