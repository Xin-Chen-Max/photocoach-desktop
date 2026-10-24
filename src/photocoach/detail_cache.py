"""Worker-owned, bounded native geometry reuse for straightened detail views."""
from __future__ import annotations

import weakref

import numpy as np

from .composition import output_size, sample_composition


GEOMETRY_CACHE_BYTES = 64 * 1024 * 1024


class DetailGeometryCache:
    """Keep one pre-tonal patch, with a small margin for nearby pans.

    Source identity is weak: the patch must not keep an evicted RAW allocation
    alive. Exposure and other tone values deliberately do not enter the key.
    Callers reserve combined-cache space before allocation; large requests use
    the existing strip renderer without retaining a patch.
    """

    def __init__(self):
        self.hits = 0
        self.builds = 0
        self.clear()

    @property
    def retained_bytes(self):
        return 0 if self._pixels is None else self._pixels.nbytes

    def clear(self):
        self._source = None
        self._key = None
        self._rect = None
        self._pixels = None

    def region(self, linear, identity, composition, rect, *, max_bytes, reserve):
        width, height = output_size((linear.shape[1], linear.shape[0]), composition)
        left, top, right, bottom = rect
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError("Invalid detail region.")
        # Unrotated crops are already cheap native array slices.
        limit = min(GEOMETRY_CACHE_BYTES, max_bytes)
        if (composition.angle == 0 or linear.dtype != np.float32
                or (right-left) * (bottom-top) * 12 > limit):
            self.clear()
            return None
        key = (identity, composition)
        if (self._source is not None and self._source() is linear and self._key == key
                and self.retained_bytes <= limit):
            l, t, r, b = self._rect
            if l <= left and t <= top and r >= right and b >= bottom:
                reserve(self.retained_bytes)
                self.hits += 1
                return self._pixels[top-t:bottom-t, left-l:right-l]
        self.clear()
        margin = 128
        while True:
            l, t = max(0, left-margin), max(0, top-margin)
            r, b = min(width, right+margin), min(height, bottom+margin)
            needed = (r-l) * (b-t) * 12
            if needed <= limit:
                break
            margin //= 2
        reserve(needed)
        try:
            pixels = np.empty((b-t, r-l, 3), dtype=np.float32)
            for start in range(t, b, 128):
                end = min(start+128, b)
                pixels[start-t:end-t] = sample_composition(
                    linear, composition, rect=(l, start, r, end))
        except MemoryError:
            # Reuse is optional; the original bounded strip renderer still works.
            return None
        pixels.setflags(write=False)
        self._pixels = pixels
        self._source = weakref.ref(linear)
        self._key = key
        self._rect = (l, t, r, b)
        self.builds += 1
        return pixels[top-t:bottom-t, left-l:right-l]
