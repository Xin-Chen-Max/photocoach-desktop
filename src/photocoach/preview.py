"""Float-only preview reduction; no JPEG intermediary in the edit pipeline."""
from __future__ import annotations

import numpy as np
from PIL import Image
from dataclasses import dataclass
import math
from concurrent.futures import CancelledError
from concurrent.futures import ThreadPoolExecutor
import os

from .adjustments import render_edit
from .imaging import quantize_srgb
from .state import EditValues
from .composition import output_size, sample_composition
from .render_timing import timed
from .scratch import ScratchPool
from .gradients import RenderGeometry


LINEAR_CACHE_BYTES = 512 * 1024 * 1024
SMALL_RESERVE_BYTES = 1600 * 1600 * 3 * 4
DISPLAY_CACHE_BYTES = 64 * 1024 * 1024

# One image job remains active. Within it, at most four independent native row
# strips can run; never submit the whole image to an unbounded future queue.
_STRIP_WORKERS = min(4, max(1, os.cpu_count() or 1))
_STRIP_POOL = ThreadPoolExecutor(max_workers=_STRIP_WORKERS, thread_name_prefix="photo-strip")


def oriented_size(metadata: dict) -> tuple[int, int]:
    """LibRaw visible_size is unrotated; output_size is already oriented."""
    if metadata.get("half_size") is False:
        return tuple(metadata["output_size"])
    width, height = metadata["visible_size"]
    return (height, width) if int(metadata.get("orientation_flip", 0)) & 4 else (width, height)


@dataclass(frozen=True)
class DetailRequest:
    generation: int
    full_size: tuple[int, int]
    rect: tuple[int, int, int, int]  # exclusive right/bottom, oriented source pixels
    step: int = 1
    source_size: tuple[int, int] | None = None
    adjusting_edge: int | None = None  # Display-only cap, captured at gesture start.


def visible_region(full_size, viewport, center, zoom: float, dpr: float) -> tuple[int, int, int, int]:
    """Viewport dimensions are logical; zoom counts physical pixels/source pixel."""
    width, height = full_size
    half_w, half_h = viewport[0] * dpr / zoom / 2, viewport[1] * dpr / zoom / 2
    return (max(0, math.floor(center[0] - half_w)), max(0, math.floor(center[1] - half_h)),
            min(width, math.ceil(center[0] + half_w)), min(height, math.ceil(center[1] + half_h)))


def render_region(linear: np.ndarray, values: EditValues, request: DetailRequest, *,
                  composed_region: np.ndarray | None = None, cancel=None, workers=None,
                  timer=None) -> np.ndarray:
    """Pointwise full-resolution processing in strips; display-only BOX reduction.

    At 100% (step=1), these are exactly the full export's 8-bit pixels. Smaller
    scales average encoded pixels after editing, never before the tone curve.
    No retained NumPy slice pins the full decoded allocation in the GUI.
    """
    if output_size((linear.shape[1], linear.shape[0]), values.composition) != request.full_size:
        raise ValueError("Full-size RAW dimensions differ from the preview geometry.")
    left, top, right, bottom = request.rect
    step = request.step
    if not (isinstance(step, int) and 1 <= step and 0 <= left < right <= request.full_size[0]
            and 0 <= top < bottom <= request.full_size[1]):
        raise ValueError("Invalid detail region.")
    width, height = right - left, bottom - top
    if composed_region is not None and (composed_region.shape != (height, width, 3)
                                       or composed_region.dtype != np.float32):
        raise ValueError("Cached geometry differs from the requested native region.")
    output_w, output_h = math.ceil(width / step), math.ceil(height / step)
    # Leave room in the display budget for the separate <=1600px quick image.
    if output_w * output_h * 3 + 1600 * 1600 * 3 > DISPLAY_CACHE_BYTES:
        raise MemoryError("The visible detail exceeds the display buffer budget. Resize the window or zoom in.")
    result = np.empty((output_h, output_w, 3), dtype=np.uint8)
    rows = max(step, (128 // step) * step)
    pool = ScratchPool()
    def strip(offset):
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        scratch = pool.for_thread()
        count = min(rows, height - offset)
        with timed(timer, "geometry"):
            region = (sample_composition(linear, values.composition,
                                        rect=(left, top+offset, right, top+offset+count),
                                        scratch=scratch)
                      if composed_region is None else composed_region[offset:offset+count])
        rendered = render_edit(region, values, cancel=cancel, timer=timer, scratch=scratch,
            geometry=RenderGeometry((linear.shape[1],linear.shape[0]),values.composition,
                                    (left,top+offset,right,top+offset+count),(width,count)))
        if step != 1:
            size = (output_w, math.ceil(count / step))
            with timed(timer, "reduction"):
                rendered = np.stack([np.asarray(Image.fromarray(rendered[:, :, channel]).resize(
                    size, Image.Resampling.BOX)) for channel in range(3)], axis=2)
        with timed(timer, "quantize"):
            return offset, count, quantize_srgb(rendered)
    # Small views avoid thread handoff. Identical strips/reduction boundaries
    # preserve exact pixels at every scale, independent of completion order.
    parallel = _STRIP_WORKERS if workers is None else max(1, min(_STRIP_WORKERS, workers))
    if width * height < 2_000_000:
        parallel = 1
    offsets = range(0, height, rows)
    for batch in range(0, len(offsets), parallel):
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        selected = offsets[batch:batch+parallel]
        futures = []
        try:
            if parallel > 1:
                for offset in selected:
                    futures.append(_STRIP_POOL.submit(strip, offset))
            for item in (future.result() for future in futures) if futures else map(strip, selected):
                offset, count, pixels = item
                result[offset // step:(offset + count + step - 1) // step] = pixels
        finally:
            # Do not let obsolete strips outlive this image job or its decoded
            # buffers. A cancelled worker checks between bounded tone blocks.
            for future in futures:
                future.cancel()
            for future in futures:
                if not future.cancelled():
                    try:
                        future.result()
                    except Exception:
                        pass
    result.setflags(write=False)
    return result


def reduce_linear(image: np.ndarray, max_edge: int = 1600) -> np.ndarray:
    if isinstance(max_edge, bool) or not isinstance(max_edge, int) or max_edge < 1:
        raise ValueError("Preview max_edge must be a positive integer.")
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("Preview input must be a nonempty H x W x 3 RGB array.")
    if not np.issubdtype(image.dtype, np.floating) or not np.isfinite(image).all():
        raise ValueError("Preview input must contain finite floating-point RGB values.")
    height, width, _ = image.shape
    factor = min(1.0, max_edge / max(height, width))
    if factor == 1:
        return image
    size = (max(1, round(width * factor)), max(1, round(height * factor)))
    result = np.stack([
        np.asarray(Image.fromarray(image[:, :, channel]).resize(size, Image.Resampling.BOX))
        for channel in range(3)
    ], axis=2).astype(np.float32, copy=False)
    result.setflags(write=False)
    return result
