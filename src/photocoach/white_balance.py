"""Reproducible sensor-WB tools, with no invented Kelvin calibration."""
import math
from pathlib import Path

import numpy as np

from .imaging import RawOptions, effective_whitebalance, _fingerprint
from .composition import canvas_size, crop_bounds, output_size
from .safe_io import sha256_file


def balance_coordinates(gains):
    r, g, b = gains
    red, blue = math.log2(r/g), math.log2(b/g)
    return (red-blue)/2, (red+blue)/2


def balance_gains(warmth, tint):
    """Positive warmth favors R over B; positive tint favors R/B over G.

    A common scale fits equivalent gains inside RawOptions bounds, without
    clipping individual channels or treating a common multiplier as exposure.
    """
    if not all(math.isfinite(v) for v in (warmth, tint)):
        raise ValueError("White balance adjustments must be finite.")
    gains = (2**(tint+warmth), 1., 2**(tint-warmth))
    low, high = .25/min(gains), 4/max(gains)
    if low > high + 1e-12:
        raise ValueError("This warmth/tint combination exceeds the supported RGB gain range. Reduce either adjustment.")
    scale = min(high, max(low, 1.))
    return tuple(min(4., max(.25, v*scale)) for v in gains)


def sensor_position(size, flip, composition, position):
    """Map a composed, oriented pixel center back to visible sensor coordinates.

    LibRaw flip bits transpose first, then reverse sensor row/column. The same
    inverse rotation/crop geometry as sample_composition is used before this.
    """
    w, h = size
    if flip not in (0, 1, 2, 3, 4, 5, 6, 7):
        raise ValueError("Unsupported RAW orientation for white balance sampling.")
    oriented = (h, w) if flip & 4 else (w, h)
    ow, oh = output_size(oriented, composition)
    px, py = position
    if not (math.isfinite(px) and math.isfinite(py) and 0 <= px < ow and 0 <= py < oh):
        raise ValueError("Choose a neutral patch inside the photo.")
    cl, ct, _, _ = crop_bounds(oriented, composition)
    cw, ch = canvas_size(oriented, composition.angle)
    x, y = cl+px+.5-cw/2, ct+py+.5-ch/2
    a = math.radians(composition.angle)
    sx = math.cos(a)*x + math.sin(a)*y + oriented[0]/2 - .5
    sy = -math.sin(a)*x + math.cos(a)*y + oriented[1]/2 - .5
    col, row = (sy, sx) if flip & 4 else (sx, sy)
    if flip & 2:
        row = h-1-row
    if flip & 1:
        col = w-1-col
    return min(w-1, max(0, round(col))), min(h-1, max(0, round(row)))


def estimate_neutral_gains(patch, colors, black, white, camera_wb):
    """Estimate WB from a small matte neutral CFA patch, not rendered RGB.

    Reject unreliable samples instead of silently discarding clipped channels
    and biasing the estimate. This assumes the chosen object really is neutral.
    """
    camera = effective_whitebalance(camera_wb, RawOptions())
    means = []
    for channel in (0, 1, 2):
        mask = (colors == channel) if channel != 1 else ((colors == 1) | (colors == 3))
        if np.count_nonzero(mask) < 8:
            raise ValueError("The sample contains too few sensor pixels. Choose a larger interior patch.")
        levels = np.asarray(black, np.float64)[colors[mask]]
        values = (patch[mask].astype(np.float64)-levels)/(white-levels)
        if not np.isfinite(values).all() or values.mean() < .008:
            raise ValueError("The sample is too dark for reliable white balance. Choose a brighter neutral patch.")
        if np.any(values >= .95):
            raise ValueError("The sample is near sensor clipping. Choose a less bright neutral patch.")
        if values.std()/values.mean() > .35:
            raise ValueError("The sample is not uniform enough. Choose a flat, evenly lit neutral patch.")
        means.append(float(values.mean()))
    relative = (means[1]/means[0]/camera[0], 1., means[1]/means[2]/camera[2])
    warmth, tint = balance_coordinates(relative)
    return balance_gains(warmth, tint)


def sample_white_balance(state, position):
    """One explicit background read per click; never postprocess or modify RAW."""
    import rawpy
    source = Path(state.source_path).resolve(strict=True)
    before = _fingerprint(source)
    if sha256_file(source) != state.photo_id:
        raise RuntimeError("The source changed. Reopen the photo before sampling white balance.")
    with source.open("rb") as stream, rawpy.imread(stream) as raw:
        effective_whitebalance(raw.camera_whitebalance, RawOptions(),
            color_desc=raw.color_desc.decode("ascii"), num_colors=raw.num_colors)
        if raw.raw_pattern is None or raw.raw_pattern.shape != (2, 2):
            raise ValueError("The white balance picker currently supports Bayer RGBG sensors only.")
        data = raw.raw_image_visible
        colors = raw.raw_colors_visible
        x, y = sensor_position((data.shape[1], data.shape[0]), int(raw.sizes.flip), state.composition, position)
        rect = (slice(max(0, y-16), min(data.shape[0], y+17)),
                slice(max(0, x-16), min(data.shape[1], x+17)))
        gains = estimate_neutral_gains(data[rect], colors[rect], raw.black_level_per_channel,
                                       raw.white_level, raw.camera_whitebalance)
    if _fingerprint(source) != before:
        raise RuntimeError("The source changed while sampling. Reopen the photo.")
    return RawOptions("custom", gains)
