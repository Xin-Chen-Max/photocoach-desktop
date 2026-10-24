"""Immutable eight-band HSL controls and a bounded float32 display-space transform."""
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

from .scratch import Scratch

COLOR_NAMES = ("Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta")
COLOR_KEYS = tuple(name.lower() for name in COLOR_NAMES)
CENTERS = np.array((0, 30, 60, 120, 180, 240, 270, 300, 360), np.float32) / 360
CENTERS_LO = np.ascontiguousarray(CENTERS[:-1])
CENTERS_SPAN = np.ascontiguousarray(CENTERS[1:] - CENTERS[:-1])
CENTERS_INNER = tuple(np.float32(center) for center in CENTERS[1:8])
AXES = ("hue", "saturation", "lightness")
NEUTRAL_BANDS = ((0., 0., 0.),) * 8


@lru_cache(maxsize=64)
def _prepared_bands(bands: tuple[tuple[float, float, float], ...]):
    table = np.array((*bands, bands[0]), np.float32)
    active = []
    for axis in range(3):
        column = table[:, axis]
        if np.any(column):
            active.append((axis, np.ascontiguousarray(column[:-1]), np.ascontiguousarray(column[1:])))
    return tuple(active)


@dataclass(frozen=True)
class ColorMixer:
    bands: tuple[tuple[float, float, float], ...] = NEUTRAL_BANDS

    def __post_init__(self):
        if not isinstance(self.bands, (list, tuple)) or len(self.bands) != 8:
            raise ValueError("Color Mixer requires eight color bands.")
        result = []
        for band in self.bands:
            if not isinstance(band, (list, tuple)) or len(band) != 3:
                raise ValueError("Each color requires Hue, Saturation and Lightness.")
            for value, limit in zip(band, (30, 100, 100)):
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or abs(value) > limit:
                    raise ValueError("Invalid Color Mixer value: Hue ±30°, Saturation/Lightness ±100.")
            result.append(tuple(float(value) for value in band))
        object.__setattr__(self, "bands", tuple(result))

    @property
    def neutral(self):
        return self.bands == NEUTRAL_BANDS

    def with_value(self, color, axis, value):
        bands = [list(band) for band in self.bands]
        bands[color][axis] = value
        return ColorMixer(bands)

    def to_dict(self):
        return {key: dict(zip(AXES, band)) for key, band in zip(COLOR_KEYS, self.bands)}

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) != set(COLOR_KEYS):
            raise ValueError("Invalid Color Mixer colors.")
        for band in data.values():
            if not isinstance(band, dict) or set(band) != set(AXES):
                raise ValueError("Invalid Color Mixer axes.")
        return cls(tuple(tuple(data[key][axis] for axis in AXES) for key in COLOR_KEYS))


def apply_color_mixer(rgb, mixer, *, scratch: Scratch | None = None):
    """Modify one float32 row block in place. Selection uses original encoded hue.

    Cubic smoothstep blends adjacent band centers around the hue circle. Neutral
    protection fades from zero at S=0 to full strength at S=.1. S/L move toward
    available headroom; positive values are not exposure multipliers.
    """
    if mixer.neutral:
        return
    if scratch is None:
        scratch = Scratch()
    shape = rgb.shape[:2]
    red = scratch.get("hsl_r", shape, np.float32)
    green = scratch.get("hsl_g", shape, np.float32)
    blue = scratch.get("hsl_b", shape, np.float32)
    np.copyto(red, rgb[..., 0])
    np.copyto(green, rgb[..., 1])
    np.copyto(blue, rgb[..., 2])
    channels = (red, green, blue)
    maximum = scratch.get("hsl_max", shape, np.float32)
    minimum = scratch.get("hsl_min", shape, np.float32)
    chroma = scratch.get("hsl_chroma", shape, np.float32)
    lightness = scratch.get("hsl_light", shape, np.float32)
    saturation = scratch.get("hsl_sat", shape, np.float32)
    hue = scratch.get("hsl_hue", shape, np.float32)
    buf0 = scratch.get("hsl_b0", shape, np.float32)
    buf1 = scratch.get("hsl_b1", shape, np.float32)
    buf2 = scratch.get("hsl_b2", shape, np.float32)
    colored = scratch.get("hsl_col", shape, np.bool_)
    mask = scratch.get("hsl_mask", shape, np.bool_)
    index = scratch.get("hsl_idx", shape, np.uint8)
    mask_u8 = mask.view(np.uint8)

    np.maximum(red, green, out=maximum)
    np.maximum(maximum, blue, out=maximum)
    np.minimum(red, green, out=minimum)
    np.minimum(minimum, blue, out=minimum)
    np.subtract(maximum, minimum, out=chroma)
    np.add(maximum, minimum, out=lightness)
    lightness *= np.float32(.5)
    np.greater(chroma, 0, out=colored)
    saturation.fill(0)
    # At almost-white/black pixels float32 can round L to an endpoint while
    # chroma is still nonzero. HSL S is bounded; avoid an endpoint division by 0.
    np.multiply(lightness, np.float32(2), out=buf0)
    np.subtract(buf0, np.float32(1), out=buf0)
    np.abs(buf0, out=buf0)
    np.subtract(np.float32(1), buf0, out=buf0)
    np.maximum(buf0, chroma, out=buf0)
    np.divide(chroma, buf0, out=saturation, where=colored)
    hue.fill(0)
    difference = buf0
    off = buf1
    off.fill(0)
    for channel, offset in ((0, 0), (1, 2), (2, 4)):
        np.equal(maximum, channels[channel], out=mask)
        np.logical_and(colored, mask, out=mask)
        np.subtract(channels[(channel+1) % 3], channels[(channel+2) % 3], out=buf2)
        np.copyto(difference, buf2, where=mask)
        np.copyto(off, np.float32(offset), where=mask)
    np.divide(difference, chroma, out=hue, where=colored)
    np.add(hue, off, out=hue, where=colored)
    hue /= np.float32(6)
    np.less(hue, np.float32(0), out=mask)
    np.add(hue, np.float32(1), out=hue, where=mask)
    hue += np.float32(0)
    index.fill(0)
    for center in CENTERS_INNER:
        np.greater_equal(hue, center, out=mask)
        index += mask_u8
    blend = buf0
    protection = buf1
    tmp = buf2
    np.take(CENTERS_LO, index, out=blend)
    np.subtract(hue, blend, out=blend)
    np.take(CENTERS_SPAN, index, out=tmp)
    blend /= tmp
    np.multiply(blend, np.float32(2), out=tmp)
    np.subtract(np.float32(3), tmp, out=tmp)
    np.multiply(blend, tmp, out=tmp)
    blend *= tmp
    np.divide(saturation, np.float32(.1), out=protection)
    np.clip(protection, 0, 1, out=protection)
    np.multiply(protection, np.float32(2), out=tmp)
    np.subtract(np.float32(3), tmp, out=tmp)
    np.multiply(protection, tmp, out=tmp)
    protection *= tmp
    one_minus_blend = chroma
    np.subtract(np.float32(1), blend, out=one_minus_blend)
    amount = maximum
    for axis, band_lo, band_hi in _prepared_bands(mixer.bands):
        np.take(band_lo, index, out=amount)
        amount *= one_minus_blend
        np.take(band_hi, index, out=tmp)
        tmp *= blend
        amount += tmp
        amount *= protection
        if axis == 0:
            amount /= np.float32(360)
            hue += amount
            np.greater_equal(hue, np.float32(1), out=mask)
            np.subtract(hue, np.float32(1), out=hue, where=mask)
            np.less(hue, np.float32(0), out=mask)
            np.add(hue, np.float32(1), out=hue, where=mask)
            hue += np.float32(0)
        else:
            amount /= np.float32(100)
            target = saturation if axis == 1 else lightness
            np.subtract(np.float32(1), target, out=tmp)
            np.less(amount, 0, out=mask)
            np.copyto(tmp, target, where=mask)
            np.multiply(amount, tmp, out=tmp)
            target += tmp
            np.clip(target, 0, 1, out=target)
    amplitude = buf0
    np.subtract(np.float32(1), lightness, out=amplitude)
    np.minimum(lightness, amplitude, out=amplitude)
    amplitude *= saturation
    hue12 = buf1
    k = buf2
    tmp = maximum
    np.multiply(hue, np.float32(12), out=hue12)
    for channel, offset in enumerate((0, 8, 4)):
        np.add(np.float32(offset), hue12, out=k)
        np.greater_equal(k, np.float32(12), out=mask)
        np.subtract(k, np.float32(12), out=k, where=mask)
        np.subtract(k, np.float32(3), out=tmp)
        np.subtract(np.float32(9), k, out=k)
        np.minimum(tmp, k, out=tmp)
        np.minimum(tmp, np.float32(1), out=tmp)
        np.maximum(np.float32(-1), tmp, out=tmp)
        tmp *= amplitude
        np.subtract(lightness, tmp, out=channels[channel])
        np.clip(channels[channel], 0, 1, out=rgb[..., channel])
