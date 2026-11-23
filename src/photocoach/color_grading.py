"""Immutable 3-way Color Grading model and bounded float32 display-space transform.

Luma zone weights use the encoded display-space proxy Y = 0.2126*R + 0.7152*G + 0.0722*B
(not physical scene luminance). Pure color-wheel tints (luminance=0) subtract their own
luma proxy so chromatic shifts stay luma-neutral before SDR [0, 1] clipping.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np

from .scratch import Scratch

WHEEL_NAMES = ("shadows", "midtones", "highlights")
WHEEL_FIELDS = ("hue", "saturation", "luminance")

_TINT_STRENGTH = np.float32(0.35)
_LUMINANCE_STRENGTH = np.float32(0.15)
_CENTER_SHADOWS = np.float32(1.0 / 3.0)
_CENTER_HIGHLIGHTS = np.float32(2.0 / 3.0)
_HALF_WIDTH_MIN = np.float32(0.18)
_HALF_WIDTH_SPAN = np.float32(0.42)


def _finite_bound(
    value: Any,
    low: float,
    high: float,
    name: str,
    *,
    upper_exclusive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number.")
    val = float(value)
    in_range = (low <= val < high) if upper_exclusive else (low <= val <= high)
    if not math.isfinite(val) or not in_range:
        upper = ")" if upper_exclusive else "]"
        raise ValueError(f"{name} must be finite and in [{low:g}, {high:g}{upper}.")
    return 0.0 if val == 0.0 else val


@dataclass(frozen=True)
class ColorWheel:
    hue: float = 0.0
    saturation: float = 0.0
    luminance: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "hue", _finite_bound(self.hue, 0.0, 360.0, "Hue", upper_exclusive=True))
        object.__setattr__(self, "saturation", _finite_bound(self.saturation, 0.0, 1.0, "Saturation"))
        object.__setattr__(self, "luminance", _finite_bound(self.luminance, -1.0, 1.0, "Luminance"))

    @property
    def is_identity(self) -> bool:
        return self.saturation == 0.0 and self.luminance == 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "hue": self.hue,
            "saturation": self.saturation,
            "luminance": self.luminance,
        }

    @classmethod
    def from_dict(cls, value: dict) -> ColorWheel:
        if not isinstance(value, dict) or set(value) != set(WHEEL_FIELDS):
            raise ValueError("Invalid Color Wheel parameters.")
        return cls(
            hue=value["hue"],
            saturation=value["saturation"],
            luminance=value["luminance"],
        )


@dataclass(frozen=True)
class ColorGrading:
    shadows: ColorWheel = ColorWheel()
    midtones: ColorWheel = ColorWheel()
    highlights: ColorWheel = ColorWheel()
    blend: float = 0.5
    balance: float = 0.0

    def __post_init__(self) -> None:
        for name in WHEEL_NAMES:
            if not isinstance(getattr(self, name), ColorWheel):
                raise ValueError(f"Invalid Color Grading {name} wheel.")
        object.__setattr__(self, "blend", _finite_bound(self.blend, 0.0, 1.0, "Blend"))
        object.__setattr__(self, "balance", _finite_bound(self.balance, -1.0, 1.0, "Balance"))

    @property
    def is_identity(self) -> bool:
        return (
            self.shadows.is_identity
            and self.midtones.is_identity
            and self.highlights.is_identity
        )

    def with_wheel(self, zone: str, wheel: ColorWheel) -> ColorGrading:
        if zone not in WHEEL_NAMES:
            raise ValueError(f"Unknown Color Grading zone: {zone}")
        if not isinstance(wheel, ColorWheel):
            raise ValueError(f"Invalid Color Grading {zone} wheel.")
        return replace(self, **{zone: wheel})

    def to_dict(self) -> dict[str, object]:
        return {
            "shadows": self.shadows.to_dict(),
            "midtones": self.midtones.to_dict(),
            "highlights": self.highlights.to_dict(),
            "blend": self.blend,
            "balance": self.balance,
        }

    @classmethod
    def from_dict(cls, value: dict) -> ColorGrading:
        expected = {"shadows", "midtones", "highlights", "blend", "balance"}
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("Unknown or missing Color Grading parameters.")
        return cls(
            shadows=ColorWheel.from_dict(value["shadows"]),
            midtones=ColorWheel.from_dict(value["midtones"]),
            highlights=ColorWheel.from_dict(value["highlights"]),
            blend=value["blend"],
            balance=value["balance"],
        )


def _hue_to_rgb(hue_deg: float) -> tuple[float, float, float]:
    """Convert hue angle in [0, 360) degrees to pure saturated RGB on [0, 1]."""
    sector = (float(hue_deg) % 360.0) / 60.0
    index = int(math.floor(sector)) % 6
    frac = sector - math.floor(sector)
    if index == 0:
        return (1.0, frac, 0.0)
    if index == 1:
        return (1.0 - frac, 1.0, 0.0)
    if index == 2:
        return (0.0, 1.0, frac)
    if index == 3:
        return (0.0, 1.0 - frac, 1.0)
    if index == 4:
        return (frac, 0.0, 1.0)
    return (1.0, 0.0, 1.0 - frac)


def _wheel_delta(wheel: ColorWheel) -> tuple[np.float32, np.float32, np.float32]:
    """Compute display-space RGB offset for one wheel; tint is luma-proxy neutral."""
    if wheel.saturation > 0.0:
        cr, cg, cb = _hue_to_rgb(wheel.hue)
        yc = 0.2126 * cr + 0.7152 * cg + 0.0722 * cb
        scale = float(_TINT_STRENGTH) * wheel.saturation
        tr = scale * (cr - yc)
        tg = scale * (cg - yc)
        tb = scale * (cb - yc)
    else:
        tr = tg = tb = 0.0
    lum = float(_LUMINANCE_STRENGTH) * wheel.luminance
    return (np.float32(tr + lum), np.float32(tg + lum), np.float32(tb + lum))


def _compute_zone_weights_into(
    y: np.ndarray,
    u: np.ndarray,
    w_s: np.ndarray,
    w_m: np.ndarray,
    w_h: np.ndarray,
    tmp: np.ndarray,
    blend: float,
    balance: float,
) -> None:
    """Populate smooth partition-of-unity shadow/midtone/highlight weights in float32."""
    if balance == 0.0:
        np.copyto(u, y)
    else:
        k = np.float32(2.0 ** float(balance))
        np.multiply(y, np.float32(k - 1.0), out=tmp)
        tmp += np.float32(1.0)
        np.multiply(y, k, out=u)
        u /= tmp

    h = _HALF_WIDTH_MIN + _HALF_WIDTH_SPAN * np.float32(blend)
    inv_2h = np.float32(1.0 / (2.0 * float(h)))

    # Shadow -> Midtone transition S_a(u)
    np.subtract(u, _CENTER_SHADOWS - h, out=tmp)
    tmp *= inv_2h
    np.clip(tmp, np.float32(0.0), np.float32(1.0), out=tmp)
    np.multiply(tmp, np.float32(2.0), out=w_m)
    np.subtract(np.float32(3.0), w_m, out=w_m)
    w_m *= tmp
    w_m *= tmp  # w_m now holds sa = S(a)
    np.subtract(np.float32(1.0), w_m, out=w_s)

    # Midtone -> Highlight transition S_b(u)
    np.subtract(u, _CENTER_HIGHLIGHTS - h, out=tmp)
    tmp *= inv_2h
    np.clip(tmp, np.float32(0.0), np.float32(1.0), out=tmp)
    np.multiply(tmp, np.float32(2.0), out=w_h)
    np.subtract(np.float32(3.0), w_h, out=w_h)
    w_h *= tmp
    w_h *= tmp  # w_h now holds sb = S(b)

    w_m -= w_h
    np.maximum(w_m, np.float32(0.0), out=w_m)


def zone_weights(
    luma: np.ndarray,
    *,
    blend: float = 0.5,
    balance: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return smooth (shadows, midtones, highlights) weights summing to 1 over luma in [0, 1]."""
    blend_v = _finite_bound(blend, 0.0, 1.0, "Blend")
    balance_v = _finite_bound(balance, -1.0, 1.0, "Balance")
    y = np.clip(np.asarray(luma, dtype=np.float32), np.float32(0.0), np.float32(1.0))
    shape = y.shape
    u = np.empty(shape, dtype=np.float32)
    w_s = np.empty(shape, dtype=np.float32)
    w_m = np.empty(shape, dtype=np.float32)
    w_h = np.empty(shape, dtype=np.float32)
    tmp = np.empty(shape, dtype=np.float32)
    _compute_zone_weights_into(y, u, w_s, w_m, w_h, tmp, blend_v, balance_v)
    return w_s, w_m, w_h


def apply_color_grading(
    encoded_rgb: np.ndarray,
    grading: ColorGrading,
    *,
    scratch: Scratch | None = None,
    in_place: bool = False,
) -> np.ndarray:
    """Apply 3-way Color Grading in float32 encoded sRGB space.

    When grading.is_identity is True, short-circuits without allocating or modifying pixels.
    """
    if not isinstance(grading, ColorGrading):
        raise ValueError("Color Grading requires a validated ColorGrading instance.")
    if in_place:
        if not isinstance(encoded_rgb, np.ndarray):
            raise ValueError("In-place Color Grading requires a NumPy array.")
        arr = encoded_rgb
    else:
        arr = np.asarray(encoded_rgb)
    if arr.ndim != 3 or arr.shape[2] != 3 or not np.issubdtype(arr.dtype, np.floating):
        raise ValueError("Color Grading requires an H x W x 3 floating-point RGB array.")
    if in_place and (arr.dtype != np.float32 or (not grading.is_identity and arr.size > 0 and not arr.flags.writeable)):
        raise ValueError("In-place Color Grading requires a writeable float32 RGB array.")

    if grading.is_identity or arr.size == 0:
        if in_place or arr.dtype == np.float32:
            return arr
        return arr.astype(np.float32, copy=False)

    src = arr if arr.dtype == np.float32 else arr.astype(np.float32, copy=False)
    if scratch is None:
        scratch = Scratch()
    shape = src.shape[:2]
    y = scratch.get("cg_y", shape, np.float32)
    u = scratch.get("cg_u", shape, np.float32)
    w_s = scratch.get("cg_ws", shape, np.float32)
    w_m = scratch.get("cg_wm", shape, np.float32)
    w_h = scratch.get("cg_wh", shape, np.float32)
    env = scratch.get("cg_env", shape, np.float32)
    tmp = scratch.get("cg_tmp", shape, np.float32)

    # Display-space luma proxy Y in [0, 1]
    np.multiply(src[..., 0], np.float32(0.2126), out=y)
    np.multiply(src[..., 1], np.float32(0.7152), out=tmp)
    y += tmp
    np.multiply(src[..., 2], np.float32(0.0722), out=tmp)
    y += tmp
    np.clip(y, np.float32(0.0), np.float32(1.0), out=y)

    _compute_zone_weights_into(y, u, w_s, w_m, w_h, tmp, grading.blend, grading.balance)

    # Smooth endpoint envelope 4 * y * (1 - y) anchors pure black/white and prevents gamut blowups.
    np.subtract(np.float32(1.0), y, out=env)
    env *= y
    env *= np.float32(4.0)

    ds = _wheel_delta(grading.shadows)
    dm = _wheel_delta(grading.midtones)
    dh = _wheel_delta(grading.highlights)

    out = src if in_place else np.empty(src.shape, dtype=np.float32)
    for channel in range(3):
        np.multiply(w_s, ds[channel], out=tmp)
        if dm[channel] != 0.0:
            np.multiply(w_m, dm[channel], out=u)
            tmp += u
        if dh[channel] != 0.0:
            np.multiply(w_h, dh[channel], out=u)
            tmp += u
        tmp *= env
        np.add(src[..., channel], tmp, out=out[..., channel])
        np.clip(out[..., channel], np.float32(0.0), np.float32(1.0), out=out[..., channel])
    return out
