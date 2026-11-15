"""Validated spatial detail controls: luminance USM sharpening, edge masking, and noise reduction."""
from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import dataclass
import math
import numpy as np

from .scratch import Scratch

DETAIL_HALO_PX = 6


@dataclass(frozen=True)
class DetailControls:
    sharpening: float = 0.0
    masking: float = 0.0
    luminance_nr: float = 0.0
    chroma_nr: float = 0.0

    def __post_init__(self) -> None:
        for name in ("sharpening", "masking", "luminance_nr", "chroma_nr"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 100.0
            ):
                raise ValueError(f"{name} must be finite and between 0 and 100.")
            object.__setattr__(self, name, float(value))

    @property
    def active(self) -> bool:
        return bool(self.sharpening or self.luminance_nr or self.chroma_nr)

    def to_dict(self) -> dict[str, float]:
        return {
            "sharpening": self.sharpening,
            "masking": self.masking,
            "luminance_nr": self.luminance_nr,
            "chroma_nr": self.chroma_nr,
        }

    @classmethod
    def from_dict(cls, value: dict) -> DetailControls:
        if not isinstance(value, dict) or set(value) != {"sharpening", "masking", "luminance_nr", "chroma_nr"}:
            raise ValueError("Unknown or missing detail parameters.")
        return cls(**value)


CAMERA_RAW_DETAIL = DetailControls(sharpening=40.0, masking=10.0, luminance_nr=0.0, chroma_nr=25.0)


def _sep_conv_3(plane: np.ndarray) -> np.ndarray:
    """Separable 3x3 [1, 2, 1]/4 Gaussian filter with edge-clamped boundary."""
    p = np.pad(plane, ((1, 1), (1, 1)), mode="edge")
    h = (p[:, 0:-2] + np.float32(2.0) * p[:, 1:-1] + p[:, 2:]) * np.float32(0.25)
    v = (h[0:-2, :] + np.float32(2.0) * h[1:-1, :] + h[2:, :]) * np.float32(0.25)
    return v.astype(np.float32, copy=False)


def _sep_conv_5(plane: np.ndarray) -> np.ndarray:
    """Separable 5x5 [1, 4, 6, 4, 1]/16 Gaussian filter with edge-clamped boundary."""
    p = np.pad(plane, ((2, 2), (2, 2)), mode="edge")
    h = (
        p[:, 0:-4]
        + np.float32(4.0) * p[:, 1:-3]
        + np.float32(6.0) * p[:, 2:-2]
        + np.float32(4.0) * p[:, 3:-1]
        + p[:, 4:]
    ) * np.float32(1.0 / 16.0)
    v = (
        h[0:-4, :]
        + np.float32(4.0) * h[1:-3, :]
        + np.float32(6.0) * h[2:-2, :]
        + np.float32(4.0) * h[3:-1, :]
        + h[4:, :]
    ) * np.float32(1.0 / 16.0)
    return v.astype(np.float32, copy=False)


def _detail_patch(srgb: np.ndarray, detail: DetailControls) -> np.ndarray:
    """Apply opponent chroma NR, wavelet luminance NR, and edge-masked luminance USM on one patch."""
    r = srgb[..., 0]
    g = srgb[..., 1]
    b = srgb[..., 2]
    y0 = (np.float32(0.2126) * r + np.float32(0.7152) * g + np.float32(0.0722) * b).astype(np.float32)
    cr = (r - y0).astype(np.float32)
    cb = (b - y0).astype(np.float32)
    y = y0

    if detail.chroma_nr > 0 or (detail.sharpening > 0 and detail.masking > 0):
        yp = np.pad(y0, ((1, 1), (1, 1)), mode="edge")
        gx = (yp[1:-1, 2:] - yp[1:-1, :-2]) * np.float32(0.5)
        gy = (yp[2:, 1:-1] - yp[:-2, 1:-1]) * np.float32(0.5)
        grad2 = gx * gx + gy * gy
    else:
        grad2 = None

    if detail.chroma_nr > 0:
        sc = np.float32(detail.chroma_nr / 100.0)
        cr_blur = _sep_conv_5(_sep_conv_5(cr))
        cb_blur = _sep_conv_5(_sep_conv_5(cb))
        tau_c2 = np.float32((0.08 + 0.12 * float(sc)) ** 2)
        assert grad2 is not None
        w_edge = tau_c2 / (grad2 + tau_c2)
        blend = sc * w_edge
        cr = cr + blend * (cr_blur - cr)
        cb = cb + blend * (cb_blur - cb)

    if detail.luminance_nr > 0:
        sl = np.float32(detail.luminance_nr / 100.0)
        b1 = _sep_conv_3(y)
        b2 = _sep_conv_5(b1)
        d1 = y - b1
        d2 = b1 - b2
        t1_sq = np.float32((0.022 * float(sl)) ** 2)
        t2_sq = np.float32((0.012 * float(sl)) ** 2)
        d1_keep = d1 * (d1 * d1) / (d1 * d1 + t1_sq + np.float32(1e-12))
        d2_keep = d2 * (d2 * d2) / (d2 * d2 + t2_sq + np.float32(1e-12))
        y = np.clip(b2 + d2_keep + d1_keep, 0.0, 1.0)

    if detail.sharpening > 0:
        amount = np.float32((detail.sharpening / 100.0) * 2.5)
        blur = np.float32(0.55) * _sep_conv_3(y) + np.float32(0.45) * _sep_conv_5(y)
        high = y - blur
        core_sq = np.float32(0.0020 ** 2)
        high_cored = high * (high * high) / (high * high + core_sq)
        if detail.masking > 0:
            assert grad2 is not None
            g2 = _sep_conv_3(grad2)
            tm_sq = np.float32((0.06 * (detail.masking / 100.0) ** 2) ** 2 + 1e-12)
            edge_mask = g2 / (g2 + tm_sq)
        else:
            edge_mask = np.float32(1.0)
        halo_limited = high_cored / (np.float32(1.0) + np.abs(high_cored) / np.float32(0.14))
        y = np.clip(y + amount * edge_mask * halo_limited, 0.0, 1.0)

    out = np.empty_like(srgb, dtype=np.float32)
    out[..., 0] = np.clip(y + cr, 0.0, 1.0)
    out[..., 2] = np.clip(y + cb, 0.0, 1.0)
    out[..., 1] = np.clip(
        y - (np.float32(0.2126) * cr + np.float32(0.0722) * cb) / np.float32(0.7152),
        0.0,
        1.0,
    )
    return out


def apply_detail(
    srgb: np.ndarray,
    detail: DetailControls,
    *,
    cancel=None,
    scratch: Scratch | None = None,
) -> np.ndarray:
    """Process an encoded sRGB buffer in bounded 128-row halo strips; neutral is a no-op."""
    if not detail.active or srgb.size == 0:
        return srgb
    h = srgb.shape[0]
    if h <= 128:
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        srgb[:] = _detail_patch(srgb, detail)
        return srgb
    source = srgb.copy()
    for top in range(0, h, 128):
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        bottom = min(h, top + 128)
        pad_top = max(0, top - DETAIL_HALO_PX)
        pad_bottom = min(h, bottom + DETAIL_HALO_PX)
        filtered = _detail_patch(source[pad_top:pad_bottom], detail)
        srgb[top:bottom] = filtered[top - pad_top : top - pad_top + (bottom - top)]
    return srgb
