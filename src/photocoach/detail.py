"""Validated spatial detail controls: luminance USM sharpening, edge masking, and noise reduction."""
from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import dataclass
import math
import numpy as np

from .scratch import Scratch

DETAIL_HALO_PX = 20


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


def _box_mean_stack(stack: np.ndarray, radius: int) -> np.ndarray:
    """Fast shift-invariant edge-clamped 2D box average on a (N, H, W) float32 stack."""
    if radius <= 0:
        return stack
    _, h, w = stack.shape
    inv = np.float32(1.0 / (2 * radius + 1))
    px = np.pad(stack, ((0, 0), (0, 0), (radius, radius)), mode="edge")
    acc = px[:, :, :w].copy()
    for k in range(1, 2 * radius + 1):
        acc += px[:, :, k : k + w]
    acc *= inv
    py = np.pad(acc, ((0, 0), (radius, radius), (0, 0)), mode="edge")
    out = py[:, :h, :].copy()
    for k in range(1, 2 * radius + 1):
        out += py[:, k : k + h, :]
    out *= inv
    return out


def _box_mean(plane: np.ndarray, radius: int) -> np.ndarray:
    return _box_mean_stack(plane[None, ...], radius)[0]


def _guided_filter_pair(
    guide: np.ndarray,
    src1: np.ndarray,
    src2: np.ndarray | None,
    radius: int,
    eps: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Batched Kaiming He guided filter for 1 or 2 targets sharing a guide (support radius = 2 * radius)."""
    if src2 is None:
        means = _box_mean_stack(np.stack([guide, guide * guide, src1, guide * src1], axis=0), radius)
        mean_i, mean_ii, mean_p1, mean_ip1 = means
        var_i = np.maximum(np.float32(0.0), mean_ii - mean_i * mean_i)
        denom = var_i + eps + np.float32(1e-12)
        a1 = (mean_ip1 - mean_i * mean_p1) / denom
        b1 = mean_p1 - a1 * mean_i
        ab = _box_mean_stack(np.stack([a1, b1], axis=0), radius)
        return (ab[0] * guide + ab[1]).astype(np.float32, copy=False), None, var_i
    means = _box_mean_stack(
        np.stack([guide, guide * guide, src1, src2, guide * src1, guide * src2], axis=0),
        radius,
    )
    mean_i, mean_ii, mean_p1, mean_p2, mean_ip1, mean_ip2 = means
    var_i = np.maximum(np.float32(0.0), mean_ii - mean_i * mean_i)
    denom = var_i + eps + np.float32(1e-12)
    a1 = (mean_ip1 - mean_i * mean_p1) / denom
    b1 = mean_p1 - a1 * mean_i
    a2 = (mean_ip2 - mean_i * mean_p2) / denom
    b2 = mean_p2 - a2 * mean_i
    ab = _box_mean_stack(np.stack([a1, b1, a2, b2], axis=0), radius)
    out1 = (ab[0] * guide + ab[1]).astype(np.float32, copy=False)
    out2 = (ab[2] * guide + ab[3]).astype(np.float32, copy=False)
    return out1, out2, var_i


def _detail_patch(srgb: np.ndarray, detail: DetailControls) -> np.ndarray:
    """Apply 2-scale chroma NR, shadow-adaptive luma NR, and edge-masked dual-band USM on one patch."""
    r = srgb[..., 0]
    g = srgb[..., 1]
    b = srgb[..., 2]
    y0 = (np.float32(0.2126) * r + np.float32(0.7152) * g + np.float32(0.0722) * b).astype(np.float32)
    cr = (r - y0).astype(np.float32)
    cb = (b - y0).astype(np.float32)
    y = y0

    if detail.chroma_nr > 0:
        sc = float(detail.chroma_nr / 100.0)
        y_guide = _sep_conv_3(y0)
        eps_c1 = np.float32((0.025 + 0.095 * sc) ** 2)
        cr1, cb1, _ = _guided_filter_pair(y_guide, cr, cb, 3, eps_c1)
        assert cb1 is not None

        # Pass 2: wider support handles 6-16px AHD/low-light purple-green blotches along luma edges.
        eps_c2 = np.float32((0.018 + 0.080 * sc) ** 2)
        cr2, cb2, _ = _guided_filter_pair(y_guide, cr1, cb1, 6, eps_c2)
        assert cb2 is not None

        # Chroma self-guided branch protects genuine isoluminant color edges (where var(y_guide) ~ 0)
        # across the full r=6 halo without letting Pass 2 bleed color across sharp hue boundaries.
        cr_s0 = _sep_conv_5(cr)
        cb_s0 = _sep_conv_5(cb)
        eps_iso = np.float32((0.018 + 0.024 * sc) ** 2)
        cr_c, _, var_cr = _guided_filter_pair(cr_s0, cr, None, 3, eps_iso)
        cb_c, _, var_cb = _guided_filter_pair(cb_s0, cb, None, 3, eps_iso)
        c_var_wide = _box_mean(var_cr + var_cb, 6)
        tau_iso2 = np.float32((0.036 + 0.026 * sc) ** 2)
        cv2 = c_var_wide * c_var_wide
        w_iso_edge = cv2 / (cv2 + tau_iso2 * tau_iso2 + np.float32(1e-18))

        cr_y = cr1 + np.float32(sc) * (cr2 - cr1)
        cb_y = cb1 + np.float32(sc) * (cb2 - cb1)
        cr_clean = cr_y + w_iso_edge * (cr_c - cr_y)
        cb_clean = cb_y + w_iso_edge * (cb_c - cb_y)
        chroma_blend = np.float32(min(1.0, sc * 1.45))
        cr = cr + chroma_blend * (cr_clean - cr)
        cb = cb + chroma_blend * (cb_clean - cb)

    if detail.luminance_nr > 0:
        sl = float(detail.luminance_nr / 100.0)
        s_eff = sl ** 0.92
        # Mildly smoothed structural guide attenuates 1px uncorrelated shot noise variance by ~3.4x
        # while preserving thin 1-2px ink strokes, foliage, and edges.
        b3_y = _sep_conv_3(y)
        g1 = np.float32(0.35) * y + np.float32(0.65) * b3_y
        # Photon shot noise in sRGB is stronger in dark shadows; highlights remain gentler.
        shadow_mult = np.float32(0.65) + np.float32(1.55) * (np.float32(1.0) - np.clip(b3_y, 0.0, 1.0)) ** 2
        eps1 = np.float32((0.0040 + 0.025 * s_eff) ** 2) * shadow_mult
        y1, _, var_g1 = _guided_filter_pair(g1, y, None, 2, eps1)

        # Stage 2 (r=4, 9x9) removes 4-8px flat shadow mottle; var_y2 measures denoised structure variance
        # weighted inversely by shadow_mult so mid/bright texture stays free of milky haze.
        eps2 = np.float32((0.0025 + 0.014 * (s_eff ** 1.1)) ** 2) * shadow_mult
        y2, _, var_y2 = _guided_filter_pair(y1, y1, None, 4, eps2)

        macro_keep = np.float32(0.25 + 0.72 * s_eff)
        struct_keep = var_y2 / (var_y2 + np.float32(0.00008 + 0.00018 * s_eff) * shadow_mult + np.float32(1e-12))
        y_mottle = y1 + macro_keep * (np.float32(1.0) - struct_keep * np.float32(0.90)) * (y2 - y1)
        # Restore crisp micro-edge acutance using the noise-suppressed Stage-1 guide variance.
        crisp_keep = (var_g1 / (var_g1 + np.float32(0.0014) * shadow_mult + np.float32(1e-12))) * np.float32(0.50)
        y_nr = y_mottle + crisp_keep * (y - y1)
        luma_blend = np.float32(min(1.0, sl * 2.0))
        y = np.clip(y + luma_blend * (y_nr - y), np.float32(0.0), np.float32(1.0))

    if detail.sharpening > 0:
        amount = np.float32((detail.sharpening / 100.0) * 2.85)
        b3 = _sep_conv_3(y)
        b5 = _sep_conv_5(b3)
        high = np.float32(0.62) * (y - b3) + np.float32(0.38) * (b3 - b5)
        core_t = 0.0030 + 0.0045 * float(detail.luminance_nr / 100.0)
        core_sq = np.float32(core_t * core_t)
        high_sq = high * high
        high_cored = high * high_sq / (high_sq + core_sq + np.float32(1e-12))

        if detail.masking > 0:
            yp = np.pad(b3, ((1, 1), (1, 1)), mode="edge")
            gx = (yp[1:-1, 2:] - yp[1:-1, :-2]) * np.float32(0.5)
            gy = (yp[2:, 1:-1] - yp[:-2, 1:-1]) * np.float32(0.5)
            grad_mag = _sep_conv_3(np.sqrt(gx * gx + gy * gy))
            m = float(detail.masking / 100.0)
            tau_m = np.float32(0.0020 + 0.017 * (m ** 1.35))
            g2 = grad_mag * grad_mag
            t2 = np.float32(float(tau_m) ** 2 + 1e-12)
            edge_mask = g2 / (g2 + t2)
        else:
            edge_mask = np.float32(1.0)

        halo_lim = np.where(high_cored >= 0.0, np.float32(0.14), np.float32(0.18))
        halo_limited = high_cored / (np.float32(1.0) + np.abs(high_cored) / halo_lim)
        end_guard = np.clip(np.float32(4.2) * y * (np.float32(1.0) - y) + np.float32(0.25), np.float32(0.25), np.float32(1.0))
        y = np.clip(y + amount * edge_mask * halo_limited * end_guard, np.float32(0.0), np.float32(1.0))

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
    """Process an encoded sRGB buffer in bounded 256-row halo strips; neutral is a no-op."""
    if not detail.active or srgb.size == 0:
        return srgb
    h = srgb.shape[0]
    if h <= 256:
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        srgb[:] = _detail_patch(srgb, detail)
        return srgb
    source = srgb.copy()
    for top in range(0, h, 256):
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        bottom = min(h, top + 256)
        pad_top = max(0, top - DETAIL_HALO_PX)
        pad_bottom = min(h, bottom + DETAIL_HALO_PX)
        filtered = _detail_patch(source[pad_top:pad_bottom], detail)
        srgb[top:bottom] = filtered[top - pad_top : top - pad_top + (bottom - top)]
    return srgb
