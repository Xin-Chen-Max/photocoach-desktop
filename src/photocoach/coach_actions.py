"""Executable one-click Coach actions, including Local Mask artificial lighting recipes."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re
from typing import Any
from uuid import uuid4

import numpy as np

from .color_mixer import AXES, COLOR_KEYS
from .composition import output_size
from .gradients import LinearGradient, RadialGradient, validate_gradients, source_point
from .state import EditValues, validate_curve
from .imaging import RawOptions


def _clamp(value: float, low: float, high: float) -> float:
    return round(max(low, min(high, float(value))), 2)


@dataclass(frozen=True)
class CoachAction:
    id: str
    label: str
    parameter: str
    new_value: Any
    focus_control: str = ""
    changes: tuple[tuple[str, Any], ...] = ()

    def is_applied(self, values: EditValues | None) -> bool:
        if values is None:
            return False
        targets = self.changes or ((self.parameter, self.new_value),)
        return all(getattr(values, key) == target for key, target in targets)


LIGHTING_RECIPE_META: tuple[tuple[str, str, str], ...] = (
    ("mountain_beam", "Mountain / Ridge Light Beam", "Scene-targeted Radial Mountain Beam (+0.68 EV) + Valley Falloff (-0.28 EV)"),
    ("spotlight", "3D Subject Spotlight + Falloff", "Radial Key Light (+0.45 EV) + inverted Ambient Falloff (-0.28 EV)"),
    ("sunbeam", "Top-Down Sunbeam Glow", "Soft overhead Radial Sunbeam (+0.55 EV, Highlights +0.22, Shadows +0.28)"),
    ("window_light", "Directional Side Window Light", "Diagonal Window Light (+0.52 EV) + opposite Shadow Falloff (-0.26 EV)"),
    ("rim_backlight", "Backlit Rim / Halo Glow", "Upper-center Radial Backlight (+0.60 EV, Highlights +0.28, Shadows +0.35)"),
    ("sky_filter", "Graduated Sky Filter", "Top-down Linear Sky Filter (-0.45 EV, Highlights -0.35)"),
    ("foreground_fill", "Foreground Fill Light", "Bottom-up Linear Foreground Fill (+0.38 EV, Shadows +0.32)"),
)


def _box_blur_2d(arr: np.ndarray, ry: int, rx: int) -> np.ndarray:
    """Fast O(H*W) edge-padded 2D box average using pure NumPy cumulative sums."""
    arr = np.asarray(arr, dtype=np.float32)
    h, w = arr.shape[:2]
    ry = max(1, min(int(ry), max(1, h // 2)))
    rx = max(1, min(int(rx), max(1, w // 2)))
    pad_x = np.pad(arr, ((0, 0), (rx + 1, rx)), mode="edge")
    cs_x = np.cumsum(pad_x, axis=1, dtype=np.float64)
    horiz = (cs_x[:, 2 * rx + 1 :] - cs_x[:, :w]) / float(2 * rx + 1)
    pad_y = np.pad(horiz, ((ry + 1, ry), (0, 0)), mode="edge")
    cs_y = np.cumsum(pad_y, axis=0, dtype=np.float64)
    vert = (cs_y[2 * ry + 1 :, :] - cs_y[:h, :]) / float(2 * ry + 1)
    return vert.astype(np.float32)


def _detect_foreground_subjects(
    luma: np.ndarray,
    edge_energy: np.ndarray,
    horizon_y: float,
    *,
    max_candidates: int = 5,
) -> list[dict[str, Any]]:
    """Detect salient foreground subject blobs (including dark subjects on brighter ground)."""
    h, w = luma.shape[:2]
    if h < 20 or w < 20:
        return []
    y0 = max(1, min(h - 8, int(round(max(0.10, min(0.32, horizon_y * 0.70)) * h))))
    y1 = max(y0 + 6, int(round(h * 0.92)))
    bg = _box_blur_2d(luma, max(4, h // 5), max(4, w // 5))
    dark_contrast = np.maximum(0.0, bg - luma)
    abs_contrast = np.abs(luma - bg)
    sal_map = 1.45 * dark_contrast + 0.55 * abs_contrast + 0.30 * edge_energy

    sub_sal = sal_map[y0:y1, :]
    sub_luma = luma[y0:y1, :]
    sub_bg = bg[y0:y1, :]
    gh = min(24, sub_sal.shape[0])
    gw = min(36, sub_sal.shape[1])
    r_edges = np.linspace(0, sub_sal.shape[0], gh + 1, dtype=int)
    c_edges = np.linspace(0, sub_sal.shape[1], gw + 1, dtype=int)

    grid_sal = np.zeros((gh, gw), dtype=np.float32)
    grid_lum = np.zeros((gh, gw), dtype=np.float32)
    grid_bg = np.zeros((gh, gw), dtype=np.float32)
    for r in range(gh):
        r0, r1 = int(r_edges[r]), max(int(r_edges[r]) + 1, int(r_edges[r + 1]))
        for c in range(gw):
            c0, c1 = int(c_edges[c]), max(int(c_edges[c]) + 1, int(c_edges[c + 1]))
            grid_sal[r, c] = float(sub_sal[r0:r1, c0:c1].mean())
            grid_lum[r, c] = float(sub_luma[r0:r1, c0:c1].mean())
            grid_bg[r, c] = float(sub_bg[r0:r1, c0:c1].mean())

    max_sal = float(grid_sal.max())
    if max_sal < 0.04:
        return []
    thresh = max(float(np.percentile(grid_sal, 82)), float(grid_sal.mean() + 0.95 * grid_sal.std()), 0.05)
    active = grid_sal >= thresh
    if not np.any(active):
        return []

    visited = np.zeros((gh, gw), dtype=bool)
    scored: list[tuple[float, dict[str, Any]]] = []
    span_y = (y1 - y0) / float(h)
    top_norm_base = y0 / float(h)
    for r in range(gh):
        for c in range(gw):
            if not active[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                # 4-connectivity keeps distinct nearby subjects from merging across diagonal sand pixels
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < gh and 0 <= nc < gw and not visited[nr, nc] and active[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            if len(cells) < 2 and max_sal > 0.08:
                continue
            rs = [cell[0] for cell in cells]
            cs = [cell[1] for cell in cells]
            left = min(cs) / float(gw)
            right = (max(cs) + 1) / float(gw)
            top = top_norm_base + (min(rs) / float(gh)) * span_y
            bottom = top_norm_base + ((max(rs) + 1) / float(gh)) * span_y
            bw = right - left
            bh = bottom - top
            # Reject horizon/background bands spanning almost the entire frame width
            if bw > 0.72 and bh < 0.20:
                continue
            if bw > 0.78 or bh > 0.65:
                continue
            weights = [float(grid_sal[cr, cc]) for cr, cc in cells]
            w_sum = sum(weights) + 1e-6
            cx = sum(((cc + 0.5) / float(gw)) * wt for (_, cc), wt in zip(cells, weights)) / w_sum
            cy = sum((top_norm_base + ((cr + 0.5) / float(gh)) * span_y) * wt for (cr, _), wt in zip(cells, weights)) / w_sum
            m_lum = sum(float(grid_lum[cr, cc]) for cr, cc in cells) / float(len(cells))
            m_bg = sum(float(grid_bg[cr, cc]) for cr, cc in cells) / float(len(cells))
            contrast = max(0.01, float(np.mean(weights)))
            dist_center = math.hypot(cx - 0.5, cy - 0.52)
            score = (len(cells) ** 0.75) * contrast * max(0.35, 1.30 - 0.75 * dist_center)
            crest_y = min(bottom - 0.01, top + 0.18 * bh)
            scored.append((score, {
                "center": [round(cx, 3), round(cy, 3)],
                "bbox": [round(max(0.01, left), 3), round(max(0.01, top), 3),
                         round(min(0.99, right), 3), round(min(0.99, bottom), 3)],
                "crest": [round(cx, 3), round(crest_y, 3)],
                "size": [round(bw, 3), round(bh, 3)],
                "mean_luma": round(m_lum, 3),
                "bg_luma": round(m_bg, 3),
                "is_dark_subject": bool(m_lum < m_bg - 0.025),
            }))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    results: list[dict[str, Any]] = []
    for idx, (_, item) in enumerate(scored[:max_candidates], start=1):
        results.append({"id": idx, **item})
    return results


def build_overhead_spotlight_actions(
    bbox: tuple[float, float, float, float] | list[float],
    source_pt: tuple[float, float] = (0.35, 0.14),
    *,
    is_dark_subject: bool = True,
    direction: str = "auto",
    strength: str = "moderate",
    recipe: str = "spotlight",
    max_masks: int = 4,
) -> list[dict[str, Any]]:
    """Construct ACTION_SCHEMA-compliant create_mask actions anchored to a subject crest & overhead beam."""
    left, top, right, bottom = [float(v) for v in bbox]
    if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
        raise ValueError("Invalid target_box coordinates.")
    if max_masks < 1:
        raise MaskCapacityError("No free local mask slots remain (maximum 4).")
    cx = (left + right) / 2.0
    bw = max(0.06, right - left)
    bh = max(0.05, bottom - top)
    crest_y = min(bottom, top + 0.16 * bh)

    scales = {"subtle": 0.75, "moderate": 1.0, "dramatic": 1.28}
    scale = scales.get(strength, 1.0)

    sx, sy = float(source_pt[0]), float(source_pt[1])
    if direction == "overhead":
        sx, sy = cx, max(0.03, top - 0.26)
    elif direction == "top_left":
        sx, sy = max(0.06, cx - 0.28), max(0.03, top - 0.24)
    elif direction == "top_right":
        sx, sy = min(0.94, cx + 0.28), max(0.03, top - 0.24)

    dx_light = cx - sx
    dy_light = max(0.05, crest_y - sy)
    beam_angle = 0.0 if direction == "overhead" else round(_clamp(math.degrees(math.atan2(dx_light, dy_light)) * 0.42, -30.0, 30.0), 1)

    if recipe == "window_light":
        from_right = sx > cx
        k_start = [round(min(0.96, cx + 0.34) if from_right else max(0.04, cx - 0.34), 3), round(max(0.06, crest_y - 0.18), 3)]
        k_end = [round(max(0.18, cx - 0.16) if from_right else min(0.82, cx + 0.16), 3), round(min(0.92, bottom + 0.14), 3)]
        actions: list[dict[str, Any]] = [{
            "operation": "create_mask",
            "kind": "linear",
            "name": "Window Key Light",
            "start": k_start,
            "end": k_end,
            "exposure_ev": round(_clamp((0.48 if is_dark_subject else 0.54) * scale, 0.20, 1.20), 2),
            "highlights": round(_clamp(0.18 * scale, 0.05, 0.45), 2),
            "shadows": round(_clamp((0.34 if is_dark_subject else 0.22) * scale, 0.05, 0.65), 2),
            "warmth": round(_clamp(0.34 * scale, 0.10, 0.65), 2),
            "inverted": False,
            "angle": 0.0,
            "feather": 0.0,
        }]
        if max_masks >= 2:
            actions.append({
                "operation": "create_mask",
                "kind": "linear",
                "name": "Shadow Falloff",
                "start": [round(max(0.04, 1.0 - k_start[0]), 3), round(min(0.94, bottom + 0.16), 3)],
                "end": [round(cx, 3), round(crest_y, 3)],
                "exposure_ev": round(_clamp(-0.26 * scale, -0.60, -0.10), 2),
                "highlights": -0.10,
                "shadows": -0.14,
                "warmth": -0.12,
                "inverted": False,
                "angle": 0.0,
                "feather": 0.0,
            })
        return actions

    if recipe == "sunbeam":
        span_w = max(0.42, min(0.76, bw * 2.5))
        span_h = max(0.36, min(0.70, (bottom - sy) * 0.95))
        beam_cx = max(0.16, min(0.84, 0.68 * cx + 0.32 * sx))
        beam_cy = max(0.16, min(0.76, 0.58 * crest_y + 0.42 * max(0.08, sy + 0.10)))
    elif recipe == "rim_backlight":
        span_w = max(0.32, min(0.64, bw * 1.9))
        span_h = max(0.22, min(0.48, bh * 1.9))
        beam_cx = cx
        beam_cy = max(0.08, top + 0.06 * bh)
    elif recipe == "mountain_beam":
        span_w = max(0.48, min(0.88, bw * 2.4))
        span_h = max(0.30, min(0.62, bh * 2.2))
        beam_cx = cx
        beam_cy = max(0.14, min(0.78, crest_y + 0.12 * bh))
    else:
        # Overhead 3D spotlight: anchor upper-crest core slightly above subject spine so the
        # top surface gets peak illumination, side slopes roll off gently, and ground shadows hold.
        span_w = max(0.24, min(0.50, bw * 1.65))
        span_h = max(0.18, min(0.42, bh * 2.10))
        beam_cx = max(0.10, min(0.90, cx + (sx - cx) * 0.10))
        beam_cy = max(0.10, min(0.84, top + 0.18 * bh - min(0.012, 0.08 * bh)))

    b_x0 = round(_clamp(beam_cx - span_w / 2.0, 0.02, 0.74), 3)
    b_y0 = round(_clamp(beam_cy - span_h / 2.0, 0.02, 0.80), 3)
    b_x1 = round(_clamp(b_x0 + span_w, min(0.98, b_x0 + 0.22), 0.98), 3)
    b_y1 = round(_clamp(b_y0 + span_h, min(0.98, b_y0 + 0.18), 0.98), 3)

    key_ev = round(_clamp((0.40 if is_dark_subject else 0.48) * scale, 0.24, 1.10), 2)
    key_hl = round(_clamp((0.20 if recipe != "sunbeam" else 0.24) * scale, 0.06, 0.45), 2)
    key_sh = round(_clamp((0.36 if is_dark_subject else 0.22) * scale, 0.12, 0.65), 2)
    key_warmth = round(_clamp((0.32 if recipe == "spotlight" else 0.44) * scale, 0.15, 0.65), 2)
    key_feather = 0.84 if recipe in ("sunbeam", "mountain_beam") else 0.82

    key_name = {
        "sunbeam": "Sunbeam Shaft",
        "rim_backlight": "Rim Backlight",
        "mountain_beam": "Overhead Ridge Beam",
    }.get(recipe, "Overhead Key Light")

    actions = [{
        "operation": "create_mask",
        "kind": "radial",
        "name": key_name,
        "start": [b_x0, b_y0],
        "end": [b_x1, b_y1],
        "exposure_ev": key_ev,
        "highlights": key_hl,
        "shadows": key_sh,
        "warmth": key_warmth,
        "inverted": False,
        "angle": beam_angle,
        "feather": key_feather,
    }]
    if max_masks >= 2 and recipe != "rim_backlight":
        f_x0 = round(_clamp(b_x0 - 0.06, 0.01, 0.70), 3)
        f_y0 = round(_clamp(b_y0 - 0.05, 0.01, 0.72), 3)
        f_x1 = round(_clamp(b_x1 + 0.06, min(0.99, f_x0 + 0.26), 0.99), 3)
        f_y1 = round(_clamp(b_y1 + 0.06, min(0.99, f_y0 + 0.22), 0.99), 3)
        actions.append({
            "operation": "create_mask",
            "kind": "radial",
            "name": "Ambient Falloff",
            "start": [f_x0, f_y0],
            "end": [f_x1, f_y1],
            "exposure_ev": round(_clamp(-0.28 * scale, -0.55, -0.14), 2),
            "highlights": -0.14,
            "shadows": -0.12,
            "warmth": -0.10,
            "inverted": True,
            "angle": beam_angle,
            "feather": 0.78,
        })
    if max_masks >= 3 and recipe == "spotlight":
        # Focused upper-crest rim catch light: peaks on the upward-facing top surface and fades
        # cleanly across the side slope without illuminating the ground beneath.
        c_w = max(0.18, min(0.36, bw * 1.25))
        c_h = max(0.08, min(0.22, bh * 1.15))
        c_cy = max(0.06, min(0.88, top + 0.22 * bh))
        c_x0 = round(_clamp(cx - c_w / 2.0, 0.03, 0.78), 3)
        c_y0 = round(_clamp(c_cy - c_h / 2.0, 0.03, 0.86), 3)
        c_x1 = round(_clamp(c_x0 + c_w, min(0.97, c_x0 + 0.18), 0.97), 3)
        c_y1 = round(_clamp(c_y0 + c_h, min(0.97, c_y0 + 0.08), 0.97), 3)
        actions.append({
            "operation": "create_mask",
            "kind": "radial",
            "name": "Top Crest Highlight",
            "start": [c_x0, c_y0],
            "end": [c_x1, c_y1],
            "exposure_ev": round(_clamp((0.30 if is_dark_subject else 0.24) * scale, 0.14, 0.65), 2),
            "highlights": round(_clamp(0.26 * scale, 0.08, 0.45), 2),
            "shadows": round(_clamp((0.32 if is_dark_subject else 0.16) * scale, 0.10, 0.55), 2),
            "warmth": round(_clamp(0.22 * scale, 0.08, 0.45), 2),
            "inverted": False,
            "angle": beam_angle,
            "feather": 0.85,
        })
    return actions


def analyze_scene_lighting(preview: np.ndarray, question: str = "", goal: str = "") -> dict[str, Any]:
    """Analyze preview pixels to locate the ridge contour, peak, natural light vector, and sunlit slope."""
    h, w = preview.shape[:2]
    if h < 20 or w < 20:
        return {}
    rgb = preview.astype(np.float32) / 255.0
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    if float(luma.max() - luma.min()) < 0.02:
        return {}
    row_luma = luma.mean(axis=1)

    # 1. Trace 1D skyline / mountain ridge profile across horizontal bins
    mean_luma = float(luma.mean())
    p15_luma = float(np.percentile(luma, 15))
    p90_luma = float(np.percentile(luma, 90))
    if mean_luma < 0.30:
        # preview luma is sRGB-encoded (~Y^(1/2.2)), so linear EV scales by ~2.2 * log2(target / mean_luma)
        rec_exposure_ev = _clamp(2.2 * math.log2(0.285 / max(0.11, mean_luma)) * 0.68, 0.35, 1.10)
    elif mean_luma > 0.56:
        rec_exposure_ev = _clamp(2.2 * math.log2(0.46 / mean_luma) * 0.50, -0.55, -0.10)
    else:
        rec_exposure_ev = 0.25
    rec_shadows = _clamp(0.34 if p15_luma < 0.14 else 0.24, 0.16, 0.42)
    rec_highlights = _clamp(-0.22 if p90_luma > 0.65 else -0.16, -0.35, -0.10)
    r_lo, r_hi = max(2, int(round(h * 0.16))), max(4, min(h - 2, int(round(h * 0.70))))
    gx_full = np.abs(np.diff(luma, axis=1, prepend=luma[:, :1]))
    gy_full = np.abs(np.diff(luma, axis=0, prepend=luma[:1, :]))
    edge_energy = gx_full + 1.4 * gy_full
    n_bins = max(4, min(16, w // 2))
    ridge_ys: list[float] = []
    ridge_xs: list[float] = []
    for b in range(n_bins):
        c0 = min(w - 1, int(round(b * w / n_bins)))
        c1 = max(c0 + 1, min(w, int(round((b + 1) * w / n_bins))))
        col_luma = luma[:, c0:c1].mean(axis=1)
        col_edge = edge_energy[:, c0:c1].mean(axis=1)
        best_r_bin = int(round(h * 0.40))
        best_s = -1e9
        for r in range(r_lo, r_hi):
            above_l = float(col_luma[:r].mean())
            r_end = max(r + 1, min(h, r + max(1, int(round(h * 0.18)))))
            below_l = float(col_luma[r:r_end].mean())
            above_e = float(col_edge[:r].mean())
            local_e = float(col_edge[max(0, r - 2):min(h, r + 3)].mean())
            drop = float(col_luma[max(0, r - 3)] - col_luma[min(h - 1, r + 3)])
            s = (above_l - below_l) * 1.6 + drop * 1.2 + (local_e - above_e) * 2.2
            if s > best_s:
                best_s = s
                best_r_bin = r
        ridge_ys.append(_clamp(best_r_bin / max(1, h - 1), 0.18, 0.68))
        ridge_xs.append(round((b + 0.5) / n_bins, 3))

    # Dominant peak in central bins
    central_indices = range(max(0, n_bins // 6), max(1, n_bins - max(1, n_bins // 6)))
    peak_idx = min(central_indices, key=lambda idx: ridge_ys[idx])
    peak_x = _clamp(ridge_xs[peak_idx], 0.25, 0.75)
    peak_y = _clamp(ridge_ys[peak_idx], 0.18, 0.58)
    horizon_y = _clamp(float(np.median(ridge_ys)), 0.22, 0.66)
    best_r = int(round(horizon_y * (h - 1)))
    ridge_relief_span = float(max(ridge_ys) - min(ridge_ys))
    q_combined = f"{question} {goal}"
    wants_center_subject = bool(re.search(
        r"\b(?:center|middle|car|vehicle|road|street|subject|person|seal|sea ?lion|animal|bird|dog|cat|wildlife|spotlight|foreground)s?\b|"
        r"中间|中央|车|车辆|车道|道路|马路|街道|主体|人物|海狮|海豹|动物|聚光|前景",
        q_combined, re.I,
    ))
    wants_mountain = bool(re.search(
        r"\b(?:mountain|ridge|slope|peak|cliff)\b|山|山体|山峰|山脊|山坡",
        q_combined, re.I,
    ))
    is_mountain_scene = wants_mountain or (ridge_relief_span >= 0.07 and not wants_center_subject)

    # 2. Locate natural sky & horizon light source
    sky_rows = max(2, int(round(min(0.28, max(peak_y, horizon_y)) * h)))
    sky_warmth = 0.35 * luma[:sky_rows, :] + 1.80 * np.maximum(0.0, rgb[:sky_rows, :, 0] - rgb[:sky_rows, :, 2] * 0.78)
    col_weights = np.maximum(0.005, sky_warmth.mean(axis=0))
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)
    top_mask = col_weights >= np.percentile(col_weights, 70)
    source_x = _clamp(float((xs[top_mask] * col_weights[top_mask]).sum() / col_weights[top_mask].sum()), 0.14, 0.86)

    # Check existing sunlit vs shadowed face
    y_top = max(0, min(h - 2, int(round(peak_y * h))))
    y_bot = min(h, max(y_top + 2, int(round(min(0.84, peak_y + 0.36) * h))))
    mid_c = max(1, min(w - 1, int(round((0.50 if not is_mountain_scene else peak_x) * w))))
    span_c = max(1, int(w * 0.22))
    left_face_luma = float((luma[y_top:y_bot, max(0, mid_c - span_c):mid_c] + 0.8 * (rgb[y_top:y_bot, max(0, mid_c - span_c):mid_c, 0] - rgb[y_top:y_bot, max(0, mid_c - span_c):mid_c, 2])).mean())
    right_face_luma = float((luma[y_top:y_bot, mid_c:min(w, mid_c + span_c)] + 0.8 * (rgb[y_top:y_bot, mid_c:min(w, mid_c + span_c), 0] - rgb[y_top:y_bot, mid_c:min(w, mid_c + span_c), 2])).mean())

    # 3. Locate target subject (car/road in central corridor or sun-facing mountain slope)
    if not is_mountain_scene:
        c_lo, c_hi = max(0, int(round(w * 0.33))), min(w, max(2, int(round(w * 0.67))))
        r_sub_lo = max(0, min(h - 2, int(round(max(0.45, horizon_y + 0.05) * h))))
        r_sub_hi = min(h, max(r_sub_lo + 2, int(round(h * 0.76))))
        sub_edge = edge_energy[r_sub_lo:r_sub_hi, c_lo:c_hi]
        if sub_edge.size > 0 and float(sub_edge.sum()) > 1e-6:
            yy, xx = np.meshgrid(
                np.linspace(r_sub_lo / max(1, h - 1), (r_sub_hi - 1) / max(1, h - 1), sub_edge.shape[0]),
                np.linspace(c_lo / max(1, w - 1), (c_hi - 1) / max(1, w - 1), sub_edge.shape[1]),
                indexing="ij",
            )
            p80 = np.percentile(sub_edge, 80)
            hot = sub_edge >= p80
            w_hot = np.maximum(1e-5, sub_edge[hot])
            target_x = _clamp(float((xx[hot] * w_hot).sum() / w_hot.sum()), 0.36, 0.64)
            target_y = _clamp(float((yy[hot] * w_hot).sum() / w_hot.sum()), 0.48, 0.68)
        else:
            target_x, target_y = 0.45, 0.56
        if left_face_luma >= right_face_luma:
            source_x = _clamp(min(source_x, target_x - 0.22), 0.14, 0.32)
        elif abs(source_x - target_x) < 0.18:
            source_x = _clamp(target_x + (0.22 if source_x >= target_x else -0.22), 0.14, 0.86)
        source_y = _clamp(horizon_y - 0.12, 0.16, 0.30)
    else:
        slope_bias_x = 0.05 if (right_face_luma >= left_face_luma * 0.98 and source_x <= 0.54) else (-0.05 if source_x > 0.56 else 0.03)
        band = luma[y_top:y_bot, :]
        gx = np.abs(np.diff(band, axis=1, prepend=band[:, :1]))
        gy = np.abs(np.diff(band, axis=0, prepend=band[:1, :]))
        relief = (gx + gy) * (1.20 - np.abs(band - 0.28))
        if relief.size > 0 and float(relief.sum()) > 1e-6:
            yy, xx = np.meshgrid(
                np.linspace(y_top / max(1, h - 1), (y_bot - 1) / max(1, h - 1), band.shape[0]),
                np.linspace(0.0, 1.0, w),
                indexing="ij",
            )
            p82 = np.percentile(relief, 82)
            salient = relief >= p82
            w_sal = np.maximum(1e-5, relief[salient])
            sal_x = float((xx[salient] * w_sal).sum() / w_sal.sum())
            sal_y = float((yy[salient] * w_sal).sum() / w_sal.sum())
        else:
            sal_x = peak_x
            sal_y = min(0.68, peak_y + 0.16)
        target_x = _clamp(0.58 * peak_x + 0.42 * sal_x + slope_bias_x, 0.28, 0.72)
        target_y = _clamp(max(peak_y + 0.17, 0.55 * (peak_y + 0.18) + 0.45 * sal_y), peak_y + 0.15, 0.72)
        if abs(source_x - target_x) < 0.15:
            source_x = _clamp(target_x + (-0.20 if slope_bias_x >= 0 else 0.20), 0.16, 0.84)
        source_y = _clamp(peak_y - 0.04, 0.14, 0.34)

    # 4. Compute scene-anchored directional mask geometry & grazing angle
    if source_x <= target_x:
        dir_cn, dir_en = "左上方天际破云斜射光", "upper-left breaking sky light"
    else:
        dir_cn, dir_en = "右上方天际斜射霞光", "upper-right grazing twilight light"
    dx_px = (target_x - source_x) * w
    dy_px = (target_y - source_y) * h
    raw_deg = math.degrees(math.atan2(dy_px, dx_px))
    if raw_deg > 90.0:
        raw_deg -= 180.0
    elif raw_deg < -90.0:
        raw_deg += 180.0
    ridge_left = float(np.mean(ridge_ys[:4]))
    ridge_center = float(np.mean(ridge_ys[5:11]))
    ridge_right = float(np.mean(ridge_ys[12:]))
    if is_mountain_scene:
        # Center the directional light beam on the detected sunlit ridge/slope (target_x, target_y)
        ridge_top = _clamp(max(0.24, min(peak_y, horizon_y) - 0.01), 0.24, 0.42)
        ridge_bot = _clamp(max(target_y + 0.24, ridge_top + 0.48), 0.72, 0.90)
        left_x = _clamp(target_x - 0.44, 0.02, 0.22)
        right_x = _clamp(target_x + 0.44, max(left_x + 0.64, 0.78), 0.98)
        beam_start = (round(left_x, 2), round(ridge_top, 2))
        beam_end = (round(right_x, 2), round(ridge_bot, 2))
        ridge_tilt = (ridge_right - ridge_left) * 42.0
        grazing_sign = 1.0 if source_x <= target_x else -1.0
        beam_angle = round(_clamp(ridge_tilt + grazing_sign * 10.0, -18.0, 18.0), 1)
    else:
        # Place strictly below the sky horizon along the street/subject corridor (never floating in the open sky)
        top_y = _clamp(max(0.38, horizon_y + 0.01), 0.36, 0.52)
        bot_y = _clamp(max(target_y + 0.26, top_y + 0.40), 0.74, 0.88)
        left_x = _clamp(min(source_x, target_x - 0.28), 0.14, 0.28)
        right_x = _clamp(max(target_x + 0.32, left_x + 0.58), 0.76, 0.90)
        beam_start = (round(left_x, 2), round(top_y, 2))
        beam_end = (round(right_x, 2), round(bot_y, 2))
        beam_angle = round(_clamp(raw_deg * 0.45, -24.0, 24.0), 1)
    terrain_luma = float(luma[best_r:, :].mean()) if best_r < h else mean_luma
    sky_luma = float(luma[:best_r, :].mean()) if best_r > 0 else mean_luma
    subject_candidates = _detect_foreground_subjects(luma, edge_energy, horizon_y)
    if (not is_mountain_scene) and subject_candidates:
        primary = subject_candidates[0]
        target_x = _clamp(primary["center"][0], 0.08, 0.92)
        target_y = _clamp(primary["center"][1], 0.14, 0.90)
        primary_box = tuple(primary["bbox"])
        primary_dark = bool(primary["is_dark_subject"])
    else:
        primary_box = (
            _clamp(target_x - 0.16, 0.05, 0.72),
            _clamp(target_y - 0.12, 0.08, 0.72),
            _clamp(target_x + 0.16, 0.28, 0.95),
            _clamp(target_y + 0.12, 0.28, 0.92),
        )
        primary_dark = bool(mean_luma < 0.35)
    recommended_recipes = {
        "spotlight": build_overhead_spotlight_actions(
            primary_box,
            (round(source_x, 2), round(source_y, 2)),
            is_dark_subject=primary_dark,
            recipe="spotlight",
        ),
        "sunbeam": build_overhead_spotlight_actions(
            primary_box,
            (round(source_x, 2), round(source_y, 2)),
            is_dark_subject=primary_dark,
            recipe="sunbeam",
        ),
    }

    return {
        "horizon_y": horizon_y,
        "peak_x": peak_x,
        "peak_y": peak_y,
        "ridge_y_left": round(ridge_left, 2),
        "ridge_y_center": round(ridge_center, 2),
        "ridge_y_right": round(ridge_right, 2),
        "source_x": round(source_x, 2),
        "source_y": round(source_y, 2),
        "dir_cn": dir_cn,
        "dir_en": dir_en,
        "target_x": round(target_x, 2),
        "target_y": round(target_y, 2),
        "beam_start": (round(beam_start[0], 2), round(beam_start[1], 2)),
        "beam_end": (round(beam_end[0], 2), round(beam_end[1], 2)),
        "beam_angle": beam_angle,
        "terrain_luma": terrain_luma,
        "sky_luma": sky_luma,
        "mean_luma": mean_luma,
        "rec_exposure_ev": rec_exposure_ev,
        "rec_shadows": rec_shadows,
        "rec_highlights": rec_highlights,
        "is_mountain_scene": is_mountain_scene,
        "subject_candidates": subject_candidates,
        "recommended_recipes": recommended_recipes,
    }


def _upsert_gradients(existing: tuple[LinearGradient, ...],
                      new_masks: list[LinearGradient]) -> tuple[LinearGradient, ...]:
    check_mask_space(existing, sum(all(m.id != g.id for g in existing) for m in new_masks))
    result = list(existing)
    for mask in new_masks:
        idx = next((i for i, g in enumerate(result)
                    if g.id == mask.id), None)
        if idx is not None:
            result[idx] = replace(mask, id=result[idx].id)
        else:
            result.append(mask)
    validate_gradients(tuple(result))
    return tuple(result)


def make_lighting_action(recipe_id: str, values: EditValues,
                         custom_ev: float | None = None,
                         custom_start: tuple[float, float] | None = None,
                         custom_end: tuple[float, float] | None = None,
                         *,
                         custom_warmth: float | None = None,
                         custom_angle: float | None = None,
                         custom_highlights: float | None = None,
                         custom_shadows: float | None = None,
                         custom_feather: float | None = None,
                         preview: np.ndarray | None = None,
                         question: str = "", source_size=None) -> CoachAction:
    """Build a single-transaction Local Mask artificial lighting action."""
    existing = values.local_gradients
    scene = analyze_scene_lighting(preview, question=question) if isinstance(preview, np.ndarray) and preview.ndim == 3 else None
    def upsert(masks):
        return _upsert_gradients(existing, [map_preview_mask(mask, values, source_size) for mask in masks])

    if recipe_id in ("mountain_beam", "light_beam"):
        is_mountain = (
            recipe_id == "mountain_beam"
            and not re.search(r"\b(?:car|road|street|vehicle|person)\b|车|车辆|车道|道路|马路|街道|人物", question, re.I)
            and (not scene or scene.get("is_mountain_scene", True) or bool(re.search(r"\b(?:mountain|ridge|slope|peak|cliff)\b|山|山体|山峰|山脊|山坡", question, re.I)))
        )
        ev = 0.68 if custom_ev is None else custom_ev
        start = custom_start or (scene["beam_start"] if scene else ((0.04, 0.29) if is_mountain else (0.16, 0.40)))
        end = custom_end or (scene["beam_end"] if scene else ((0.96, 0.75) if is_mountain else (0.86, 0.84)))
        warmth = (0.56 if is_mountain else 0.52) if custom_warmth is None else custom_warmth
        angle = _clamp(custom_angle if custom_angle is not None else (scene["beam_angle"] if scene else (12.0 if is_mountain else 18.0)), -90.0, 90.0)
        hl = _clamp(custom_highlights if custom_highlights is not None else 0.16, -1.0, 1.0)
        sh = _clamp(custom_shadows if custom_shadows is not None else 0.30, -1.0, 1.0)
        feather = _clamp(custom_feather if custom_feather is not None else 0.82, 0.0, 1.0)
        beam_id = "coach-mountain-beam" if is_mountain else "coach-light-beam"
        beam_name = "Mountain Beam" if is_mountain else "Light Beam"
        falloff_id = "coach-valley-falloff" if is_mountain else "coach-ambient-falloff"
        falloff_name = "Valley Falloff" if is_mountain else "Ambient Falloff"
        beam = RadialGradient(
            id=beam_id,
            name=beam_name,
            start=start,
            end=end,
            exposure_ev=ev,
            highlights=hl,
            shadows=sh,
            feather=feather,
            warmth=warmth,
            angle=angle,
        )
        if is_mountain:
            falloff: LinearGradient = LinearGradient(
                id=falloff_id,
                name=falloff_name,
                start=(0.50, 0.02),
                end=(0.50, _clamp(start[1] + 0.08, 0.32, 0.46)),
                exposure_ev=-0.28,
                highlights=-0.26,
                shadows=-0.08,
                warmth=-0.16,
            )
        else:
            falloff = RadialGradient(
                id=falloff_id,
                name=falloff_name,
                start=(_clamp(start[0] - 0.06, 0.04, 0.80), _clamp(start[1] - 0.04, 0.28, 0.80)),
                end=(_clamp(end[0] + 0.06, 0.20, 0.96), _clamp(end[1] + 0.06, 0.40, 0.96)),
                exposure_ev=-0.26,
                highlights=-0.16,
                shadows=-0.14,
                inverted=True,
                feather=0.84,
                warmth=-0.16,
                angle=angle,
            )
        masks = [beam, falloff]
        updated = upsert(masks)
        target = beam.id
        coord_changes: list[tuple[str, Any]] = []
        if abs(values.exposure_ev) < 0.15:
            base_ev = scene["rec_exposure_ev"] if scene else (0.50 if is_mountain else 0.40)
            coord_changes.append(("exposure_ev", base_ev))
        if abs(values.highlights) < 0.08:
            coord_changes.append(("highlights", scene["rec_highlights"] if scene else -0.16))
        if abs(values.shadows) < 0.08:
            coord_changes.append(("shadows", scene["rec_shadows"] if scene else 0.26))
        if abs(values.whites) < 0.08:
            coord_changes.append(("whites", 0.18 if is_mountain else 0.15))
        if abs(values.blacks) < 0.05:
            coord_changes.append(("blacks", -0.06))
        if abs(values.contrast) < 0.06:
            coord_changes.append(("contrast", 0.06))
        if abs(values.vibrance) < 0.08:
            coord_changes.append(("vibrance", 0.24 if is_mountain else 0.20))
        if values.rgb_curve == ((0.0, 0.0), (1.0, 1.0)):
            coord_changes.append(("rgb_curve", ((0.0, 0.0), (0.08, 0.07), (0.28, 0.32), (0.56, 0.62), (0.82, 0.86), (1.0, 1.0))))
        coord_changes.append(("local_gradients", updated))
        label = f"Apply {beam_name} + {falloff_name} & Tonal Grade ({ev:+.2f} EV, Warmth {warmth:+.2f})"
        return CoachAction(
            id=f"lighting.{recipe_id}",
            label=label,
            parameter="local_gradients",
            new_value=updated,
            focus_control=f"local_gradients.{target}",
            changes=tuple(coord_changes),
        )
    if recipe_id == "sunbeam":
        ev = _clamp(custom_ev if custom_ev is not None else 0.58, -3.0, 3.0)
        if scene:
            sx, sy = scene["source_x"], scene["source_y"]
            tx, ty = scene["target_x"], scene["target_y"]
            s_pt = (_clamp(min(sx, tx) - 0.22, 0.06, 0.45), _clamp(min(sy - 0.10, 0.06), 0.02, 0.18))
            e_pt = (_clamp(max(sx, tx) + 0.28, 0.55, 0.94), _clamp(max(ty + 0.16, 0.64), 0.58, 0.86))
            ang = _clamp(scene["beam_angle"], -35.0, 35.0)
        else:
            s_pt, e_pt, ang = (0.18, 0.02), (0.82, 0.68), 0.0
        mask = RadialGradient(
            id="coach-sunbeam",
            name="Sunbeam Glow",
            start=custom_start or s_pt,
            end=custom_end or e_pt,
            exposure_ev=ev,
            highlights=0.22,
            shadows=0.28,
            feather=0.82,
            warmth=0.48,
            angle=ang,
        )
        updated = upsert([mask])
        target = mask.id
        return CoachAction(
            id="lighting.sunbeam",
            label=f"Add Sunbeam Glow mask ({ev:+.2f} EV)",
            parameter="local_gradients",
            new_value=updated,
            focus_control=f"local_gradients.{target}",
        )

    if recipe_id == "window_light":
        ev = _clamp(custom_ev if custom_ev is not None else 0.52, -3.0, 3.0)
        from_right = bool(scene and scene["source_x"] > scene["target_x"])
        if scene:
            tx, ty = scene["target_x"], scene["target_y"]
            k_start = (0.94, _clamp(ty - 0.24, 0.10, 0.40)) if from_right else (0.06, _clamp(ty - 0.24, 0.10, 0.40))
            k_end = (_clamp(tx - 0.14, 0.22, 0.52) if from_right else _clamp(tx + 0.14, 0.48, 0.78), _clamp(ty + 0.14, 0.50, 0.82))
            f_start = (0.06, _clamp(ty + 0.22, 0.58, 0.88)) if from_right else (0.94, _clamp(ty + 0.22, 0.58, 0.88))
            f_end = (_clamp(tx + 0.06, 0.35, 0.65) if from_right else _clamp(tx - 0.06, 0.35, 0.65), _clamp(ty - 0.06, 0.32, 0.62))
        else:
            k_start, k_end = (0.06, 0.22), (0.68, 0.66)
            f_start, f_end = (0.94, 0.78), (0.42, 0.42)
        key_mask = LinearGradient(
            id="coach-window-key",
            name="Window Light",
            start=custom_start or k_start,
            end=custom_end or k_end,
            exposure_ev=ev,
            highlights=0.18,
            shadows=0.22,
            warmth=0.36,
        )
        falloff_mask = LinearGradient(
            id="coach-window-falloff",
            name="Shadow Falloff",
            start=f_start,
            end=f_end,
            exposure_ev=-0.26,
            highlights=-0.10,
            shadows=-0.15,
            warmth=-0.12,
        )
        masks = [key_mask, falloff_mask]
        updated = upsert(masks)
        target = key_mask.id
        return CoachAction(
            id="lighting.window_light",
            label=f"Add Side Window Light ({ev:+.2f} EV)",
            parameter="local_gradients",
            new_value=updated,
            focus_control=f"local_gradients.{target}",
        )

    if recipe_id == "rim_backlight":
        ev = _clamp(custom_ev if custom_ev is not None else 0.60, -3.0, 3.0)
        if scene:
            px, py = scene["peak_x"], scene["peak_y"]
            s_pt = (_clamp(px - 0.30, 0.08, 0.46), _clamp(py - 0.14, 0.06, 0.36))
            e_pt = (_clamp(px + 0.30, 0.54, 0.92), _clamp(py + 0.26, 0.42, 0.78))
        else:
            s_pt, e_pt = (0.22, 0.08), (0.78, 0.64)
        mask = RadialGradient(
            id="coach-rim-backlight",
            name="Backlight Glow",
            start=custom_start or s_pt,
            end=custom_end or e_pt,
            exposure_ev=ev,
            highlights=0.28,
            shadows=0.35,
            feather=0.85,
            warmth=0.42,
        )
        updated = upsert([mask])
        target = mask.id
        return CoachAction(
            id="lighting.rim_backlight",
            label=f"Add Backlight Glow mask ({ev:+.2f} EV)",
            parameter="local_gradients",
            new_value=updated,
            focus_control=f"local_gradients.{target}",
        )

    if recipe_id == "sky_filter":
        ev = _clamp(custom_ev if custom_ev is not None else -0.45, -3.0, 3.0)
        end_y = _clamp(scene["horizon_y"] + 0.06, 0.28, 0.62) if scene else 0.48
        mask = LinearGradient(
            id="coach-sky-filter",
            name="Sky Filter",
            start=custom_start or (0.50, 0.04),
            end=custom_end or (0.50, end_y),
            exposure_ev=ev,
            highlights=-0.35,
            shadows=0.0,
            warmth=-0.12,
        )
        updated = upsert([mask])
        target = mask.id
        return CoachAction(
            id="lighting.sky_filter",
            label=f"Add Sky Filter mask ({ev:+.2f} EV)",
            parameter="local_gradients",
            new_value=updated,
            focus_control=f"local_gradients.{target}",
        )

    if recipe_id == "foreground_fill":
        ev = _clamp(custom_ev if custom_ev is not None else 0.38, -3.0, 3.0)
        end_y = _clamp(max(scene["horizon_y"] + 0.08, scene["target_y"] - 0.04), 0.38, 0.68) if scene else 0.50
        mask = LinearGradient(
            id="coach-foreground-fill",
            name="Foreground Fill",
            start=custom_start or (0.50, 0.94),
            end=custom_end or (0.50, end_y),
            exposure_ev=ev,
            highlights=0.0,
            shadows=0.32,
            warmth=0.22,
        )
        updated = upsert([mask])
        target = mask.id
        return CoachAction(
            id="lighting.foreground_fill",
            label=f"Add Foreground Fill mask ({ev:+.2f} EV)",
            parameter="local_gradients",
            new_value=updated,
            focus_control=f"local_gradients.{target}",
        )

    # Default recipe: Overhead 3D Subject Spotlight + Ambient Falloff (anchored on upper subject crest)
    ev = _clamp(custom_ev if custom_ev is not None else 0.46, -3.0, 3.0)
    ang = _clamp(custom_angle if custom_angle is not None else 0.0, -90.0, 90.0)
    if scene and scene.get("subject_candidates"):
        cand = scene["subject_candidates"][0]
        rig = build_overhead_spotlight_actions(
            cand["bbox"],
            (scene["source_x"], scene["source_y"]),
            is_dark_subject=bool(cand.get("is_dark_subject", True)),
            recipe="spotlight",
        )
        k_spec, f_spec = rig[0], rig[1]
        k_start = tuple(k_spec["start"])
        k_end = tuple(k_spec["end"])
        a_start = tuple(f_spec["start"])
        a_end = tuple(f_spec["end"])
        if custom_ev is None:
            ev = float(k_spec["exposure_ev"])
        if custom_angle is None:
            ang = float(k_spec["angle"])
        hl = float(k_spec["highlights"])
        sh = float(k_spec["shadows"])
        feather = float(k_spec["feather"])
        w_spot = float(k_spec["warmth"])
    elif scene:
        tx, ty = scene["target_x"], scene["target_y"]
        k_start = (_clamp(tx - 0.24, 0.08, 0.52), _clamp(ty - 0.28, 0.08, 0.52))
        k_end = (_clamp(tx + 0.24, 0.48, 0.92), _clamp(ty + 0.22, 0.46, 0.92))
        a_start = (_clamp(k_start[0] - 0.06, 0.04, 0.46), _clamp(k_start[1] - 0.06, 0.04, 0.48))
        a_end = (_clamp(k_end[0] + 0.06, 0.54, 0.96), _clamp(k_end[1] + 0.06, 0.52, 0.96))
        hl, sh, feather, w_spot = 0.20, 0.34, 0.78, 0.34
    else:
        k_start, k_end = (0.24, 0.16), (0.76, 0.84)
        a_start, a_end = (0.18, 0.12), (0.82, 0.88)
        hl, sh, feather, w_spot = 0.14, 0.28, 0.72, 0.0
    key_light = RadialGradient(
        id="coach-key-light",
        name="Key Light",
        start=custom_start or k_start,
        end=custom_end or k_end,
        exposure_ev=ev,
        highlights=hl if custom_highlights is None else _clamp(custom_highlights, -1.0, 1.0),
        shadows=sh if custom_shadows is None else _clamp(custom_shadows, -1.0, 1.0),
        feather=feather if custom_feather is None else _clamp(custom_feather, 0.0, 1.0),
        warmth=w_spot if custom_warmth is None else _clamp(custom_warmth, -1.0, 1.0),
        angle=ang,
    )
    ambient_falloff = RadialGradient(
        id="coach-ambient-falloff",
        name="Ambient Falloff",
        start=a_start,
        end=a_end,
        exposure_ev=-0.28,
        highlights=-0.14,
        shadows=-0.12,
        inverted=True,
        feather=0.78,
        warmth=-0.10,
        angle=ang,
    )
    masks = [key_light, ambient_falloff]
    updated = upsert(masks)
    target = key_light.id
    label = f"Add 3D Spotlight + Falloff ({ev:+.2f} EV)"
    return CoachAction(
        id="lighting.spotlight",
        label=label,
        parameter="local_gradients",
        new_value=updated,
        focus_control=f"local_gradients.{target}",
    )



class MaskCapacityError(ValueError):
    pass


def check_mask_space(existing, needed):
    free = 4 - len(existing)
    if needed > free:
        raise MaskCapacityError(f"This suggestion needs {needed} free mask slots. Available: {free}.")


def map_preview_mask(mask, values, size):
    """Map normalized preview geometry into oriented source pixels."""
    comp, distortion = values.composition, values.lens_corrections.distortion
    if comp.angle == 0 and comp.crop == (0, 0, 1, 1) and not distortion:
        return mask
    if not size or min(size) <= 0:
        raise ValueError("Original image dimensions are needed to position this mask.")
    canvas = np.asarray(output_size(size, comp))

    def mapped(point):
        return np.asarray(source_point(point, size, comp, distortion=distortion))

    start, end = np.asarray(mask.start) * canvas, np.asarray(mask.end) * canvas
    if not isinstance(mask, RadialGradient) or (not comp.angle and not distortion):
        return replace(mask, start=tuple(mapped(start)/size), end=tuple(mapped(end)/size))
    center = (start + end) / 2
    theta = math.radians(mask.angle)
    c, s = math.cos(theta), math.sin(theta)
    rx, ry = abs(end - start) / 2
    axes = np.array([[rx*c, rx*s], [-ry*s, ry*c]])
    origin = mapped(center)
    # A radial mask remains an ellipse. Fit its axes from the mapped diameters.
    basis = np.column_stack([(mapped(center+a) - mapped(center-a))/2 for a in axes])
    rotation, radii, _ = np.linalg.svd(basis)
    angle = (math.degrees(math.atan2(rotation[1, 0], rotation[0, 0])) + 90) % 180 - 90
    return replace(mask, start=tuple((origin-radii)/size), end=tuple((origin+radii)/size), angle=angle)


SCALARS = {"exposure_ev", "contrast", "saturation", "highlights", "shadows", "whites", "blacks", "vibrance"}


def strict_object(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


NUMBER = {"type": "number"}
POINT = {"type": "array", "items": NUMBER}
ACTION_SCHEMA = {"type": "array", "items": {"anyOf": [
    strict_object({"operation": {"type": "string", "enum": ["set", "add"]},
                   "parameter": {"type": "string"}, "value": NUMBER}),
    strict_object({"operation": {"type": "string", "enum": ["set_curve"]},
                   "points": {"type": "array", "items": POINT}}),
    strict_object({"operation": {"type": "string", "enum": ["set_wb"]},
                   "mode": {"type": "string", "enum": ["camera", "custom"]}, "gains": POINT}),
    strict_object({"operation": {"type": "string", "enum": ["create_mask"]},
                   "kind": {"type": "string", "enum": ["radial", "linear"]},
                   "name": {"type": "string"}, "start": POINT, "end": POINT,
                   "exposure_ev": NUMBER, "warmth": NUMBER, "highlights": NUMBER, "shadows": NUMBER,
                   "inverted": {"type": "boolean"}, "angle": NUMBER, "feather": NUMBER}),
]}}


def resolve_actions(items, baseline, size=None):
    """Resolve the complete proposal once against the request's immutable edit state."""
    if not isinstance(items, list) or len(items) > 24:
        raise ValueError("Invalid action list.")
    values, targets, labels, seen = baseline, {}, [], set()
    focus_control = ""
    masks_needed = sum(isinstance(item, dict) and item.get("operation") == "create_mask" for item in items)
    check_mask_space(values.local_gradients, masks_needed)
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Invalid action.")
        op = item.get("operation")
        if op in ("set", "add") and set(item) == {"operation", "parameter", "value"}:
            path, number = item["parameter"], item["value"]
            if not isinstance(path, str) or path in seen or type(number) not in (int, float) or not math.isfinite(number):
                raise ValueError("Invalid numeric adjustment.")
            seen.add(path)
            parts = path.split(".")
            name = parts[0]
            control = ".".join(parts[:2]) if name in ("local_gradients", "color_mixer") else path
            def target(current):
                return number if op == "set" else current + number
            if len(parts) == 1 and name in SCALARS:
                value = result = target(getattr(values, name))
            elif len(parts) == 2 and (name, parts[1]) in {
                ("lens_corrections", "distortion"), ("lens_corrections", "vignetting_ev"),
                *(("detail", key) for key in ("sharpening", "masking", "luminance_nr", "chroma_nr"))}:
                group = getattr(values, name)
                result = target(getattr(group, parts[1]))
                value = replace(group, **{parts[1]: result})
            elif len(parts) == 3 and name == "color_mixer" and parts[1] in COLOR_KEYS and parts[2] in AXES:
                bands = values.color_mixer.to_dict()
                result = target(bands[parts[1]][parts[2]])
                bands[parts[1]][parts[2]] = result
                value = type(values.color_mixer).from_dict(bands)
            elif len(parts) == 3 and name == "local_gradients":
                mask = next((g for g in baseline.local_gradients if g.id == parts[1]), None)
                if mask is None or parts[2] not in ("exposure_ev","warmth","highlights","shadows","angle","feather") or not hasattr(mask, parts[2]):
                    raise ValueError("Unknown mask control.")
                result = target(getattr(mask, parts[2]))
                value = tuple(replace(g, **{parts[2]: result}) if g.id == mask.id else g for g in values.local_gradients)
            else:
                raise ValueError("Unsupported editing control.")
            unit = " EV" if parts[-1] in ("exposure_ev", "vignetting_ev") else ""
            label = f"{path} → {result:g}{unit}"
        elif op == "set_curve" and set(item) == {"operation", "points"}:
            name, value = "rgb_curve", validate_curve(item["points"])
            control = name
            label = f"RGB curve → {value}"
        elif op == "set_wb" and set(item) == {"operation", "mode", "gains"}:
            if not isinstance(item["gains"], list) or any(type(g) not in (float, int) for g in item["gains"]):
                raise ValueError("White balance needs numeric gains.")
            name, value = "raw_options", RawOptions(item["mode"], tuple(item["gains"]))
            control = "white_balance"
            label = f"White balance → {value.wb_mode} {value.wb_gains}"
        elif op == "create_mask" and set(item) == set(ACTION_SCHEMA["items"]["anyOf"][-1]["required"]):
            fields = {k: item[k] for k in ("name","start","end","inverted","exposure_ev","highlights","shadows","warmth")}
            fields["id"] = "coach-" + uuid4().hex
            if item["kind"] == "radial":
                mask = RadialGradient(**fields, angle=item["angle"], feather=item["feather"])
            elif (item["kind"] == "linear" and all(type(item[k]) in (int, float) and item[k] == 0
                                                   for k in ("angle", "feather"))):
                mask = LinearGradient(**fields)
            else:
                raise ValueError("Invalid mask shape.")
            name = "local_gradients"
            control = f"{name}.{mask.id}"
            value = (*values.local_gradients, map_preview_mask(mask, baseline, size))
            label = (f"{mask.name} ({item['kind']}) {mask.start} → {mask.end}, exposure {mask.exposure_ev:g} EV, "
                     f"warmth {mask.warmth:g}, highlights {mask.highlights:g}, shadows {mask.shadows:g}, "
                     f"invert {mask.inverted}, angle {item['angle']:g}°, feather {item['feather']:g}")
        else:
            raise ValueError("Unknown or incomplete coach action.")
        if op in ("set_curve", "set_wb") and name in targets:
            raise ValueError("Conflicting actions.")
        values = replace(values, **{name: value})
        targets[name] = getattr(values, name)
        labels.append(label)
        if not focus_control or name == "local_gradients":
            focus_control = control
    if values == baseline:
        return ()
    first = next(iter(targets))
    return (CoachAction(first, " · ".join(labels), first, targets[first], focus_control, tuple(targets.items())),)
