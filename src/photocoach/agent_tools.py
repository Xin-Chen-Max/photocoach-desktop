"""Read-only multimodal inspection and sandbox trial tools for PhotoCoach Agent."""
from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import dataclass, field
import math
import threading
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw

from .coach import CoachRequest
from .coach_actions import (ACTION_SCHEMA, MaskCapacityError, analyze_scene_lighting,
                            build_overhead_spotlight_actions, resolve_actions)
from .coach_agent import (PLAN_SCHEMA, REGIONS, AgentPlan, AgentReport,
                          _region, build_reflexion_prompt_summary,
                          evaluate_trial_candidate, parse_plan)
from .color_mixer import COLOR_KEYS
from .gradients import LinearGradient, RadialGradient
from .state import EditState, EditValues


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    read_only: bool = True
    consumes_trial: bool = False


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolImage:
    label: str
    pixels: np.ndarray = field(repr=False)

    def __post_init__(self):
        arr = np.asarray(self.pixels)
        if (arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3
                or min(arr.shape[:2]) < 1 or max(arr.shape[:2]) > 768):
            raise ValueError("Tool images must be bounded RGB uint8 arrays <= 768px.")
        copy = np.array(arr, copy=True)
        copy.setflags(write=False)
        object.__setattr__(self, "pixels", copy)


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    ok: bool
    payload: dict[str, Any]
    images: tuple[ToolImage, ...] = ()
    error: str = ""


@dataclass
class ToolContext:
    """Mutable per-run sandbox state; holds zero StateStore or credential access."""
    request: CoachRequest
    state: EditState | None = None
    render_trial: Callable[[EditValues], np.ndarray] | None = None
    render_crop_100pct: Callable[..., tuple[np.ndarray, dict[str, Any]]] | None = None
    inspect_cfa: Callable[..., dict[str, Any]] | None = None
    playbook: Any | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    max_trials: int = 3
    inspected_100pct: bool = False
    inspected_crops: list[tuple[str, np.ndarray]] = field(default_factory=list)
    trial_count: int = 0
    trial_history: list[dict[str, Any]] = field(default_factory=list)
    best_trial: tuple[tuple[Any, ...], AgentPlan, AgentReport] | None = None
    last_trial: tuple[tuple[Any, ...], AgentPlan, AgentReport] | None = None


def _validate_value(value: Any, schema: dict[str, Any], path: str) -> Any:
    if "anyOf" in schema:
        for branch in schema["anyOf"]:
            try:
                return _validate_value(value, branch, path)
            except ValueError:
                continue
        raise ValueError(f"Invalid value for {path}.")
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be an object.")
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        if not required.issubset(value.keys()):
            missing = sorted(required - set(value.keys()))
            raise ValueError(f"{path} missing required field(s): {', '.join(missing)}.")
        if schema.get("additionalProperties", True) is False:
            extra = set(value.keys()) - set(props.keys())
            if extra:
                raise ValueError(f"{path} contains unsupported field(s): {', '.join(sorted(extra))}.")
        out = {}
        for k, v in value.items():
            if k in props:
                out[k] = _validate_value(v, props[k], f"{path}.{k}")
            else:
                out[k] = v
        return out
    if kind == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path} must be an array.")
        max_items = schema.get("maxItems", 32)
        min_items = schema.get("minItems", 0)
        if not min_items <= len(value) <= max_items:
            raise ValueError(f"{path} length must be between {min_items} and {max_items}.")
        item_schema = schema.get("items", {})
        return [_validate_value(item, item_schema, f"{path}[{i}]") for i, item in enumerate(value)]
    if kind == "string":
        if not isinstance(value, str):
            raise ValueError(f"{path} must be a string.")
        max_len = schema.get("maxLength", 1200)
        min_len = schema.get("minLength", 0)
        if not min_len <= len(value) <= max_len:
            raise ValueError(f"{path} string length out of range.")
        if "enum" in schema and value not in schema["enum"]:
            raise ValueError(f"{path} must be one of {', '.join(map(str, schema['enum']))}.")
        return value
    if kind == "integer":
        if isinstance(value, bool) or type(value) is not int:
            raise ValueError(f"{path} must be an integer.")
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path} must be >= {schema['minimum']}.")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path} must be <= {schema['maximum']}.")
        if "enum" in schema and value not in schema["enum"]:
            raise ValueError(f"{path} has unsupported integer value.")
        return value
    if kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path} must be a finite number.")
        num = float(value)
        if not math.isfinite(num):
            raise ValueError(f"{path} must be finite.")
        if "minimum" in schema and num < schema["minimum"]:
            raise ValueError(f"{path} must be >= {schema['minimum']}.")
        if "maximum" in schema and num > schema["maximum"]:
            raise ValueError(f"{path} must be <= {schema['maximum']}.")
        return num
    if kind == "boolean":
        if type(value) is not bool:
            raise ValueError(f"{path} must be a boolean.")
        return value
    return value


def validate_tool_arguments(spec: ToolSpec, arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    return _validate_value(arguments, spec.parameters, spec.name)


_HUE_CENTERS = (0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 270.0, 300.0)


def analyze_color_zones(pixels: np.ndarray, region: str = "whole") -> dict[str, Any]:
    """Compute 8-band hue distribution and approximate CIELAB stats on SDR sRGB pixels."""
    if region not in REGIONS:
        raise ValueError("Unsupported region name.")
    patch = _region(pixels, region)
    srgb = patch.astype(np.float32) / 255.0
    maxc = srgb.max(axis=2)
    minc = srgb.min(axis=2)
    delta = maxc - minc
    luma = 0.2126 * srgb[..., 0] + 0.7152 * srgb[..., 1] + 0.0722 * srgb[..., 2]

    # Standard HSL saturation & hue in degrees [0, 360)
    lightness = 0.5 * (maxc + minc)
    denom = 1.0 - np.abs(2.0 * lightness - 1.0)
    sat = np.where(delta > 1e-6, delta / np.maximum(1e-6, denom), 0.0)
    hue = np.zeros_like(delta)
    nonzero = delta > 1e-6
    r, g, b = srgb[..., 0], srgb[..., 1], srgb[..., 2]
    mask_r = nonzero & (maxc == r)
    mask_g = nonzero & (maxc == g) & (~mask_r)
    mask_b = nonzero & (~mask_r) & (~mask_g)
    hue[mask_r] = (60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0))
    hue[mask_g] = (60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0))
    hue[mask_b] = (60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0))
    hue = np.mod(hue, 360.0)

    colored = sat > 0.08
    centers_arr = np.asarray(_HUE_CENTERS, dtype=np.float32)
    # Assign each colored pixel to its nearest hue center across the 360-deg seam.
    dists = np.abs(((hue[..., None] - centers_arr + 180.0) % 360.0) - 180.0)
    nearest_band = np.argmin(dists, axis=-1)
    bands: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(COLOR_KEYS):
        in_band = colored & (nearest_band == idx)
        frac = float(in_band.mean())
        mean_sat = float(sat[in_band].mean()) if np.any(in_band) else 0.0
        bands[name] = {"fraction": round(frac, 4), "mean_saturation": round(mean_sat, 4)}

    # Approximate sRGB -> linear -> XYZ(D65) -> CIELAB
    lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    x = (0.4124564 * lin[..., 0] + 0.3575761 * lin[..., 1] + 0.1804375 * lin[..., 2]) / 0.95047
    y = (0.2126729 * lin[..., 0] + 0.7151522 * lin[..., 1] + 0.0721750 * lin[..., 2]) / 1.00000
    z = (0.0193339 * lin[..., 0] + 0.1191920 * lin[..., 1] + 0.9503041 * lin[..., 2]) / 1.08883
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    def f_lab(t):
        return np.where(t > eps, np.cbrt(np.maximum(0.0, t)), (kappa * t + 16.0) / 116.0)

    fx, fy, fz = f_lab(x), f_lab(y), f_lab(z)
    lab_l = 116.0 * fy - 16.0
    lab_a = 500.0 * (fx - fy)
    lab_b = 200.0 * (fy - fz)
    chroma = np.hypot(lab_a, lab_b)
    neutral_luma = (luma >= 0.20) & (luma <= 0.85)
    warm_skin_band = colored & (hue >= 12.0) & (hue <= 42.0) & (sat <= 0.75)
    sky_blue_band = colored & (hue >= 185.0) & (hue <= 245.0)
    return {
        "region": region,
        "pixels": int(luma.size),
        "lab_mean": {
            "L": round(float(lab_l.mean()), 2),
            "a": round(float(lab_a.mean()), 2),
            "b": round(float(lab_b.mean()), 2),
        },
        "midtone_cast_ab": {
            "a": round(float(lab_a[neutral_luma].mean()), 2) if np.any(neutral_luma) else round(float(lab_a.mean()), 2),
            "b": round(float(lab_b[neutral_luma].mean()), 2) if np.any(neutral_luma) else round(float(lab_b.mean()), 2),
        },
        "chroma_p95": round(float(np.percentile(chroma, 95)), 2),
        "neutral_fraction": round(float((~colored).mean()), 4),
        "warm_orange_hue_fraction": round(float(warm_skin_band.mean()), 4),
        "cool_blue_hue_fraction": round(float(sky_blue_band.mean()), 4),
        "hue_bands": bands,
        "note": "Hue-band and CIELAB statistics of the SDR preview; not semantic skin or sky detection.",
    }


def _handle_inspect_region_100pct(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    if ctx.render_crop_100pct is None:
        return ToolResult("", "inspect_region_100pct", False,
                          {"error": "Native 100% crop inspection is unavailable in this environment."},
                          error="unavailable")
    x_norm = float(args["x_norm"])
    y_norm = float(args["y_norm"])
    size_px = int(args.get("size_px", 512))
    apply_edits = bool(args.get("apply_current_edits", True))
    patch, metrics = ctx.render_crop_100pct(x_norm, y_norm, size_px, apply_current_edits=apply_edits)
    image = ToolImage(f"100% native crop at ({x_norm:.2f}, {y_norm:.2f})", patch)
    ctx.inspected_100pct = True
    if len(ctx.inspected_crops) < 4:
        ctx.inspected_crops.append((image.label, image.pixels))
    return ToolResult("", "inspect_region_100pct", True,
                      {"x_norm": x_norm, "y_norm": y_norm, "apply_current_edits": apply_edits, **metrics},
                      images=(image,))


def _handle_inspect_raw_headroom(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    if ctx.inspect_cfa is None:
        return ToolResult("", "inspect_raw_headroom", False,
                          {"error": "Raw Bayer CFA inspection is unavailable in this environment."},
                          error="unavailable")
    x_norm = float(args.get("x_norm", 0.5))
    y_norm = float(args.get("y_norm", 0.5))
    patch_px = int(args.get("patch_px", 64))
    stats = ctx.inspect_cfa(x_norm, y_norm, patch_px)
    return ToolResult("", "inspect_raw_headroom", True, stats)


def _handle_analyze_color_zones(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    region = args.get("region", "whole")
    return ToolResult("", "analyze_color_zones", True, analyze_color_zones(ctx.request.current_preview, region))


def _render_som_lighting_overlay(preview: np.ndarray, anchors: dict[str, Any]) -> np.ndarray:
    """Create a read-only Set-of-Mark numbered overlay for model visual grounding."""
    img = Image.fromarray(preview).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    # Subtle normalized grid ticks (0.2, 0.4, 0.6, 0.8) along edges
    grid_color = (165, 195, 225)
    for frac in (0.2, 0.4, 0.6, 0.8):
        gx = int(round(frac * (w - 1)))
        gy = int(round(frac * (h - 1)))
        for x in range(0, w, 12):
            draw.line([(x, gy), (min(w - 1, x + 5), gy)], fill=grid_color, width=1)
        for y in range(0, h, 12):
            draw.line([(gx, y), (gx, min(h - 1, y + 5))], fill=grid_color, width=1)

    sx = float(anchors.get("source_x", 0.5)) * (w - 1)
    sy = float(anchors.get("source_y", 0.15)) * (h - 1)
    candidates = anchors.get("subject_candidates") or ()
    palette = ((80, 225, 255), (255, 210, 90), (140, 240, 145), (255, 155, 200), (200, 175, 255))
    for idx, cand in enumerate(candidates):
        bbox = cand.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x0 = max(1, min(w - 2, int(round(float(bbox[0]) * (w - 1)))))
        y0 = max(1, min(h - 2, int(round(float(bbox[1]) * (h - 1)))))
        x1 = max(x0 + 2, min(w - 2, int(round(float(bbox[2]) * (w - 1)))))
        y1 = max(y0 + 2, min(h - 2, int(round(float(bbox[3]) * (h - 1)))))
        color = palette[idx % len(palette)]
        draw.rectangle([x0 - 1, y0 - 1, x1 + 1, y1 + 1], outline=(0, 0, 0), width=3)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        crest = cand.get("crest")
        if isinstance(crest, (list, tuple)) and len(crest) == 2:
            cx = int(round(float(crest[0]) * (w - 1)))
            cy = int(round(float(crest[1]) * (h - 1)))
            draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(255, 225, 95), outline=(0, 0, 0))
            if idx == 0:
                draw.line([(int(round(sx)), int(round(sy))), (cx, cy)], fill=(255, 225, 95), width=2)
        tag = f"[{cand.get('id', idx + 1)}]"
        tx = x0
        ty = max(2, y0 - 13)
        draw.rectangle([tx, ty, min(w - 2, tx + 24), min(h - 2, ty + 12)], fill=(18, 22, 28))
        draw.text((tx + 2, ty + 1), tag, fill=color)

    out = np.asarray(img, dtype=np.uint8)
    out.setflags(write=False)
    return out


def _draw_mask_outline(draw: ImageDraw.ImageDraw, mask: Any, x_off: int, y_off: int, pw: int, ph: int) -> None:
    color = (255, 200, 95) if mask.inverted else (80, 225, 255)
    if isinstance(mask, RadialGradient):
        cx = (mask.start[0] + mask.end[0]) * pw / 2.0
        cy = (mask.start[1] + mask.end[1]) * ph / 2.0
        rx = max(2.0, abs(mask.end[0] - mask.start[0]) * pw / 2.0)
        ry = max(2.0, abs(mask.end[1] - mask.start[1]) * ph / 2.0)
        rad = math.radians(mask.angle)
        ca, sa = math.cos(rad), math.sin(rad)
        pts = []
        steps = 48
        for k in range(steps + 1):
            th = (2.0 * math.pi * k) / steps
            lx = rx * math.cos(th)
            ly = ry * math.sin(th)
            px = x_off + cx + ca * lx + sa * ly
            py = y_off + cy - sa * lx + ca * ly
            pts.append((px, py))
        for k in range(0, steps, 2):
            draw.line([pts[k], pts[k + 1]], fill=color, width=2)
    else:
        p0 = (x_off + mask.start[0] * pw, y_off + mask.start[1] * ph)
        p1 = (x_off + mask.end[0] * pw, y_off + mask.end[1] * ph)
        draw.line([p0, p1], fill=color, width=2)


def _render_trial_comparison_image(
    before: np.ndarray,
    trial: np.ndarray,
    new_masks: tuple[Any, ...],
) -> np.ndarray:
    """Create a bounded <=768px BEFORE | TRIAL diagnostic preview with mask outlines on TRIAL."""
    h, w = trial.shape[:2]
    b_img = Image.fromarray(before).convert("RGB")
    t_img = Image.fromarray(trial).convert("RGB")
    if b_img.size != (w, h):
        b_img = b_img.resize((w, h), Image.Resampling.BILINEAR)

    gap = 4
    # Compare total pixels within the <=768px ToolImage cap for vertical vs horizontal stack
    v_scale = min(1.0, 768.0 / max(1, w), (768.0 - gap) / max(1, 2 * h))
    h_scale = min(1.0, (768.0 - gap) / max(1, 2 * w), 768.0 / max(1, h))
    vertical = (v_scale >= h_scale)
    scale = v_scale if vertical else h_scale
    pw = max(16, min(760, int(round(w * scale))))
    ph = max(16, min(760, int(round(h * scale))))
    if vertical and 2 * ph + gap > 768:
        ph = (768 - gap) // 2
    if (not vertical) and 2 * pw + gap > 768:
        pw = (768 - gap) // 2

    b_panel = b_img.resize((pw, ph), Image.Resampling.LANCZOS)
    t_panel = t_img.resize((pw, ph), Image.Resampling.LANCZOS)
    canvas_w = pw if vertical else 2 * pw + gap
    canvas_h = 2 * ph + gap if vertical else ph
    canvas = Image.new("RGB", (canvas_w, canvas_h), (24, 24, 24))
    t_x, t_y = (0, ph + gap) if vertical else (pw + gap, 0)
    canvas.paste(b_panel, (0, 0))
    canvas.paste(t_panel, (t_x, t_y))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle([2, 2, 54, 15], fill=(16, 16, 16))
    draw.text((5, 3), "BEFORE", fill=(220, 220, 220))
    draw.rectangle([t_x + 2, t_y + 2, t_x + 54, t_y + 15], fill=(16, 16, 16))
    draw.text((t_x + 5, t_y + 3), "TRIAL", fill=(140, 220, 255))
    for mask in new_masks:
        if isinstance(mask, LinearGradient):
            _draw_mask_outline(draw, mask, t_x, t_y, pw, ph)

    out = np.asarray(canvas, dtype=np.uint8)
    out.setflags(write=False)
    return out


def _handle_detect_lighting(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    anchors = analyze_scene_lighting(ctx.request.current_preview, ctx.request.question, ctx.request.goal)
    if not anchors:
        return ToolResult("", "detect_scene_lighting_anchors", False,
                          {"error": "Preview has too little contrast or resolution for lighting anchors."},
                          error="insufficient_contrast")
    som_img = _render_som_lighting_overlay(ctx.request.current_preview, anchors)
    image = ToolImage("Scene lighting anchors & numbered subjects [1..N] (diagnostic overlay)", som_img)
    return ToolResult("", "detect_scene_lighting_anchors", True, anchors, images=(image,))


def _handle_build_lighting_rig(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    recipe = str(args.get("recipe", "spotlight"))
    direction = str(args.get("light_direction", "auto"))
    strength = str(args.get("strength", "moderate"))
    baseline = EditValues.from_dict(ctx.request.parameters)
    free_slots = 4 - len(baseline.local_gradients)
    if free_slots <= 0:
        return ToolResult("", "build_lighting_rig", False,
                          {"error": "All 4 local adjustment mask slots are already used."},
                          error="mask_capacity")
    anchors = analyze_scene_lighting(ctx.request.current_preview, ctx.request.question, ctx.request.goal) or {}
    candidates = anchors.get("subject_candidates") or []
    src = (float(anchors.get("source_x", 0.35)), float(anchors.get("source_y", 0.14)))
    is_dark = True
    chosen_id: int | None = None
    target_box = args.get("target_box")
    if target_box is not None:
        bbox = [float(v) for v in target_box]
        if not (len(bbox) == 4 and 0.0 <= bbox[0] < bbox[2] <= 1.0 and 0.0 <= bbox[1] < bbox[3] <= 1.0):
            return ToolResult("", "build_lighting_rig", False,
                              {"error": "target_box must be [left, top, right, bottom] inside [0, 1]."},
                              error="invalid_box")
    else:
        req_id = args.get("subject_id")
        cand = None
        if req_id is not None:
            cand = next((c for c in candidates if c.get("id") == int(req_id)), None)
            if cand is None and candidates:
                return ToolResult("", "build_lighting_rig", False,
                                  {"error": f"subject_id={req_id} not found; valid IDs: {[c['id'] for c in candidates]}."},
                                  error="invalid_subject")
        elif candidates:
            cand = candidates[0]
        if cand is not None:
            bbox = cand["bbox"]
            is_dark = bool(cand.get("is_dark_subject", True))
            chosen_id = int(cand["id"])
        else:
            tx = float(anchors.get("target_x", 0.5))
            ty = float(anchors.get("target_y", 0.5))
            bbox = [max(0.05, tx - 0.16), max(0.08, ty - 0.12), min(0.95, tx + 0.16), min(0.92, ty + 0.12)]
            is_dark = bool(anchors.get("mean_luma", 0.3) < 0.35)

    try:
        actions = build_overhead_spotlight_actions(
            bbox, src, is_dark_subject=is_dark, direction=direction,
            strength=strength, recipe=recipe, max_masks=free_slots,
        )
    except (ValueError, MaskCapacityError) as exc:
        return ToolResult("", "build_lighting_rig", False, {"error": str(exc)}, error="invalid_rig")

    return ToolResult("", "build_lighting_rig", True, {
        "recipe": recipe,
        "subject_id": chosen_id,
        "subject_box": bbox,
        "light_direction": direction,
        "strength": strength,
        "actions": actions,
        "suggested_workflow_checks": [
            {
                "region": "whole",
                "metric": "visual",
                "direction": "visual",
                "expectation": "Confirm overhead illumination peaks on the subject crest and rolls off naturally into the surrounding ground.",
            }
        ],
    })


def _handle_search_playbook(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    if ctx.playbook is None:
        return ToolResult("", "search_editing_playbook", False,
                          {"error": "Playbook knowledge base is not configured."}, error="unavailable")
    query = args["query"]
    top_k = int(args.get("top_k", 3))
    hits = ctx.playbook.search(query, top_k=top_k)
    return ToolResult("", "search_editing_playbook", True, {"query": query, "matches": hits})


def _handle_propose_trial(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    if ctx.render_trial is None:
        return ToolResult("", "propose_and_preview_trial", False,
                          {"error": "Trial renderer is unavailable."}, error="unavailable")
    if ctx.trial_count >= ctx.max_trials:
        return ToolResult("", "propose_and_preview_trial", False,
                          {"error": f"Per-run trial render limit ({ctx.max_trials}) reached. Deliver your final answer."},
                          error="trial_limit")
    raw_actions = args["actions"]
    if len(raw_actions) > 6:
        return ToolResult("", "propose_and_preview_trial", False,
                          {"error": "Reduce the proposal to at most six explicit adjustments for one objective."},
                          error="too_many_actions")
    try:
        plan = parse_plan(args["workflow"])
        baseline = EditValues.from_dict(ctx.request.parameters)
        actions = resolve_actions(raw_actions, baseline, ctx.request.source_size)
    except (ValueError, MaskCapacityError) as exc:
        return ToolResult("", "propose_and_preview_trial", False,
                          {"error": str(exc)}, error="invalid_proposal")
    if not actions and plan.decision == "recommend":
        return ToolResult("", "propose_and_preview_trial", False,
                          {"error": "Proposed actions produced no parameter change against current state."},
                          error="no_op")
    report, metrics_delta = evaluate_trial_candidate(
        ctx.request, plan, actions, ctx.render_trial,
        cancel=ctx.cancel, allow_detail=ctx.inspected_100pct,
    )
    rendered_attempted = report.trial_preview is not None or "render_error" in metrics_delta.get("failed_checks", ())
    if rendered_attempted:
        ctx.trial_count += 1
    reflexion = build_reflexion_prompt_summary(report, metrics_delta)
    record = {
        "trial_index": ctx.trial_count,
        "rendered": rendered_attempted,
        "can_apply": report.can_apply,
        "status": report.status,
        "metrics_delta": metrics_delta,
        "checks": [
            {"region": c.check.region, "metric": c.check.metric, "direction": c.check.direction,
             "verdict": c.verdict, "before": c.before, "after": c.after}
            for c in report.checks
        ],
        "action_label": actions[0].label if actions else "",
        "reflexion": reflexion,
    }
    ctx.trial_history.append(record)
    ctx.last_trial = (actions, plan, report)
    if report.can_apply:
        ctx.best_trial = (actions, plan, report)
    images: tuple[ToolImage, ...] = ()
    if report.trial_preview is not None:
        baseline = EditValues.from_dict(ctx.request.parameters)
        old_ids = {g.id for g in baseline.local_gradients}
        new_masks: tuple[Any, ...] = ()
        if actions:
            candidate = actions[0].new_value
            if isinstance(actions[0].changes, tuple):
                for key, val in actions[0].changes:
                    if key == "local_gradients":
                        candidate = val
                        break
            if isinstance(candidate, tuple):
                new_masks = tuple(g for g in candidate if getattr(g, "id", None) not in old_ids)
        if report.current_preview is not None:
            comp_img = _render_trial_comparison_image(report.current_preview, report.trial_preview, new_masks)
            images = (ToolImage(
                f"Before vs Rendered trial #{ctx.trial_count} (with cyan/amber mask outline on TRIAL — diagnostic only)",
                comp_img,
            ),)
        else:
            images = (ToolImage(f"Rendered trial #{ctx.trial_count} preview (not applied)", report.trial_preview),)
    return ToolResult("", "propose_and_preview_trial", True, record, images=images)


ToolHandler = Callable[[dict[str, Any], ToolContext], ToolResult]


class ToolRegistry:
    """Validated, read-only tool registry shared by AgentRuntime and MCP Server."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[ToolSpec, ToolHandler]] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        if not spec.read_only:
            raise ValueError("Only read-only or sandbox trial tools may be registered.")
        self._entries[spec.name] = (spec, handler)

    def get(self, name: str) -> ToolSpec | None:
        entry = self._entries.get(name)
        return entry[0] if entry else None

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(spec for spec, _ in self._entries.values())

    def json_schemas(self) -> list[dict[str, Any]]:
        return [{"name": s.name, "description": s.description, "parameters": s.parameters} for s in self.specs()]

    def execute(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        if ctx.cancel.is_set():
            raise CancelledError()
        entry = self._entries.get(call.name)
        if entry is None:
            return ToolResult(call.call_id, call.name, False,
                              {"error": f"Unknown tool '{call.name}'."}, error="unknown_tool")
        spec, handler = entry
        try:
            validated = validate_tool_arguments(spec, call.arguments)
        except ValueError as exc:
            return ToolResult(call.call_id, call.name, False, {"error": str(exc)}, error="invalid_arguments")
        try:
            result = handler(validated, ctx)
            return ToolResult(call.call_id, call.name, result.ok, result.payload, result.images, result.error)
        except CancelledError:
            raise
        except ValueError as exc:
            return ToolResult(call.call_id, call.name, False, {"error": str(exc)}, error="tool_error")
        except Exception:
            return ToolResult(call.call_id, call.name, False,
                              {"error": "Tool execution failed safely without modifying edit state."},
                              error="tool_error")


def build_default_tool_registry(*, include_trial: bool = True) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="inspect_region_100pct",
        description=(
            "Render and return an actual 1:1 native-scale RGB crop (up to 512x512) at normalized "
            "(x_norm, y_norm) coordinates plus sharpness/noise/clipping metrics. Required before proposing "
            "Detail (sharpening or noise reduction) edits."
        ),
        parameters={
            "type": "object",
            "properties": {
                "x_norm": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                           "description": "Horizontal center in [0, 1] of current preview."},
                "y_norm": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                           "description": "Vertical center in [0, 1] of current preview."},
                "size_px": {"type": "integer", "minimum": 128, "maximum": 512,
                            "description": "Square crop edge in source pixels (128..512)."},
                "apply_current_edits": {"type": "boolean",
                                        "description": "True to inspect current edited render, False for baseline."},
            },
            "required": ["x_norm", "y_norm"],
            "additionalProperties": False,
        },
    ), _handle_inspect_region_100pct)

    registry.register(ToolSpec(
        name="inspect_raw_headroom",
        description=(
            "Read-only Bayer sensor CFA headroom and clipping check on the source RAW file, plus local "
            "neutral-patch uniformity metrics around (x_norm, y_norm)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "x_norm": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "y_norm": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "patch_px": {"type": "integer", "minimum": 16, "maximum": 128},
            },
            "required": [],
            "additionalProperties": False,
        },
    ), _handle_inspect_raw_headroom)

    registry.register(ToolSpec(
        name="analyze_color_zones",
        description=(
            "Measure 8-band HSL hue distribution (matching Color Mixer bands) and approximate CIELAB "
            "color cast statistics on a preview region."
        ),
        parameters={
            "type": "object",
            "properties": {
                "region": {"type": "string", "enum": list(REGIONS)},
            },
            "required": [],
            "additionalProperties": False,
        },
    ), _handle_analyze_color_zones)

    registry.register(ToolSpec(
        name="detect_scene_lighting_anchors",
        description=(
            "Measure horizon/ridge profile, brightest sky/horizon source direction, and numbered [1..N] "
            "foreground subject candidates with upper crest coordinates, plus a diagnostic Set-of-Mark "
            "image overlay and ready-to-use spotlight/sunbeam recipes."
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    ), _handle_detect_lighting)

    registry.register(ToolSpec(
        name="build_lighting_rig",
        description=(
            "Construct a geometry-accurate, multi-mask 3D lighting rig (e.g., an overhead key beam anchored "
            "to the subject's upper crest plus an inverted ambient falloff mask) that returns ready-to-use "
            "ACTION_SCHEMA create_mask items for propose_and_preview_trial."
        ),
        parameters={
            "type": "object",
            "properties": {
                "recipe": {
                    "type": "string",
                    "enum": ["spotlight", "sunbeam", "window_light", "rim_backlight", "mountain_beam"],
                },
                "subject_id": {"type": "integer", "minimum": 1, "maximum": 5},
                "target_box": {
                    "type": "array",
                    "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "light_direction": {
                    "type": "string",
                    "enum": ["auto", "overhead", "top_left", "top_right"],
                },
                "strength": {
                    "type": "string",
                    "enum": ["subtle", "moderate", "dramatic"],
                },
            },
            "required": ["recipe"],
            "additionalProperties": False,
        },
    ), _handle_build_lighting_rig)

    registry.register(ToolSpec(
        name="search_editing_playbook",
        description=(
            "Search PhotoCoach's grounded editing playbook for pipeline-exact tonal, color, lighting, "
            "and detail recipes matching a photo issue or goal."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 300},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ), _handle_search_playbook)

    if include_trial:
        registry.register(ToolSpec(
            name="propose_and_preview_trial",
            description=(
                "Run a non-mutating sandbox trial render of up to 6 coordinated adjustments with explicit "
                "verification checks. Returns measured deltas and the actual rendered trial preview image "
                "for visual self-critique before finalizing."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "actions": ACTION_SCHEMA,
                    "workflow": PLAN_SCHEMA,
                },
                "required": ["actions", "workflow"],
                "additionalProperties": False,
            },
            consumes_trial=True,
        ), _handle_propose_trial)

    return registry
