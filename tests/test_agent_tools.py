"""Tests for Milestone 2 & 3 multimodal inspection tools, color/CFA diagnostics, and strict validation."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from photocoach.agent_tools import (
    ToolCall,
    ToolContext,
    analyze_color_zones,
    build_default_tool_registry,
)
from photocoach.app import PreviewEngine
from photocoach.coach import build_request
from photocoach.composition import Composition
from photocoach.detail import DetailControls
from photocoach.imaging import DecodedPhoto, RawOptions
from photocoach.lens import LensCorrections
from photocoach.preview import DetailRequest, render_region
from photocoach.state import StateStore
from photocoach.white_balance import measure_cfa_headroom


def _synthetic_decoded(w=240, h=160):
    rng = np.random.default_rng(42)
    linear = rng.uniform(0.05, 0.35, size=(h, w, 3)).astype(np.float32)
    # Put a crisp vertical step edge in the center
    linear[:, w // 2 :, :] += 0.25
    linear.setflags(write=False)
    meta = {
        "sha256": "synthetic-photo-id",
        "photo_id": "synthetic-photo-id",
        "half_size": False,
        "output_size": (w, h),
        "visible_size": (w, h),
        "orientation_flip": 0,
    }
    return DecodedPhoto(linear, meta, RawOptions(), False, Path("synthetic.ARW"))


def test_inspect_region_100pct_matches_formal_render_region_under_crop_and_detail(monkeypatch):
    decoded = _synthetic_decoded(400, 300)
    store = StateStore()
    store.activate(Path("synthetic.ARW"), "synthetic-photo-id")
    store.set_parameter("composition", Composition((0.1, 0.1, 0.9, 0.9), 4.0))
    store.set_parameter("lens_corrections", LensCorrections(distortion=-12.0, vignetting_ev=-0.2))
    store.set_parameter("detail", DetailControls(sharpening=35.0, masking=10.0))
    state = store.current

    engine = PreviewEngine()
    monkeypatch.setattr(engine.cache, "get", lambda *a, **kw: decoded)
    patch, metrics = engine.inspect_region_100pct(state, 0.5, 0.5, size_px=128, apply_current_edits=True)
    assert patch.shape == (128, 128, 3) and not patch.flags.writeable
    l, t, r, b = metrics["rect_px"]
    full = tuple(metrics["full_size"])
    expected = render_region(
        decoded.linear_rgb, state.values,
        DetailRequest(0, full, (l, t, r, b), 1, (400, 300), native_detail=True),
    )
    assert np.array_equal(patch, expected)
    assert metrics["edge_energy_mean"] > 0 and metrics["laplacian_sharpness"] > 0

    # Baseline inspection ignores tone/detail edits
    base_patch, _ = engine.inspect_region_100pct(state, 0.5, 0.5, size_px=128, apply_current_edits=False)
    assert not np.array_equal(patch, base_patch)


def test_measure_cfa_headroom_computes_ev_and_clipping_accurately():
    # Bayer RGGB pattern (40x40), black=512, white=16383
    colors = np.zeros((40, 40), dtype=np.uint8)
    colors[0::2, 0::2] = 0
    colors[0::2, 1::2] = 1
    colors[1::2, 0::2] = 3
    colors[1::2, 1::2] = 2
    black = [512.0, 512.0, 512.0, 512.0]
    white = 16383.0
    # Half-scale linear signal (norm = 0.25 -> ~2.0 EV headroom)
    val = int(512 + 0.25 * (white - 512))
    whole = np.full((40, 40), val, dtype=np.uint16)
    local = np.full((16, 16), val, dtype=np.uint16)
    stats = measure_cfa_headroom(whole, colors, local, colors[:16, :16], black, white)
    assert stats["whole_sensor_clipped_fraction"] == 0.0
    assert stats["whole_channels"]["g"]["headroom_ev"] == pytest.approx(2.0, abs=0.05)
    assert stats["is_reliable_neutral_candidate"] is True

    # Clipped sensor region rejects neutral candidate and reports ~0 EV headroom
    clipped = np.full((16, 16), 16300, dtype=np.uint16)
    stats2 = measure_cfa_headroom(clipped, colors[:16, :16], clipped, colors[:16, :16], black, white)
    assert stats2["patch_sensor_clipped_fraction"] == 1.0
    assert stats2["patch_channels"]["r"]["headroom_ev"] == 0.0
    assert stats2["is_reliable_neutral_candidate"] is False


def test_analyze_color_zones_and_registry_validation():
    neutral = np.full((30, 30, 3), 128, dtype=np.uint8)
    res_gray = analyze_color_zones(neutral, "whole")
    assert res_gray["neutral_fraction"] == 1.0
    assert abs(res_gray["lab_mean"]["a"]) < 0.5 and abs(res_gray["lab_mean"]["b"]) < 0.5

    sky = np.full((30, 30, 3), [70, 140, 230], dtype=np.uint8)
    res_sky = analyze_color_zones(sky, "center")
    assert res_sky["cool_blue_hue_fraction"] == 1.0
    assert res_sky["hue_bands"]["blue"]["fraction"] > 0.8

    store = StateStore()
    state = store.activate(Path("s.ARW"), "s-id")
    req = build_request(state, state, [], sky, sky)
    ctx = ToolContext(request=req, state=state)
    reg = build_default_tool_registry()

    # Rejects NaN, bool-as-number, unknown keys, out-of-range coordinates
    for bad_args in (
        {"x_norm": float("nan"), "y_norm": 0.5},
        {"x_norm": True, "y_norm": 0.5},
        {"x_norm": 1.2, "y_norm": 0.5},
        {"x_norm": 0.5, "y_norm": 0.5, "extra": 1},
    ):
        out = reg.execute(ToolCall("c1", "inspect_region_100pct", bad_args), ctx)
        assert not out.ok and out.error == "invalid_arguments"

    out_color = reg.execute(ToolCall("c2", "analyze_color_zones", {"region": "top_center"}), ctx)
    assert out_color.ok and out_color.payload["cool_blue_hue_fraction"] == 1.0


@pytest.mark.real_raw
def test_real_raw_active_inspection_is_strictly_readonly():
    raw_path = Path(r"G:\TestPhotos\7C204108.ARW")
    if not raw_path.is_file():
        pytest.skip("Real Sony A7C II sample not present.")
    before_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    engine = PreviewEngine()
    photo, _ = engine.photo(raw_path, RawOptions())
    store = StateStore()
    state = store.activate(raw_path, photo.metadata["sha256"])
    crop, metrics = engine.inspect_region_100pct(state, 0.5, 0.5, size_px=256)
    headroom = engine.inspect_raw_headroom(state, 0.5, 0.5, patch_px=64)
    assert crop.shape == (256, 256, 3) and metrics["rect_px"][2] - metrics["rect_px"][0] == 256
    assert "whole_sensor_clipped_fraction" in headroom and "patch_channels" in headroom
    after_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert before_sha == after_sha


def test_detect_lighting_som_overlay_build_rig_and_trial_comparison():
    from photocoach.adjustments import render_edit
    from photocoach.coach_actions import analyze_scene_lighting, resolve_actions
    from photocoach.imaging import quantize_srgb

    # Synthetic beach photo: warm bright sand (135) with dark foreground sea-lion ellipse (48) at center
    h, w = 180, 270
    scene = np.full((h, w, 3), 135, dtype=np.uint8)
    scene[:36, :] = [110, 92, 74]  # dark bluff top
    yy, xx = np.ogrid[:h, :w]
    seal_mask = ((xx - 135) / 36.0) ** 2 + ((yy - 104) / 16.0) ** 2 <= 1.0
    scene[seal_mask] = [52, 48, 44]

    anchors = analyze_scene_lighting(scene, question="Add an overhead spotlight on the sea lion")
    cands = anchors.get("subject_candidates", [])
    assert len(cands) >= 1
    assert cands[0]["is_dark_subject"] is True
    assert 0.35 <= cands[0]["center"][0] <= 0.65 and 0.42 <= cands[0]["center"][1] <= 0.72
    assert cands[0]["crest"][1] < cands[0]["center"][1]
    assert "spotlight" in anchors.get("recommended_recipes", {})

    store = StateStore()
    state = store.activate(Path("synthetic-beach.ARW"), "beach-id")
    req = build_request(state, state, [], scene, scene, question="Spotlight the center sea lion", source_size=(w, h))
    reg = build_default_tool_registry()

    def render_trial(values):
        return quantize_srgb(render_edit(np.full((h, w, 3), 0.14, dtype=np.float32), values,
                                         geometry=None if not values.local_gradients else __import__(
                                             "photocoach.gradients", fromlist=["RenderGeometry"]
                                         ).RenderGeometry.full((w, h), values.composition, (w, h))))

    ctx = ToolContext(request=req, state=state, render_trial=render_trial)
    det = reg.execute(ToolCall("c1", "detect_scene_lighting_anchors", {}), ctx)
    assert det.ok and len(det.images) == 1
    assert det.images[0].pixels.shape[:2] == (h, w)

    rig = reg.execute(ToolCall("c2", "build_lighting_rig", {
        "recipe": "spotlight",
        "subject_id": 1,
        "light_direction": "overhead",
        "strength": "moderate",
    }), ctx)
    assert rig.ok and 2 <= len(rig.payload["actions"]) <= 3
    # Generated actions resolve cleanly through resolve_actions
    resolved = resolve_actions(rig.payload["actions"], state.values, (w, h))
    assert len(resolved) == 1

    # Proposing the generated rig produces a bounded Before/Trial comparison image
    trial_res = reg.execute(ToolCall("c3", "propose_and_preview_trial", {
        "actions": rig.payload["actions"],
        "workflow": {
            "version": 1,
            "decision": "recommend",
            "objective": "Overhead 3D spotlight on center dark subject.",
            "findings": [{
                "region": "center",
                "observation": "Dark subject blends into surrounding midtone sand.",
                "interpretation": "Overhead key beam + crest highlight lifts subject relief.",
                "category": "preference",
                "confidence": "high",
            }],
            "checks": rig.payload["suggested_workflow_checks"],
            "uncertainty": "Subject spotlight balance depends on visual preference.",
        },
    }), ctx)
    assert trial_res.ok and trial_res.payload["can_apply"] is True
    assert not trial_res.payload["reflexion"]["needs_repair"]
    assert len(trial_res.images) == 1 and max(trial_res.images[0].pixels.shape[:2]) <= 768


def test_build_lighting_rig_isolates_sky_backed_silhouette():
    from photocoach.coach_actions import analyze_scene_lighting, resolve_actions

    h, w = 180, 270
    scene = np.empty((h, w, 3), dtype=np.uint8)
    scene[:, :] = [64, 80, 104]  # twilight sky backdrop
    scene[55:145, 95:175] = [18, 16, 14]  # dark silhouette subject
    anchors = analyze_scene_lighting(scene, question="Relight the central silhouette")
    cands = anchors.get("subject_candidates", [])
    assert cands and cands[0]["sky_backed_silhouette"] is True

    store = StateStore()
    state = store.activate(Path("dusk.ARW"), "dusk-id")
    req = build_request(state, state, [], scene, scene, question="Relight subject", source_size=(w, h))
    reg = build_default_tool_registry()
    ctx = ToolContext(request=req, state=state, render_trial=lambda _: scene)
    rig = reg.execute(ToolCall("c1", "build_lighting_rig", {"recipe": "spotlight", "subject_id": 1}), ctx)
    assert rig.ok and rig.payload["sky_backed_silhouette"] is True
    assert len(rig.payload["actions"]) == 2  # No stacked 3rd positive crest orb
    key = rig.payload["actions"][0]
    assert key["edge_aware"] is True
    assert 0.16 <= key["luma_max"] <= 0.42
    assert key["warmth"] <= 0.12
    assert resolve_actions(rig.payload["actions"], state.values, (w, h))
