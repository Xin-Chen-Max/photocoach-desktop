"""Tests for Roadmap #4A: 3-way Color Grading math, state, pipeline, and Schema 10 migration."""
from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import numpy as np
import pytest

from photocoach.adjustments import render_edit, render_photo
from photocoach.api_coach import GEMINI_ENDPOINT, ApiConfig, build_payload, request_context
from photocoach.coach import MockCoach, build_request
from photocoach.color_grading import ColorGrading, ColorWheel, apply_color_grading, zone_weights
from photocoach.color_mixer import ColorMixer
from photocoach.composition import Composition, output_size
from photocoach.detail import DetailControls
from photocoach.imaging import quantize_srgb, render_srgb
from photocoach.lens import LensCorrections
from photocoach.preview import DetailRequest, render_region
from photocoach.project import (
    SCHEMA_VERSION,
    read_project,
    snapshot_project,
    upgrade_project,
    validate_project,
    write_project,
)
from photocoach.render_timing import StageTimer
from photocoach.safe_io import sha256_file
from photocoach.scratch import Scratch
from photocoach.state import EDIT_PIPELINE_VERSION, EditValues, StateStore


def _make_store(tmp_path: Path, name: str = "fixture.ARW") -> StateStore:
    source = tmp_path / name
    source.write_bytes(b"readonly synthetic raw fixture")
    store = StateStore()
    store.activate(source, sha256_file(source))
    return store


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hue": -0.01},
        {"hue": 360.0},
        {"hue": 400.0},
        {"hue": float("nan")},
        {"hue": float("inf")},
        {"hue": True},
        {"hue": "45"},
        {"saturation": -0.01},
        {"saturation": 1.01},
        {"saturation": float("nan")},
        {"saturation": False},
        {"luminance": -1.01},
        {"luminance": 1.01},
        {"luminance": float("-inf")},
        {"luminance": None},
    ],
)
def test_color_wheel_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        ColorWheel(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"blend": -0.01},
        {"blend": 1.01},
        {"blend": float("nan")},
        {"blend": True},
        {"balance": -1.01},
        {"balance": 1.01},
        {"balance": float("inf")},
        {"balance": False},
        {"shadows": {}},
        {"midtones": None},
        {"highlights": (0.0, 0.0, 0.0)},
    ],
)
def test_color_grading_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        ColorGrading(**kwargs)


def test_color_grading_boundaries_immutability_and_serialization():
    wheel = ColorWheel(hue=359.999, saturation=1.0, luminance=-1.0)
    assert ColorWheel.from_dict(wheel.to_dict()) == wheel
    zero_neg = ColorWheel(hue=-0.0, saturation=-0.0, luminance=-0.0)
    assert zero_neg.to_dict() == {"hue": 0.0, "saturation": 0.0, "luminance": 0.0}

    grading = ColorGrading(
        shadows=ColorWheel(210.0, 0.45, -0.2),
        midtones=ColorWheel(30.0, 0.25, 0.1),
        highlights=ColorWheel(45.0, 0.6, 0.3),
        blend=1.0,
        balance=-1.0,
    )
    assert ColorGrading.from_dict(grading.to_dict()) == grading
    assert grading.with_wheel("midtones", ColorWheel()).midtones == ColorWheel()

    with pytest.raises(FrozenInstanceError):
        wheel.hue = 10.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        grading.blend = 0.2  # type: ignore[misc]

    for bad_wheel in ({}, {"hue": 0.0}, dict(wheel.to_dict(), extra=1)):
        with pytest.raises(ValueError):
            ColorWheel.from_dict(bad_wheel)
    for bad_grading in ({}, {"shadows": wheel.to_dict()}, dict(grading.to_dict(), extra=1)):
        with pytest.raises(ValueError):
            ColorGrading.from_dict(bad_grading)
    with pytest.raises(ValueError):
        grading.with_wheel("unknown", wheel)
    with pytest.raises(ValueError):
        EditValues(color_grading={})  # type: ignore[arg-type]


def test_neutral_identity_and_zero_overhead_short_circuit():
    neutral_variants = [
        ColorGrading(),
        ColorGrading(
            shadows=ColorWheel(hue=215.0, saturation=0.0, luminance=0.0),
            midtones=ColorWheel(hue=45.0, saturation=0.0, luminance=0.0),
            highlights=ColorWheel(hue=120.0, saturation=0.0, luminance=0.0),
            blend=0.95,
            balance=-0.75,
        ),
    ]
    rng = np.random.default_rng(17)
    encoded = rng.random((48, 64, 3), dtype=np.float32)
    encoded.setflags(write=False)
    linear = rng.random((48, 64, 3), dtype=np.float32)
    linear.setflags(write=False)
    baseline_edit = render_edit(linear, EditValues())

    for grading in neutral_variants:
        assert grading.is_identity
        out = apply_color_grading(encoded, grading)
        assert out is encoded
        assert np.array_equal(out.view(np.uint32), encoded.view(np.uint32))
        edited = render_edit(linear, EditValues(color_grading=grading))
        assert np.array_equal(edited.view(np.uint32), baseline_edit.view(np.uint32))

    # In-place identity short-circuit succeeds even on a read-only float32 array.
    assert apply_color_grading(encoded, ColorGrading(), in_place=True) is encoded


def test_zone_weights_partition_duality_and_monotone_blend_balance():
    luma = np.linspace(0.0, 1.0, 1025, dtype=np.float32)
    for blend in (0.0, 0.25, 0.5, 0.75, 1.0):
        for balance in (-1.0, -0.5, 0.0, 0.5, 1.0):
            ws, wm, wh = zone_weights(luma, blend=blend, balance=balance)
            assert np.all((ws >= 0.0) & (ws <= 1.0))
            assert np.all((wm >= 0.0) & (wm <= 1.0))
            assert np.all((wh >= 0.0) & (wh <= 1.0))
            np.testing.assert_allclose(ws + wm + wh, 1.0, atol=1e-6)

            ws_flip, wm_flip, wh_flip = zone_weights(1.0 - luma, blend=blend, balance=-balance)
            np.testing.assert_allclose(ws_flip, wh, atol=1e-6)
            np.testing.assert_allclose(wh_flip, ws, atol=1e-6)
            np.testing.assert_allclose(wm_flip, wm, atol=1e-6)

    # Blend monotonically increases overlap at u=0.5 and softens zone-weight transition slope.
    mid = np.array([0.5], dtype=np.float32)
    ws_at_half = [float(zone_weights(mid, blend=b, balance=0.0)[0][0]) for b in (0.0, 0.25, 0.5, 0.75, 1.0)]
    wh_at_half = [float(zone_weights(mid, blend=b, balance=0.0)[2][0]) for b in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(a < b for a, b in zip(ws_at_half, ws_at_half[1:]))
    assert all(a < b for a, b in zip(wh_at_half, wh_at_half[1:]))

    max_weight_slopes_s = []
    max_weight_slopes_h = []
    for blend in (0.0, 0.25, 0.5, 0.75, 1.0):
        ws, _, wh = zone_weights(luma, blend=blend, balance=0.0)
        max_weight_slopes_s.append(float(np.max(np.abs(np.diff(ws)))))
        max_weight_slopes_h.append(float(np.max(np.abs(np.diff(wh)))))
    assert all(a > b for a, b in zip(max_weight_slopes_s, max_weight_slopes_s[1:]))
    assert all(a > b for a, b in zip(max_weight_slopes_h, max_weight_slopes_h[1:]))

    # Balance monotonically shifts midtones between cool shadows and warm highlights.
    mid_gray = np.full((1, 1, 3), 0.5, dtype=np.float32)
    warmth_deltas = []
    for balance in (-1.0, -0.5, 0.0, 0.5, 1.0):
        split = ColorGrading(
            shadows=ColorWheel(210.0, 0.8, 0.0),
            highlights=ColorWheel(35.0, 0.8, 0.0),
            blend=0.5,
            balance=balance,
        )
        out = apply_color_grading(mid_gray, split)
        warmth_deltas.append(float(out[0, 0, 0] - out[0, 0, 2]))
    assert all(a < b for a, b in zip(warmth_deltas, warmth_deltas[1:]))


def test_zone_color_casts_luma_neutrality_endpoints_and_hue_seam():
    # Dark shadow, midtone, bright highlight grays
    zones = np.array([[[0.0, 0.0, 0.0], [0.16, 0.16, 0.16], [0.5, 0.5, 0.5], [0.84, 0.84, 0.84], [1.0, 1.0, 1.0]]], np.float32)

    blue_shadows = apply_color_grading(zones, ColorGrading(shadows=ColorWheel(240.0, 0.8, 0.0), blend=0.25))
    assert blue_shadows[0, 1, 2] > blue_shadows[0, 1, 0] + 0.05
    np.testing.assert_allclose(blue_shadows[0, 3], 0.84, atol=1e-4)
    np.testing.assert_array_equal(blue_shadows[0, 0], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(blue_shadows[0, 4], [1.0, 1.0, 1.0])

    amber_highlights = apply_color_grading(zones, ColorGrading(highlights=ColorWheel(35.0, 0.8, 0.0), blend=0.25))
    assert amber_highlights[0, 3, 0] > amber_highlights[0, 3, 2] + 0.05
    np.testing.assert_allclose(amber_highlights[0, 1], 0.16, atol=1e-4)

    green_mids = apply_color_grading(zones, ColorGrading(midtones=ColorWheel(120.0, 0.8, 0.0), blend=0.25))
    assert green_mids[0, 2, 1] > green_mids[0, 2, 0] + 0.05

    # Pure tint (luminance=0) preserves display luma proxy before clipping.
    mid_patches = np.random.default_rng(11).uniform(0.3, 0.7, (16, 16, 3)).astype(np.float32)
    luma_before = 0.2126 * mid_patches[..., 0] + 0.7152 * mid_patches[..., 1] + 0.0722 * mid_patches[..., 2]
    tinted = apply_color_grading(
        mid_patches,
        ColorGrading(
            shadows=ColorWheel(200.0, 0.4, 0.0),
            midtones=ColorWheel(35.0, 0.4, 0.0),
            highlights=ColorWheel(310.0, 0.4, 0.0),
        ),
    )
    luma_after = 0.2126 * tinted[..., 0] + 0.7152 * tinted[..., 1] + 0.0722 * tinted[..., 2]
    np.testing.assert_allclose(luma_after, luma_before, atol=1e-6)

    # Pure luminance preserves neutral grayscale and ramp monotonicity.
    ramp = np.repeat(np.linspace(0.0, 1.0, 1025, dtype=np.float32)[None, :, None], 3, axis=2)
    for lum_grading in (
        ColorGrading(shadows=ColorWheel(0.0, 0.0, 1.0)),
        ColorGrading(midtones=ColorWheel(0.0, 0.0, -1.0)),
        ColorGrading(highlights=ColorWheel(0.0, 0.0, -1.0)),
    ):
        out = apply_color_grading(ramp, lum_grading)
        np.testing.assert_allclose(out[..., 0], out[..., 1], atol=1e-7)
        np.testing.assert_allclose(out[..., 1], out[..., 2], atol=1e-7)
        assert np.diff(out[0, :, 0]).min() >= -1e-6
        assert np.all((out >= 0.0) & (out <= 1.0))

    # Hue seam continuity near 0 / 360 degrees.
    a = apply_color_grading(mid_patches, ColorGrading(midtones=ColorWheel(0.0, 0.7, 0.0)))
    b = apply_color_grading(mid_patches, ColorGrading(midtones=ColorWheel(359.999, 0.7, 0.0)))
    assert np.max(np.abs(a - b)) < 1e-5


def test_apply_color_grading_scratch_reuse_read_only_and_validation():
    rng = np.random.default_rng(77)
    grading = ColorGrading(
        shadows=ColorWheel(205.0, 0.6, -0.15),
        midtones=ColorWheel(45.0, 0.35, 0.1),
        highlights=ColorWheel(28.0, 0.55, 0.2),
        blend=0.65,
        balance=-0.25,
    )
    scratch = Scratch()
    for shape in ((1, 1, 3), (32, 48, 3), (128, 17, 3)):
        src = rng.random(shape, dtype=np.float32)
        src.setflags(write=False)
        before = src.copy()
        direct = apply_color_grading(src, grading)
        reused = apply_color_grading(src, grading, scratch=scratch)
        np.testing.assert_array_equal(direct.view(np.uint32), reused.view(np.uint32))
        np.testing.assert_array_equal(src, before)

    ro = np.zeros((4, 4, 3), dtype=np.float32)
    ro.setflags(write=False)
    with pytest.raises(ValueError):
        apply_color_grading(ro, grading, in_place=True)
    with pytest.raises(ValueError):
        apply_color_grading(np.zeros((4, 4, 3), dtype=np.uint8), grading)
    with pytest.raises(ValueError):
        apply_color_grading(np.zeros((4, 4, 3), dtype=np.float32), {})  # type: ignore[arg-type]


def test_pipeline_order_strip_roi_and_export_parity():
    rng = np.random.default_rng(91)
    linear = rng.uniform(0.02, 0.88, (300, 360, 3)).astype(np.float32)
    linear.setflags(write=False)
    grading = ColorGrading(
        shadows=ColorWheel(210.0, 0.55, -0.1),
        midtones=ColorWheel(45.0, 0.25, 0.05),
        highlights=ColorWheel(35.0, 0.5, 0.15),
        blend=0.6,
        balance=0.2,
    )

    # Verify placement right before Color Mixer & RGB curve.
    base_only = render_edit(linear, EditValues(exposure_ev=0.2, contrast=0.1))
    expected = apply_color_grading(base_only, grading)
    actual = render_edit(linear, EditValues(exposure_ev=0.2, contrast=0.1, color_grading=grading))
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    # Full pipeline parity across crop + straighten + distortion + detail + color mixer + curve.
    values = EditValues(
        base_rendering="natural_v1",
        exposure_ev=0.15,
        contrast=0.12,
        vibrance=0.2,
        color_grading=grading,
        color_mixer=ColorMixer().with_value(2, 2, 15).with_value(5, 1, -20),
        rgb_curve=((0.0, 0.0), (0.35, 0.32), (0.75, 0.8), (1.0, 1.0)),
        composition=Composition((0.05, 0.06, 0.95, 0.94), angle=2.5),
        lens_corrections=LensCorrections(distortion=-12.0, vignetting_ev=0.2),
        detail=DetailControls(sharpening=35.0, masking=15.0, chroma_nr=20.0),
    )
    timer = StageTimer()
    full_float = render_photo(linear, values, timer=timer)
    assert "color_grading" in timer.milliseconds()
    full_u8 = quantize_srgb(full_float, bits=8)
    size = (full_u8.shape[1], full_u8.shape[0])
    rect = (25, 90, min(260, size[0]), min(220, size[1]))
    roi_serial = render_region(linear, values, DetailRequest(1, size, rect, 1), workers=1)
    roi_parallel = render_region(linear, values, DetailRequest(1, size, rect, 1), workers=4)
    np.testing.assert_array_equal(roi_serial, roi_parallel)
    np.testing.assert_array_equal(roi_serial, full_u8[rect[1]:rect[3], rect[0]:rect[2]])


def test_state_history_coach_and_schema10_roundtrip(tmp_path: Path):
    store = _make_store(tmp_path)
    before = store.current
    assert before is not None

    for i in range(1, 6):
        store.begin_drag("color_grading")
        store.set_parameter(
            "color_grading",
            ColorGrading(
                shadows=ColorWheel(205.0, i * 0.1, -0.05),
                highlights=ColorWheel(40.0, i * 0.08, 0.05),
                blend=0.55,
                balance=-0.1,
            ),
        )
        store.end_drag()
    assert len(store.undo_stack) == 5
    final = store.current.values

    # Cancelled drag leaves undo stack unchanged.
    store.begin_drag("color_grading")
    store.set_parameter("color_grading", ColorGrading())
    store.end_drag(cancel=True)
    assert store.current.values == final
    assert len(store.undo_stack) == 5

    # Project save & reload round-trip.
    project_path = tmp_path / "grading.photocoach"
    snap = snapshot_project(store)
    assert snap["schema_version"] == SCHEMA_VERSION == 14
    assert snap["pipeline_version"] == EDIT_PIPELINE_VERSION == "phase1-sdr-srgb-v14"
    write_project(project_path, snap)
    restored = read_project(project_path)
    assert restored.current.values == final
    assert restored.undo_stack == store.undo_stack
    assert not restored.load_warnings

    # Coach request context, API payload, and MockCoach summary.
    pixels = np.full((16, 16, 3), 120, dtype=np.uint8)
    req = build_request(store.current, before, store.events, pixels, pixels)
    assert req.parameters["color_grading"] == final.color_grading.to_dict()
    ctx = request_context(req)
    assert ctx["parameters"]["color_grading"] == final.color_grading.to_dict()
    cfg = ApiConfig(GEMINI_ENDPOINT, "gemini-3.5-flash", "gemini")
    assert "color_grading" in json.dumps(build_payload(cfg, req))
    reply = MockCoach().analyze(req)
    assert "Color Grading" in reply.main_change

    # Reset & undo restore exact ColorGrading state.
    store.reset()
    assert store.current.color_grading == ColorGrading()
    store.undo()
    assert store.current.values == final


@pytest.mark.parametrize("with_detail", [False, True])
@pytest.mark.parametrize("base_rendering", ["legacy_v1", "natural_v1"])
def test_schema9_migration_preserves_history_base_detail_and_pixels(
    tmp_path: Path, with_detail: bool, base_rendering: str
):
    store = _make_store(tmp_path, "v9_source.ARW")
    store.activate(store.current.source_path, store.current.photo_id, base_rendering=base_rendering)
    store.set_exposure(0.35)
    if with_detail:
        store.set_parameter("detail", DetailControls(40.0, 10.0, 5.0, 25.0))
    store.reset()
    store.undo()

    v9 = snapshot_project(store)
    v9["source"]["byte_length"] = store.current.source_path.stat().st_size

    def strip_v10(node):
        if isinstance(node, dict):
            node.pop("color_grading", None)
            node.pop("spots", None)
            node.pop("compare_snapshot_id", None)
            node.pop("snapshots", None)
            for child in node.values():
                strip_v10(child)
        elif isinstance(node, list):
            for child in node:
                strip_v10(child)

    strip_v10(v9)
    v9["schema_version"] = 9
    v9["pipeline_version"] = "phase1-sdr-srgb-v9"
    original = copy.deepcopy(v9)

    upgraded = upgrade_project(v9)
    assert v9 == original
    assert upgraded["schema_version"] == 14
    assert upgraded["pipeline_version"] == "phase1-sdr-srgb-v14"
    undo, redo = validate_project(upgraded)
    assert undo == store.undo_stack and redo == store.redo_stack

    legacy_file = tmp_path / "legacy_v9.photocoach"
    legacy_file.write_text(json.dumps(v9), encoding="utf-8")
    digest = sha256_file(legacy_file)
    loaded = read_project(legacy_file)
    assert sha256_file(legacy_file) == digest
    assert loaded.current.values == store.current.values
    assert loaded.current.base_rendering == base_rendering
    assert "format v14" in loaded.load_warnings[0]

    pixels = np.random.default_rng(13).uniform(0.05, 0.85, (16, 20, 3)).astype(np.float32)
    np.testing.assert_array_equal(
        render_photo(pixels, loaded.current.values).view(np.uint32),
        render_photo(pixels, store.current.values).view(np.uint32),
    )

    # Strict schema checks: v9 cannot already carry color_grading; v10 cannot miss or tamper it.
    bad_v9 = copy.deepcopy(v9)
    bad_v9["state"]["parameters"]["color_grading"] = ColorGrading().to_dict()
    with pytest.raises(ValueError):
        upgrade_project(bad_v9)

    bad_v10 = copy.deepcopy(upgraded)
    bad_v10["state"]["parameters"].pop("color_grading")
    with pytest.raises(ValueError):
        validate_project(bad_v10)

    tampered_v10 = copy.deepcopy(upgraded)
    tampered_v10["state"]["parameters"]["color_grading"]["blend"] = True
    with pytest.raises(ValueError):
        validate_project(tampered_v10)
