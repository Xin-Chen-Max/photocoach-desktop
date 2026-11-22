"""Tests for Milestone 3 Detail controls: state, math, strip/ROI equivalence, and UI."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from photocoach.adjustments import render_edit, render_photo
from photocoach.api_coach import request_context
from photocoach.app import MainWindow
from photocoach.coach import MockCoach, build_request
from photocoach.coach_view import related_control
from photocoach.composition import Composition
from photocoach.detail import CAMERA_RAW_DETAIL, DetailControls, _detail_patch, apply_detail
from photocoach.imaging import quantize_srgb
from photocoach.lens import LensCorrections
from photocoach.preview import DetailRequest, render_region
from photocoach.project import read_project, snapshot_project, write_project
from photocoach.safe_io import sha256_file
from photocoach.settings import SettingsStore
from photocoach.state import EditValues, StateStore


@pytest.mark.parametrize("name", ["sharpening", "masking", "luminance_nr", "chroma_nr"])
@pytest.mark.parametrize("bad", [True, False, None, "40", float("nan"), float("inf"), -0.01, 100.01])
def test_detail_invalid_values_rejected(name, bad):
    with pytest.raises(ValueError):
        DetailControls(**{name: bad})


def test_detail_immutability_and_serialization():
    d = DetailControls(40, 15, 10, 25)
    assert d.active
    assert DetailControls.from_dict(d.to_dict()) == d
    assert not DetailControls(0, 50, 0, 0).active
    with pytest.raises(FrozenInstanceError):
        d.sharpening = 10  # type: ignore[misc]
    for bad in ({}, {"sharpening": 10}, dict(d.to_dict(), extra=1)):
        with pytest.raises(ValueError):
            DetailControls.from_dict(bad)
    with pytest.raises(ValueError):
        EditValues(detail={})  # type: ignore[arg-type]


def test_detail_state_history_project_roundtrip_and_coach(tmp_path: Path):
    source = tmp_path / "sample.ARW"
    source.write_bytes(b"readonly raw fixture")
    store = StateStore()
    store.activate(source, sha256_file(source))
    before = store.current
    assert before is not None

    for i in range(1, 6):
        store.begin_drag("detail")
        store.set_parameter("detail", DetailControls(i * 10, i * 5, i * 2, i * 5))
        store.end_drag()
    assert len(store.undo_stack) == 5
    final = store.current.values
    assert final.detail == DetailControls(50, 25, 10, 25)

    project_file = tmp_path / "detail_session.photocoach"
    write_project(project_file, snapshot_project(store))
    restored = read_project(project_file)
    assert restored.current.values == final
    assert restored.undo_stack == store.undo_stack

    pixels = np.full((32, 32, 3), 120, dtype=np.uint8)
    req = build_request(store.current, before, store.events, pixels, pixels, "Is sharpening too strong?")
    ctx = request_context(req)
    assert ctx["parameters"]["detail"] == final.detail.to_dict()
    reply = MockCoach().analyze(req)
    assert "Sharpening 50" in reply.main_change

    store.reset()
    assert store.current.detail == DetailControls()
    store.undo()
    assert store.current.values == final


def test_detail_neutral_identity_and_signal_properties():
    rng = np.random.default_rng(42)
    linear = rng.uniform(0.02, 0.85, (96, 128, 3)).astype(np.float32)
    base = render_edit(linear, EditValues())
    masked_only = render_edit(linear, EditValues(detail=DetailControls(0, 80, 0, 0)))
    assert np.array_equal(base, masked_only)

    # Grayscale step edge: luminance USM must increase gradient energy without introducing color fringes (R==G==B).
    gray_edge = np.full((64, 64, 3), 0.35, dtype=np.float32)
    gray_edge[:, 32:, :] = 0.65
    unsharp = render_edit(gray_edge, EditValues())
    sharpened = render_edit(gray_edge, EditValues(detail=DetailControls(sharpening=60, masking=0)))
    gx_before = np.abs(unsharp[:, 32, 0] - unsharp[:, 31, 0]).mean()
    gx_after = np.abs(sharpened[:, 32, 0] - sharpened[:, 31, 0]).mean()
    assert gx_after > gx_before * 1.15
    assert np.allclose(sharpened[..., 0], sharpened[..., 1], atol=1e-6)
    assert np.allclose(sharpened[..., 1], sharpened[..., 2], atol=1e-6)

    # Chroma NR reduces opponent chroma speckle while preserving luminance step.
    noisy_chroma = gray_edge.copy()
    speckle = rng.normal(0.0, 0.04, (64, 64)).astype(np.float32)
    noisy_chroma[..., 0] = np.clip(noisy_chroma[..., 0] + speckle, 0.01, 0.99)
    noisy_chroma[..., 2] = np.clip(noisy_chroma[..., 2] - speckle, 0.01, 0.99)
    out_raw = render_edit(noisy_chroma, EditValues())
    out_cnr = render_edit(noisy_chroma, EditValues(detail=DetailControls(chroma_nr=60)))
    cr_raw = float(np.std(out_raw[..., 0] - out_raw[..., 1]))
    cr_cnr = float(np.std(out_cnr[..., 0] - out_cnr[..., 1]))
    assert cr_cnr < cr_raw * 0.65


def test_detail_strip_and_roi_zero_lsb_equivalence():
    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:300, 0:360].astype(np.float32)
    base = 0.25 + 0.35 * (0.5 + 0.5 * np.sin(x * 0.09) * np.cos(y * 0.08))
    noise = rng.normal(0.0, 0.015, (300, 360, 3)).astype(np.float32)
    linear = np.clip(np.stack([base * 1.05, base, base * 0.95], axis=-1) + noise, 0.01, 0.95).astype(np.float32)

    values = EditValues(
        exposure_ev=0.15,
        contrast=0.1,
        composition=Composition((0.05, 0.05, 0.95, 0.95), angle=2.5),
        lens_corrections=LensCorrections(distortion=-15.0, vignetting_ev=0.2),
        detail=CAMERA_RAW_DETAIL,
    )
    full_float = render_photo(linear, values)
    full_u8 = quantize_srgb(full_float, bits=8)

    # Arbitrary ROI spanning across the 128-row strip boundary (y = 90..210, x = 50..260)
    full_size = (full_u8.shape[1], full_u8.shape[0])
    rect = (50, 90, 260, 210)
    roi_u8 = render_region(
        linear,
        values,
        DetailRequest(generation=1, full_size=full_size, rect=rect, step=1),
    )
    expected_roi = full_u8[rect[1]:rect[3], rect[0]:rect[2]]
    assert np.array_equal(roi_u8, expected_roi)


def test_detail_ui_preset_reset_and_navigation(tmp_path: Path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    w = MainWindow(settings_store=SettingsStore(tmp_path / "unused.json"))
    monkeypatch.setattr(w, "request_preview", lambda: None)
    monkeypatch.setattr(w, "request_detail", lambda: None)
    w.store.activate(Path("synthetic.ARW"), "photo-a")
    w._set_controls(True)
    w.resize(1280, 800)
    w.show()
    app.processEvents()
    try:
        assert not w.detail_section.toggle.isChecked()
        for target, expected in (
            ("Increase Sharpening slightly", "detail.sharpening"),
            ("Raise Edge masking to protect sky", "detail.masking"),
            ("Apply Color NR for chroma mottling", "detail.chroma_nr"),
            ("Add mild Luminance NR", "detail.luminance_nr"),
        ):
            assert related_control(target) == expected

        w._locate_control("detail.chroma_nr")
        app.processEvents()
        assert w.detail_section.toggle.isChecked()

        w.detail_editor.preset.click()
        app.processEvents()
        assert w.store.current.detail == CAMERA_RAW_DETAIL
        assert len(w.store.undo_stack) == 1

        w.detail_editor.reset.click()
        app.processEvents()
        assert w.store.current.detail == DetailControls()
        assert len(w.store.undo_stack) == 2
    finally:
        w._close_approved = True
        w.close()
        w.deleteLater()
        app.processEvents()


def test_heavy_detail_strip_and_roi_zero_lsb_equivalence_8bit_and_16bit():
    rng = np.random.default_rng(19)
    y, x = np.mgrid[0:320, 0:360].astype(np.float32)
    base = 0.22 + 0.38 * (0.5 + 0.5 * np.sin(x * 0.08) * np.cos(y * 0.07))
    noise = rng.normal(0.0, 0.02, (320, 360, 3)).astype(np.float32)
    linear = np.clip(np.stack([base * 1.04, base, base * 0.96], axis=-1) + noise, 0.01, 0.95).astype(np.float32)
    heavy = DetailControls(sharpening=75.0, masking=70.0, luminance_nr=85.0, chroma_nr=90.0)

    values = EditValues(
        exposure_ev=0.15,
        contrast=0.1,
        composition=Composition((0.05, 0.05, 0.95, 0.95), angle=2.5),
        lens_corrections=LensCorrections(distortion=-15.0, vignetting_ev=0.2),
        detail=heavy,
    )
    full_float = render_photo(linear, values)
    full_u8 = quantize_srgb(full_float, bits=8)

    # ROI crosses both y=128 (render_region strip) and y=256 (apply_detail chunk) boundaries
    full_size = (full_u8.shape[1], full_u8.shape[0])
    rect = (40, 96, min(260, full_u8.shape[1]), min(280, full_u8.shape[0]))
    assert rect[1] < 128 < rect[3] and rect[1] < 256 < rect[3]
    roi_u8 = render_region(
        linear,
        values,
        DetailRequest(generation=1, full_size=full_size, rect=rect, step=1),
    )
    assert np.array_equal(roi_u8, full_u8[rect[1]:rect[3], rect[0]:rect[2]])

    # 16-bit strip-boundary equivalence across apply_detail's 256-row seam
    srgb = np.clip(np.stack([base[:320, :220] * 1.05, base[:320, :220], base[:320, :220] * 0.94], axis=-1)
                   + noise[:320, :220], 0.0, 1.0).astype(np.float32)
    strip_16 = quantize_srgb(apply_detail(srgb.copy(), heavy), bits=16)
    single_16 = quantize_srgb(_detail_patch(srgb, heavy), bits=16)
    assert np.array_equal(strip_16, single_16)


def test_luminance_nr_reduces_flat_grain_while_preserving_step_edge():
    rng = np.random.default_rng(23)
    yy, xx = np.mgrid[0:160, 0:160].astype(np.float32)
    base = np.where(xx < 80, np.float32(0.25), np.float32(0.65))
    flat_mask = (np.abs(xx - 80.0) > 10.0).astype(np.float32)
    mottle = np.float32(0.015) * np.sin(xx * np.float32(0.5)) * np.cos(yy * np.float32(0.5)) * flat_mask
    shot = rng.normal(0.0, 0.03, (160, 160)).astype(np.float32)
    gray = np.clip(base + mottle + shot, 0.0, 1.0).astype(np.float32)
    src = np.repeat(gray[..., None], 3, axis=2)

    out = apply_detail(src.copy(), DetailControls(luminance_nr=80.0))

    assert np.allclose(out[..., 0], out[..., 1], atol=1e-6)
    assert np.allclose(out[..., 0], out[..., 2], atol=1e-6)

    base_flat_std = float(np.std(src[16:144, 16:64, 0]))
    out_flat_std = float(np.std(out[16:144, 16:64, 0]))
    assert out_flat_std <= 0.45 * base_flat_std

    base_step = float(np.mean(src[16:144, 81:84, 0]) - np.mean(src[16:144, 76:79, 0]))
    out_step = float(np.mean(out[16:144, 81:84, 0]) - np.mean(out[16:144, 76:79, 0]))
    assert out_step >= 0.85 * base_step


def test_color_nr_suppresses_coarse_chroma_blotches_while_preserving_isoluminant_edge():
    rng = np.random.default_rng(31)
    yy, xx = np.mgrid[0:160, 0:180].astype(np.float32)
    y_plane = np.full((160, 180), 0.45, dtype=np.float32)

    blotch_cr = np.float32(0.09) * np.sin(xx * np.float32(np.pi / 6.0)) * np.cos(yy * np.float32(np.pi / 6.0))
    blotch_cb = np.float32(0.09) * np.cos(xx * np.float32(np.pi / 6.0)) * np.sin(yy * np.float32(np.pi / 6.0))
    blotch_cr += rng.normal(0.0, 0.025, (160, 180)).astype(np.float32)
    blotch_cb += rng.normal(0.0, 0.025, (160, 180)).astype(np.float32)

    iso_cr = np.where(yy < 80, np.float32(0.18), np.float32(-0.12))
    iso_cb = np.where(yy < 80, np.float32(-0.06), np.float32(0.12))

    cr = np.where(xx < 90, blotch_cr, iso_cr)
    cb = np.where(xx < 90, blotch_cb, iso_cb)
    src = np.stack(
        [
            np.clip(y_plane + cr, 0.0, 1.0),
            np.clip(y_plane - (np.float32(0.2126) * cr + np.float32(0.0722) * cb) / np.float32(0.7152), 0.0, 1.0),
            np.clip(y_plane + cb, 0.0, 1.0),
        ],
        axis=-1,
    ).astype(np.float32)

    out = apply_detail(src.copy(), DetailControls(chroma_nr=90.0))

    src_rb = src[..., 0] - src[..., 2]
    out_rb = out[..., 0] - out[..., 2]
    base_chroma_std = float(np.std(src_rb[20:140, 16:72]))
    out_chroma_std = float(np.std(out_rb[20:140, 16:72]))
    assert out_chroma_std <= 0.30 * base_chroma_std

    base_iso_jump = float(np.mean(src_rb[74:78, 108:164]) - np.mean(src_rb[82:86, 108:164]))
    out_iso_jump = float(np.mean(out_rb[74:78, 108:164]) - np.mean(out_rb[82:86, 108:164]))
    assert out_iso_jump >= 0.80 * base_iso_jump


def test_edge_masking_suppresses_flat_noise_amplification_while_keeping_edges():
    rng = np.random.default_rng(41)
    _, xx = np.mgrid[0:160, 0:160].astype(np.float32)
    flat_noisy = np.float32(0.45) + rng.normal(0.0, 0.015, (160, 160)).astype(np.float32)
    step_clean = np.where(xx < 120, np.float32(0.25), np.float32(0.70))
    gray = np.clip(np.where(xx < 75, flat_noisy, step_clean), 0.0, 1.0).astype(np.float32)
    src = np.repeat(gray[..., None], 3, axis=2)

    out_unmasked = apply_detail(src.copy(), DetailControls(sharpening=80.0, masking=0.0))
    out_masked = apply_detail(src.copy(), DetailControls(sharpening=80.0, masking=85.0))

    flat_std_unmasked = float(np.std(out_unmasked[16:144, 16:60, 0]))
    flat_std_masked = float(np.std(out_masked[16:144, 16:60, 0]))
    assert flat_std_masked < 0.90 * flat_std_unmasked

    neutral_edge_grad = float(np.mean(np.abs(np.diff(src[16:144, 117:124, 0], axis=1))))
    masked_edge_grad = float(np.mean(np.abs(np.diff(out_masked[16:144, 117:124, 0], axis=1))))
    assert masked_edge_grad > 1.15 * neutral_edge_grad
