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
from photocoach.detail import CAMERA_RAW_DETAIL, DetailControls, apply_detail
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
