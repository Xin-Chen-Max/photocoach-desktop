"""Versioned base rendering, without modifying creative controls or old projects."""
import copy
from dataclasses import replace
import json

import numpy as np
import pytest

from photocoach.adjustments import render_photo
from photocoach.composition import Composition
from photocoach.imaging import export_image, quantize_srgb, render_srgb
from photocoach.preview import DetailRequest, render_region
from photocoach.project import read_project, snapshot_project, upgrade_project, validate_project, write_project
from photocoach.safe_io import sha256_file
from photocoach.state import EditValues, StateStore


def test_natural_monotone_neutral_axis_exposure_and_pointwise_mapping():
    ramp = np.repeat(np.linspace(0, 4, 4097, dtype=np.float32)[:, None, None], 3, axis=2)
    natural = render_srgb(ramp, base_rendering="natural_v1")
    assert np.isfinite(natural).all() and natural.min() == 0 and natural.max() == pytest.approx(1)
    assert np.min(np.diff(natural[:, 0, 0])) >= 0
    np.testing.assert_array_equal(natural[..., 0], natural[..., 1])
    np.testing.assert_array_equal(natural[..., 1], natural[..., 2])
    for ev in (-2, -1, 0, 1, 2):
        small = ramp[:200]
        np.testing.assert_array_equal(render_srgb(small, ev, base_rendering="natural_v1"),
                                      render_srgb(small * np.float32(2**ev), base_rendering="natural_v1"))
    np.testing.assert_array_equal(natural[29:95], render_srgb(ramp[29:95], base_rendering="natural_v1"))
    assert natural[184, 0, 0] > render_srgb(ramp)[184, 0, 0]
    np.testing.assert_array_equal(render_srgb(ramp), render_srgb(ramp, base_rendering="neutral_v1"))


def test_natural_preserves_hue_direction_and_compresses_saturated_highlights():
    from photocoach.imaging import apply_base_rendering
    from photocoach.scratch import Scratch
    rgb = np.array([[[.05, .1, .2], [.9, .2, .1], [0, 0, 1], [0, 1, 0], [1, 1, 1]]], np.float32)
    out = rgb.copy()
    apply_base_rendering(out, "natural_v1", Scratch())
    assert out.min() >= 0 and out.max() <= 1
    # RGB channel differences keep the hue direction through gamut compression.
    for original, mapped in zip(rgb[0, :4], out[0, :4]):
        a, b = original - original.mean(), mapped - mapped.mean()
        assert np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b) == pytest.approx(1, abs=1e-6)
    assert np.all(out[0, 2] > [0, 0, .99])
    np.testing.assert_allclose(out[0, -1], 1)


def test_natural_detail_matches_tiff_with_crop_rotation_and_local_edit(tmp_path):
    import tifffile
    from photocoach.gradients import RadialGradient
    linear = np.random.default_rng(12).uniform(0, .9, (181, 237, 3)).astype(np.float32)
    saved = linear.copy()
    values = EditValues(base_rendering="natural_v1", exposure_ev=.3, highlights=-.2,
                        composition=Composition((.1, .1, .9, .9), 3),
                        local_gradients=(RadialGradient("m", "Subject", (.2, .2), (.6, .8), exposure_ev=.1),))
    output = render_photo(linear, values)
    size = (output.shape[1], output.shape[0])
    rect = (20, 17, 140, 100)
    detail = render_region(linear, values, DetailRequest(1, size, rect))
    path = export_image(tmp_path / "natural.tiff", output)
    tiff = tifffile.imread(path)
    # Quantize the shared rendered float directly for the exact 8-bit contract.
    np.testing.assert_array_equal(detail, quantize_srgb(output)[17:100, 20:140])
    assert np.max(np.abs(tiff.astype(np.float32)/65535 - output)) <= 1/65535
    np.testing.assert_array_equal(linear, saved)


def test_schema8_migration_keeps_every_legacy_history_and_pixel(tmp_path):
    path = tmp_path / "original.ARW"
    path.write_bytes(b"readonly fixture")
    store = StateStore()
    store.activate(path, sha256_file(path))
    store.set_exposure(.35)
    store.reset()
    store.undo()
    old = snapshot_project(store)
    old["source"]["byte_length"] = path.stat().st_size
    def remove_mode(value):
        if isinstance(value, dict):
            value.pop("base_rendering", None)
            for child in value.values():
                remove_mode(child)
        elif isinstance(value, list):
            for child in value:
                remove_mode(child)
    remove_mode(old)
    old.update(schema_version=8, pipeline_version="phase1-sdr-srgb-v8")
    original = copy.deepcopy(old)
    upgraded = upgrade_project(old)
    assert old == original and upgraded["schema_version"] == 9
    undo, redo = validate_project(upgraded)
    assert undo == store.undo_stack and redo == store.redo_stack
    document = tmp_path / "old.photocoach"
    document.write_text(json.dumps(old), encoding="utf-8")
    digest = sha256_file(document)
    restored = read_project(document)
    assert restored.current.values == store.current.values
    assert sha256_file(document) == digest
    pixels = np.random.default_rng(7).random((10, 15, 3), dtype=np.float32)
    np.testing.assert_array_equal(render_photo(pixels, restored.current.values), render_srgb(pixels, .35))
    broken = copy.deepcopy(upgraded)
    broken["state"]["parameters"].pop("base_rendering")
    with pytest.raises(ValueError):
        validate_project(broken)


def test_new_natural_project_switch_reset_and_roundtrip(tmp_path):
    source = tmp_path / "new.ARW"
    source.write_bytes(b"new fixture")
    store = StateStore()
    store.activate(source, sha256_file(source), base_rendering="natural_v1")
    initial = store.current.values
    assert replace(initial, base_rendering="legacy_v1") == EditValues()
    store.set_parameter("base_rendering", "neutral_v1")
    assert len(store.undo_stack) == 1
    store.undo()
    assert store.current.values == initial
    store.set_exposure(.35)
    store.reset()
    assert store.current.values == initial
    target = tmp_path / "new.photocoach"
    write_project(target, snapshot_project(store))
    loaded = read_project(target)
    assert loaded.current.values == initial and not loaded.load_warnings
    assert loaded.undo_stack == store.undo_stack
    assert loaded.events[0]["event_id"] == store.events[0]["event_id"]
    with pytest.raises(ValueError):
        replace(initial, base_rendering="future")


def test_base_selector_ignores_teardown_signals(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from photocoach.app import MainWindow
    from photocoach.settings import SettingsStore
    app = QApplication.instance() or QApplication([])
    window = MainWindow(settings_store=SettingsStore(tmp_path / "settings.json"))
    monkeypatch.setattr(window, "request_preview", lambda **kwargs: None)
    source = tmp_path / "fixture.ARW"
    source.write_bytes(b"fixture")
    window.store.activate(source, sha256_file(source), base_rendering="natural_v1")
    window._set_controls(True)
    window._sync_controls()
    initial = window.store.current
    window._closing = True
    window.base_rendering.setCurrentIndex(1)
    window._closing = False
    window.base_rendering.clear()
    assert window.store.current == initial and not window.store.undo_stack
    window._close_approved = True
    window.close()
    window.deleteLater()
    app.processEvents()
