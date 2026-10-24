"""HSL math, audit migration, formal pixel agreement and passive color selection."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import colorsys
import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from photocoach.color_mixer import ColorMixer, apply_color_mixer, COLOR_KEYS
from photocoach.adjustments import render_edit, render_photo
from photocoach.imaging import render_srgb, quantize_srgb
from photocoach.state import EditState, EditValues, StateStore
from photocoach.project import snapshot_project, upgrade_project, validate_project, write_project, read_project
from photocoach.preview import DetailRequest, render_region
from photocoach.composition import Composition, output_size
from photocoach.safe_io import sha256_file
from photocoach.coach import build_request, MockCoach
from photocoach.coach_view import related_control


@pytest.mark.parametrize("bad", [(), ((0, 0, 0),)*7, ((31, 0, 0),)*8,
    ((0, -101, 0),)*8, ((0, 0, float("nan")),)*8, ((True, 0, 0),)*8])
def test_strict_values(bad):
    with pytest.raises(ValueError):
        ColorMixer(bad)


def test_serialization_is_immutable_and_strict():
    values = [[0, 0, 0] for _ in range(8)]
    mixer = ColorMixer(values)
    values[0][0] = 12
    assert mixer.neutral
    assert ColorMixer.from_dict(mixer.to_dict()) == mixer
    for bad in ({}, dict(mixer.to_dict(), cyan={}), {key: {} for key in COLOR_KEYS}):
        with pytest.raises(ValueError):
            ColorMixer.from_dict(bad)


def test_neutral_exact_old_pixel_pipeline():
    linear = np.random.default_rng(7).random((130, 150, 3), dtype=np.float32)
    values = EditValues(exposure_ev=.3, highlights=-.2, shadows=.1, contrast=.2,
                        saturation=1.13, rgb_curve=((0, 0), (.4, .5), (1, 1)))
    old = render_srgb(linear, .3, highlights=-.2, shadows=.1)
    old -= np.float32(.5)
    old *= np.float32(2**.2)
    old += np.float32(.5)
    np.clip(old, 0, 1, out=old)
    gray = (old[...,0]*.2126+old[...,1]*.7152+old[...,2]*.0722)[...,None]
    old -= gray
    old *= np.float32(1.13)
    old += gray
    np.clip(old, 0, 1, out=old)
    x, y = np.asarray(values.rgb_curve, np.float32).T
    old[:] = np.interp(old, x, y)
    reference = render_edit(linear, values)
    np.testing.assert_array_equal(reference, old)
    np.testing.assert_array_equal(reference, render_edit(linear, replace(values, color_mixer=ColorMixer())))


@pytest.mark.parametrize("axis,amount", [(0, 30), (0, -30), (1, 50), (1, -100), (2, 40), (2, -40)])
def test_uniform_controls_match_colorsys_reference(axis, amount):
    samples = np.random.default_rng(3).uniform(.05, .95, (1, 37, 3)).astype(np.float32)
    original = samples.copy()
    band = [0., 0., 0.]
    band[axis] = amount
    mixer = ColorMixer((tuple(band),)*8)
    apply_color_mixer(samples, mixer)
    expected = []
    for rgb in original[0]:
        h, l, s = colorsys.rgb_to_hls(*map(float, rgb))
        p = min(s/.1, 1)
        p = p*p*(3-2*p)
        a = amount*p
        if axis == 0:
            h = (h+a/360) % 1
        else:
            v = s if axis == 1 else l
            v += a/100*(1-v) if a >= 0 else a/100*v
            if axis == 1: s = v
            else: l = v
        expected.append(colorsys.hls_to_rgb(h, l, s))
    np.testing.assert_allclose(samples[0], expected, atol=5e-7)


def test_gray_neutral_fade_extremes_and_hue_seam():
    edge_pixels = np.array([[[1, 1, np.nextafter(np.float32(1), np.float32(0))],
                              [0, 0, np.nextafter(np.float32(0), np.float32(1))]]], np.float32)
    with np.errstate(all="raise", under="ignore"):
        apply_color_mixer(edge_pixels, ColorMixer(((30, 100, 100),)*8))
    assert np.isfinite(edge_pixels).all()
    for hue in (-30, 30):
        mixer = ColorMixer(((hue, 100, 100),)*8)
        gray = np.repeat(np.linspace(0, 1, 101, dtype=np.float32)[:,None,None], 3, axis=2)
        original = gray.copy()
        apply_color_mixer(gray, mixer)
        np.testing.assert_array_equal(gray, original)
    mixer = ColorMixer().with_value(0, 0, 30).with_value(7, 2, -100)
    points = np.array([[colorsys.hls_to_rgb(h, .5, .8) for h in (1-1e-6, 0, 1e-6)]], np.float32)
    apply_color_mixer(points, mixer)
    assert np.max(np.abs(points[0,0]-points[0,2])) < 3e-5
    rgb = np.random.default_rng(2).random((128, 300, 3), dtype=np.float32)
    apply_color_mixer(rgb, ColorMixer(((30, -100, -100),)*8))
    assert np.isfinite(rgb).all() and rgb.min() >= 0 and rgb.max() <= 1


def test_color_range_boundaries_are_continuous_and_other_colors_stay_put():
    mixer = ColorMixer().with_value(2, 2, 60)
    hue = np.array([colorsys.hls_to_rgb(h/360, .4, .7) for h in
                    (29.999, 30, 30.001, 59.999, 60, 60.001, 119.999, 120, 120.001, 240)], np.float32)[None]
    before = hue.copy()
    apply_color_mixer(hue, mixer)
    for i in (0, 3, 6):
        assert np.max(np.abs(hue[0,i]-hue[0,i+2])) < 1e-4
    np.testing.assert_allclose(hue[0,-1], before[0,-1], atol=2e-7)


@pytest.mark.parametrize("angle", [0, 3.25])
def test_hsl_native_roi_matches_formal_export(angle):
    linear = np.random.default_rng(19).random((190, 250, 3), dtype=np.float32)
    copy_of_source = linear.copy()
    values = EditValues(color_mixer=ColorMixer().with_value(2, 2, 20).with_value(5, 1, -30),
        composition=Composition((.05, .1, .95, .9), angle), rgb_curve=((0, 0), (.4, .5), (1, 1)))
    size = output_size((250, 190), values.composition)
    request = DetailRequest(1, size, (20, 17, 170, 110))
    roi = render_region(linear, values, request)
    np.testing.assert_array_equal(roi, quantize_srgb(render_photo(linear, values))[17:110,20:170])
    np.testing.assert_array_equal(linear, copy_of_source)


def store_for(tmp_path):
    source = tmp_path/"synthetic.ARW"
    source.write_bytes(b"test original")
    store = StateStore()
    store.activate(source, sha256_file(source))
    return store


@pytest.mark.parametrize("version", [1, 2, 3])
def test_legacy_history_migration_is_complete_strict_and_read_only(tmp_path, version):
    store = store_for(tmp_path)
    store.begin_drag("exposure_ev")
    store.set_exposure(.4)
    store.end_drag()
    store.reset()
    store.undo()
    data = snapshot_project(store)
    def strip(value):
        if isinstance(value, dict):
            if "white_balance" in value:
                value.pop("color_mixer")
                value.pop("local_gradients")
                for name in ("whites", "blacks", "vibrance"): value.pop(name)
                if version < 3: value.pop("composition")
                if version < 2:
                    value.pop("highlights")
                    value.pop("shadows")
            for item in value.values(): strip(item)
        elif isinstance(value, list):
            for item in value: strip(item)
    strip(data)
    data["schema_version"] = version
    data["pipeline_version"] = f"phase1-sdr-srgb-v{version}"
    original = copy.deepcopy(data)
    upgraded = upgrade_project(data)
    undo, redo = validate_project(upgraded)
    assert data == original and upgraded["schema_version"] == 7
    assert undo == store.undo_stack and redo == store.redo_stack
    bad = copy.deepcopy(data)
    bad["state"]["parameters"]["color_mixer"] = {}
    with pytest.raises(ValueError): upgrade_project(bad)


def test_ten_edits_history_restore_and_ai_context(tmp_path):
    store = store_for(tmp_path)
    before = store.current
    for i in range(10):
        store.set_parameter("color_mixer", ColorMixer().with_value(2, 2, i+1))
    final = store.current.values
    store.undo(); store.redo()
    assert store.current.values == final and len(store.undo_stack) == 10
    target = tmp_path/"session.photocoach"
    write_project(target, snapshot_project(store))
    loaded = read_project(target)
    assert loaded.current.values == final and loaded.undo_stack == store.undo_stack
    assert json.loads(target.read_text())["schema_version"] == 7
    pixels = np.zeros((4, 6, 3), np.uint8)
    request = build_request(store.current, before, store.events, pixels, pixels)
    assert request.parameters["color_mixer"]["yellow"]["lightness"] == 10
    reply = MockCoach().analyze(request)
    assert "Yellow" in reply.main_change and "Lightness +10" in reply.main_change
    assert related_control(reply.next_step_or_stop) == "color_mixer.yellow"
    assert related_control("Adjust blue saturation in the Color Mixer") == "color_mixer.blue"
    loaded.reset()
    assert loaded.current.color_mixer.neutral
    loaded.undo()
    assert loaded.current.color_mixer == final.color_mixer


def test_widget_passive_selection_gesture_resets_baseline_and_locate(tmp_path, monkeypatch):
    from photocoach.app import MainWindow
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    w.store = store_for(tmp_path)
    w._set_controls(True)
    w._sync_controls()
    monkeypatch.setattr(w, "request_preview", lambda **kw: None)
    widget = w.color_mixer
    widget.toggle.setChecked(True)
    widget.color.setCurrentIndex(2)
    assert not w.store.dirty and w.api_budget.attempts == 0
    w.change_compare(True)
    w._begin_parameter_gesture("color_mixer")
    for i in range(1, 11): widget.sliders[2].setValue(i)
    w._finish_gesture()
    assert not w.store.current.compare_baseline
    assert len(w.store.undo_stack) == 1 and w.store.current.color_mixer.bands[2][2] == 10
    widget.reset_color.click()
    assert w.store.current.color_mixer.neutral and len(w.store.undo_stack) == 2
    w.undo()
    assert widget.sliders[2].value() == 10
    widget.reset_all.click()
    assert len(w.store.undo_stack) == 2 and w.store.current.color_mixer.neutral
    revision = w.store.current.revision
    w._locate_control("color_mixer.blue")
    assert widget.color.currentIndex() == 5 and widget.toggle.isChecked()
    assert w.store.current.revision == revision
    w._settle_timer.stop(); w._render_timer.stop(); w._detail_timer.stop()
    w._close_approved = True
    w.close()
