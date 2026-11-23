"""Non-destructive brush masks: bounded strokes, strip parity, Schema 12 migration, and grouped Undo UI."""
from __future__ import annotations

import copy
import json
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from photocoach.adjustments import render_edit, render_photo
from photocoach.app import MainWindow, PreviewEngine
from photocoach.coach import MockCoach, build_request
from photocoach.coach_view import related_control
from photocoach.composition import Composition, sample_composition
from photocoach.gradients import (
    BrushMask,
    BrushStroke,
    DEFAULT_BRUSH_FEATHER,
    DEFAULT_BRUSH_RADIUS,
    LinearGradient,
    MAX_BRUSH_RADIUS,
    MAX_POINTS_PER_STROKE,
    MAX_STROKES_PER_MASK,
    MIN_BRUSH_RADIUS,
    OverlaySpec,
    RadialGradient,
    RenderGeometry,
    gradient_from_dict,
    gradient_weight,
    overlay_pixels,
)
from photocoach.histogram import sampled_histogram
from photocoach.i18n import apply_workspace_language
from photocoach.imaging import DecodedPhoto, RawOptions, quantize_srgb
from photocoach.preview import DetailRequest, render_region
from photocoach.project import (
    SCHEMA_VERSION,
    read_project,
    snapshot_project,
    upgrade_project,
    validate_project,
    write_project,
)
from photocoach.safe_io import sha256_file
from photocoach.settings import SettingsStore
from photocoach.state import EDIT_PIPELINE_VERSION, EditState, EditValues, StateStore


def _stroke(points=((0.25, 0.3), (0.75, 0.7)), **fields) -> BrushStroke:
    return BrushStroke(points=points, **fields)


def _brush(**fields) -> BrushMask:
    defaults = dict(
        id="b1",
        name="Brush 1",
        strokes=(_stroke(),),
    )
    defaults.update(fields)
    return BrushMask(**defaults)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"points": ()},
        {"points": ((0.5, 0.5),) * (MAX_POINTS_PER_STROKE + 1)},
        {"points": ((-0.01, 0.5),)},
        {"points": ((1.01, 0.5),)},
        {"points": ((float("nan"), 0.5),)},
        {"points": ((True, 0.5),)},
        {"points": ((0.5,),)},
        {"radius": MIN_BRUSH_RADIUS * 0.5},
        {"radius": MAX_BRUSH_RADIUS + 0.01},
        {"radius": True},
        {"feather": -0.05},
        {"feather": 1.05},
        {"feather": False},
        {"erase": 1},
    ],
)
def test_brush_stroke_rejects_invalid_or_unbounded_inputs(kwargs):
    params = dict(points=((0.3, 0.4),), radius=DEFAULT_BRUSH_RADIUS, feather=DEFAULT_BRUSH_FEATHER, erase=False)
    params.update(kwargs)
    with pytest.raises(ValueError):
        BrushStroke(**params)


def test_brush_mask_validation_caps_single_dab_and_roundtrip():
    dab = _brush(strokes=(_stroke(((0.4, 0.6),)),), exposure_ev=0.75, warmth=0.25)
    assert isinstance(dab, LinearGradient)
    assert dab.start == (0.4, 0.6) and dab.end == (0.4, 0.6)

    serialized = dab.to_dict()
    assert serialized["kind"] == "brush"
    assert gradient_from_dict(serialized) == dab
    assert EditValues.from_dict(EditValues(local_gradients=(dab,)).to_dict()).local_gradients == (dab,)

    # Empty or oversized stroke lists are rejected.
    with pytest.raises(ValueError):
        _brush(strokes=())
    with pytest.raises(ValueError):
        _brush(strokes=tuple(_stroke(((0.5, 0.5),)) for _ in range(MAX_STROKES_PER_MASK + 1)))

    # Malformed serialized stroke dicts fail strictly.
    bad_dict = copy.deepcopy(serialized)
    bad_dict["strokes"][0]["extra"] = 1
    with pytest.raises(ValueError):
        gradient_from_dict(bad_dict)

    # Shared 4-mask budget across Linear, Radial, and Brush masks.
    mixed_four = (
        LinearGradient("l1", "Linear 1", (0.1, 0.1), (0.9, 0.9)),
        RadialGradient("r1", "Radial 1", (0.2, 0.2), (0.8, 0.8)),
        _brush(id="b1", name="Brush 1"),
        _brush(id="b2", name="Brush 2"),
    )
    assert EditValues(local_gradients=mixed_four).local_gradients == mixed_four
    with pytest.raises(ValueError):
        EditValues(local_gradients=(*mixed_four, _brush(id="b3", name="Brush 3")))


def test_brush_weight_dab_polyline_isotropic_feather_erase_and_inversion():
    geometry = RenderGeometry.full((400, 200), Composition(), (400, 200))
    dab = _brush(strokes=(_stroke(((0.5, 0.5),), radius=0.10, feather=0.5),))
    w = gradient_weight(dab, geometry)
    assert w.shape == (200, 400) and w.dtype == np.float32
    assert w[100, 200] == pytest.approx(1.0)
    # Radius is 0.10 * min(400, 200) = 20 original pixels in both X and Y (isotropic).
    assert w[100, 225] == 0.0 and w[125, 200] == 0.0
    assert 0.0 < w[100, 215] < 1.0 and 0.0 < w[115, 200] < 1.0
    assert np.allclose(w + gradient_weight(replace(dab, inverted=True), geometry), 1.0)

    hard = gradient_weight(
        _brush(strokes=(_stroke(((0.2, 0.5), (0.8, 0.5)), radius=0.10, feather=0.0),)),
        geometry,
    )
    assert set(np.unique(hard)) == {0.0, 1.0}
    assert np.all(hard[100, 80:320] == 1.0)

    # Sequential paint -> erase -> re-paint compositing.
    painted_erased_repainted = _brush(
        strokes=(
            _stroke(((0.5, 0.5),), radius=0.15, feather=0.0, erase=False),
            _stroke(((0.5, 0.5),), radius=0.08, feather=0.0, erase=True),
            _stroke(((0.5, 0.5),), radius=0.03, feather=0.0, erase=False),
        )
    )
    comp_w = gradient_weight(painted_erased_repainted, geometry)
    assert comp_w[100, 200] == 1.0  # re-painted inner dot (r=6px)
    assert comp_w[100, 210] == 0.0  # erased ring (6px < d=10px <= 16px)
    assert comp_w[100, 224] == 1.0  # surviving outer ring (16px < d=24px <= 30px)
    assert comp_w[100, 240] == 0.0  # outside outer stroke

    # Multi-strip (>128 rows) matches explicit 128-row strip slicing identically.
    big_geom = RenderGeometry.full((320, 300), Composition(), (320, 300))
    big_w = gradient_weight(dab, big_geom)
    strip_a = gradient_weight(dab, big_geom.rows(0, 128))
    strip_b = gradient_weight(dab, big_geom.rows(128, 256))
    strip_c = gradient_weight(dab, big_geom.rows(256, 300))
    assert np.array_equal(big_w, np.vstack([strip_a, strip_b, strip_c]))


@pytest.mark.parametrize("mode", ["neutral", "disabled"])
def test_brush_neutral_and_disabled_bypass_bit_exact(mode):
    linear = np.random.default_rng(31).uniform(0.02, 0.85, (145, 211, 3)).astype(np.float32)
    linear.setflags(write=False)
    mask = _brush(exposure_ev=1.5 if mode == "disabled" else 0.0, enabled=(mode != "disabled"))
    base = EditValues(contrast=0.12, shadows=0.25)
    a = render_edit(linear, base)
    b = render_edit(linear, replace(base, local_gradients=(mask,)))
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))


@pytest.mark.parametrize("angle", [0.0, 4.5, -7.0])
@pytest.mark.parametrize("distortion", [0.0, -25.0, 20.0])
def test_brush_strip_and_roi_8bit_16bit_exact_with_crop_rotation_distortion(angle, distortion):
    linear = np.random.default_rng(42).uniform(0.02, 0.85, (289, 413, 3)).astype(np.float32)
    original = linear.copy()
    linear.setflags(write=False)
    masks = (
        BrushMask(
            "b1",
            "Brush 1",
            strokes=(
                _stroke(((0.22, 0.28), (0.52, 0.55), (0.76, 0.42)), radius=0.12, feather=0.65, erase=False),
                _stroke(((0.48, 0.48), (0.58, 0.58)), radius=0.05, feather=0.35, erase=True),
            ),
            exposure_ev=0.85,
            highlights=-0.25,
            shadows=0.35,
            warmth=0.3,
        ),
        RadialGradient("r1", "Radial 1", (0.15, 0.15), (0.7, 0.8), exposure_ev=-0.35, inverted=True),
    )
    from photocoach.lens import LensCorrections

    values = EditValues(
        exposure_ev=0.15,
        composition=Composition((0.06, 0.07, 0.94, 0.95), angle),
        lens_corrections=LensCorrections(distortion=distortion),
        local_gradients=masks,
    )
    full = render_photo(linear, values)
    size = (full.shape[1], full.shape[0])
    rect = (11, 17, size[0] - 9, size[1] - 13)
    roi_1 = render_region(linear, values, DetailRequest(0, size, rect, 1), workers=1)
    roi_4 = render_region(linear, values, DetailRequest(0, size, rect, 1), workers=4)
    assert np.array_equal(roi_1, roi_4)
    assert np.array_equal(roi_1, quantize_srgb(full)[rect[1] : rect[3], rect[0] : rect[2]])

    geometry = RenderGeometry((413, 289), values.composition, rect, (rect[2] - rect[0], rect[3] - rect[1]), distortion)
    sampled = sample_composition(linear, values.composition, rect=rect, distortion=distortion)
    roi_float = render_edit(sampled, values, geometry=geometry)
    assert np.array_equal(
        quantize_srgb(roi_float, bits=16),
        quantize_srgb(full, bits=16)[rect[1] : rect[3], rect[0] : rect[2]],
    )
    assert np.array_equal(linear, original) and np.isfinite(full).all()


def test_brush_overlay_histogram_and_ai_preview_isolation(monkeypatch):
    linear = np.full((32, 48, 3), 0.12, dtype=np.float32)
    linear.setflags(write=False)
    brush = _brush(strokes=(_stroke(((0.3, 0.4), (0.7, 0.6)), radius=0.18, feather=0.5),), exposure_ev=1.2)
    values = EditValues(local_gradients=(brush,))

    # Overlay matches brush weights and is display-only.
    disp = quantize_srgb(render_photo(linear, values))
    disp_orig = disp.copy()
    geom = RenderGeometry.full((48, 32), Composition(), (48, 32))
    ov = overlay_pixels(disp, geom, OverlaySpec("mask", brush))
    assert np.array_equal(disp, disp_orig)
    assert np.array_equal(ov[..., 3], np.rint(gradient_weight(brush, geom) * 90).astype(np.uint8))

    # Histogram reflects brush brightness lift.
    hist_base = sampled_histogram(linear, EditValues())
    hist_brush = sampled_histogram(linear, values)
    assert hist_brush["rgb"] != hist_base["rgb"]

    # AI request preview uses real BrushMask output without display overlay leakage.
    photo = DecodedPhoto(linear, {"sha256": "a", "half_size": False, "output_size": [48, 32]}, RawOptions(), False, Path("test.ARW"))
    engine = PreviewEngine()
    monkeypatch.setattr(engine, "photo", lambda *args: (photo, linear))
    engine.overlay_spec = OverlaySpec("mask", brush)
    before = EditState("a", Path("test.ARW"), "s")
    current = replace(before, revision=1, local_gradients=(brush,))
    req = engine.coach_request(current, before, [], "Review Brush 1")
    assert np.array_equal(req.current_preview, disp)
    assert req.parameters["local_gradients"][0]["kind"] == "brush"
    reply = MockCoach().analyze(req)
    assert "Brush 1" in reply.main_change


def test_schema12_roundtrip_and_strict_legacy_migration(tmp_path: Path):
    source = tmp_path / "photo.ARW"
    source.write_bytes(b"readonly arw bytes")
    digest = sha256_file(source)

    store = StateStore()
    store.activate(source, digest, base_rendering="natural_v1")
    brush = _brush(
        strokes=(
            _stroke(((0.2, 0.3), (0.6, 0.5)), radius=0.08, feather=0.5, erase=False),
            _stroke(((0.4, 0.4),), radius=0.04, feather=0.25, erase=True),
        ),
        exposure_ev=0.65,
        warmth=0.2,
    )
    store.set_parameter("local_gradients", (brush,))
    snap = store.save_snapshot("Brush Look")

    doc = snapshot_project(store)
    assert doc["schema_version"] == SCHEMA_VERSION == 14
    assert doc["pipeline_version"] == EDIT_PIPELINE_VERSION == "phase1-sdr-srgb-v14"

    project_path = tmp_path / "brush_v12.photocoach"
    write_project(project_path, doc)
    restored = read_project(project_path)
    assert restored.current.values == store.current.values
    assert restored.snapshots == [snap]
    assert restored.undo_stack == store.undo_stack
    assert not restored.load_warnings
    assert sha256_file(source) == digest

    # Schema 11 project (with snapshots) upgrades cleanly in memory to Schema 13 without modifying disk file.
    v11_store = StateStore()
    v11_store.activate(source, digest, base_rendering="natural_v1")
    v11_store.set_parameter("local_gradients", (RadialGradient("r1", "Radial 1", (0.2, 0.2), (0.8, 0.8), exposure_ev=0.4),))
    v11_store.save_snapshot("Before Brush")
    v11_doc = snapshot_project(v11_store)
    v11_doc["source"]["byte_length"] = source.stat().st_size
    def _strip_spots(node):
        if isinstance(node, dict):
            node.pop("spots", None)
            for child in node.values():
                _strip_spots(child)
        elif isinstance(node, list):
            for child in node:
                _strip_spots(child)
    _strip_spots(v11_doc)
    v11_doc["schema_version"] = 11
    v11_doc["pipeline_version"] = "phase1-sdr-srgb-v10"
    v11_orig = copy.deepcopy(v11_doc)

    upgraded = upgrade_project(v11_doc)
    assert v11_doc == v11_orig
    assert upgraded["schema_version"] == 14
    assert upgraded["pipeline_version"] == "phase1-sdr-srgb-v14"
    validate_project(upgraded)

    v11_file = tmp_path / "legacy_v11.photocoach"
    v11_file.write_text(json.dumps(v11_doc), encoding="utf-8")
    v11_hash = sha256_file(v11_file)
    loaded_v11 = read_project(v11_file)
    assert sha256_file(v11_file) == v11_hash
    assert loaded_v11.current.values == v11_store.current.values
    assert loaded_v11.snapshots == v11_store.snapshots
    assert "format v14" in loaded_v11.load_warnings[0]

    # Reject brush payloads smuggled into Schema 11 state or snapshots.
    bad_v11_state = copy.deepcopy(v11_doc)
    bad_v11_state["state"]["parameters"]["local_gradients"] = [brush.to_dict()]
    with pytest.raises(ValueError):
        upgrade_project(bad_v11_state)

    bad_v11_snap = copy.deepcopy(v11_doc)
    bad_v11_snap["snapshots"][0]["values"]["local_gradients"] = [brush.to_dict()]
    with pytest.raises(ValueError):
        upgrade_project(bad_v11_snap)

    # Reject brush payloads in Schema 6, 7, and 10 as well.
    for legacy_ver, pipe_ver in ((6, "phase1-sdr-srgb-v6"), (7, "phase1-sdr-srgb-v7"), (10, "phase1-sdr-srgb-v10")):
        legacy_doc = copy.deepcopy(v11_doc)
        legacy_doc.pop("snapshots")
        legacy_doc["state"].pop("compare_snapshot_id")
        legacy_doc["schema_version"] = legacy_ver
        legacy_doc["pipeline_version"] = pipe_ver
        if legacy_ver < 10:
            def _strip(node):
                if isinstance(node, dict):
                    node.pop("color_grading", None)
                    if legacy_ver < 9:
                        node.pop("base_rendering", None)
                    if legacy_ver < 8:
                        node.pop("lens_corrections", None)
                    for child in node.values():
                        _strip(child)
                elif isinstance(node, list):
                    for child in node:
                        _strip(child)
            _strip(legacy_doc)
        legacy_doc["state"]["parameters"]["local_gradients"] = [brush.to_dict()]
        with pytest.raises(ValueError):
            upgrade_project(legacy_doc)


@pytest.fixture
def window(tmp_path: Path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    w = MainWindow(settings_store=SettingsStore(tmp_path / "unused.json"))
    monkeypatch.setattr(w, "request_preview", lambda **kwargs: None)
    monkeypatch.setattr(w, "request_detail", lambda: None)
    monkeypatch.setattr(w.network_pool, "start", lambda *a: pytest.fail("No network allowed"))
    w.store.activate(Path("synthetic.ARW"), "photo-brush")
    w.preview.configure_photo((1200, 800))
    w.preview.display(np.full((400, 600, 3), 85, np.uint8))
    w._set_controls(True)
    w._sync_controls()
    w.resize(1280, 800)
    w.show()
    app.processEvents()
    yield w
    w._close_approved = True
    w.close()
    w.deleteLater()
    app.processEvents()


def test_brush_ui_passive_tools_grouped_undo_erase_cancel_and_i18n(window: MainWindow):
    w = window
    ge = w.local_gradients
    view = w.preview

    assert [ge.new_shape.itemText(i) for i in range(3)] == ["Linear", "Radial", "Brush"]
    assert ge.brush_panel.isHidden()

    # Choosing Brush and adjusting size/feather/erase before drawing is passive.
    before_state = w.store.current
    before_events = list(w.store.events)
    ge.new_shape.setCurrentText("Brush")
    ge.brush_size.setValue(90)
    ge.brush_feather.setValue(30)
    ge.brush_erase.setChecked(True)
    ge.brush_paint.setChecked(True)
    assert w.store.current is before_state
    assert w.store.events == before_events
    assert ge.brush_panel.isHidden()

    # Clicking Add mask enters Brush draw mode and exposes brush controls.
    ge.add.click()
    assert not ge.brush_panel.isHidden()
    assert ge.edit_on_photo.isChecked() and ge.show_mask.isChecked()

    # Continuous multi-move paint drag commits ONE Undo transaction.
    p0 = view.rect().center() + QPoint(-60, -25)
    p1 = p0 + QPoint(45, 15)
    p2 = p1 + QPoint(50, 20)
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=p0)
    QTest.mouseMove(view, p1)
    QTest.mouseMove(view, p2)
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=p2)

    assert len(w.store.undo_stack) == 1
    assert len(w.store.current.local_gradients) == 1
    mask = w.store.current.local_gradients[0]
    assert isinstance(mask, BrushMask)
    assert len(mask.strokes) == 1
    assert not mask.strokes[0].erase
    assert len(mask.strokes[0].points) >= 2
    assert mask.strokes[0].radius == pytest.approx(0.09)
    assert mask.strokes[0].feather == pytest.approx(0.30)

    # Adjust exposure slider, then append a second stroke in Erase mode (1 Undo per stroke).
    ge.sliders["exposure_ev"].setValue(75)
    assert len(w.store.undo_stack) == 2

    ge.brush_erase.setChecked(True)
    ge.brush_size.setValue(50)
    e0 = view.rect().center() + QPoint(-20, -10)
    e1 = e0 + QPoint(35, 15)
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=e0)
    QTest.mouseMove(view, e1)
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=e1)

    assert len(w.store.undo_stack) == 3
    after_erase = w.store.current.local_gradients[0]
    assert isinstance(after_erase, BrushMask) and len(after_erase.strokes) == 2
    assert after_erase.strokes[1].erase is True
    assert after_erase.strokes[1].radius == pytest.approx(0.05)

    # Single click dab works and commits 1 Undo entry.
    ge.brush_paint.setChecked(True)
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=view.rect().center() + QPoint(40, 30))
    assert len(w.store.undo_stack) == 4
    assert len(w.store.current.local_gradients[0].strokes) == 3

    # Undo/Redo steps cleanly stroke-by-stroke.
    w.undo()
    assert len(w.store.current.local_gradients[0].strokes) == 2
    w.redo()
    assert len(w.store.current.local_gradients[0].strokes) == 3

    # Esc mid-drag cancels the entire stroke cleanly without creating an Undo entry.
    c0 = view.rect().center() + QPoint(10, -40)
    c1 = c0 + QPoint(40, 20)
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=c0)
    QTest.mouseMove(view, c1)
    QTest.keyClick(view, Qt.Key.Key_Escape)
    assert len(w.store.undo_stack) == 4
    assert len(w.store.current.local_gradients[0].strokes) == 3
    assert not ge.edit_on_photo.isChecked()

    # Redraw mask replaces strokes while retaining tonal adjustments.
    ge.redraw.click()
    r0 = view.rect().center() + QPoint(-30, 10)
    r1 = r0 + QPoint(60, 0)
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=r0)
    QTest.mouseMove(view, r1)
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=r1)
    redrawn = w.store.current.local_gradients[0]
    assert len(redrawn.strokes) == 1
    assert redrawn.exposure_ev == pytest.approx(0.75)

    # Space + drag pans without painting a stroke when zoomed in.
    view.actual_size(view.rect().center())
    center_before = view.center
    stroke_count_before = len(w.store.current.local_gradients[0].strokes)
    ge.edit_on_photo.setChecked(True)
    QTest.keyPress(view, Qt.Key.Key_Space)
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=view.rect().center())
    QTest.mouseMove(view, view.rect().center() + QPoint(30, 20))
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=view.rect().center() + QPoint(30, 20))
    QTest.keyRelease(view, Qt.Key.Key_Space)
    assert view.center != center_before
    assert len(w.store.current.local_gradients[0].strokes) == stroke_count_before

    # Adding a Linear mask hides brush_panel when the Linear mask is selected.
    ge.new_shape.setCurrentText("Linear")
    ge.add.click()
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=QPoint(250, 180))
    QTest.mouseMove(view, QPoint(450, 300))
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=QPoint(450, 300))
    assert type(ge.selected) is LinearGradient
    assert ge.brush_panel.isHidden()

    # Bilingual ZH/EN toggle translates brush UI without modifying state or calling AI.
    ge.list.setCurrentIndex(0)
    assert isinstance(ge.selected, BrushMask) and not ge.brush_panel.isHidden()
    state_before_i18n = w.store.current
    apply_workspace_language(w, "zh")
    assert ge.new_shape.itemText(2) == "画笔 Brush"
    assert ge.brush_paint.text() == "涂抹"
    assert ge.brush_erase.text() == "擦除"
    assert "画笔大小" in ge.brush_size_label.name.text()
    assert "画笔羽化" in ge.brush_feather_label.name.text()
    assert w.store.current is state_before_i18n

    apply_workspace_language(w, "en")
    assert ge.new_shape.itemText(2) == "Brush"
    assert ge.brush_paint.text() == "Paint"
    assert ge.brush_erase.text() == "Erase"

    # related_control handles Brush masks in EN & ZH.
    assert related_control(f"Adjust {redrawn.name} Exposure.", w.store.current.local_gradients) == f"local_gradients.{redrawn.id}"
    assert related_control("Use a brush mask on the foreground.", ()) == "local_gradients"
    assert related_control("用画笔蒙版提亮前景。", ()) == "local_gradients"
    assert w.coach.calls == w.api_budget.attempts == 0


def test_brush_stroke_cap_shows_hint_without_crashing(window: MainWindow):
    w = window
    ge = w.local_gradients
    view = w.preview
    full_strokes = tuple(_stroke(((0.2 + 0.005 * i, 0.5),), radius=0.04) for i in range(MAX_STROKES_PER_MASK))
    capped = _brush(strokes=full_strokes, exposure_ev=0.5)
    w.change_parameter("local_gradients", (capped,), sync=True)
    ge.edit_on_photo.setChecked(True)
    before = w.store.current
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=view.rect().center())
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=view.rect().center())
    assert w.store.current is before
    assert str(MAX_STROKES_PER_MASK) in ge.hint.text()


def test_long_brush_drag_decimates_within_max_points_per_stroke():
    from photocoach.gradient_view import _decimate_points

    dense = [(i / 1000.0, 0.5) for i in range(1001)]
    decimated = _decimate_points(dense, MAX_POINTS_PER_STROKE)
    assert len(decimated) == MAX_POINTS_PER_STROKE
    assert decimated[0] == dense[0] and decimated[-1] == dense[-1]
    stroke = BrushStroke(tuple(decimated))
    assert len(stroke.points) == MAX_POINTS_PER_STROKE
