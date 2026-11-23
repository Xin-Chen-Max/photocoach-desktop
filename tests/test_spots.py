"""Non-destructive circular spot removal (heal/clone): strip/ROI parity, Schema 13 migration, and canvas UI."""
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
from photocoach.app import MainWindow, PreviewEngine, display_render_spec
from photocoach.coach import MockCoach, build_request
from photocoach.coach_view import related_control
from photocoach.composition import Composition, sample_composition
from photocoach.detail import DetailControls
from photocoach.gradients import BrushMask, BrushStroke, LinearGradient, RadialGradient, RenderGeometry
from photocoach.histogram import sampled_histogram
from photocoach.i18n import apply_workspace_language
from photocoach.imaging import DecodedPhoto, RawOptions, quantize_srgb
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
from photocoach.safe_io import sha256_file
from photocoach.settings import SettingsStore
from photocoach.spots import (
    DEFAULT_SPOT_FEATHER,
    DEFAULT_SPOT_OPACITY,
    DEFAULT_SPOT_RADIUS,
    MAX_SPOT_RADIUS,
    MAX_SPOTS,
    MIN_SPOT_RADIUS,
    SpotRemoval,
    apply_spots,
    suggest_spot_source,
)
from photocoach.state import EDIT_PIPELINE_VERSION, EditState, EditValues, StateStore


def _spot(**fields) -> SpotRemoval:
    defaults = dict(
        id="s1",
        target=(0.35, 0.40),
        source=(0.55, 0.40),
        radius=0.05,
        feather=0.5,
        mode="heal",
        opacity=1.0,
        enabled=True,
    )
    defaults.update(fields)
    return SpotRemoval(**defaults)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"id": ""},
        {"id": "bad id!"},
        {"target": (0.5,)},
        {"target": (-0.01, 0.5)},
        {"target": (0.5, 1.01)},
        {"target": (float("nan"), 0.5)},
        {"target": (True, 0.5)},
        {"source": (0.5,)},
        {"source": (-0.1, 0.5)},
        {"source": (1.1, 0.5)},
        {"source": (0.5, False)},
        {"radius": MIN_SPOT_RADIUS * 0.5},
        {"radius": MAX_SPOT_RADIUS + 0.01},
        {"radius": True},
        {"feather": -0.01},
        {"feather": 1.01},
        {"feather": False},
        {"mode": "airbrush"},
        {"mode": 1},
        {"opacity": -0.05},
        {"opacity": 1.05},
        {"opacity": True},
        {"enabled": 1},
    ],
)
def test_spot_validation_rejects_invalid_inputs(kwargs):
    params = dict(
        id="spot1",
        target=(0.3, 0.4),
        source=(0.5, 0.4),
        radius=DEFAULT_SPOT_RADIUS,
        feather=DEFAULT_SPOT_FEATHER,
        mode="heal",
        opacity=DEFAULT_SPOT_OPACITY,
        enabled=True,
    )
    params.update(kwargs)
    with pytest.raises(ValueError):
        SpotRemoval(**params)


def test_spot_caps_duplicate_ids_and_roundtrip():
    s = _spot(mode="clone", opacity=0.75, feather=0.25)
    serialized = s.to_dict()
    assert SpotRemoval.from_dict(serialized) == s
    assert EditValues.from_dict(EditValues(spots=(s,)).to_dict()).spots == (s,)

    bad = dict(serialized, extra=1)
    with pytest.raises(ValueError):
        SpotRemoval.from_dict(bad)

    bad_list = dict(serialized, target=(0.3, 0.4))
    with pytest.raises(ValueError):
        SpotRemoval.from_dict(bad_list)

    many = tuple(_spot(id=f"s{i}", target=(0.1 + 0.01 * i, 0.5)) for i in range(MAX_SPOTS))
    assert len(EditValues(spots=many).spots) == MAX_SPOTS
    with pytest.raises(ValueError):
        EditValues(spots=(*many, _spot(id="extra")))

    with pytest.raises(ValueError):
        EditValues(spots=(_spot(id="dup"), _spot(id="dup", target=(0.6, 0.6))))


@pytest.mark.parametrize(
    "target",
    [(0.5, 0.5), (0.02, 0.02), (0.98, 0.02), (0.02, 0.98), (0.98, 0.98)],
)
def test_suggest_spot_source_is_deterministic_clean_and_bounded(target):
    s1 = suggest_spot_source(target, DEFAULT_SPOT_RADIUS, (600, 400))
    s2 = suggest_spot_source(target, DEFAULT_SPOT_RADIUS, (600, 400))
    assert s1 == s2
    assert 0.0 <= s1[0] <= 1.0 and 0.0 <= s1[1] <= 1.0
    assert s1 != target


@pytest.mark.parametrize("mode", ["empty", "disabled", "zero_opacity"])
def test_spot_idle_bypass_is_bit_exact(mode):
    linear = np.random.default_rng(17).uniform(0.02, 0.85, (148, 213, 3)).astype(np.float32)
    linear.setflags(write=False)
    base = EditValues(exposure_ev=0.1, contrast=0.15)
    if mode == "empty":
        spots = ()
    elif mode == "disabled":
        spots = (_spot(enabled=False, opacity=1.0),)
    else:
        spots = (_spot(enabled=True, opacity=0.0),)

    a = render_edit(linear, base)
    b = render_edit(linear, replace(base, spots=spots))
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))

    geom = RenderGeometry.full((213, 148), Composition(), (213, 148))
    assert apply_spots(linear, spots, geom, linear) is linear


def test_heal_matches_target_luminance_and_gradient_better_than_clone():
    # Vertical sky gradient from dark (top) to bright (bottom), plus a dark sensor dust speck.
    h, w = 240, 320
    y_ramp = np.linspace(0.18, 0.52, h, dtype=np.float32)[:, None, None]
    clean = np.broadcast_to(y_ramp, (h, w, 3)).copy()
    # Add subtle uniform texture so source/target texture transfers meaningfully.
    noise = np.random.default_rng(9).normal(0.0, 0.004, (h, w, 3)).astype(np.float32)
    clean = np.clip(clean + noise, 0.02, 0.95)

    # Place a dark dust blob near (0.5, 0.72) where sky is brighter (~0.42),
    # and sample source from (0.5, 0.25) where sky is much darker (~0.26).
    yy, xx = np.ogrid[:h, :w]
    tx, ty = int(0.5 * w), int(0.72 * h)
    r_px = 0.06 * min(w, h)
    dust_mask = (xx - tx) ** 2 + (yy - ty) ** 2 <= (r_px * 0.65) ** 2
    dirty = clean.copy()
    dirty[dust_mask] = 0.03
    dirty.setflags(write=False)

    heal_spot = SpotRemoval("h1", target=(0.5, 0.72), source=(0.5, 0.25), radius=0.06, feather=0.5, mode="heal")
    clone_spot = replace(heal_spot, mode="clone")
    half_spot = replace(heal_spot, opacity=0.5)

    geom = RenderGeometry.full((w, h), Composition(), (w, h))
    healed = apply_spots(dirty, (heal_spot,), geom, dirty)
    cloned = apply_spots(dirty, (clone_spot,), geom, dirty)
    half_healed = apply_spots(dirty, (half_spot,), geom, dirty)

    assert np.isfinite(healed).all() and np.isfinite(cloned).all()
    # Dust center should be lifted close to clean target sky (~0.42).
    assert abs(float(healed[ty, tx].mean()) - float(clean[ty, tx].mean())) < 0.03
    # Clone transfers the darker top sky (~0.26) directly, leaving a strong brightness mismatch.
    assert abs(float(cloned[ty, tx].mean()) - float(clean[ty, tx].mean())) > 0.12
    # Healed seam error inside the spot vs clean reference is far lower than clone.
    disc = (xx - tx) ** 2 + (yy - ty) ** 2 <= r_px ** 2
    heal_err = float(np.abs(healed[disc] - clean[disc]).mean())
    clone_err = float(np.abs(cloned[disc] - clean[disc]).mean())
    assert heal_err < clone_err * 0.25
    # Partial opacity blends smoothly halfway between dirty and full heal.
    assert np.allclose(half_healed[ty, tx], 0.5 * dirty[ty, tx] + 0.5 * healed[ty, tx], atol=1e-5)


@pytest.mark.parametrize("angle", [0.0, 4.5, -6.0])
@pytest.mark.parametrize("distortion", [0.0, -25.0, 20.0])
def test_spot_strip_and_roi_8bit_16bit_exact_with_crop_angle_distortion_detail(angle, distortion):
    linear = np.random.default_rng(77).uniform(0.03, 0.85, (289, 413, 3)).astype(np.float32)
    # Inject visible dust spots.
    linear[130:144, 190:204] = 0.02
    linear[160:172, 230:242] = 0.92
    original = linear.copy()
    linear.setflags(write=False)

    # Spot 1's source (0.08, 0.10) intentionally lies OUTSIDE the cropped ROI rect!
    spots = (
        SpotRemoval("s1", target=(0.48, 0.47), source=(0.08, 0.10), radius=0.05, feather=0.5, mode="heal", opacity=1.0),
        SpotRemoval("s2", target=(0.57, 0.57), source=(0.82, 0.80), radius=0.04, feather=0.35, mode="clone", opacity=0.85),
    )
    values = EditValues(
        base_rendering="natural_v1",
        exposure_ev=0.15,
        composition=Composition((0.12, 0.12, 0.88, 0.88), angle),
        lens_corrections=LensCorrections(distortion=distortion, vignetting_ev=0.35),
        detail=DetailControls(sharpening=35.0, masking=10.0, luminance_nr=10.0, chroma_nr=20.0),
        spots=spots,
    )

    full = render_photo(linear, values)
    size = (full.shape[1], full.shape[0])
    rect = (15, 21, size[0] - 13, size[1] - 17)
    roi_1 = render_region(linear, values, DetailRequest(0, size, rect, 1), workers=1)
    roi_4 = render_region(linear, values, DetailRequest(0, size, rect, 1), workers=4)
    assert np.array_equal(roi_1, roi_4)
    assert np.array_equal(roi_1, quantize_srgb(full)[rect[1] : rect[3], rect[0] : rect[2]])

    # Without detail active, verify direct 16-bit ROI equivalence against full render_photo.
    no_detail = replace(values, detail=DetailControls())
    full_nd = render_photo(linear, no_detail)
    geom = RenderGeometry((413, 289), no_detail.composition, rect, (rect[2] - rect[0], rect[3] - rect[1]), distortion)
    sampled = sample_composition(linear, no_detail.composition, rect=rect, distortion=distortion)
    roi_float = render_edit(sampled, no_detail, geometry=geom, source_linear=linear)
    assert np.array_equal(
        quantize_srgb(roi_float, bits=16),
        quantize_srgb(full_nd, bits=16)[rect[1] : rect[3], rect[0] : rect[2]],
    )
    assert np.array_equal(linear, original) and np.isfinite(full).all()


def test_spots_histogram_baseline_and_ai_preview_isolation(monkeypatch):
    linear = np.full((64, 96, 3), 0.35, dtype=np.float32)
    linear[20:44, 36:60] = 0.01  # large dark spot
    linear.setflags(write=False)

    spot = SpotRemoval("s1", target=(0.5, 0.5), source=(0.15, 0.2), radius=0.14, feather=0.3, mode="heal")
    values = EditValues(spots=(spot,))

    hist_base = sampled_histogram(linear, EditValues())
    hist_healed = sampled_histogram(linear, values)
    assert hist_healed["rgb"] != hist_base["rgb"]

    # Baseline view ignores spots while Edited shows them.
    state = EditState("a", Path("test.ARW"), "sess", revision=1, compare_baseline=True, spots=(spot,))
    _, baseline_vals = display_render_spec(state)
    assert baseline_vals.spots == ()
    _, edited_vals = display_render_spec(replace(state, compare_baseline=False))
    assert edited_vals.spots == (spot,)

    # Coach request receives healed preview and spot metadata without overlay graphics.
    photo = DecodedPhoto(linear, {"sha256": "a", "half_size": False, "output_size": [96, 64]}, RawOptions(), False, Path("test.ARW"))
    engine = PreviewEngine()
    monkeypatch.setattr(engine, "photo", lambda *args: (photo, linear))
    before = EditState("a", Path("test.ARW"), "sess")
    req = engine.coach_request(replace(state, compare_baseline=False), before, [], "Check dust removal")
    expected_preview = quantize_srgb(render_photo(linear, values))
    assert np.array_equal(req.current_preview, expected_preview)
    assert req.parameters["spots"] == [spot.to_dict()]
    reply = MockCoach().analyze(req)
    assert "Spot removal" in reply.main_change


def test_schema13_roundtrip_and_strict_legacy_migration(tmp_path: Path):
    source = tmp_path / "photo.ARW"
    source.write_bytes(b"readonly arw bytes")
    digest = sha256_file(source)

    store = StateStore()
    store.activate(source, digest, base_rendering="natural_v1")
    s1 = _spot(id="s1", target=(0.3, 0.4), source=(0.5, 0.4), mode="heal")
    s2 = _spot(id="s2", target=(0.7, 0.6), source=(0.8, 0.6), mode="clone", opacity=0.75)
    store.set_parameter("spots", (s1, s2))
    snap = store.save_snapshot("Spots Cleaned")

    doc = snapshot_project(store)
    assert doc["schema_version"] == SCHEMA_VERSION == 14
    assert doc["pipeline_version"] == EDIT_PIPELINE_VERSION == "phase1-sdr-srgb-v14"

    project_path = tmp_path / "spots_v13.photocoach"
    write_project(project_path, doc)
    restored = read_project(project_path)
    assert restored.current.values == store.current.values
    assert restored.snapshots == [snap]
    assert restored.undo_stack == store.undo_stack
    assert not restored.load_warnings
    assert sha256_file(source) == digest

    # Schema 12 project (with BrushMask + snapshots) upgrades cleanly in memory to Schema 13 read-only.
    v12_store = StateStore()
    v12_store.activate(source, digest, base_rendering="natural_v1")
    brush = BrushMask("b1", "Brush 1", strokes=(BrushStroke(((0.2, 0.3), (0.6, 0.5))),), exposure_ev=0.5)
    v12_store.set_parameter("local_gradients", (brush,))
    v12_store.save_snapshot("With Brush")
    v12_doc = snapshot_project(v12_store)
    v12_doc["source"]["byte_length"] = source.stat().st_size

    def _strip_spots(node):
        if isinstance(node, dict):
            node.pop("spots", None)
            for child in node.values():
                _strip_spots(child)
        elif isinstance(node, list):
            for child in node:
                _strip_spots(child)

    _strip_spots(v12_doc)
    v12_doc["schema_version"] = 12
    v12_doc["pipeline_version"] = "phase1-sdr-srgb-v12"
    v12_orig = copy.deepcopy(v12_doc)

    upgraded = upgrade_project(v12_doc)
    assert v12_doc == v12_orig
    assert upgraded["schema_version"] == 14
    assert upgraded["pipeline_version"] == "phase1-sdr-srgb-v14"
    validate_project(upgraded)

    v12_file = tmp_path / "legacy_v12.photocoach"
    v12_file.write_text(json.dumps(v12_doc), encoding="utf-8")
    v12_hash = sha256_file(v12_file)
    loaded_v12 = read_project(v12_file)
    assert sha256_file(v12_file) == v12_hash
    assert loaded_v12.current.values == v12_store.current.values
    assert loaded_v12.snapshots == v12_store.snapshots
    assert "format v14" in loaded_v12.load_warnings[0]

    # Reject spots smuggled into Schema 12 state or snapshots, and into Schema 11 / 10 / 6.
    bad_v12_state = copy.deepcopy(v12_doc)
    bad_v12_state["state"]["parameters"]["spots"] = [s1.to_dict()]
    with pytest.raises(ValueError):
        upgrade_project(bad_v12_state)

    bad_v12_snap = copy.deepcopy(v12_doc)
    bad_v12_snap["snapshots"][0]["values"]["spots"] = [s1.to_dict()]
    with pytest.raises(ValueError):
        upgrade_project(bad_v12_snap)

    for legacy_ver, pipe_ver in ((6, "phase1-sdr-srgb-v6"), (10, "phase1-sdr-srgb-v10"), (11, "phase1-sdr-srgb-v10")):
        legacy_doc = copy.deepcopy(v12_doc)
        if legacy_ver < 11:
            legacy_doc.pop("snapshots")
            legacy_doc["state"].pop("compare_snapshot_id")
        if legacy_ver < 12:
            def _clean_old(node):
                if isinstance(node, dict):
                    if "local_gradients" in node:
                        node["local_gradients"] = []
                    if legacy_ver < 10:
                        node.pop("color_grading", None)
                        node.pop("base_rendering", None)
                        node.pop("lens_corrections", None)
                    for child in node.values():
                        _clean_old(child)
                elif isinstance(node, list):
                    for child in node:
                        _clean_old(child)
            _clean_old(legacy_doc)
        legacy_doc["schema_version"] = legacy_ver
        legacy_doc["pipeline_version"] = pipe_ver
        legacy_doc["state"]["parameters"]["spots"] = [s1.to_dict()]
        with pytest.raises(ValueError):
            upgrade_project(legacy_doc)


@pytest.fixture
def window(tmp_path: Path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    w = MainWindow(settings_store=SettingsStore(tmp_path / "unused.json"))
    monkeypatch.setattr(w, "request_preview", lambda **kwargs: None)
    monkeypatch.setattr(w, "request_detail", lambda: None)
    monkeypatch.setattr(w.network_pool, "start", lambda *a: pytest.fail("No network allowed"))
    w.store.activate(Path("synthetic.ARW"), "photo-spots")
    w.preview.configure_photo((1200, 800))
    w.preview.display(np.full((400, 600, 3), 90, np.uint8))
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


def test_spots_ui_direct_manipulation_grouped_undo_cancel_pan_and_i18n(window: MainWindow):
    w = window
    se = w.spots_editor
    layer = w.spots_controller.layer
    view = w.preview

    assert not w.spots_section.toggle.isChecked()
    w.spots_section.set_expanded(True)

    # Passive controls before any spot exists must NOT mutate StateStore.
    before_state = w.store.current
    before_events = list(w.store.events)
    se.mode_clone.click()
    se.sliders["radius"].setValue(40)
    se.sliders["feather"].setValue(30)
    se.sliders["opacity"].setValue(80)
    assert w.store.current is before_state
    assert w.store.events == before_events
    assert se.count_label.text() == f"0 / {MAX_SPOTS}"

    # Mutual exclusion: enabling spots edit_on_photo disables local_gradients edit_on_photo and pick_wb.
    w.local_gradients.edit_on_photo.setChecked(True)
    se.edit_on_photo.setChecked(True)
    assert not w.local_gradients.edit_on_photo.isChecked()
    assert not w.pick_wb.isChecked()

    # Click photo to place a spot (grouped 1 Undo entry).
    click_pos = view.rect().center() + QPoint(-30, -20)
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=click_pos)
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=click_pos)
    assert len(w.store.undo_stack) == 1
    assert len(w.store.current.spots) == 1
    s1 = w.store.current.spots[0]
    assert s1.mode == "clone"
    assert s1.radius == pytest.approx(0.04)
    assert s1.feather == pytest.approx(0.30)
    assert s1.opacity == pytest.approx(0.80)
    assert s1.source != s1.target
    assert se.count_label.text() == f"1 / {MAX_SPOTS}"

    # Drag target circle independently -> target moves, source stays fixed, 1 Undo step.
    tgt_scr = layer.screen_point(s1.target).toPoint()
    tgt_dst = tgt_scr + QPoint(24, 16)
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=tgt_scr)
    QTest.mouseMove(view, tgt_dst)
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=tgt_dst)
    assert len(w.store.undo_stack) == 2
    s1_after_tgt = w.store.current.spots[0]
    assert s1_after_tgt.target != s1.target
    assert s1_after_tgt.source == s1.source

    # Drag source circle independently -> source moves, target stays fixed, 1 Undo step.
    src_scr = layer.screen_point(s1_after_tgt.source).toPoint()
    src_dst = src_scr + QPoint(-18, 22)
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=src_scr)
    QTest.mouseMove(view, src_dst)
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=src_dst)
    assert len(w.store.undo_stack) == 3
    s1_after_src = w.store.current.spots[0]
    assert s1_after_src.source != s1_after_tgt.source
    assert s1_after_src.target == s1_after_tgt.target

    # Drag perimeter resize handle -> radius changes, 1 Undo step.
    handle_scr = layer.screen_point(layer.resize_handle_norm(s1_after_src)).toPoint()
    handle_dst = handle_scr + QPoint(18, 0)
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=handle_scr)
    QTest.mouseMove(view, handle_dst)
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=handle_dst)
    assert len(w.store.undo_stack) == 4
    assert w.store.current.spots[0].radius > s1_after_src.radius

    # Switch mode to Heal and adjust slider -> updates selected spot.
    se.mode_heal.click()
    assert len(w.store.undo_stack) == 5
    assert w.store.current.spots[0].mode == "heal"

    # Esc mid-drag cancels cleanly without adding an Undo step and exits Edit on photo.
    cur_tgt_scr = layer.screen_point(w.store.current.spots[0].target).toPoint()
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=cur_tgt_scr)
    QTest.mouseMove(view, cur_tgt_scr + QPoint(35, -15))
    QTest.keyClick(view, Qt.Key.Key_Escape)
    assert len(w.store.undo_stack) == 5
    assert w.store.current.spots[0].target == s1_after_src.target
    assert not se.edit_on_photo.isChecked()

    # Space + drag pans when zoomed in without placing or moving a spot.
    view.actual_size(view.rect().center())
    center_before = view.center
    se.edit_on_photo.setChecked(True)
    QTest.keyPress(view, Qt.Key.Key_Space)
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=view.rect().center())
    QTest.mouseMove(view, view.rect().center() + QPoint(30, 20))
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=view.rect().center() + QPoint(30, 20))
    QTest.keyRelease(view, Qt.Key.Key_Space)
    assert view.center != center_before
    assert len(w.store.current.spots) == 1

    # Delete spot + Undo + Reset spots + Undo.
    se.delete_button.click()
    assert len(w.store.current.spots) == 0
    w.undo()
    assert len(w.store.current.spots) == 1
    se.reset_button.click()
    assert len(w.store.current.spots) == 0
    w.undo()
    assert len(w.store.current.spots) == 1

    # Bilingual ZH/EN toggle translates Spot removal section and controls without changing state.
    state_before_i18n = w.store.current
    apply_workspace_language(w, "zh")
    assert w.spots_section.toggle.text() == "污点修复 Spots"
    assert se.mode_heal.text() == "修复 Heal"
    assert se.mode_clone.text() == "仿制 Clone"
    assert "污点大小" in se.labels["radius"].name.text()
    assert "不透明度" in se.labels["opacity"].name.text()
    assert se.delete_button.text() == "删除污点"
    assert se.reset_button.text() == "重置污点"
    assert w.store.current is state_before_i18n

    apply_workspace_language(w, "en")
    assert w.spots_section.toggle.text() == "Spot removal"
    assert se.mode_heal.text() == "Heal"
    assert se.mode_clone.text() == "Clone"

    # Coach navigation separates spot removal from spotlight masks.
    assert related_control("Remove sensor dust using Spot removal.") == "spots"
    assert related_control("用污点修复清理灰尘瑕疵。") == "spots"
    assert related_control("Add a warm spotlight on the subject.") == "local_gradients"
    assert w.coach.calls == w.api_budget.attempts == 0


def test_spot_cap_shows_hint_without_crashing(window: MainWindow):
    w = window
    se = w.spots_editor
    view = w.preview
    full_spots = tuple(_spot(id=f"s{i}", target=(0.05 + 0.014 * i, 0.1)) for i in range(MAX_SPOTS))
    w.change_parameter("spots", full_spots, sync=True)
    se.edit_on_photo.setChecked(True)
    before = w.store.current
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=view.rect().center() + QPoint(0, 80))
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=view.rect().center() + QPoint(0, 80))
    assert w.store.current is before
    assert str(MAX_SPOTS) in se.hint.text()


def test_heal_rejects_partial_ring_obstacle_outliers():
    from photocoach.spots import prepare_spots

    h, w = 240, 320
    clean_sky = np.full((h, w, 3), 0.30, dtype=np.float32)
    clean_sky[..., 2] = 0.45
    dirty = clean_sky.copy()
    # Dark dust fully inside the core (r_px * (1 - feather) = 0.5 * r_px) at target (0.35, 0.45)
    yy, xx = np.ogrid[:h, :w]
    tx, ty = int(0.35 * w), int(0.45 * h)
    r_px = 0.06 * min(w, h)
    dirty[(xx - tx) ** 2 + (yy - ty) ** 2 <= (r_px * 0.45) ** 2] = 0.02
    # Obstacle intersecting the bottom edge of the source perimeter ring around (0.65, 0.45)
    sx, sy = int(0.65 * w), int(0.45 * h)
    dirty[int(sy + r_px * 1.06) :, sx - 30 : sx + 30] = 0.01
    dirty.setflags(write=False)

    spot = SpotRemoval("o1", target=(0.35, 0.45), source=(0.65, 0.45), radius=0.06, feather=0.5, mode="heal")
    ps = prepare_spots((spot,), dirty, (w, h))[0]
    assert np.allclose(ps.gain_a, 0.0, atol=1e-5)
    assert np.allclose(ps.gain_bx, 0.0, atol=1e-5)
    assert np.allclose(ps.gain_by, 0.0, atol=1e-5)

    geom = RenderGeometry.full((w, h), Composition(), (w, h))
    healed = apply_spots(dirty, (spot,), geom, dirty)
    disc = (xx - tx) ** 2 + (yy - ty) ** 2 <= r_px ** 2
    assert float(np.abs(healed[disc] - clean_sky[disc]).max()) < 1e-4

