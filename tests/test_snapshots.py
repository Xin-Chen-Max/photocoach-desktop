"""Tests for Roadmap #5: Named Snapshots, Schema 11 Migration & A/B Comparison."""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import copy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from photocoach.adjustments import render_edit, render_photo
from photocoach.app import MainWindow, PreviewEngine, display_render_spec
from photocoach.color_grading import ColorGrading, ColorWheel
from photocoach.composition import Composition
from photocoach.detail import DetailControls
from photocoach.gradients import LinearGradient
from photocoach.imaging import RawOptions, quantize_srgb
from photocoach.lens import LensCorrections
from photocoach.preview import DetailRequest
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
from photocoach.snapshots import (
    MAX_SNAPSHOTS,
    MAX_SNAPSHOT_NAME_LENGTH,
    EditSnapshot,
    validate_snapshot_name,
    validate_snapshots,
)
from photocoach.state import EDIT_PIPELINE_VERSION, EditValues, StateStore
from photocoach.theme import configure_appearance


def _make_store(tmp_path: Path, name: str = "sample.ARW") -> StateStore:
    source = tmp_path / name
    source.write_bytes(b"readonly synthetic raw fixture for snapshot tests")
    store = StateStore()
    store.activate(source, sha256_file(source), base_rendering="natural_v1")
    return store


@pytest.fixture
def window(tmp_path: Path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    configure_appearance(app)
    win = MainWindow(settings_store=SettingsStore(tmp_path / "unused-settings.json"))
    monkeypatch.setattr(win, "request_preview", lambda **kw: None)
    monkeypatch.setattr(win, "request_detail", lambda: None)

    source = tmp_path / "synthetic.ARW"
    source.write_bytes(b"readonly synthetic raw bytes")
    state = win.store.activate(source, sha256_file(source), base_rendering="natural_v1")
    win.preview.configure_photo((600, 400))
    win.preview.display(np.full((200, 300, 3), 120, dtype=np.uint8))
    win.store.mark_preview(state)
    win._set_controls(True)
    win._sync_controls()
    win.resize(1280, 800)
    win.show()
    app.processEvents()

    yield win

    win._render_timer.stop()
    win._settle_timer.stop()
    win._detail_timer.stop()
    win._notice_timer.stop()
    win._highlight_timer.stop()
    win._close_approved = True
    win.close()
    win.deleteLater()
    app.processEvents()


def test_edit_snapshot_validation_capacity_and_roundtrip():
    values = EditValues(
        exposure_ev=0.75,
        contrast=0.25,
        color_grading=ColorGrading(shadows=ColorWheel(210.0, 0.35, 0.0)),
    )
    snap = EditSnapshot("snap-1", "  Warm  Base  ", "2026-11-22T01:00:00+00:00", values)
    assert snap.name == "Warm  Base"
    assert EditSnapshot.from_dict(snap.to_dict()) == snap

    assert validate_snapshot_name("  " + "a" * MAX_SNAPSHOT_NAME_LENGTH + "  ") == "a" * MAX_SNAPSHOT_NAME_LENGTH

    for bad_name in ("", "   ", "\t\n", "bad\x00name", "x" * (MAX_SNAPSHOT_NAME_LENGTH + 1), 123):
        with pytest.raises(ValueError):
            validate_snapshot_name(bad_name)

    for bad_snap in (
        {"id": "", "name": "ok", "created_utc": "2026-11-22T01:00:00+00:00", "values": values},
        {"id": "ok-id", "name": "ok", "created_utc": "not-a-timestamp", "values": values},
        {"id": "ok-id", "name": "ok", "created_utc": "2026-11-22T01:00:00+00:00", "values": "wrong"},
    ):
        with pytest.raises(ValueError):
            EditSnapshot(**bad_snap)

    for bad_dict in (
        {"id": "a", "name": "x", "created_utc": "2026-11-22T01:00:00+00:00"},
        {"id": "a", "name": "x", "created_utc": "2026-11-22T01:00:00+00:00", "values": values.to_dict(), "extra": 1},
    ):
        with pytest.raises(ValueError):
            EditSnapshot.from_dict(bad_dict)

    # Duplicate name (case-insensitive) and duplicate ID rejected; capacity capped at MAX_SNAPSHOTS.
    s1 = EditSnapshot("id-1", "Warm Look", "2026-11-22T01:00:00+00:00", values)
    s2_dup_name = EditSnapshot("id-2", "warm look", "2026-11-22T01:01:00+00:00", values)
    s2_dup_id = EditSnapshot("id-1", "Cool Look", "2026-11-22T01:01:00+00:00", values)
    with pytest.raises(ValueError, match="already exists"):
        validate_snapshots([s1, s2_dup_name])
    with pytest.raises(ValueError, match="Duplicate snapshot ID"):
        validate_snapshots([s1, s2_dup_id])

    many = [
        EditSnapshot(f"id-{i}", f"Look {i}", "2026-11-22T01:00:00+00:00", values)
        for i in range(MAX_SNAPSHOTS)
    ]
    assert len(validate_snapshots(many)) == MAX_SNAPSHOTS
    with pytest.raises(ValueError, match=str(MAX_SNAPSHOTS)):
        validate_snapshots([*many, EditSnapshot("extra", "Extra", "2026-11-22T01:00:00+00:00", values)])


def test_state_store_snapshots_undo_safe_restore_and_no_overwrite(tmp_path: Path):
    store = _make_store(tmp_path)
    original_photo = (store.current.photo_id, store.current.session_id, store.current.source_path)

    store.set_exposure(0.6)
    store.set_parameter("contrast", 0.3)
    warm_values = store.current.values
    store.mark_saved(store.current.revision)
    assert not store.dirty

    # Saving a snapshot marks dirty without bumping revision or pushing to undo_stack.
    rev_before = store.current.revision
    undo_len_before = len(store.undo_stack)
    warm_snap = store.save_snapshot("  Warm Base  ")
    assert warm_snap.name == "Warm Base"
    assert store.dirty
    assert store.current.revision == rev_before
    assert len(store.undo_stack) == undo_len_before

    # Case-insensitive duplicate name is rejected without overwriting existing snapshot.
    store.set_exposure(-0.4)
    with pytest.raises(ValueError, match="already exists"):
        store.save_snapshot("warm base")
    assert store.get_snapshot(warm_snap.id).values == warm_values

    cool_snap = store.save_snapshot("Cool Minus")
    assert len(store.snapshots) == 2

    # Saving while in Baseline or A/B compare still captures store.current.values.
    store.set_compare(True)
    assert store.current.compare_baseline and store.current.compare_snapshot is None
    still_cool = store.save_snapshot("Still Cool")
    assert still_cool.values.exposure_ev == -0.4

    # A/B compare is mutually exclusive with Baseline and preserves current.values.
    store.set_compare_snapshot(warm_snap.id)
    assert not store.current.compare_baseline
    assert store.current.compare_snapshot == warm_snap
    assert store.current.values.exposure_ev == -0.4

    # Restoring a snapshot exits comparison and commits ONE Undoable transaction preserving identity.
    assert store.restore_snapshot(warm_snap.id) is True
    assert store.current.values == warm_values
    assert not store.current.compare_baseline and store.current.compare_snapshot is None
    assert (store.current.photo_id, store.current.session_id, store.current.source_path) == original_photo
    assert store.undo_stack[-1].label == "Restore snapshot: Warm Base"
    assert len(store.undo_stack) == 4  # exposure, contrast, exposure(-0.4), restore

    # Restoring the identical state is a no-op.
    assert store.restore_snapshot(warm_snap.id) is False
    assert len(store.undo_stack) == 4

    # Single Undo step reverts to pre-restore state; Redo re-applies.
    store.undo()
    assert store.current.values == still_cool.values
    store.redo()
    assert store.current.values == warm_values

    # Deleting an actively compared snapshot exits A/B compare cleanly.
    store.set_compare_snapshot(cool_snap.id)
    assert store.current.compare_snapshot_id == cool_snap.id
    store.delete_snapshot(cool_snap.id)
    assert store.current.compare_snapshot is None
    assert [s.id for s in store.snapshots] == [warm_snap.id, still_cool.id]

    # Capacity guard enforces MAX_SNAPSHOTS = 24.
    for idx in range(MAX_SNAPSHOTS - len(store.snapshots)):
        store.save_snapshot(f"Extra {idx}")
    assert len(store.snapshots) == MAX_SNAPSHOTS
    with pytest.raises(ValueError, match=str(MAX_SNAPSHOTS)):
        store.save_snapshot("Overflow")

    # Opening another photo clears snapshots cleanly.
    store.activate(store.current.source_path, store.current.photo_id)
    assert store.snapshots == []
    assert not store.dirty


def test_schema11_project_roundtrip_and_schema10_migration(tmp_path: Path):
    source = tmp_path / "sunset.NEF"
    source.write_bytes(b"readonly nef source bytes")
    source_digest = sha256_file(source)

    store = StateStore()
    store.activate(source, source_digest, base_rendering="natural_v1")
    store.set_exposure(0.5)
    store.set_parameter(
        "color_grading",
        ColorGrading(shadows=ColorWheel(215.0, 0.4, -0.05), blend=0.7),
    )
    snap_a = store.save_snapshot("Teal Dusk")

    store.set_parameter(
        "local_gradients",
        (LinearGradient("g1", "Sky", (0.5, 0.1), (0.5, 0.6), exposure_ev=-0.6),),
    )
    store.set_parameter("detail", DetailControls(35.0, 20.0, 10.0, 15.0))
    snap_b = store.save_snapshot("Dramatic Sky")

    store.restore_snapshot(snap_a.id)
    store.undo()
    store.redo()
    store.set_compare_snapshot(snap_b.id)
    store.delete_snapshot(snap_a.id)
    store.mark_preview(store.current)

    project_path = tmp_path / "snapshots.photocoach"
    doc = snapshot_project(store)
    assert doc["schema_version"] == SCHEMA_VERSION == 14
    assert doc["pipeline_version"] == EDIT_PIPELINE_VERSION == "phase1-sdr-srgb-v14"
    assert len(doc["snapshots"]) == 1
    assert doc["state"]["compare_snapshot_id"] == snap_b.id

    write_project(project_path, doc)
    restored = read_project(project_path)
    assert not restored.dirty
    assert restored.snapshots == store.snapshots
    assert restored.current.values == store.current.values
    assert restored.current.compare_snapshot == snap_b
    assert restored.undo_stack == store.undo_stack
    assert restored.redo_stack == store.redo_stack
    assert not restored.load_warnings
    assert sha256_file(source) == source_digest

    # Corrupting compare_snapshot_id or missing snapshots fails strict validation.
    missing_snaps = copy.deepcopy(doc)
    missing_snaps.pop("snapshots")
    with pytest.raises(ValueError):
        validate_project(missing_snaps)

    dangling_compare = copy.deepcopy(doc)
    dangling_compare["snapshots"] = []
    with pytest.raises(ValueError):
        validate_project(dangling_compare)

    tampered_snap = copy.deepcopy(doc)
    tampered_snap["snapshots"][0]["values"]["exposure_ev"] = 99.0
    with pytest.raises(ValueError):
        validate_project(tampered_snap)

    # Schema 10 -> 11 migration preserves everything without writing legacy file.
    v10_store = _make_store(tmp_path, "legacy10.ARW")
    v10_store.set_exposure(0.45)
    v10_store.set_parameter("color_grading", ColorGrading(highlights=ColorWheel(40.0, 0.3, 0.0)))
    v10_doc = snapshot_project(v10_store)
    v10_doc["source"]["byte_length"] = v10_store.current.source_path.stat().st_size
    v10_doc.pop("snapshots")
    v10_doc["state"].pop("compare_snapshot_id")
    def _strip_spots(node):
        if isinstance(node, dict):
            node.pop("spots", None)
            for child in node.values():
                _strip_spots(child)
        elif isinstance(node, list):
            for child in node:
                _strip_spots(child)
    _strip_spots(v10_doc)
    v10_doc["schema_version"] = 10
    v10_doc["pipeline_version"] = "phase1-sdr-srgb-v10"
    v10_original = copy.deepcopy(v10_doc)

    upgraded = upgrade_project(v10_doc)
    assert v10_doc == v10_original
    assert upgraded["schema_version"] == 14
    assert upgraded["snapshots"] == []
    assert upgraded["state"]["compare_snapshot_id"] is None
    undo, redo = validate_project(upgraded)
    assert undo == v10_store.undo_stack and redo == v10_store.redo_stack

    v10_file = tmp_path / "v10.photocoach"
    v10_file.write_text(json.dumps(v10_doc), encoding="utf-8")
    v10_hash = sha256_file(v10_file)
    loaded_v10 = read_project(v10_file)
    assert sha256_file(v10_file) == v10_hash
    assert loaded_v10.snapshots == []
    assert loaded_v10.current.compare_snapshot is None
    assert loaded_v10.current.values == v10_store.current.values
    assert "format v14" in loaded_v10.load_warnings[0]

    bad_v10 = copy.deepcopy(v10_doc)
    bad_v10["snapshots"] = []
    with pytest.raises(ValueError):
        upgrade_project(bad_v10)


def test_preview_engine_renders_snapshot_ab_without_altering_export_or_coach(tmp_path: Path):
    source = tmp_path / "render.ARW"
    source.write_bytes(b"synthetic raw bytes")
    digest = sha256_file(source)

    store = StateStore()
    store.activate(source, digest, base_rendering="natural_v1")
    store.set_exposure(-0.8)
    store.set_parameter("composition", Composition((0.1, 0.1, 0.9, 0.9), 2.0))
    dark_snap = store.save_snapshot("Dark Cropped")

    store.set_exposure(1.2)
    store.set_parameter("composition", Composition((0.0, 0.0, 1.0, 1.0), 0.0))
    store.set_compare_snapshot(dark_snap.id)

    # display_render_spec aligns crop/distortion to current view while applying snapshot tone/color.
    options, display_vals = display_render_spec(store.current)
    assert options == dark_snap.values.raw_options
    assert display_vals.exposure_ev == pytest.approx(-0.8)
    assert display_vals.composition == store.current.composition
    assert store.current.values.exposure_ev == pytest.approx(1.2)

    linear = np.full((64, 96, 3), 0.25, dtype=np.float32)
    linear.setflags(write=False)
    fake_photo = SimpleNamespace(
        source_path=source,
        options=RawOptions(),
        half_size=False,
        linear_rgb=linear,
        metadata={
            "sha256": digest,
            "photo_id": digest,
            "visible_size": (96, 64),
            "output_size": (96, 64),
            "orientation_flip": 0,
            "half_size": False,
        },
    )
    engine = PreviewEngine()
    engine.cache.get = lambda path, opt, half_size=True, **kw: fake_photo

    ab_pixels, _, _ = engine.render(store.current)
    edited_pixels, _, _ = engine.render(replace(store.current, compare_snapshot=None))
    export_pixels = quantize_srgb(render_photo(linear, store.current.values))
    detail_ab, _, _ = engine.detail(store.current, DetailRequest(1, (96, 64), (8, 8, 56, 48), 1, (96, 64)))

    assert ab_pixels.mean() < edited_pixels.mean() - 40
    assert detail_ab.mean() < edited_pixels.mean() - 40
    assert export_pixels.mean() == pytest.approx(edited_pixels.mean(), abs=2.0)

    # Coach 100% crop inspection and trial preview ignore A/B compare and use actual edit values.
    _, metrics_edited = engine.inspect_region_100pct(store.current, 0.5, 0.5, 64, apply_current_edits=True)
    _, metrics_plain = engine.inspect_region_100pct(
        replace(store.current, compare_snapshot=None), 0.5, 0.5, 64, apply_current_edits=True
    )
    assert metrics_edited["mean_luminance"] == metrics_plain["mean_luminance"]


def test_snapshots_ui_compare_ab_preserves_zoom_pan_returns_on_edit_and_i18n(window, tmp_path: Path):
    app = QApplication.instance()
    panel = window.snapshots_panel
    assert panel.isEnabled()
    assert panel.count_label.text() == f"0 / {MAX_SNAPSHOTS}"

    # Passive tab switch and empty-list checks do not alter state or call AI.
    window.tabs.setCurrentIndex(1)
    app.processEvents()
    assert not panel.restore_button.isEnabled()
    assert not panel.delete_button.isEnabled()
    assert not panel.compare_button.isEnabled()

    # Create two snapshots with distinct edits.
    window.change_exposure(80)
    window._finish_gesture()
    panel.name_input.setText("  Bright Warm  ")
    panel.save_button.click()
    app.processEvents()
    assert len(window.store.snapshots) == 1
    assert panel.count_label.text() == f"1 / {MAX_SNAPSHOTS}"
    assert panel.name_input.text() == ""
    snap_bright = window.store.snapshots[0]

    # Duplicate name shows inline error and refuses to overwrite.
    panel.name_input.setText("bright warm")
    panel.save_button.click()
    app.processEvents()
    assert len(window.store.snapshots) == 1
    assert not panel.feedback_label.isHidden()

    window.change_exposure(-50)
    window.change_parameter("color_grading", ColorGrading(shadows=ColorWheel(210.0, 0.4, 0.0)), sync=True)
    panel.name_input.setText("Moody Blue")
    panel.save_button.click()
    app.processEvents()
    assert len(window.store.snapshots) == 2
    snap_moody = window.store.snapshots[1]

    # Selecting another list row while NOT comparing is passive (no state/revision/AI change).
    rev_before = window.store.current.revision
    panel.list_widget.setCurrentRow(0)
    app.processEvents()
    assert window.store.current.revision == rev_before
    assert window.store.current.compare_snapshot is None
    assert window.api_budget.attempts == 0 and window.coach.calls == 0

    # Zoom to 100% and pan, then toggle Compare A/B -> exact (center, zoom) preserved!
    window.preview.actual_size()
    window.preview.pan(24, -18)
    pose = (window.preview.center, window.preview.zoom)

    panel.compare_button.click()
    app.processEvents()
    assert window.store.current.compare_snapshot == snap_bright
    assert not window.edited_button.isChecked()
    assert not window.compare.isChecked()
    assert (window.preview.center, window.preview.zoom) == pose
    assert "Comparing A/B: Bright Warm" in window.view_notice.text()
    assert "[A/B] Bright Warm" in panel.list_widget.item(0).text()

    # Switching selection while A/B compare is active updates compared snapshot cleanly.
    panel.list_widget.setCurrentRow(1)
    app.processEvents()
    assert window.store.current.compare_snapshot == snap_moody
    assert (window.preview.center, window.preview.zoom) == pose
    assert "Comparing A/B: Moody Blue" in window.view_notice.text()

    # Clicking Edited exits A/B compare without changing zoom/pan or edit values.
    window.edited_button.click()
    app.processEvents()
    assert window.store.current.compare_snapshot is None
    assert window.edited_button.isChecked()
    assert (window.preview.center, window.preview.zoom) == pose

    # Re-enter A/B compare, then drag Exposure slider -> returns automatically to Edited without moving view.
    panel.list_widget.setCurrentRow(0)
    panel.compare_button.click()
    app.processEvents()
    assert window.store.current.compare_snapshot == snap_bright
    undo_before = len(window.store.undo_stack)
    window.exposure_slider.sliderPressed.emit()
    window.exposure_slider.setValue(30)
    window.exposure_slider.sliderReleased.emit()
    app.processEvents()
    assert window.store.current.compare_snapshot is None
    assert window.edited_button.isChecked()
    assert (window.preview.center, window.preview.zoom) == pose
    assert len(window.store.undo_stack) == undo_before + 1
    assert window.store.current.exposure_ev == pytest.approx(0.30)

    # Restore snapshot commits 1 Undo step and syncs all UI controls.
    panel.list_widget.setCurrentRow(0)
    panel.restore_button.click()
    app.processEvents()
    assert window.store.current.exposure_ev == pytest.approx(0.80)
    assert window.exposure_slider.value() == 80
    assert window.store.current.color_grading == ColorGrading()
    assert len(window.store.undo_stack) == undo_before + 2

    # Undo reverts restore in 1 step.
    window.undo()
    app.processEvents()
    assert window.store.current.exposure_ev == pytest.approx(0.30)
    assert window.store.current.color_grading == ColorGrading(shadows=ColorWheel(210.0, 0.4, 0.0))

    # Bilingual EN <-> ZH switching updates panel & active A/B notice without altering state/AI.
    panel.compare_button.click()
    app.processEvents()
    state_before_lang = window.store.current
    window.set_language("zh")
    assert panel.header_label.text() == "命名快照"
    assert panel.save_button.text() == "保存快照"
    assert panel.compare_button.text() == "A/B 对比"
    assert panel.restore_button.text() == "恢复"
    assert panel.delete_button.text() == "删除"
    assert "A/B 对比：Bright Warm" in window.view_notice.text()

    window.set_language("en")
    assert panel.header_label.text() == "Snapshots"
    assert panel.save_button.text() == "Save snapshot"
    assert panel.compare_button.text() == "Compare A/B"
    assert "Comparing A/B: Bright Warm" in window.view_notice.text()
    assert window.store.current is state_before_lang

    # Delete removes snapshot and cleanly exits A/B compare.
    panel.delete_button.click()
    app.processEvents()
    assert len(window.store.snapshots) == 1
    assert window.store.current.compare_snapshot is None
    assert window.edited_button.isChecked()
    assert window.api_budget.attempts == 0 and window.coach.calls == 0

    # _set_controls(False) disables the panel; True restores it.
    window._set_controls(False)
    assert not panel.isEnabled() and not panel.save_button.isEnabled()
    window._set_controls(True)
    assert panel.isEnabled() and panel.save_button.isEnabled()
