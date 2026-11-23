"""Graphite layout/navigation must stay separate from edits and cloud requests."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import pytest
from PySide6.QtWidgets import QApplication
from photocoach.app import MainWindow
from photocoach.settings import SettingsStore


@pytest.fixture
def window(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    view = MainWindow(settings_store=SettingsStore(tmp_path / "unused.json"))
    monkeypatch.setattr(view, "request_preview", lambda: None)
    monkeypatch.setattr(view, "request_detail", lambda: None)
    view.store.activate(Path("synthetic.ARW"), "photo-a")
    view._set_controls(True)
    view.resize(1280, 800)
    view.show()
    app.processEvents()
    yield view
    view._close_approved = True
    view.close()
    view.deleteLater()
    app.processEvents()


def test_initial_groups_and_passive_navigation(window):
    assert window.light_section.toggle.isChecked()
    assert not window.color_section.toggle.isChecked()
    assert not window.curve_section.toggle.isChecked()
    assert not window.color_mixer.toggle.isChecked() and not window.advanced_wb.isChecked()
    state, events = window.store.current, list(window.store.events)
    for name in ("saturation", "rgb_curve", "color_mixer.blue", "white_balance", "exposure_ev", "composition"):
        window._locate_control(name)
        QApplication.processEvents()
        if name != "composition":
            target = (window.color_mixer.color if name.startswith("color_mixer") else
                      window.wb_mode if name == "white_balance" else
                      window.curve if name == "rgb_curve" else
                      window.exposure_slider if name == "exposure_ev" else window.adjustment_sliders[name])
            viewport = window.edit_scroll.viewport()
            assert viewport.rect().contains(target.mapTo(viewport, target.rect().center()))
    assert all(section.toggle.isChecked() for section in
               (window.light_section, window.color_section, window.curve_section))
    assert window.color_mixer.color.currentText() == "Blue"
    assert window.store.current is state and window.store.events == events
    assert window.api_budget.attempts == window.coach.calls == 0
    assert window.curve.isVisible() and window.color_mixer.color.isVisible()


def test_topbar_info_and_sidebar_width(window):
    app = QApplication.instance()
    for size in ((1280, 800), (1920, 1080)):
        window.resize(*size)
        for _ in range(4): app.processEvents()
        assert 320 <= window.tabs.width() <= 350
        for widget in (window.undo_button, window.redo_button, window.photo_info_button, window.composition_button):
            assert widget.isVisible()
    window.photo_info_button.click()
    assert window.photo_info.isVisible()
    window.photo_info.reject()
    assert not window.photo_info_button.isChecked()


@pytest.mark.parametrize("requested_height", [180, 250])
def test_coach_menu_and_height_inputs_survive_navigation(window, requested_height):
    app = QApplication.instance()
    assert not window.pause_coach.isVisible() and window.coach_menu_button.isVisible()
    window.coach_collapse.click()
    app.processEvents()
    assert not window.pause_coach.isVisible() and window.coach_menu_button.isVisible()
    rectangles = [widget.geometry() for widget in (window.coach_summary, window.coach_selector, window.coach_menu_button, window.coach_collapse)]
    assert all(not a.intersects(b) for i, a in enumerate(rectangles) for b in rectangles[i+1:])
    assert [a.text() for a in window.coach_menu_button.menu().actions()] == [
        "Edit goal", "Usage and diagnostics"]
    window.goal.setText("Natural light")
    window.question.setText("How does the curve work?")
    window.workspace_splitter.setSizes([700-requested_height, requested_height])
    app.processEvents()
    height = window.coach_panel.height()
    for name in ("rgb_curve", "saturation", "exposure_ev"):
        window._locate_control(name)
        app.processEvents()
        assert window.coach_panel.height() == height
    window.coach_collapse.click(); window.coach_collapse.click()
    app.processEvents()
    assert abs(window.coach_panel.height() - height) <= 1
    assert window.goal.text() == "Natural light"
    assert window.question.text() == "How does the curve work?"
    window.coach_menu_button.menu().actions()[1].trigger()
    assert window.coach_details.isVisible() and (window.coach_mode.isVisible() if window.dev_mode else not window.coach_mode.isVisible())
    assert window.api_budget.attempts == window.coach.calls == 0


def test_crop_straighten_entry_points_summary_and_undoable_reset(window):
    from photocoach.composition import Composition
    assert window.composition_button.text() == "Crop / Straighten"
    assert window.crop_section.toggle.text() == "Crop and straighten"
    assert window.crop_open_button.text() == "Crop and straighten…"
    assert not window.crop_section.toggle.isChecked() and window.crop_section.body.isHidden()
    assert window.crop_summary.text() == "Full frame · Angle +0.00°"
    assert window.crop_open_button.isEnabled() and not window.crop_reset_button.isEnabled()
    depth = len(window.store.undo_stack)
    window.crop_reset_button.click()  # Neutral: no history entry.
    assert len(window.store.undo_stack) == depth
    window.change_parameter("composition", Composition((.1, .2, .9, .8), -3.5), sync=True)
    assert window.crop_summary.text() == "Cropped 80% × 60% · Angle -3.50°"
    assert window.crop_reset_button.isEnabled()
    edited = window.store.current.composition
    window.reset_composition()
    assert window.store.current.composition == Composition()
    assert len(window.store.undo_stack) == depth + 2
    assert window.crop_summary.text() == "Full frame · Angle +0.00°" and not window.crop_reset_button.isEnabled()
    window.undo()
    assert window.store.current.composition == edited and window.crop_reset_button.isEnabled()
    window._set_controls(False)
    assert not window.crop_open_button.isEnabled() and not window.crop_reset_button.isEnabled()
    assert window.api_budget.attempts == window.coach.calls == 0


def test_edit_sections_toggle_without_horizontal_or_zoom_jitter(window):
    import numpy as np
    from PySide6.QtCore import Qt
    app = QApplication.instance()
    assert window.detail_editor.note.wordWrap()
    assert window.lens_editor.note.wordWrap()
    assert window.edit_scroll.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOn
    assert window.edit_scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff

    window.preview.configure_photo((6000, 4000))
    window.preview.display(np.full((400, 600, 3), 120, dtype=np.uint8))
    sections = (
        window.light_section,
        window.crop_section,
        window.lens_section,
        window.detail_section,
        window.local_section,
        window.spots_section,
        window.color_section,
        window.curve_section,
    )
    for size in ((1280, 800), (1920, 1080)):
        window.resize(*size)
        for lang in ("en", "zh"):
            if (lang == "zh") != window.lang_button.isChecked():
                window.toggle_language()
            for _ in range(3):
                app.processEvents()
            baseline_vp_w = window.edit_scroll.viewport().width()
            baseline_splitter = window.editor_splitter.sizes()
            baseline_preview_size = window.preview.size()
            baseline_zoom = window.preview.zoom
            for section in sections:
                section.toggle.click()
                for _ in range(3):
                    app.processEvents()
                assert window.edit_scroll.viewport().width() == baseline_vp_w
                assert window.editor_splitter.sizes() == baseline_splitter
                assert window.preview.size() == baseline_preview_size
                assert abs(window.preview.zoom - baseline_zoom) < 1e-9
                section.toggle.click()
                for _ in range(3):
                    app.processEvents()
    if window.lang_button.isChecked():
        window.toggle_language()
