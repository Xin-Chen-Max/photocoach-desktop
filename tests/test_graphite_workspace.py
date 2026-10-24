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
    assert not window.pause_coach.isVisible() and not window.coach_menu_button.isVisible()
    window.coach_collapse.click()
    app.processEvents()
    assert window.pause_coach.isVisible() and window.coach_menu_button.isVisible()
    rectangles = [widget.geometry() for widget in (window.coach_summary, window.pause_coach,
        window.coach_model_status, window.coach_menu_button, window.coach_collapse)]
    assert all(not a.intersects(b) for i, a in enumerate(rectangles) for b in rectangles[i+1:])
    assert [a.text() for a in window.coach_menu_button.menu().actions()] == [
        "Connection…", "Usage and send confirmation…"]
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
