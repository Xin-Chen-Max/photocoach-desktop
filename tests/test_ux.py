"""Offline interaction contracts, separate from physical desktop acceptance."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from photocoach.app import MainWindow
from photocoach.settings import SettingsStore


@pytest.fixture
def window(tmp_path, monkeypatch):
    application = QApplication.instance() or QApplication([])
    view = MainWindow(settings_store=SettingsStore(tmp_path / "unused.json"))
    monkeypatch.setattr(view, "request_preview", lambda: None)
    monkeypatch.setattr(view, "request_detail", lambda: None)
    view.store.activate(Path("synthetic.ARW"), "photo-a")
    view._set_controls(True)
    view.resize(1280, 800)
    view.show()
    application.processEvents()
    view.preview.configure_photo((6000,4000))
    view.preview.display(np.zeros((1000,1500,3), np.uint8))
    view.preview.actual_size()
    view.preview.pan(60,40)
    yield view
    view._render_timer.stop()
    view._stop_detail()
    view._close_approved = True
    view.close()
    view.deleteLater()
    application.processEvents()


@pytest.mark.parametrize("action", ["exposure", "contrast", "saturation", "curve", "wb", "reset", "undo", "redo"])
def test_edit_returns_to_edited_without_moving_view(window, action):
    window.change_exposure(40)
    if action == "redo":
        window.undo()
    window.change_compare(True)
    center, zoom = window.preview.center, window.preview.zoom
    start = len(window.store.events)
    if action == "exposure": window.change_exposure(60)
    elif action in ("contrast", "saturation"): window.change_parameter(action, .2)
    elif action == "curve": window.change_parameter("rgb_curve", ((0.,0.),(.5,.6),(1.,1.)))
    elif action == "wb":
        window.wb_mode.setCurrentIndex(1)
        window.gains[0].setValue(1.1)
        window.change_wb()
    elif action == "reset": window.reset_edits()
    elif action == "undo": window.undo()
    else: window.redo()
    assert not window.store.current.compare_baseline
    assert window.edited_button.isChecked() and not window.compare.isChecked()
    assert window.store.events[start]["type"] == "compare"
    assert window.preview.center == center and window.preview.zoom == zoom
    assert window.view_notice.text() == "Showing edited photo"
    assert window.api_budget.attempts == 0


def test_drag_switch_precedes_one_undo_transaction(window):
    window.change_compare(True)
    window.exposure_slider.sliderPressed.emit()
    for value in range(10, 101, 10): window.exposure_slider.setValue(value)
    window.exposure_slider.sliderReleased.emit()
    assert len(window.store.undo_stack) == 1
    assert window.store.current.exposure_ev == 1

    assert not window.store.current.compare_baseline
    window.undo()
    assert window.store.current.exposure_ev == 0
    window.redo()
    assert window.store.current.exposure_ev == 1


@pytest.mark.parametrize("size", [(1280, 800), (1920, 1080)])
@pytest.mark.parametrize("fit", [False, True])
def test_histogram_updates_do_not_move_slider_or_blink_plot(window, size, fit):
    from photocoach.histogram import sampled_histogram
    application = QApplication.instance()
    window.resize(*size)
    if fit:
        window.preview.fit()
    data = sampled_histogram(np.full((12, 18, 3), .18, np.float32), window.store.current.values)
    window.histogram.display(data)
    for _ in range(4): application.processEvents()
    slider = window.adjustment_sliders["shadows"]
    window.edit_scroll.ensureWidgetVisible(slider)
    for _ in range(4): application.processEvents()
    origin = slider.mapTo(window, slider.rect().topLeft())
    height = window.histogram.height()
    view = (window.preview.center, window.preview.zoom, window.preview.size())
    window._begin_parameter_gesture("shadows")
    for value in range(10, 101, 10):
        slider.setValue(value)
        for _ in range(4): application.processEvents()
        assert slider.mapTo(window, slider.rect().topLeft()) == origin
        assert window.histogram.height() == height
        assert window.histogram.plot.data is data
        assert "Updating" in window.histogram.title.text()
        assert (window.preview.center, window.preview.zoom, window.preview.size()) == view
        window.histogram.display(data)
        for _ in range(4): application.processEvents()
        assert slider.mapTo(window, slider.rect().topLeft()) == origin
    window._finish_gesture()
    assert len(window.store.undo_stack) == 1
    assert window.api_budget.attempts == 0
    window.histogram.clear("Opening photo…")
    assert window.histogram.plot.data is None
    for _ in range(4): application.processEvents()
    assert window.histogram.height() == height
    window.histogram.display(data)
    window.histogram.mark_updating()
    window.histogram.clear("Unavailable")
    for _ in range(4): application.processEvents()
    assert window.histogram.height() == height and window.histogram.plot.data is None


def test_drafts_comparison_navigation_and_noop_history_do_not_edit_or_call_ai(window):
    window.change_compare(True)
    params = window.store.current.values
    window.undo()
    window.redo()
    assert window.store.current.compare_baseline
    window.goal.setText("Keep natural light")
    window.change_goal()
    window.question.setText("How should I adjust RGB?")
    window.wb_mode.setCurrentIndex(1)
    window.gains[0].setValue(1.3)
    window.preview.fit()
    window.preview.actual_size()
    window.coach_collapse.click()
    window.coach_collapse.click()
    assert window.store.current.compare_baseline
    assert window.store.current.values == replace(params, goal="Keep natural light")
    assert window.question.text() == "How should I adjust RGB?"
    assert window.goal.text() == "Keep natural light"
    assert window.coach.calls == window.api_budget.attempts == 0


def test_toolbar_states_and_diagnostics(window):
    window.preview.set_quality("Loading detail")
    assert window.detail_busy.isVisible() and window.detail_busy.maximum() == 0
    window.preview.set_quality("Detail unavailable")
    assert window.retry_detail_button.isVisible() and not window.detail_busy.isVisible()
    window.preview.set_quality("Full detail")
    assert not window.retry_detail_button.isVisible()
    window.change_exposure(50)
    assert window.unsaved_label.isVisible()
    window.store.mark_saved(window.store.current.revision)
    window._refresh_history()
    assert not window.unsaved_label.isVisible()
    assert [a.text() for a in window.open_button.menu().actions()] == ["Open ARW…", "Open project…"]
    assert "Save diagnostics…" in [a.text() for a in window.more_button.menu().actions()]
    assert not window.version_label.isVisible()


def test_responsive_toolbar_keeps_actions_visible(window):
    application = QApplication.instance()
    for width in (600, 950, 600):
        window.photo_toolbar.setFixedWidth(width)
        application.processEvents()
        if width == 600:
            assert window.photo_toolbar.two_rows
        for part in window.photo_toolbar.parts:
            assert window.photo_toolbar.rect().contains(part.geometry())
        a,b,c = [part.geometry() for part in window.photo_toolbar.parts]
        assert not a.intersects(b) and not a.intersects(c) and not b.intersects(c)
