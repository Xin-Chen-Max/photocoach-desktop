"""Synthetic headless integration checks; these do not validate GUI appearance."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtGui import QIcon, QImage, QPalette
from PySide6.QtWidgets import QApplication

from photocoach import app as desktop
from photocoach.imaging import RawOptions, render_srgb


def spin_until(application: QApplication, condition, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        application.processEvents()
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for a headless Qt worker.")
        time.sleep(0.002)
    application.processEvents()


@pytest.fixture(scope="module")
def application():
    instance = QApplication.instance() or QApplication([])
    yield instance


@pytest.fixture
def window(application, monkeypatch):
    view = desktop.MainWindow()
    # No actual RAW, filesystem output, or visible desktop interaction is used.
    state = view.store.activate(Path("C:/photocoach-tests/synthetic-a.ARW"), "photo-a")
    view.store.mark_preview(state)
    view._set_controls(True)
    view._show_metadata({"visible_size": [6, 4], "photo_id": "photo-a"})
    # Final gesture settling now uses formal detail, including at Fit. Keep this
    # synthetic integration fixture offline on that path as well as quick rendering.
    monkeypatch.setattr(view.engine, "detail", lambda state, request: render_result(state))
    yield view
    view._render_timer.stop()
    spin_until(application, lambda: view._active is None and view._pending is None)
    view.close()
    view.deleteLater()
    application.processEvents()


def render_result(state):
    return (
        np.full((4, 6, 3), state.revision, dtype=np.uint8),
        0.001,
        {"visible_size": [6, 4], "photo_id": state.photo_id},
    )


def test_visual_identity_assets_and_main_window_surface(application, tmp_path):
    icon_path = Path(desktop.__file__).with_name("assets") / "PhotoCoach.ico"
    assert icon_path.is_file()
    icon = QIcon(str(icon_path))
    assert not icon.isNull()
    assert icon.availableSizes()
    assert not QImage(str(icon_path.with_suffix(".png"))).isNull()
    for filename in ("chevron-down.svg", "chevron-up.svg", "check.svg"):
        indicator = QIcon(str(icon_path.with_name(filename)))
        assert not indicator.pixmap(14, 14).isNull()
    previous = (application.styleSheet(), application.style().objectName(),
                QPalette(application.palette()), QIcon(application.windowIcon()))
    view = None
    try:
        desktop.configure_appearance(application)
        view = desktop.MainWindow(settings_store=desktop.SettingsStore(tmp_path / "unused-settings.json"))
        view.show()
        application.processEvents()
        assert not view.windowIcon().isNull()
        assert view.windowIcon().availableSizes() == icon.availableSizes()
        assert view.centralWidget().objectName() == "appBody"
        assert view.findChild(type(view.preview), "photoCanvas") is view.preview
        assert view.findChild(type(view.workspace_splitter), "workspaceSplitter") is view.workspace_splitter
        view.open_button.setFocus()
        assert view.open_button.hasFocus()
        assert view.coach_collapse.text() == "Open coach"
        view.resize(1280, 800)
        view.coach_collapse.click()
        view.goal.setEnabled(True)
        view.question.setEnabled(True)
        view.goal.setText("Natural light")
        view.question.setText("How should I adjust the shared RGB curve?")
        view.question.setFocus()
        application.processEvents()
        height = view.coach_panel.height()
        for field in (view.goal, view.question):
            assert field.height() >= field.sizeHint().height()
        view.coach_text.setPlainText("A long answer. " * 300)
        application.processEvents()
        assert view.question.hasFocus() and view.coach_panel.height() == height
        view.coach_collapse.click()
        view.coach_collapse.click()
        assert view.goal.text() == "Natural light"
        assert view.question.text() == "How should I adjust the shared RGB curve?"
        scroll = view.coach_composer_scroll
        scroll.ensureWidgetVisible(view.coach_usage)
        application.processEvents()
        assert scroll.viewport().rect().contains(view.coach_usage.mapTo(
            scroll.viewport(), view.coach_usage.rect().center()))
        assert view.api_budget.attempts == 0
    finally:
        if view is not None:
            view.close()
            view.deleteLater()
        application.setStyleSheet(previous[0])
        application.setStyle(previous[1])
        application.setPalette(previous[2])
        application.setWindowIcon(previous[3])
        application.processEvents()


@pytest.mark.parametrize("name", ["highlights", "shadows"])
def test_zone_slider_baseline_gesture_history_and_histogram_invalidation(window, monkeypatch, name):
    monkeypatch.setattr(window, "request_preview", lambda: None)
    window.preview.configure_photo((6000,4000))
    window.preview.display(np.zeros((1000,1500,3),np.uint8))
    window.preview.actual_size()
    window.preview.pan(20,40)
    geometry = (window.preview.center, window.preview.zoom)
    window.store.set_compare(True)
    slider = window.adjustment_sliders[name]
    slider.sliderPressed.emit()
    slider.setValue(10)
    slider.setValue(40)
    slider.sliderReleased.emit()
    assert not window.store.current.compare_baseline
    assert len(window.store.undo_stack)==1 and getattr(window.store.current,name)==.4
    assert geometry == (window.preview.center,window.preview.zoom)
    assert window.histogram.plot.data is None and "Updating" in window.histogram.title.text()
    window.undo()
    assert getattr(window.store.current,name)==0
    window.redo()
    assert getattr(window.store.current,name)==.4
    assert window.api_budget.attempts==window.coach.calls==0


def test_histogram_samples_full_photo_once_per_render_state_not_per_pan(monkeypatch):
    from photocoach.imaging import DecodedPhoto
    from photocoach.preview import DetailRequest
    from photocoach.histogram import sampled_histogram
    from photocoach.state import StateStore
    engine = desktop.PreviewEngine()
    store = StateStore()
    state = store.activate(Path("synthetic.ARW"),"synthetic")
    pixels = np.full((80,120,3),.18,np.float32)
    pixels[:,90:] = 1
    pixels.setflags(write=False)
    photo = DecodedPhoto(pixels,{"sha256":"synthetic"},RawOptions(),False,state.source_path)
    monkeypatch.setattr(engine.cache,"get",lambda *a,**k:photo)
    calls=[]
    def histogram(*args):
        calls.append(True)
        return sampled_histogram(*args)
    monkeypatch.setattr(desktop,"sampled_histogram",histogram)
    first = engine.detail(state,DetailRequest(1,(120,80),(0,0,50,60)))[2]["display_histogram"]
    second = engine.detail(state,DetailRequest(2,(120,80),(50,0,100,60)))[2]["display_histogram"]
    assert first is second and first["white_percent"]==25 and len(calls)==1
    store.set_parameter("highlights",-1)
    third = engine.detail(store.current,DetailRequest(3,(120,80),(0,0,50,60)))[2]["display_histogram"]
    assert third["white_percent"] < 25 and len(calls)==2


def test_session_confirmation_uses_fresh_requests_and_preserves_limits(window, monkeypatch):
    from photocoach.api_coach import ApiConfig, ApiError
    from photocoach.settings import ApiSettings
    from photocoach.coach import build_request
    window.api_settings = ApiSettings(ApiConfig(model="test-model"), {}, "dummy")
    window.coach_mode.setCurrentIndex(1)
    window.api_budget.configure(call_limit=3)
    assert window._review_settings is None
    reviews, sent = [], []

    class Review:
        include_before = SimpleNamespace(isChecked=lambda: False)
        skip_session = SimpleNamespace(isChecked=lambda: True)
        def __init__(self, *args):
            reviews.append(True)
        def exec(self):
            return desktop.QDialog.DialogCode.Accepted
        def deleteLater(self):
            pass

    class Adapter:
        def __init__(self, *args):
            pass
        def analyze(self, request, **kwargs):
            sent.append((request, kwargs["include_before"]))
            raise ApiError("unavailable")

    monkeypatch.setattr(desktop, "SendReviewDialog", Review)
    monkeypatch.setattr(desktop, "ApiCoach", Adapter)
    # Run the fake worker synchronously; it never makes a network connection.
    monkeypatch.setattr(window.network_pool, "start", lambda worker: worker.run())
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    def send():
        state = window.store.current
        request = build_request(state, state, window.store.events, image, image, window.question.text())
        job = desktop.Job("api_prepare", window._generation, lambda: None, state, window._coach_generation)
        window.review_api_request(request, job)

    send()
    assert len(reviews) == len(sent) == 1
    assert window.analyze_button.text() == "Send to coach"
    window.store.set_exposure(.5)
    window.question.setText("What about this new exposure?")
    send()
    assert len(reviews) == 1 and len(sent) == 2
    assert sent[-1][0].state_version == window.store.current.revision
    assert sent[-1][0].question == window.question.text()
    assert sent[-1][0].parameters["exposure_ev"] == .5
    assert all(not include for _, include in sent)
    window.set_coach_paused(True)
    send()
    assert len(sent) == 2
    window.set_coach_paused(False)
    window.restore_send_confirmation()
    send()
    assert len(reviews) == 2 and len(sent) == 3
    send()
    assert len(sent) == 3 and window.api_budget.attempts == 3


def test_compact_review_defaults_and_details_preserve_preview_selection(application):
    from photocoach.api_coach import ApiConfig
    from photocoach.budget import SessionBudget
    from photocoach.coach import build_request
    from photocoach.coach_ui import SendReviewDialog
    from photocoach.state import StateStore
    from PySide6.QtWidgets import QDialogButtonBox
    state = StateStore().activate(Path("synthetic.ARW"), "synthetic")
    pixels = np.zeros((60, 90, 3), np.uint8)
    request = build_request(state, state, [], pixels, pixels, "How should I edit this?")
    dialog = SendReviewDialog(None, request, ApiConfig(model="test-model"), SessionBudget())
    dialog.resize(580, 380)
    dialog.show()
    application.processEvents()
    assert dialog.include_before.isChecked() and not dialog.skip_session.isChecked()
    assert dialog.details.isHidden()
    dialog.details_button.click()
    dialog.include_before.setChecked(False)
    application.processEvents()
    assert not dialog.details.isHidden() and dialog.before_label.isHidden()
    assert "How should I edit this?" in dialog.context.toPlainText()
    buttons = dialog.findChild(QDialogButtonBox)
    assert dialog.rect().contains(buttons.geometry())
    assert dialog.body_scroll.verticalScrollBar().maximum() > 0
    dialog.reject()
    dialog.deleteLater()


def test_open_clears_previous_detail_error_and_opening_messages(window, application, monkeypatch):
    window.preview.set_quality("Detail unavailable")
    window.detail_label.setToolTip("Old photo failed")
    path = Path("new-photo.ARW")
    photo = SimpleNamespace(source_path=path, metadata={"photo_id":"new", "visible_size":[6000,4000],
        "output_size":[3000,2000], "half_size":True, "orientation_flip":0})
    monkeypatch.setattr(window.engine, "load", lambda _: (photo, np.zeros((1000,1500,3), np.uint8)))
    window.open_raw(path)
    assert window.preview.quality == "Opening photo…"
    assert window.detail_label.toolTip() == ""
    assert window.retry_detail_button.isHidden()
    spin_until(application, lambda: window._active is None)
    assert window.status.text() == ""
    assert "Photo ready" in window.coach_text.toPlainText()
    assert "opening" not in window.coach_summary.toolTip().lower()
    assert window.store.current.photo_id == "new" and window.preview.fit_mode


def test_rapid_edits_discard_active_stale_result_and_replace_pending_job(window, application, monkeypatch):
    started = threading.Event()
    release = threading.Event()
    calls = []
    displayed = []
    render_threads = []
    display_threads = []
    gui_thread = threading.get_ident()

    def render(state):
        calls.append(state.revision)
        render_threads.append(threading.get_ident())
        if len(calls) == 1:
            started.set()
            if not release.wait(5):
                raise TimeoutError("Synthetic worker was not released.")
        return render_result(state)

    def display(pixels):
        displayed.append(int(pixels[0, 0, 0]))
        display_threads.append(threading.get_ident())

    monkeypatch.setattr(window.engine, "render", render)
    monkeypatch.setattr(window.preview, "display", display)
    try:
        window.store.set_exposure(0.1)
        window.request_preview()
        spin_until(application, started.is_set)
        window.store.set_exposure(0.2)
        window.request_preview()
        window.store.set_exposure(0.3)
        window.request_preview()
    finally:
        release.set()
    spin_until(application, lambda: window._active is None and window._pending is None)
    assert calls == [1, 3], "The intermediate pending edit must not become another expensive job."
    assert displayed == [3], "The late result for revision 1 must never be displayed."
    assert all(thread != gui_thread for thread in render_threads)
    assert display_threads == [gui_thread], "Worker completion must update widgets on the GUI thread."
    assert window.store.preview_revision == window.store.current.revision == 3


def test_switching_photo_discards_previous_photos_inflight_preview(window, application, monkeypatch):
    started = threading.Event()
    release = threading.Event()
    displayed = []
    second_path = Path("C:/photocoach-tests/synthetic-b.ARW")

    def old_render(state):
        started.set()
        if not release.wait(5):
            raise TimeoutError("Synthetic worker was not released.")
        return render_result(state)

    def new_load(path):
        photo = SimpleNamespace(
            source_path=path,
            metadata={"photo_id": "photo-b", "visible_size": [6, 4]},
        )
        return photo, np.full((4, 6, 3), 42, dtype=np.uint8)

    monkeypatch.setattr(window.engine, "render", old_render)
    monkeypatch.setattr(window.engine, "load", new_load)
    monkeypatch.setattr(window.preview, "display", lambda pixels: displayed.append(int(pixels[0, 0, 0])))
    try:
        window.store.set_exposure(0.1)
        window.request_preview()
        spin_until(application, started.is_set)
        window.open_raw(second_path)
    finally:
        release.set()
    spin_until(application, lambda: window._active is None and window._pending is None)
    assert displayed == [42]
    assert window.store.current.photo_id == "photo-b"
    assert window.store.current.source_path == second_path
    assert window.store.preview_revision == 0
    assert window._metadata_binding["photo_id"] == "photo-b"


@pytest.mark.parametrize("export_fails", [False, True])
def test_export_uses_latest_parameters_and_restores_dropped_preview(window, application, monkeypatch, export_fails):
    """Regression: freezing output used to lose the queued current preview."""
    started = threading.Event()
    release = threading.Event()
    renders = []
    decodes = []
    exports = []
    displayed = []
    linear = np.full((8, 12, 3), 0.08, dtype=np.float32)
    destination = Path("C:/photocoach-tests/synthetic-never-written.tif")

    def slow_first_render(state):
        renders.append(state.revision)
        if len(renders) == 1:
            started.set()
            if not release.wait(5):
                raise TimeoutError("Synthetic worker was not released.")
        return render_result(state)

    def full_decode(path, options, *, half_size):
        decodes.append((path, options, half_size))
        return SimpleNamespace(linear_rgb=linear, metadata={"sha256": "photo-a"})

    def export(path, pixels, *, source_path):
        exports.append((path, pixels.copy(), source_path))
        if export_fails:
            raise OSError("Synthetic output failure")
        return path

    monkeypatch.setattr(window.engine, "render", slow_first_render)
    monkeypatch.setattr(window.preview, "display", lambda pixels: displayed.append(int(pixels[0, 0, 0])))
    monkeypatch.setattr(desktop.QFileDialog, "getSaveFileName", lambda *args: (str(destination), "16-bit RGB TIFF (*.tif)"))
    monkeypatch.setattr(desktop, "validate_destination", lambda path, **kwargs: path)
    monkeypatch.setattr(desktop, "decode_raw", full_decode)
    monkeypatch.setattr(desktop, "sha256_file", lambda path: "photo-a")
    monkeypatch.setattr(desktop, "export_image", export)
    try:
        window.store.set_exposure(0.1)
        window.request_preview()
        spin_until(application, started.is_set)
        window.store.set_exposure(0.75)
        options = RawOptions("custom", (1.1, 1.0, 0.95))
        window.store.set_whitebalance(options)
        window.store.set_compare(True)
        latest = window.store.current
        window.request_preview()
        window.choose_export()
        assert window._exporting
        assert not window.exposure_slider.isEnabled()
    finally:
        release.set()
    spin_until(application, lambda: window._active is None and window._pending is None)
    assert decodes == [(latest.source_path, options, False)]
    assert len(exports) == 1
    np.testing.assert_array_equal(exports[0][1], render_srgb(linear, 0.75))
    assert exports[0][0] == destination
    assert displayed == [latest.revision]
    assert renders == [1, latest.revision]
    assert window.store.preview_revision == latest.revision
    assert not window._exporting
    assert window.exposure_slider.isEnabled()
    assert window.open_button.isEnabled()
    assert not window.output_status.isHidden()
    if export_fails:
        assert "Synthetic output failure" in window.output_status.text(), "A successful preview must not hide an export error."
    else:
        assert str(destination) in window.output_status.text()


@pytest.mark.parametrize("action", ["edit", "switch", "pause", "failure", "malformed"])
def test_coach_late_result_and_failure_never_block_editing(window, application, monkeypatch, action):
    from photocoach.coach import build_request, MockCoach
    window.coach_mode.setCurrentIndex(0)
    started, release = threading.Event(), threading.Event()
    image = np.zeros((4, 6, 3), dtype=np.uint8)

    def request(state, before, events, question):
        return build_request(state, before, events, image, image, question)

    def analyze(request):
        started.set()
        if not release.wait(5):
            raise TimeoutError("Test worker timeout")
        if action == "failure":
            raise RuntimeError("Synthetic adapter unavailable")
        if action == "malformed":
            return {"unexpected": "response"}
        return MockCoach().analyze(request)

    monkeypatch.setattr(window.engine, "coach_request", request)
    monkeypatch.setattr(window.coach, "analyze", analyze)
    monkeypatch.setattr(window.engine, "render", render_result)
    try:
        window.analyze()
        spin_until(application, started.is_set)
        assert window.exposure_slider.isEnabled()
        if action == "edit":
            window.exposure_slider.setValue(50)
        elif action == "pause":
            window.set_coach_paused(True)
        elif action == "switch":
            monkeypatch.setattr(window.engine, "load", lambda path: (
                SimpleNamespace(source_path=path, metadata={"photo_id": "photo-b", "visible_size": [6, 4]}), image))
            window.open_raw(Path("C:/photocoach-tests/synthetic-b.ARW"))
    finally:
        release.set()
    spin_until(application, lambda: window._active is None and window._pending is None and not window._render_timer.isActive())
    assert "MOCK · Local rules" not in window.coach_text.toPlainText()
    if action == "failure":
        assert "Synthetic adapter unavailable" in window.coach_text.toPlainText()
    if action == "malformed":
        assert "invalid feedback" in window.coach_text.toPlainText()
    assert window.exposure_slider.isEnabled()
    window.exposure_slider.setValue(75)
    spin_until(application, lambda: window.store.preview_revision == window.store.current.revision)


def test_curve_hover_readout_is_passive_and_drag_uses_actual_point(application):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtTest import QTest
    from photocoach.curve import CurveEditor
    curve = CurveEditor()
    curve.resize(300, 180)
    curve.set_points(((0., 0.), (.4, .6), (1., 1.)))
    changes = []
    curve.pointsChanged.connect(changes.append)
    curve.show()
    QTest.mouseMove(curve, curve._screen((.4, .6)).toPoint())
    assert curve.readout.text() == "Point · Input 0.400 / Output 0.600"
    assert curve.readout.width() <= curve.width() - 12
    QTest.mouseMove(curve, curve._screen((.5, .2)).toPoint())
    assert curve.readout.text().startswith("Cursor · Input 0.500 / Output")
    assert not changes
    QTest.mousePress(curve, Qt.MouseButton.LeftButton, pos=curve._screen((0., 0.)).toPoint())
    QTest.mouseMove(curve, curve._screen((.3, .5)).toPoint())
    assert curve.readout.text() == "Point · Input 0.000 / Output 0.500"
    QTest.mouseRelease(curve, Qt.MouseButton.LeftButton)
    QApplication.sendEvent(curve, QEvent(QEvent.Type.Leave))
    assert curve.readout.isHidden()
    curve.close()


def test_curve_widget_add_drag_delete_and_history(window, application, monkeypatch):
    from PySide6.QtCore import Qt, QPoint
    from PySide6.QtTest import QTest
    monkeypatch.setattr(window.engine, "render", render_result)
    curve = window.curve
    curve.resize(300, 180)
    middle = curve._screen((.5, .5)).toPoint()
    QTest.mouseDClick(curve, Qt.MouseButton.LeftButton, pos=middle)
    assert len(window.store.current.rgb_curve) == 3
    assert len(window.store.undo_stack) == 1
    added = window.store.current.rgb_curve
    QTest.mousePress(curve, Qt.MouseButton.LeftButton, pos=middle)
    QTest.mouseMove(curve, middle + QPoint(10, -20))
    QTest.mouseMove(curve, middle + QPoint(15, -25))
    QTest.mouseRelease(curve, Qt.MouseButton.LeftButton, pos=middle + QPoint(15, -25))
    assert len(window.store.undo_stack) == 2
    window.undo()
    assert window.store.current.rgb_curve == added
    window.redo()
    QTest.keyClick(curve, Qt.Key.Key_Delete)
    assert len(window.store.current.rgb_curve) == 2
    assert len(window.store.undo_stack) == 3
    window.undo()
    assert len(window.store.current.rgb_curve) == 3


@pytest.mark.parametrize("review_outcome", ["cancel", "changed", "question_changed", "missing_key"])
def test_api_review_never_sends_without_current_confirmed_context(window, application, monkeypatch, review_outcome):
    from photocoach.api_coach import ApiConfig
    from photocoach.coach_ui import ApiSettings
    from photocoach.coach import build_request
    window.api_settings = ApiSettings(ApiConfig(model="test-model"), {"call_limit": 5},
                                     "" if review_outcome == "missing_key" else "dummy")
    window.coach_mode.setCurrentIndex(1)
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    state = window.store.current
    request = build_request(state, state, window.store.events, image, image)
    job = desktop.Job("api_prepare", window._generation, lambda: None, state, window._coach_generation)
    monkeypatch.delenv("PHOTOCOACH_API_KEY", raising=False)
    monkeypatch.setattr(window.engine, "render", render_result)
    class Review:
        include_before = SimpleNamespace(isChecked=lambda: True)
        skip_session = SimpleNamespace(isChecked=lambda: True)
        def __init__(self, *args):
            pass
        def exec(self):
            if review_outcome == "changed":
                window.exposure_slider.setValue(30)
            elif review_outcome == "question_changed":
                window.question.setText("A new question after the review was prepared")
            return desktop.QDialog.DialogCode.Rejected if review_outcome == "cancel" else desktop.QDialog.DialogCode.Accepted
        def deleteLater(self):
            pass
    monkeypatch.setattr(desktop, "SendReviewDialog", Review)
    window.review_api_request(request, job)
    assert window._network_active is None and window.api_budget.attempts == 0
    if review_outcome != "missing_key":
        assert window._review_settings is None


@pytest.mark.parametrize("outcome", ["success", "edit", "question", "switch", "pause", "failure", "timeout_connecting", "timeout_waiting"])
def test_api_uses_separate_worker_and_stale_results_are_discarded(window, application, monkeypatch, outcome):
    from photocoach.api_coach import ApiConfig, ApiCoach, ApiError, FEEDBACK_FIELDS
    from photocoach.coach_ui import ApiSettings
    from photocoach.coach import build_request
    import json
    window.api_settings = ApiSettings(ApiConfig(model="test-model"), {"call_limit": 5}, "dummy-test-key")
    window.coach_mode.setCurrentIndex(1)
    started, release = threading.Event(), threading.Event()
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    monkeypatch.setattr(window.engine, "render", render_result)
    monkeypatch.setattr(window.engine, "coach_request", lambda state, before, events, question:
                        build_request(state, before, events, image, image, question))
    class Review:
        include_before = SimpleNamespace(isChecked=lambda: False)
        skip_session = SimpleNamespace(isChecked=lambda: False)
        def __init__(self, *args):
            pass
        def exec(self):
            return desktop.QDialog.DialogCode.Accepted
        def deleteLater(self):
            pass
    def transport(config, key, payload, cancel):
        assert threading.get_ident() != gui_thread
        assert "dummy-test-key" not in payload.decode()
        started.set()
        if not release.wait(5):
            raise TimeoutError("test")
        if outcome == "failure":
            raise OSError("never-display-secret-key")
        if outcome.startswith("timeout_"):
            raise ApiError("timeout", transport_stage=outcome.removeprefix("timeout_"),
                           elapsed_seconds=30, timeout_seconds=30)
        fields = {name: "Test photographic feedback." for name in FEEDBACK_FIELDS}
        result = {"status": "completed", "output": [{"type": "message", "content": [
            {"type": "output_text", "text": json.dumps(fields)}]}],
            "usage": {"input_tokens": 100, "output_tokens": 80}}
        return 200, json.dumps(result).encode()
    gui_thread = threading.get_ident()
    monkeypatch.setattr(desktop, "SendReviewDialog", Review)
    monkeypatch.setattr(desktop, "ApiCoach", lambda config, key: ApiCoach(config, key, transport))
    try:
        window.analyze()
        spin_until(application, started.is_set)
        assert not window.analyze_button.isEnabled()
        assert window.exposure_slider.isEnabled()
        assert window.api_budget.attempts == 1
        window._network_progress[0].advance("waiting")
        window._update_network_progress()
        assert "Waiting for response headers" in window.coach_text.toPlainText()
        assert "30s total connection limit" in window.coach_text.toPlainText()
        if outcome == "edit":
            window.exposure_slider.setValue(40)
            # This completes while the network worker is still blocked.
            spin_until(application, lambda: window.store.preview_revision == window.store.current.revision)
            assert window._network_active is not None
        elif outcome == "question":
            original_state = window.store.current
            window.question.setText("Now answer a different question")
            assert window.store.current is original_state
            assert window._network_active[2].is_set()
        elif outcome == "pause":
            window.set_coach_paused(True)
        elif outcome == "switch":
            monkeypatch.setattr(window.engine, "load", lambda path: (
                SimpleNamespace(source_path=path, metadata={"photo_id": "photo-b", "visible_size": [6, 4]}), image))
            window.open_raw(Path("C:/photocoach-tests/synthetic-b.ARW"))
            spin_until(application, lambda: window.store.current.photo_id == "photo-b")
            assert window._network_active is not None
        if outcome in ("edit", "question", "pause", "switch"):
            text_before_tick = window.coach_text.toPlainText()
            window._update_network_progress()
            assert window.coach_text.toPlainText() == text_before_tick
    finally:
        release.set()
    spin_until(application, lambda: window._network_active is None and window._active is None and window._pending is None)
    text = window.coach_text.toPlainText()
    assert ("AI · test-model" in text) == (outcome == "success")
    assert "never-display-secret-key" not in text
    assert window.api_budget.attempts == 1 and window.api_budget.in_flight == 0
    assert not window._network_timer.isActive() and window._network_progress is None
    assert window.exposure_slider.isEnabled()
    if outcome == "failure":
        assert "Editing remains available" in text
    if outcome == "timeout_connecting":
        assert "No analysis request was sent" in text and "charges may still apply" not in text
    if outcome == "timeout_waiting":
        assert "upload completed locally" in text and "charges are unknown" in text


def test_gemini_settings_preset_does_not_send_and_validates_model(application):
    from photocoach.coach_ui import ApiSettingsDialog
    from photocoach.budget import SessionBudget
    from photocoach.api_coach import GEMINI_ENDPOINT
    budget = SessionBudget(call_limit=1)
    dialog = ApiSettingsDialog(None, None, budget)
    dialog.protocol.setCurrentIndex(dialog.protocol.findData("gemini"))
    assert dialog.endpoint.text() == GEMINI_ENDPOINT
    assert dialog.environment.text() == "GEMINI_API_KEY"
    assert dialog.timeout.value() == 60
    dialog.model.setText("gemini-test-model")
    dialog.validate_and_accept()
    assert dialog.result_settings.config.protocol == "gemini"
    assert budget.attempts == 0
    dialog.deleteLater()


@pytest.mark.parametrize("outcome", ["success", "failure", "close"])
def test_gemini_metadata_check_is_explicit_async_and_separate_from_analysis_budget(application, monkeypatch, outcome):
    from photocoach import coach_ui
    from photocoach.api_coach import ModelCheckResult, ModelCheckError, ApiError
    from photocoach.budget import SessionBudget
    budget = SessionBudget(call_limit=1)
    ticket = budget.reserve()
    budget.settle(ticket, success=False)
    dialog = coach_ui.ApiSettingsDialog(None, None, budget)
    dialog.protocol.setCurrentIndex(dialog.protocol.findData("gemini"))
    dialog.model.setText("gemini-test-model")
    dialog.key.setText("dummy-test-key")
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    gui_thread = threading.get_ident()
    calls = []

    def check(config, key, cancel):
        assert threading.get_ident() != gui_thread
        assert config.protocol == "gemini" and key == "dummy-test-key"
        calls.append(cancel)
        started.set()
        try:
            assert release.wait(5)
            if outcome == "failure":
                raise ModelCheckError(ApiError("authentication", http_status=403))
            return ModelCheckResult(True)
        finally:
            finished.set()

    monkeypatch.setattr(coach_ui, "check_gemini_model", check)
    assert not calls
    dialog.show()
    try:
        dialog.check_model()
        spin_until(application, started.is_set)
        dialog.check_model()  # No concurrent or queued duplicate.
        assert len(calls) == 1 and not dialog.model.isEnabled() and not dialog.save_button.isEnabled()
        if outcome == "close":
            dialog.reject()
            assert calls[0].is_set()
    finally:
        release.set()
    spin_until(application, lambda: finished.is_set() and dialog._check_worker is None)
    assert budget.attempts == 1 and budget.in_flight == 0 and budget.unresolved == 1
    text = dialog.model_check_status.text()
    assert "dummy-test-key" not in text
    if outcome == "success":
        assert "lookup succeeded" in text and dialog.save_button.isEnabled()
        dialog.model.setText("gemini-different-test")
        assert "Settings changed" in dialog.model_check_status.text()
        assert "lookup succeeded" not in dialog.model_check_status.text()
        assert len(calls) == 1
    elif outcome == "failure":
        assert "HTTP 403" in text and dialog.model_check_button.isEnabled()
    else:
        assert "lookup succeeded" not in text
    dialog.close()
    dialog.deleteLater()


def _configure_text_probe(view):
    from photocoach.api_coach import ApiConfig, GEMINI_ENDPOINT
    from photocoach.coach_ui import ApiSettings
    from photocoach.budget import SessionBudget
    view.api_settings = ApiSettings(ApiConfig(GEMINI_ENDPOINT, "gemini-test-model", "gemini"),
                                    {"call_limit": 1}, "dummy-probe-key")
    view.api_budget = SessionBudget(call_limit=1, input_usd_per_million="1", output_usd_per_million="2")
    view.coach_mode.setCurrentIndex(1)
    view._update_coach_status()


def _probe_review(monkeypatch, action=None, accepted=True):
    class Review:
        def __init__(self, *args):
            pass

        def exec(self):
            if action:
                action()
            return desktop.QDialog.DialogCode.Accepted if accepted else desktop.QDialog.DialogCode.Rejected

        def deleteLater(self):
            pass
    monkeypatch.setattr(desktop, "TextProbeReviewDialog", Review)


@pytest.mark.parametrize("outcome", ["cancel", "changed", "missing_key", "call_cap", "dollar_cap"])
def test_text_probe_review_budget_and_credentials_gate_sending(window, monkeypatch, outcome):
    from photocoach.coach_ui import ApiSettings
    _configure_text_probe(window)
    monkeypatch.setattr(desktop, "run_text_probe", lambda *a, **kw: pytest.fail("Must not send"))
    _probe_review(monkeypatch, action=(lambda: window.set_coach_paused(True)) if outcome == "changed" else None,
                  accepted=outcome != "cancel")
    if outcome == "missing_key":
        window.api_settings = ApiSettings(window.api_settings.config, {"call_limit": 1}, "")
        monkeypatch.delenv("PHOTOCOACH_API_KEY", raising=False)
    elif outcome == "call_cap":
        window.api_budget.configure(call_limit=0)
    elif outcome == "dollar_cap":
        window.api_budget.configure(budget_usd="0.01", reserve_usd="0.02")
    window.test_text_only()
    assert window._network_active is None and window.api_budget.attempts == 0


@pytest.mark.parametrize("outcome", ["ok", "other_text", "output_limit", "failure", "timeout", "edit", "pause", "switch"])
def test_text_probe_uses_shared_budget_worker_and_stale_result_rules(window, application, monkeypatch, outcome):
    from photocoach.text_probe import TextProbeResult
    from photocoach.api_coach import ApiError
    _configure_text_probe(window)
    _probe_review(monkeypatch)
    started, release = threading.Event(), threading.Event()
    calls = []
    gui_thread = threading.get_ident()
    result = TextProbeResult(outcome if outcome in ("ok", "other_text", "output_limit") else "ok", 10, 5)

    def run(config, key, cancel, progress=None):
        assert threading.get_ident() != gui_thread and key == "dummy-probe-key"
        calls.append(cancel)
        progress.advance("waiting")
        started.set()
        assert release.wait(5)
        if outcome == "failure":
            raise RuntimeError("secret-should-never-display")
        if outcome == "timeout":
            raise ApiError("timeout", transport_stage="waiting", elapsed_seconds=30, timeout_seconds=30)
        return result

    monkeypatch.setattr(desktop, "run_text_probe", run)
    monkeypatch.setattr(window.engine, "render", render_result)
    monkeypatch.setattr(window.engine, "coach_request", lambda *a: pytest.fail("No photos should be prepared"))
    try:
        window.test_text_only()
        spin_until(application, started.is_set)
        window._update_network_progress()
        assert "Text-only test" in window.coach_text.toPlainText()
        assert not window.text_probe_button.isEnabled() and not window.analyze_button.isEnabled()
        window.test_text_only()
        assert len(calls) == 1 and window.api_budget.attempts == 1 and window.api_budget.in_flight == 1
        if outcome == "edit":
            window.exposure_slider.setValue(40)
            spin_until(application, lambda: window.store.preview_revision == window.store.current.revision)
            assert window._network_active is not None
        elif outcome == "pause":
            window.set_coach_paused(True)
        elif outcome == "switch":
            image = np.zeros((4, 6, 3), dtype=np.uint8)
            monkeypatch.setattr(window.engine, "load", lambda path: (
                SimpleNamespace(source_path=path, metadata={"photo_id": "photo-b", "visible_size": [6, 4]}), image))
            window.open_raw(Path("C:/photocoach-tests/synthetic-b.ARW"))
            spin_until(application, lambda: window.store.current.photo_id == "photo-b")
        if outcome in ("edit", "pause", "switch"):
            old_text = window.coach_text.toPlainText()
            window._update_network_progress()
            assert window.coach_text.toPlainText() == old_text
    finally:
        release.set()
    spin_until(application, lambda: window._network_active is None)
    assert window.api_budget.attempts == 1 and window.api_budget.in_flight == 0
    assert not window._network_timer.isActive()
    text = window.coach_text.toPlainText()
    assert "secret-should-never-display" not in text
    if outcome in ("ok", "other_text", "output_limit"):
        assert text == result.text() and window.api_budget.known_cost_usd is not None
    elif outcome in ("failure", "timeout"):
        assert "Text-only test failed" in text and window.api_budget.known_cost_usd is None
    else:
        assert text != result.text()
    window.test_text_only()  # The exhausted shared cap must stop another request.
    assert len(calls) == 1


def test_text_probe_needs_no_open_photo_and_discloses_exact_payload(application, monkeypatch):
    import json
    from photocoach.text_probe import TextProbeResult, build_text_probe_payload
    from photocoach.coach_ui import TextProbeReviewDialog
    view = desktop.MainWindow()
    _configure_text_probe(view)
    assert view.store.current is None and view.text_probe_button.isEnabled()
    dialog = TextProbeReviewDialog(view, view.api_settings.config, view.api_budget)
    assert json.loads(dialog.payload.toPlainText()) == build_text_probe_payload(view.api_settings.config)
    assert "dummy-probe-key" not in dialog.payload.toPlainText()
    dialog.reject()
    dialog.deleteLater()
    _probe_review(monkeypatch)
    monkeypatch.setattr(desktop, "run_text_probe", lambda *a, **kw: TextProbeResult("ok", 10, 5))
    view.test_text_only()
    spin_until(application, lambda: view._network_active is None)
    assert view.api_budget.attempts == 1 and view.api_budget.known_cost_usd is not None
    assert view.coach_text.toPlainText() == TextProbeResult("ok", 10, 5).text()
    view.close()
    view.deleteLater()


def test_text_probe_controls_recover_when_test_finishes_before_export(window, application, monkeypatch):
    from photocoach.text_probe import TextProbeResult
    _configure_text_probe(window)
    _probe_review(monkeypatch)
    probe_started, probe_release = threading.Event(), threading.Event()
    export_started, export_release = threading.Event(), threading.Event()

    def probe(*args, **kwargs):
        probe_started.set()
        assert probe_release.wait(5)
        return TextProbeResult("ok", 10, 5)

    def export():
        export_started.set()
        assert export_release.wait(5)
        return Path("synthetic-no-file-written.jpg")

    monkeypatch.setattr(desktop, "run_text_probe", probe)
    try:
        window.test_text_only()
        spin_until(application, probe_started.is_set)
        window._freeze_output("Synthetic export")
        window._queue(desktop.Job("export", window._generation, export, window.store.current))
        spin_until(application, export_started.is_set)
        probe_release.set()
        spin_until(application, lambda: window._network_active is None)
        assert window._exporting and not window.text_probe_button.isEnabled()
    finally:
        probe_release.set()
        export_release.set()
    spin_until(application, lambda: window._active is None)
    assert not window._exporting and window.text_probe_button.isEnabled()
