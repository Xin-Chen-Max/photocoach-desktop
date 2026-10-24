"""Pixel geometry, full-detail rendering and asynchronous view integration."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import threading
import time
from dataclasses import replace
import numpy as np
import pytest
from PySide6.QtCore import QPointF, QPoint, Qt, QEvent
from PySide6.QtGui import QImage, QWheelEvent, QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from photocoach import app as desktop
from photocoach.preview import DetailRequest, render_region, oriented_size
from photocoach.preview_view import ImageView
from photocoach.adjustments import render_edit
from photocoach.imaging import quantize_srgb
from photocoach.state import EditValues


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


def spin(app, predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while not predicate():
        app.processEvents()
        if time.monotonic() > deadline:
            raise AssertionError("Background detail did not finish")
        time.sleep(.002)
    app.processEvents()


@pytest.fixture
def canvas(application):
    view = ImageView()
    view.resize(501, 301)
    view.configure_photo((4000, 3000))
    view.display(np.zeros((300, 400, 3), np.uint8))
    yield view
    view.close()


@pytest.mark.parametrize("dpr", [1, 1.5, 2])
def test_true_100_percent_has_one_source_pixel_per_physical_pixel(canvas, monkeypatch, dpr):
    monkeypatch.setattr(canvas, "devicePixelRatioF", lambda: dpr)
    canvas.actual_size()
    a = canvas.source_at(QPointF(101, 100))
    b = canvas.source_at(QPointF(101 + 1 / dpr, 100))
    assert b[0] - a[0] == pytest.approx(1)
    origin = canvas.width() * dpr / 2 - canvas.center[0]
    assert origin == pytest.approx(round(origin))
    left, top, right, bottom = canvas.detail_request().rect
    assert right - left >= canvas.width() * dpr
    assert bottom - top >= canvas.height() * dpr


@pytest.mark.parametrize("dpr", [1, 1.5, 2])
def test_actual_qt_paint_pixels_match_native_checkerboard(canvas, monkeypatch, dpr):
    monkeypatch.setattr(canvas, "devicePixelRatioF", lambda: dpr)
    canvas.actual_size()
    request = canvas.detail_request()
    l, t, r, b = request.rect
    yy, xx = np.mgrid[t:b, l:r]
    rgb = np.stack([xx % 256, yy % 256, (xx + yy) % 2 * 255], axis=2).astype(np.uint8)
    canvas.display_detail(rgb, request)
    target = QImage(round(canvas.width() * dpr), round(canvas.height() * dpr), QImage.Format.Format_RGB888)
    target.setDevicePixelRatio(dpr)
    canvas.render(target)
    pixels = np.frombuffer(target.constBits(), dtype=np.uint8).reshape(target.height(), target.bytesPerLine())
    pixels = pixels[:, :target.width() * 3].reshape(target.height(), target.width(), 3)
    origin_x = round(canvas.width() * dpr / 2 - canvas.center[0])
    origin_y = round(canvas.height() * dpr / 2 - canvas.center[1])
    np.testing.assert_array_equal(pixels[10:60, 10:60], rgb[10-origin_y-t:60-origin_y-t, 10-origin_x-l:60-origin_x-l])


@pytest.mark.parametrize('dpr',[1,1.5,2])
@pytest.mark.parametrize('zoom',[1.,2.])
def test_temporary_raster_is_smoothed_even_while_refining(canvas,monkeypatch,dpr,zoom):
    monkeypatch.setattr(canvas,'devicePixelRatioF',lambda:dpr)
    canvas.actual_size();canvas.zoom_at(zoom,QPointF(canvas.width()/2,canvas.height()/2))
    request=canvas.detail_request()
    coarse=np.zeros((2,2,3),np.uint8);coarse[:,1]=255
    canvas.display_adjusting(coarse,request)
    key=canvas._detail.cacheKey();canvas.invalidate_edit()
    assert canvas.quality=='Refining' and canvas._detail.cacheKey()==key
    target=QImage(round(canvas.width()*dpr),round(canvas.height()*dpr),QImage.Format.Format_RGB888)
    target.setDevicePixelRatio(dpr);canvas.render(target)
    pixels=np.frombuffer(target.constBits(),np.uint8).reshape(target.height(),target.bytesPerLine())
    rgb=pixels[:,:target.width()*3].reshape(target.height(),target.width(),3)
    assert len(np.unique(rgb[target.height()//2,:,0]))>16


def test_cursor_zoom_anchor_pan_and_fit(canvas):
    canvas.actual_size()
    point = QPointF(121, 112)
    anchor = canvas.source_at(point)
    canvas.zoom_at(1.25, point)
    np.testing.assert_allclose(canvas.source_at(point), anchor)
    center = canvas.center
    canvas.pan(20, -30)
    np.testing.assert_allclose(canvas.center, [center[0] - 16, center[1] + 24])
    canvas.fit()
    assert canvas.center == (2000, 1500)
    assert canvas.fit_mode


def test_events_only_ctrl_wheel_zooms_and_space_drag_pans(canvas, application):
    canvas.show()
    canvas.actual_size()
    application.processEvents()
    def wheel(mods):
        return QWheelEvent(QPointF(100, 100), QPointF(100, 100), QPoint(), QPoint(0, 120),
                           Qt.MouseButton.NoButton, mods, Qt.ScrollPhase.NoScrollPhase, False)
    application.sendEvent(canvas, wheel(Qt.KeyboardModifier.NoModifier))
    assert canvas.zoom == 1
    application.sendEvent(canvas, wheel(Qt.KeyboardModifier.ControlModifier))
    assert canvas.zoom == 1.25
    center = canvas.center
    QTest.keyPress(canvas, Qt.Key.Key_Space)
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(100, 100))
    QTest.mouseMove(canvas, QPoint(140, 110))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(140, 110))
    QTest.keyRelease(canvas, Qt.Key.Key_Space)
    assert canvas.center != center
    assert canvas._drag is None and not canvas._space
    QTest.mouseDClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(200, 150))
    assert canvas.fit_mode
    target = canvas.source_at(QPointF(250, 150))
    QTest.mouseDClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(250, 150))
    assert canvas.zoom == 1 and not canvas.fit_mode
    np.testing.assert_allclose(canvas.center, target, atol=1)


def test_resize_edit_display_and_failed_open_keep_view(canvas, application):
    canvas.show()
    canvas.actual_size()
    canvas.pan(100, 20)
    old = canvas.center
    canvas.resize(601, 401)
    application.processEvents()
    np.testing.assert_allclose(canvas.center, old, atol=1)
    canvas.invalidate_edit()
    canvas.clear_photo()
    canvas.display(np.ones((300, 400, 3), np.uint8))
    assert canvas.zoom == 1 and not canvas.fit_mode
    np.testing.assert_allclose(canvas.center, old, atol=1)
    canvas.configure_photo((3000, 4000))
    assert canvas.fit_mode and canvas.center == (1500, 2000)


@pytest.mark.parametrize("flip,expected", [(0,(7028,4688)), (6,(4688,7028)), (5,(4688,7028))])
def test_oriented_full_geometry(flip, expected):
    assert oriented_size({"visible_size": [7028,4688], "orientation_flip": flip}) == expected


@pytest.mark.parametrize("rect", [(0,0,53,47), (17,11,53,47), (0,0,1,1)])
def test_roi_exactly_matches_full_export_pixels(rect):
    linear = np.random.default_rng(7).random((47,53,3), dtype=np.float32)
    original = linear.copy()
    linear.setflags(write=False)
    values = EditValues(exposure_ev=.7, contrast=.21, saturation=.85,
                        rgb_curve=((0.,0.), (.3,.2), (.7,.82), (1.,1.)))
    roi = render_region(linear, values, DetailRequest(1,(53,47),rect))
    left, top, right, bottom = rect
    np.testing.assert_array_equal(roi, quantize_srgb(render_edit(linear, values))[top:bottom,left:right])
    np.testing.assert_array_equal(linear, original)
    assert not np.shares_memory(roi, linear)


def test_reduced_detail_averages_after_tone_curve():
    linear = np.zeros((128,128,3), np.float32)
    linear[:,1::2] = .8
    pixels = render_region(linear, EditValues(), DetailRequest(1,(128,128),(0,0,128,128),2))
    reference = quantize_srgb(render_edit(linear, EditValues()).reshape(64,2,64,2,3).mean(axis=(1,3)))
    np.testing.assert_array_equal(pixels, reference)


@pytest.fixture
def window(application, monkeypatch):
    window = desktop.MainWindow()
    state = window.store.activate(Path("synthetic.ARW"), "photo-a")
    window.store.mark_preview(state)
    window.preview.configure_photo((4000,3000))
    window.preview.display(np.zeros((1200,1600,3), np.uint8))
    window._set_controls(True)
    metadata = {"visible_size":[4000,3000], "photo_id":"photo-a"}
    monkeypatch.setattr(window.engine, "render", lambda state:
                        (np.full((1200,1600,3), state.revision % 255, np.uint8), .01, metadata))
    monkeypatch.setattr(window.engine, "detail", lambda state, request:
                        (np.full((30,40,3), state.revision % 255, np.uint8), .01, metadata))
    monkeypatch.setattr(window.engine, "adjusting", lambda state, request:
                        (np.full((30,40,3), state.revision % 255, np.uint8), .01, {}))
    yield window
    window._stop_detail()
    window._render_timer.stop()
    spin(application, lambda: window._active is None)
    window._close_approved = True
    window.close()


def settled(window):
    return not (window._active or window._pending or window._pending_detail or
                window._protected_jobs or window._render_timer.isActive() or window._detail_timer.isActive()
                or window._settle_timer.isActive())


def test_ten_edits_undo_compare_keep_zoom_and_latest_detail(window, application):
    window.preview.actual_size()
    window.preview.pan(140, 90)
    center = window.preview.center
    for i in range(10):
        window.change_exposure(10 * (i+1))
    spin(application, lambda: settled(window))
    assert window.preview.quality == "Full detail"
    assert window.store.preview_revision == window.store.current.revision == 10
    assert window.preview.center == center and window.preview.zoom == 1
    window.undo()
    spin(application, lambda: settled(window))
    window.redo()
    spin(application, lambda: settled(window))
    window.change_compare(True)
    spin(application, lambda: settled(window))
    assert window.preview.center == center and window.preview.zoom == 1
    assert window.store.current.exposure_ev == 1


@pytest.mark.parametrize("change", ["view", "edit", "photo"])
def test_inflight_detail_never_overwrites_newer_binding(window, application, monkeypatch, change):
    started, release = threading.Event(), threading.Event()
    def slow(state, request):
        started.set()
        assert release.wait(5)
        return np.full((30,40,3), 255, np.uint8), .1, {"visible_size":[4000,3000]}
    monkeypatch.setattr(window.engine, "detail", slow)
    window.preview.actual_size()
    window._detail_timer.stop()
    window.request_detail()
    spin(application, started.is_set)
    try:
        if change == "view":
            window.preview.pan(30, 50)
        elif change == "edit":
            window.change_exposure(30)
        else:
            window._generation += 1
            window.store.activate(Path("second.ARW"), "photo-b")
            window.preview.configure_photo((3000,4000))
        window._stop_detail()
        window._render_timer.stop()
    finally:
        release.set()
    spin(application, lambda: not window._active)
    assert window.preview._detail.isNull()


def test_preview_requests_do_not_replace_protected_job(window, application, monkeypatch):
    started, release = threading.Event(), threading.Event()
    calls = []
    def active():
        started.set()
        assert release.wait(5)
    window._queue(desktop.Job("test", window._generation, active))
    spin(application, started.is_set)
    try:
        window._queue(desktop.Job("test-save", window._generation, lambda: calls.append("save")))
        for i in range(10):
            window.change_exposure(i)
            window.request_preview()
    finally:
        release.set()
    spin(application, lambda: settled(window))
    assert calls == ["save"]
    assert window.store.preview_revision == window.store.current.revision


def test_detail_failure_does_not_retry_until_user_action(window, application, monkeypatch):
    calls = []
    def fail(*args):
        calls.append(1)
        raise MemoryError("budget")
    monkeypatch.setattr(window.engine, "detail", fail)
    window.preview.actual_size()
    spin(application, lambda: settled(window))
    assert calls == [1] and window.preview.quality == "Detail unavailable"
    assert window.exposure_slider.isEnabled()
    assert window.retry_detail_button.isVisibleTo(window)
    assert window.store.current.exposure_ev == 0


def test_view_changes_are_not_edits_or_ai_requests(window, application):
    events = len(window.store.events)
    revision = window.store.current.revision
    for _ in range(10):
        window.preview.actual_size()
        window.preview.pan(3, 5)
    window._stop_detail()
    assert len(window.store.events) == events
    assert window.store.current.revision == revision
    assert window.api_budget.attempts == 0


def test_pan_during_warm_edit_still_finishes_current_state(window, application, monkeypatch):
    from photocoach.imaging import RawOptions
    window.engine.cache.full_identity = ("photo-a", RawOptions())
    window.preview.actual_size()
    spin(application, lambda: settled(window))
    started, release = threading.Event(), threading.Event()
    calls = []
    def render(state, request):
        calls.append(request.generation)
        if len(calls) == 1:
            started.set()
            assert release.wait(5)
        return np.ones((30,40,3), np.uint8), .01, {"visible_size":[4000,3000]}
    monkeypatch.setattr(window.engine, "detail", render)
    window.change_exposure(40)
    window._render_timer.stop()
    window.request_preview()
    spin(application, started.is_set)
    try:
        window.preview.pan(10, 20)
        window._detail_timer.stop()
        window.request_detail()
    finally:
        release.set()
    spin(application, lambda: settled(window))
    assert len(calls) == 2 and calls[-1] == window.preview.view_generation
    assert window.store.preview_revision == window.store.current.revision
    assert window.preview.quality == "Full detail"
    assert not window.preview.quick_current
    window.preview.fit()
    spin(application, lambda: settled(window))
    # Returning to Fit after native detail must keep formal processing quality.
    assert window.preview.quality == "Full detail"
    assert not window.preview.quick_current
    assert calls[-1] == window.preview.view_generation
    assert window.store.preview_revision == window.store.current.revision
