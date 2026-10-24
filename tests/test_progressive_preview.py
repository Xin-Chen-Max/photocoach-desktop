"""Progressive display quality never changes formal rendering or editing state."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from dataclasses import replace
import hashlib
from pathlib import Path
import threading
import time

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from photocoach.app import MainWindow, PreviewEngine, Job
from photocoach.composition import Composition, output_size
from photocoach.imaging import DecodedPhoto
from photocoach.preview import DetailRequest, LINEAR_CACHE_BYTES
from photocoach.state import EditState


def wait(app, condition):
    end = time.monotonic() + 5
    while not condition():
        app.processEvents()
        assert time.monotonic() < end
        time.sleep(.001)
    app.processEvents()


@pytest.fixture
def view(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    state = window.store.activate(Path("synthetic.ARW"), "a")
    window.preview.configure_photo((300, 200))
    window.preview.display(np.zeros((200, 300, 3), np.uint8))
    window.store.mark_preview(state)
    window._set_controls(True)
    metadata = {"visible_size": [300, 200], "photo_id": "a"}
    def render(state):
        return np.full((200, 300, 3), state.revision, np.uint8), .001, metadata
    monkeypatch.setattr(window.engine, "render", render)
    monkeypatch.setattr(window.engine, "detail", lambda state, request: render(state))
    monkeypatch.setattr(window.engine, "adjusting", lambda state, request: render(state))
    monkeypatch.setattr(window.preview, "needs_detail", lambda: False)
    monkeypatch.setattr(window.network_pool, "start", lambda *_: pytest.fail("Network forbidden"))
    yield window, app
    window._close_approved = True
    window._settle_timer.stop()
    window._render_timer.stop()
    wait(app, lambda: window._active is None and not window._pending and not window._pending_detail)
    window.close()


def test_drag_uses_temporary_quality_then_immediate_final_and_one_undo(view):
    w, app = view
    center, zoom = w.preview.center, w.preview.zoom
    w._begin_parameter_gesture("exposure_ev")
    for i in range(1, 11):
        w.change_exposure(i)
    w._render_timer.stop()
    w.request_preview(interactive=True)
    wait(app, lambda: w._active is None)
    assert w.preview.quality == "Adjusting"
    assert w.store.preview_revision == 0  # Approximation cannot certify formal pixels.
    w._finish_gesture()
    wait(app, lambda: w._active is None)
    assert w.preview.quality == "Full detail"
    assert w.store.preview_revision == 10
    assert len(w.store.undo_stack) == 1
    assert (w.preview.center, w.preview.zoom) == (center, zoom)
    assert not w._settle_timer.isActive()


def test_keyboard_settles_automatically(view):
    w, app = view
    w.change_exposure(25)
    assert w._settle_timer.interval() == 150
    wait(app, lambda: w.store.preview_revision == 1)
    assert w.preview.quality == "Full detail"


@pytest.mark.parametrize("command", ["undo", "redo", "reset_edits"])
def test_committed_commands_refine_fit_without_quality_downgrade(view, command):
    w, app = view
    w.change_exposure(25)
    wait(app, lambda: w.preview.quality == "Full detail")
    if command == "redo":
        w.undo()
        wait(app, lambda: w.store.preview_revision == w.store.current.revision)
    center, zoom = w.preview.center, w.preview.zoom
    getattr(w, command)()
    wait(app, lambda: w.store.preview_revision == w.store.current.revision)
    assert w.preview.quality == "Full detail"
    assert (w.preview.center, w.preview.zoom) == (center, zoom)


@pytest.mark.parametrize("trigger", ["request", "resume", "viewport"])
def test_formal_fit_quality_survives_nonediting_refresh(view, trigger):
    w, app = view
    w.change_exposure(25)
    wait(app, lambda: w.preview.quality == "Full detail")
    values = w.store.current.values
    if trigger == "request":
        w.request_preview()
    elif trigger == "resume":
        w._queue(w._resume_preview_job(w.store.current))
    else:
        w._view_changed()
    wait(app, lambda: w._active is None and not w._detail_timer.isActive())
    assert w.preview.quality == "Full detail"
    assert w.store.current.values == values


def test_quick_refresh_does_not_request_unsolicited_full_decode(view, monkeypatch):
    w, app = view
    details = []
    original = w.engine.detail
    monkeypatch.setattr(w.engine, "detail", lambda *args: (details.append(True), original(*args))[1])
    w.request_preview()
    wait(app, lambda: w._active is None and not w._detail_timer.isActive())
    assert w.preview.quality == "Quick preview"
    assert not details


def test_late_temporary_cannot_replace_same_revision_formal_frame(view):
    w, app = view
    snapshot = w.store.current
    request = w.preview.detail_request()
    w._final_frame_binding = w._frame_binding(snapshot, request)
    w._queue(Job("adjusting", w._generation, lambda: (np.ones((2, 3, 3), np.uint8), .01, {}),
                 snapshot, view_request=request))
    wait(app, lambda: w._active is None)
    assert w.preview.quality != "Adjusting"


def test_one_pending_render_slot_and_stale_state_view_photo(view, monkeypatch):
    w, app = view
    entered, release = threading.Event(), threading.Event()
    shown = []
    def slow(state, request):
        entered.set()
        assert release.wait(4)
        return np.ones((2, 3, 3), np.uint8), .01, {}
    monkeypatch.setattr(w.engine, "adjusting", slow)
    monkeypatch.setattr(w.preview, "display_adjusting", lambda *args: shown.append(args))
    w._begin_parameter_gesture("exposure_ev")
    w.change_exposure(10)
    w._render_timer.stop()
    w.request_preview(interactive=True)
    wait(app, entered.is_set)
    try:
        for i in range(2, 12):
            w.change_exposure(i * 10)
            w.request_preview(interactive=True)
            assert (w._pending is not None) + (w._pending_detail is not None) == 1
        w.preview.actual_size()
        w.store.activate(Path("other.ARW"), "b")
        w._pending = None
    finally:
        w._interactive_preview = False
        w._render_timer.stop()
        w._settle_timer.stop()
        release.set()
    wait(app, lambda: w._active is None)
    assert not shown


def test_temporary_allocation_failure_retains_old_image_and_final_recovers(view, monkeypatch):
    w, app = view
    monkeypatch.setattr(w.engine, "adjusting", lambda *_: (_ for _ in ()).throw(MemoryError("test budget")))
    w.change_exposure(10)
    w.request_preview(interactive=True)
    wait(app, lambda: w._active is None)
    assert w.preview.has_quick and w.preview.quality == "Adjustment preview unavailable"
    w._finalize_preview()
    wait(app, lambda: w._active is None)
    assert w.store.preview_revision == 1 and w.preview.quality == "Full detail"


def test_new_edit_cancels_obsolete_formal_worker_without_waiting_for_it(view, monkeypatch):
    from concurrent.futures import CancelledError
    w, app = view
    entered = threading.Event()
    def obsolete(state):
        entered.set()
        assert w.engine.cancel.wait(1), "New edit must cancel a running formal render"
        raise CancelledError()
    monkeypatch.setattr(w.engine, "render", obsolete)
    w.request_preview()
    wait(app, entered.is_set)
    w.change_exposure(20)
    w._render_timer.stop()
    w.request_preview(interactive=True)
    wait(app, lambda: w._active is None)

    assert w.preview.quality == "Adjusting"
    assert "Operation failed" not in w.status.text()
    monkeypatch.setattr(w.engine, "render", lambda state: (np.zeros((200,300,3), np.uint8), .001, {}))
    w._finalize_preview()
    wait(app, lambda: w._active is None)


def test_stale_overlay_is_discarded_and_toggle_queue_stays_bounded(view,monkeypatch):
    w,app=view
    entered,release=threading.Event(),threading.Event()
    calls=[];frames=[]
    def render(snapshot):
        first=not calls;calls.append(True)
        if first:
            entered.set();assert release.wait(4)
        overlay=np.zeros((200,300,4),np.uint8);overlay[...,0]=255;overlay[...,3]=170
        return np.zeros((200,300,3),np.uint8),.001,{'visible_size':[300,200],
            **({'display_overlay':(overlay,(0,0,300,200))} if first else {})}
    monkeypatch.setattr(w.engine,'render',render)
    original=w.preview.accept_overlay
    monkeypatch.setattr(w.preview,'accept_overlay',lambda frame:(frames.append(frame),original(frame)))
    w.clipping.setChecked(True);wait(app,entered.is_set)
    try:
        for value in (False,True,False,True,False):
            w.clipping.setChecked(value)
            assert (w._pending is not None)+(w._pending_detail is not None)==1
    finally:release.set()
    wait(app,lambda:w._active is None)
    assert frames and all(frame is None for frame in frames)
    assert w.preview._overlay.isNull() and w.store.current.revision==0


def test_overlay_failure_still_presents_the_photo_with_explanation(view,monkeypatch):
    w,app=view
    monkeypatch.setattr(w.engine,'render',lambda state:(np.ones((200,300,3),np.uint8),.001,
        {'visible_size':[300,200],'overlay_error':'Overlay exceeds display memory budget.'}))
    w.clipping.setChecked(True)
    wait(app,lambda:w._active is None)
    assert w.preview.has_quick and w.preview._overlay.isNull()
    assert 'budget' in w.view_notice.text()


def test_formal_strips_honor_cancellation_without_altering_source():
    from concurrent.futures import CancelledError
    from photocoach.preview import render_region
    from photocoach.state import EditValues
    cancel = threading.Event()
    cancel.set()
    source = np.full((100, 120, 3), .1, np.float32)
    with pytest.raises(CancelledError):
        render_region(source, EditValues(), DetailRequest(1, (120,100), (0,0,120,100)), cancel=cancel)
    assert np.all(source == np.float32(.1))


@pytest.mark.parametrize("angle", [0, 3.25])
@pytest.mark.parametrize("small_roi", [False, True])
def test_engine_temporary_cache_roi_bound_and_no_warm_decodes(tmp_path, monkeypatch, angle, small_roi):
    import photocoach.imaging as imaging
    source = tmp_path / "test.ARW"
    source.write_bytes(b"read only sample")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    def decode(path, options, *, half_size):
        width, height = (700, 500) if half_size else (1400, 1000)
        image = np.random.default_rng(3).random((height, width, 3), dtype=np.float32)
        image.setflags(write=False)
        return DecodedPhoto(image, {"sha256": digest, "photo_id": digest,
            "visible_size": [1400, 1000], "output_size": [width, height],
            "orientation_flip": 0, "half_size": half_size}, options, half_size, path)
    monkeypatch.setattr(imaging, "decode_raw", decode)
    engine = PreviewEngine()
    engine.load(source)
    composition = Composition(angle=angle)
    state = EditState(digest, source, "session", composition=composition)
    size = output_size((1400, 1000), composition)
    rect = (100, 100, 600, 500) if small_roi else (0, 0, *size)
    request = DetailRequest(1, size, rect, source_size=(1400, 1000))
    engine.detail(state, request)
    decodes = engine.cache.decode_count
    pixels, _, metadata = engine.adjusting(state, request)
    cached = engine._adjusting_pixels
    assert max(pixels.shape[:2]) <= 960 and not metadata
    assert cached.nbytes <= 32 * 1024**2 and not cached.flags.writeable
    engine.adjusting(replace(state, contrast=.2), request)
    assert engine._adjusting_pixels is cached
    for name in ('whites', 'blacks', 'vibrance'):
        engine.adjusting(replace(state, **{name: .2}), request)
        assert engine._adjusting_pixels is cached
    assert engine.cache.decode_count == decodes
    assert engine.retained_bytes <= LINEAR_CACHE_BYTES
    engine.detail(state, request)
    assert engine._adjusting_pixels is cached
    assert engine.retained_bytes <= LINEAR_CACHE_BYTES
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest


def test_resolution_is_captured_for_drag_and_keyboard_burst(view, monkeypatch):
    import photocoach.app as desktop
    w,app=view;caps=[]
    monkeypatch.setattr(w.engine,'adjusting',lambda state,request:
        (caps.append(request.adjusting_edge) or np.zeros((20,30,3),np.uint8),.001,{}))
    monkeypatch.setattr(desktop,'ADJUSTING_MAX_EDGE',960)
    w._begin_parameter_gesture('whites')
    for value,default in ((.1,960),(.2,1280),(0,640),(.3,640)):
        monkeypatch.setattr(desktop,'ADJUSTING_MAX_EDGE',default)
        w.change_parameter('whites',value)
        w._render_timer.stop();w.request_preview(interactive=True)
        wait(app,lambda:w._active is None)
    assert caps==[960]*4
    w._finish_gesture();wait(app,lambda:w._active is None)
    assert len(w.store.undo_stack)==1 and w.preview.quality=='Full detail'
    w.change_exposure(25)
    w._render_timer.stop();w.request_preview(interactive=True)
    wait(app,lambda:w._active is None)
    assert caps[-1]==640  # Next keyboard burst captures its own cap.
    w._finalize_preview();wait(app,lambda:w._active is None)


def test_refining_retains_temporary_pixels_and_blocks_late_adjusting(view,monkeypatch):
    w,app=view;entered,release=threading.Event(),threading.Event()
    w._begin_parameter_gesture('exposure_ev');w.change_exposure(10)
    w._render_timer.stop();w.request_preview(interactive=True)
    wait(app,lambda:w._active is None)
    frame=w.preview._detail.cacheKey();position=(w.preview.center,w.preview.zoom)
    def slow(state,request):
        entered.set();assert release.wait(4)
        return np.full((20,30,3),255,np.uint8),.001,{}
    monkeypatch.setattr(w.engine,'adjusting',slow)
    qualities=[];w.preview.statusChanged.connect(lambda:qualities.append(w.preview.quality))
    w.change_exposure(20);w._render_timer.stop();w.request_preview(interactive=True)
    wait(app,entered.is_set)
    try:
        w._finish_gesture()
        assert w.preview.quality=='Refining'
        assert w.detail_busy.isVisibleTo(w)
        assert w.preview._detail.cacheKey()==frame and w.preview._detail_temporary
        assert (w.preview.center,w.preview.zoom)==position
    finally:release.set()
    wait(app,lambda:w._active is None)
    assert w.preview.quality=='Full detail' and not w.preview._detail_temporary
    assert 'Adjusting' not in qualities[qualities.index('Refining')+1:]
    assert w.store.preview_revision==w.store.current.revision


@pytest.mark.parametrize('kind',['export','project_save'])
@pytest.mark.parametrize('fails',[False,True])
def test_output_resumes_formal_refinement_after_dropping_pending_detail(view,monkeypatch,kind,fails):
    from concurrent.futures import CancelledError
    w,app=view;entered=threading.Event()
    def obsolete(state,request):
        entered.set();assert w.engine.cancel.wait(4)
        raise CancelledError()
    monkeypatch.setattr(w.engine,'adjusting',obsolete)
    w._begin_parameter_gesture('exposure_ev');w.change_exposure(25)
    w._render_timer.stop();w.request_preview(interactive=True);wait(app,entered.is_set)
    # The output drops a pending final job before the cancelled temporary worker completes.
    w._finish_gesture();w._freeze_output('Synthetic output')
    def output():
        if fails:raise OSError('Synthetic output failure')
        return Path('synthetic-output')
    w._queue(Job(kind,w._generation,output,w.store.current))
    wait(app,lambda:w._active is None and w._pending is None and w._pending_detail is None)
    assert w.store.preview_revision==w.store.current.revision
    assert w.preview.quality=='Full detail' and not w._refining_preview
    assert not w._exporting and w.exposure_slider.isEnabled()
