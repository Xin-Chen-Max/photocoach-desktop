"""Spatial edits must use source-pixel radii before display reduction."""
from dataclasses import replace
import hashlib

import numpy as np
import pytest

from photocoach.app import PreviewEngine
from photocoach.composition import Composition, output_size
from photocoach.detail import CAMERA_RAW_DETAIL, DetailControls
from photocoach.imaging import DecodedPhoto
from photocoach.preview import DetailRequest, LINEAR_CACHE_BYTES
from photocoach.state import EditState


@pytest.fixture
def engine_photo(tmp_path, monkeypatch):
    import photocoach.imaging as imaging
    source = tmp_path / 'sample.ARW'
    source.write_bytes(b'synthetic read-only RAW')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    yy, xx = np.mgrid[:336, :504]
    gray = .2 + .12 * np.sin(xx * .16) * np.cos(yy * .11)
    rng = np.random.default_rng(98)
    linear = np.clip(gray[..., None] + rng.normal(0, .015, (336, 504, 3)), .01, .9).astype(np.float32)
    linear.setflags(write=False)
    def decode(path, options, *, half_size):
        pixels = linear[::2, ::2].copy() if half_size else linear
        pixels.setflags(write=False)
        return DecodedPhoto(pixels, {'sha256': digest, 'photo_id': digest,
            'visible_size': [504, 336], 'output_size': [pixels.shape[1], pixels.shape[0]],
            'orientation_flip': 0, 'half_size': half_size}, options, half_size, path)
    monkeypatch.setattr(imaging, 'decode_raw', decode)
    engine = PreviewEngine()
    engine.load(source)
    return engine, EditState(digest, source, 'test'), linear


@pytest.mark.parametrize('detail', [DetailControls(sharpening=80),
    DetailControls(sharpening=80, masking=90), DetailControls(luminance_nr=80),
    DetailControls(chroma_nr=80), CAMERA_RAW_DETAIL])
@pytest.mark.parametrize('angle', [0., 3.25])
def test_reduced_detail_drag_matches_native_then_reduced(engine_photo, detail, angle):
    engine, state, _ = engine_photo
    state = replace(state, detail=detail, composition=Composition(angle=angle))
    size = output_size((504, 336), state.composition)
    request = DetailRequest(1, size, (0, 0, *size), step=4, source_size=(504, 336), adjusting_edge=160)
    expected, _, _ = engine.detail(state, request)
    count = engine.cache.decode_count
    actual, _, _ = engine.adjusting(state, request)
    assert np.array_equal(actual, expected), 'A drag must not magnify a spatial filter by applying it after reduction'
    assert engine.cache.decode_count == count
    assert engine.retained_bytes <= LINEAR_CACHE_BYTES


def test_detail_adjusting_cold_native_decode_and_zero_endpoint(engine_photo):
    engine, state, original = engine_photo
    state = replace(state, detail=DetailControls(sharpening=70))
    request = DetailRequest(1, (504, 336), (0, 0, 504, 336), step=4,
                           source_size=(504, 336), adjusting_edge=160, native_detail=True)
    assert engine.cache.full_identity is None
    actual, _, _ = engine.adjusting(state, request)
    assert engine.cache.full_identity == (state.photo_id, state.raw_options)
    assert np.array_equal(actual, engine.detail(state, request)[0])
    count = engine.cache.decode_count
    neutral = replace(state, detail=DetailControls())
    assert np.array_equal(engine.adjusting(neutral, request)[0], engine.detail(neutral, request)[0])
    assert engine.cache.decode_count == count
    assert not original.flags.writeable


def test_active_detail_initial_refresh_uses_same_native_sampling(engine_photo):
    engine, state, _ = engine_photo
    state = replace(state, detail=CAMERA_RAW_DETAIL)
    # The initial quick refresh has a 1600 px cap; this source fits natively.
    expected = engine.detail(state, DetailRequest(1, (504, 336), (0, 0, 504, 336)))[0]
    count = engine.cache.decode_count
    assert np.array_equal(engine.render(state)[0], expected)
    assert engine.cache.decode_count == count


def test_native_detail_drag_still_honors_cancellation(engine_photo):
    import threading
    from concurrent.futures import CancelledError
    engine, state, _ = engine_photo
    state = replace(state, detail=CAMERA_RAW_DETAIL)
    engine.cancel = threading.Event()
    engine.cancel.set()
    with pytest.raises(CancelledError):
        engine.adjusting(state, DetailRequest(1, (504, 336), (0, 0, 504, 336), step=4))


def test_detail_slider_keeps_native_radius_at_zero_and_releases_one_gesture(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from photocoach.app import MainWindow
    from photocoach.settings import SettingsStore
    app = QApplication.instance() or QApplication([])
    window = MainWindow(settings_store=SettingsStore(tmp_path / 'unused.json'))
    queued = []
    monkeypatch.setattr(window, '_queue', queued.append)
    window.store.activate(tmp_path / 'synthetic.ARW', 'test')
    window.preview.configure_photo((504, 336))
    window.preview.display(np.zeros((336, 504, 3), np.uint8))
    window._set_controls(True)
    slider = window.detail_editor.sliders['sharpening']
    try:
        slider.setSliderDown(True)
        for value in (80, 0, 60):
            slider.setValue(value)
            window._render_timer.stop()
            window.request_preview(interactive=True)
            assert window.store.current.detail.sharpening == value
            assert queued[-1].view_request.native_detail
        slider.setSliderDown(False)
        assert len(window.store.undo_stack) == 1
        assert not window._native_detail_preview
        assert queued[-1].kind == 'detail'
    finally:
        window._close_approved = True
        window.close()
        window.deleteLater()
        app.processEvents()
