"""Stage timing is diagnostics-only and never changes rendered pixels."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import threading
import time

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from photocoach.adjustments import render_edit, render_photo
from photocoach.app import MainWindow
from photocoach.color_mixer import ColorMixer
from photocoach.composition import Composition, output_size
from photocoach.histogram import sampled_histogram
from photocoach.preview import DetailRequest, render_region
from photocoach.render_timing import StageTimer, timed
from photocoach.state import EditValues

VALUES = EditValues(contrast=.1, shadows=.3, highlights=-.2, whites=-.1, vibrance=.2,
                    rgb_curve=((0., 0.), (.4, .45), (1., 1.)),
                    color_mixer=ColorMixer().with_value(2, 2, 15).with_value(5, 1, -10),
                    composition=Composition((.05, .1, .9, .95), 3.25))


def source():
    image = np.random.default_rng(7).random((300, 420, 3), dtype=np.float32)
    image.setflags(write=False)
    return image


def test_timer_is_thread_safe_and_optional():
    timer = StageTimer()
    def work():
        for _ in range(200):
            timer.add("hsl", .001)
    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert timer.milliseconds()["hsl"] == pytest.approx(800, abs=.01)
    with timed(None, "tones"):
        pass  # A missing timer is a no-op.


@pytest.mark.parametrize("step,workers", [(1, 1), (2, 4)])
def test_timed_region_render_is_pixel_identical(step, workers):
    linear = source()
    size = output_size((420, 300), VALUES.composition)
    request = DetailRequest(0, size, (0, 0, *size), step)
    timer = StageTimer()
    plain = render_region(linear, VALUES, request, workers=workers)
    timed_pixels = render_region(linear, VALUES, request, workers=workers, timer=timer)
    np.testing.assert_array_equal(plain, timed_pixels)
    stages = timer.milliseconds()
    assert {"geometry", "tones", "hsl", "curve", "quantize"} <= set(stages)
    assert ("reduction" in stages) == (step != 1)


def test_timed_edit_photo_and_histogram_are_identical():
    linear = source()
    timer = StageTimer()
    np.testing.assert_array_equal(render_edit(linear, VALUES), render_edit(linear, VALUES, timer=timer))
    np.testing.assert_array_equal(render_photo(linear, VALUES, max_edge=200),
                                  render_photo(linear, VALUES, max_edge=200, timer=timer))
    np.testing.assert_array_equal(render_photo(linear, VALUES), render_photo(linear, VALUES, timer=timer))
    assert sampled_histogram(linear, VALUES) == sampled_histogram(linear, VALUES, timer=timer)
    assert "histogram" in timer.milliseconds()


def test_window_records_queue_stage_and_presentation_diagnostics(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        state = window.store.activate(Path("synthetic.ARW"), "a")
        window.preview.configure_photo((300, 200))
        window.preview.display(np.zeros((200, 300, 3), np.uint8))
        window.store.mark_preview(state)
        window._set_controls(True)
        metadata = {"visible_size": [300, 200], "photo_id": "a"}
        observed = []
        def render(state):
            observed.append(window.engine.timer)
            window.engine.timer.add("hsl", .002)
            return np.full((200, 300, 3), 3, np.uint8), .001, metadata
        monkeypatch.setattr(window.engine, "render", render)
        monkeypatch.setattr(window.preview, "needs_detail", lambda: False)
        monkeypatch.setattr(window.network_pool, "start", lambda *_: pytest.fail("Network forbidden"))
        window.change_exposure(3)
        window._render_timer.stop()
        window.request_preview()
        end = time.monotonic() + 5
        while window._active is not None or window._pending:
            app.processEvents()
            assert time.monotonic() < end
            time.sleep(.001)
        app.processEvents()
        details = window._render_diagnostics
        assert observed and observed[0] is not None and window.engine.timer is None
        assert details["kind"] == "preview"
        assert details["stages_ms"]["hsl"] == pytest.approx(2, abs=.01)
        assert details["queue_wait_ms"] >= 0 and details["worker_wall_ms"] >= 0
        assert details["presentation_ms"] >= 0
    finally:
        window._close_approved = True
        window._settle_timer.stop()
        window._render_timer.stop()
        window.close()
