"""Offline scheduling contracts; no model, RAW, credential or desktop access."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import threading
import time

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QDialog

from photocoach.app import MainWindow
from photocoach.auto_coach import AutoCoachSchedule
from photocoach.coach import build_request
from photocoach.color_mixer import ColorMixer


class Clock:
    now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def settle(app, condition):
    deadline = time.monotonic() + 5
    while not condition():
        app.processEvents()
        assert time.monotonic() < deadline, "Worker did not settle"
        time.sleep(.002)
    app.processEvents()


def test_latest_intent_idle_interval_and_binding():
    clock = Clock()
    schedule = AutoCoachSchedule(clock=clock)
    schedule.offer(("a", 1))
    clock.advance(1)
    schedule.offer(("a", 2))
    clock.advance(1)
    assert not schedule.take(("a", 2))
    clock.advance(.3)
    assert not schedule.take(("a", 2), blocked=True)
    assert schedule.take(("a", 2))
    assert not schedule.take(("a", 2))
    schedule.offer(("a", 3))
    clock.advance(2)
    assert not schedule.take(("a", 3))
    clock.advance(8)
    assert schedule.take(("a", 3))
    schedule.offer(("a", 4))
    assert not schedule.take(("b", 4))
    assert schedule.pending is None


def test_release_restarts_idle_and_cancel_drops_backlog():
    clock = Clock()
    schedule = AutoCoachSchedule(clock=clock)
    schedule.offer(1)
    clock.advance(10)
    schedule.released()
    assert not schedule.take(1)
    clock.advance(2)
    schedule.cancel()
    assert not schedule.take(1)


@pytest.fixture
def view(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MainWindow(dev_mode=True)
    state = window.store.activate(Path("synthetic.ARW"), "a")
    window.store.mark_preview(state)
    window._set_controls(True)
    clock = Clock()
    window.auto_schedule.clock = clock
    requests = []
    pixels = np.zeros((4, 6, 3), np.uint8)

    def prepare(state, before, events, question):
        request = build_request(state, before, events, pixels, pixels, question)
        requests.append(request)
        return request

    monkeypatch.setattr(window.engine, "coach_request", prepare)
    monkeypatch.setattr(window.engine, "render", lambda state:
                        (pixels, .001, {"visible_size": [6, 4], "photo_id": state.photo_id}))
    # A GUI-test network attempt is an immediate failure.
    monkeypatch.setattr(window.network_pool, "start", lambda *_: pytest.fail("Unexpected cloud call"))
    yield window, app, clock, requests
    window._cancel_auto_coach()
    window._render_timer.stop()
    window._close_approved = True
    settle(app, lambda: window._active is None and not window._pending)
    window.close()
    window.deleteLater()
    app.processEvents()


def tick(view, seconds=2):
    window, app, clock, _ = view
    window._render_timer.stop()
    if seconds >= .15 and window._settle_timer.isActive() and not window.store.gesture_active:
        was_busy = window._active is not None
        window._finalize_preview()
        if not was_busy:
            settle(app, lambda: window._active is None and not window._pending)
    clock.advance(seconds)
    window._poll_auto_coach()


def finish(view):
    window, app, _, _ = view
    settle(app, lambda: window._active is None and not window._pending)


def test_opt_in_latest_of_ten_edits_and_bounded_context(view):
    window, _, _, requests = view
    window.change_exposure(10)
    tick(view)
    assert not requests and not window.auto_mock.isChecked()
    window.auto_mock.setChecked(True)
    tick(view)
    assert not requests  # Opt-in itself does not analyze.
    for i in range(10):
        window.change_exposure(20 + i)
    tick(view, 1)
    assert not requests
    tick(view, .3)
    finish(view)
    assert len(requests) == window.coach.calls == 1
    request = requests[0]
    assert request.parameters == window.store.current.values.to_dict()
    assert request.state_version == window.store.current.revision
    assert len(request.recent_operations) <= 12
    assert request.current_preview.shape == request.before_preview.shape == (4, 6, 3)
    assert window.api_budget.attempts == 0
    assert "New reply" in window.coach_collapse.text()
    assert window.coach_body.isHidden()


@pytest.mark.parametrize("parameter,value", [("exposure_ev", .5),
    ("rgb_curve", ((0., 0.), (.4, .5), (1., 1.))),
    ("color_mixer", ColorMixer().with_value(2, 2, 15))])
def test_no_analysis_during_gesture_and_release_debounce(view, parameter, value):
    window, _, _, requests = view
    window.auto_mock.setChecked(True)
    window._begin_parameter_gesture(parameter)
    window.change_parameter(parameter, value)
    tick(view, 20)
    assert not requests
    window._finish_gesture()
    finish(view)  # Release now schedules the formal preview before coaching.
    tick(view, 1)
    assert not requests
    tick(view, .3)
    finish(view)
    assert len(requests) == 1 and len(window.store.undo_stack) == 1
    assert requests[0].recent_operations[-1]["after"] == window.store.events[
        next(i for i in range(len(window.store.events)-1, -1, -1)
             if window.store.events[i].get("history_entry"))]["after"]


@pytest.mark.parametrize("action", ["pause", "off", "cloud", "question", "goal_draft", "compare"])
def test_cancellation_actions_do_not_replay(view, action):
    window, _, _, requests = view
    window.auto_mock.setChecked(True)
    window.change_exposure(40)
    if action == "pause":
        window.set_coach_paused(True)
        window.set_coach_paused(False)
    elif action == "off":
        window.auto_mock.setChecked(False)
        window.auto_mock.setChecked(True)
    elif action == "cloud":
        window.coach_mode.setCurrentIndex(1)
        window.analyze(automatic=True)
        assert not window.auto_mock.isChecked() and not window.auto_mock.isEnabled()
        window.coach_mode.setCurrentIndex(0)
    elif action == "question":
        window.question.setText("A new draft")
    elif action == "goal_draft":
        window.goal.textEdited.emit("A new goal draft")
    else:
        window.change_compare(True)
    tick(view, 30)
    finish(view)
    assert not requests and window.api_budget.attempts == 0


def test_manual_mock_counts_toward_interval_and_no_repeat(view):
    window, _, _, requests = view
    window.auto_mock.setChecked(True)
    window.analyze()
    finish(view)
    window.change_exposure(50)
    tick(view, 2)
    assert len(requests) == 1
    tick(view, 8)
    finish(view)
    assert len(requests) == 2
    tick(view, 100)
    assert len(requests) == 2


def test_wb_undo_redo_reset_offer_latest_state(view):
    window, _, _, requests = view
    window.auto_mock.setChecked(True)
    window.wb_mode.setCurrentIndex(1)
    window.gains[0].setValue(1.2)
    window.change_wb()
    finish(view)
    assert window.auto_schedule.pending is not None
    window.undo()
    window.redo()
    tick(view)
    finish(view)
    assert requests[-1].parameters["white_balance"] == {"mode": "custom", "relative_rgb_gain": [1.2, 1., 1.]}
    window.reset_edits()
    tick(view, 10)
    finish(view)
    assert requests[-1].parameters == window.store.current.values.to_dict()


@pytest.mark.parametrize("change", ["edit", "photo", "pause", "off"])
def test_running_result_is_stale_and_only_latest_edit_can_follow(view, monkeypatch, change):
    window, app, _, requests = view
    entered, release = threading.Event(), threading.Event()
    real = window.coach.analyze

    def slow(request):
        entered.set()
        assert release.wait(4)
        return real(request)

    monkeypatch.setattr(window.coach, "analyze", slow)
    window.auto_mock.setChecked(True)
    window.change_exposure(30)
    tick(view)
    settle(app, entered.is_set)
    try:
        if change == "edit":
            for i in range(10):
                window.change_exposure(40 + i)
        elif change == "photo":
            window.store.activate(Path("other.ARW"), "b")
            window._invalidate_coach("Photo changed")
        elif change == "pause":
            window.set_coach_paused(True)
        else:
            window.auto_mock.setChecked(False)
        tick(view, 20)
        assert len(requests) == 1 and not window._protected_jobs
    finally:
        release.set()
    finish(view)
    assert window.coach_text._feedback is None
    if change == "edit":
        tick(view)
        finish(view)
        assert len(requests) == 2
        assert requests[-1].parameters["exposure_ev"] == .49
        assert window.coach_text._feedback.state_version == window.store.current.revision
    else:
        tick(view)
        assert len(requests) == 1


@pytest.mark.parametrize("reason", ["budget", "failure"])
def test_stop_on_limit_or_failure_without_retry(view, monkeypatch, reason):
    window, _, _, requests = view
    window.auto_mock.setChecked(True)
    if reason == "budget":
        window.coach.calls = window.coach.limit
    else:
        def fail(*_):
            raise RuntimeError("Synthetic mock failure")
        monkeypatch.setattr(window.coach, "analyze", fail)
    window.change_exposure(20)
    tick(view)
    finish(view)
    assert not window.auto_mock.isChecked()
    count = len(requests)
    tick(view, 100)
    window.change_exposure(30)
    tick(view, 100)
    assert len(requests) == count
    assert window.exposure_slider.isEnabled() and window.api_budget.attempts == 0


def test_modal_blocks_dispatch_and_timing_is_configurable(view):
    window, app, _, requests = view
    window.auto_mock.setChecked(True)
    window.auto_idle.setValue(3)
    window.auto_interval.setValue(15)
    assert window.auto_schedule.interval_seconds == 15
    window.change_exposure(20)
    tick(view, 2)
    assert not requests
    dialog = QDialog(window)
    dialog.setModal(True)
    dialog.show()
    app.processEvents()
    tick(view, 5)
    assert not requests
    dialog.reject()
    app.processEvents()
    tick(view)
    finish(view)
    assert len(requests) == 1


def test_qt_timer_dispatches_without_manual_poll_and_close_cancels(view):
    window, app, clock, requests = view
    window.auto_mock.setChecked(True)
    window.change_exposure(25)
    clock.advance(2)
    settle(app, lambda: len(requests) == 1)
    finish(view)
    window.change_exposure(30)
    window._close_approved = True
    window.close()
    assert window.auto_schedule.pending is None and not window._auto_timer.isActive()


def test_view_and_goal_changes_do_not_start_analysis(view):
    window, _, _, requests = view
    window.auto_mock.setChecked(True)
    window.goal.setText("Natural light")
    window.change_goal()
    window.preview.configure_photo((6000, 4000))
    window.preview.actual_size()
    window.preview.pan(10, 20)
    window.change_compare(True)
    window.coach_collapse.click()
    tick(view, 20)
    finish(view)
    assert not requests
