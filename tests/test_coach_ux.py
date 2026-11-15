"""Simplified Coach workflows, using isolated settings and zero network access."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
from pathlib import Path
import threading

import numpy as np
import pytest
from PySide6.QtCore import Qt, QUrl
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from photocoach.api_coach import ApiConfig, ApiResult, GEMINI_ENDPOINT
from photocoach.app import Job, MainWindow
from photocoach.coach import CoachFeedback
from photocoach.coach_actions import resolve_actions
from photocoach.settings import ApiSettings, SettingsStore


@pytest.fixture
def window(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    class Vault:
        def read(self, *_):
            pytest.fail("UI navigation must not read credentials")
    view = MainWindow(settings_store=SettingsStore(tmp_path/"settings.json"), credentials=Vault())
    monkeypatch.setattr(view, "request_preview", lambda **kwargs: None)
    monkeypatch.setattr(view, "request_detail", lambda: None)
    monkeypatch.setattr(view.network_pool, "start", lambda *_: pytest.fail("No provider request"))
    view.store.activate(Path("synthetic.ARW"), "photo-a")
    view.preview.configure_photo((1000, 800))
    view.api_settings = ApiSettings(ApiConfig(GEMINI_ENDPOINT, "gemini-test", "gemini"), {}, key_source="saved")
    view._set_controls(True)
    view._sync_controls()
    view.resize(1280, 800)
    view.show()
    view.coach_collapse.setChecked(False)
    view._update_coach_status()
    app.processEvents()
    yield view
    view._network_active = view._active = None
    view._protected_jobs.clear()
    view._close_approved = True
    view.close()
    view.deleteLater()
    app.processEvents()


def reply(view):
    state = view.store.current
    actions = resolve_actions([dict(operation="add", parameter="exposure_ev", value=.35)], state.values)
    return CoachFeedback(state.photo_id, state.session_id, state.revision, "Current edit", "Visible evidence",
                         "Try a small exposure lift.", "Watch the bright rock texture.", ("Observed: texture.",),
                         answer="The subject is relatively dark.", question=view.question.text(), actions=actions)


def test_multiline_draft_limits_shortcut_and_common_question_undo(window, monkeypatch):
    sent = []
    monkeypatch.setattr(window, "analyze", lambda: sent.append(window.question.text()))
    window.question.setFocus()
    window.question.setText("First")
    QTest.keyClick(window.question, Qt.Key.Key_Return)
    assert "\n" in window.question.text() and not sent
    QTest.keyClick(window.question, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    assert sent == [window.question.text()]
    window.question.setText("Original question")
    window.coach_question_actions[1].trigger()
    assert "shared RGB curve" in window.question.text()
    window.question.undo()
    assert window.question.text() == "Original question"
    window.question.setText("x"*2100)
    assert len(window.question.text()) == 2000
    window.question.selectAll()
    window.question.insert("y"*2100)
    assert window.question.text() == "y"*2000
    from PySide6.QtGui import QInputMethodEvent
    window.question.clear()
    composition = QInputMethodEvent()
    composition.setCommitString("中文照片")
    QApplication.sendEvent(window.question, composition)
    assert window.question.text() == "中文照片"
    window.question.setText("🙂"*1999)
    cursor = window.question.textCursor()
    cursor.movePosition(cursor.MoveOperation.End)
    window.question.setTextCursor(cursor)
    composition = QInputMethodEvent()
    composition.setCommitString("中文")
    QApplication.sendEvent(window.question, composition)
    assert window.question.text() == "🙂"*1999+"中"


def test_selection_goal_and_secondary_actions_are_local(window):
    state = window.store.current
    window.question.setText("My draft")
    window.coach_selector.click()
    assert window.coach_selection.isVisible()
    assert window.coach_model_picker.isVisible()
    assert window.local_ai_button.isVisible()
    window.coach_selection.hide()
    window.edit_goal_action.trigger()
    assert window.goal.isVisible()
    window.edit_goal_action.trigger()
    assert not window.goal.isVisible()
    assert window.store.current == state
    assert window.question.text() == "My draft" and window.api_budget.attempts == 0
    assert window.coach_light_menu_button.parentWidget() is not window.coach_body


def test_cancel_discards_late_reply_and_still_settles_request(window, monkeypatch):
    feedback = reply(window)
    window.coach_text.show_feedback(feedback)
    job = Job("api_photo", window._generation, lambda: None, window.store.current, window._coach_generation)
    cancel = threading.Event()
    ticket = window.api_budget.reserve()
    window._network_active = (None, job, cancel, ticket)
    window._network_budget = window.api_budget
    window._update_coach_status()
    assert window.analyze_button.text() == "Cancel analysis"
    token = window._coach_generation
    window.analyze_button.click()
    assert cancel.is_set() and window._coach_generation > token
    assert not window.analyze_button.isEnabled()
    window._network_done(ApiResult(feedback, 20, 10), None)
    assert window.coach_text._feedback is None
    assert window.api_budget.in_flight == 0 and window.api_budget.attempts == 1
    assert window.analyze_button.isEnabled()
    assert window.coach_summary.toolTip() == "Analysis cancelled"


def test_preparation_blocks_another_analysis_and_cancel_preserves_other_jobs(window, monkeypatch):
    prepared = Job("api_prepare", window._generation, lambda: None, window.store.current, window._coach_generation)
    keep = Job("project_save", window._generation, lambda: None)
    window._active = (None, prepared)
    window._protected_jobs = [prepared, keep]
    window._update_coach_status()
    monkeypatch.setattr(window, "_queue", lambda *_: pytest.fail("Duplicate analysis"))
    window.analyze()
    assert not window.text_probe_button.isEnabled()
    window.test_text_only()
    window._coach_primary_action()
    assert window._protected_jobs == [keep]
    assert window._active[1] is prepared and window._coach_cancel_requested


def test_reply_has_one_apply_entry_and_same_fixed_targets_in_reader(window, monkeypatch):
    import photocoach.coach_actions as actions
    monkeypatch.setattr(actions, "analyze_scene_lighting", lambda *_: pytest.fail("Redisplay must not analyze"))
    feedback = reply(window)
    window.coach_text.set_preview_pixels(np.full((1067,1600,3),128,np.uint8))
    window.coach_text.show_feedback(feedback)
    window.coach_response.open_reader()
    for language in ("zh", "en"):
        window.set_language(language)
        window.coach_text.anchorClicked.emit(QUrl("photocoach:evidence"))
        assert "0.35 EV" in window.coach_text.toPlainText()
        assert "0.35 EV" in window.coach_response.changes_label.text()
        assert "0.35 EV" in window.coach_response.reader_output.toPlainText()
        assert "photocoach:apply/" not in window.coach_text.toHtml()
        assert window.coach_response.reader_output._actions is feedback.actions
    assert window._apply_coach_suggestion()
    assert not window._apply_coach_suggestion()
    assert window.coach_response.apply_button.text() == "Applied ✓"
    assert "Analyze again before applying" not in window.coach_text.toPlainText()
    assert len(window.store.undo_stack) == 1
    window.undo()
    assert not window.coach_response.apply_button.isEnabled()
    assert "changed" in window.coach_summary.toolTip()


def test_photo_question_and_model_changes_keep_drafts_and_disable_old_actions(window):
    window.question.setText("Original question")
    window.coach_text.show_feedback(reply(window))
    window._select_coach_model("gemini-alt")
    assert window.coach_text._feedback is not None
    assert not window.coach_response.apply_button.isEnabled()
    assert not window._apply_coach_suggestion()
    assert window.coach_summary.toolTip() == "Coach changed · analyze again"
    window.coach_text.show_feedback(reply(window))
    window.question.setText("New question")
    window.question.setText("Original question")
    assert not window.coach_response.apply_button.isEnabled()
    window.coach_collapse.setChecked(True)
    window.coach_collapse.setChecked(False)
    assert window.question.text() == "Original question"
    assert window.coach_text._feedback is not None
    window.store.activate(Path("different.ARW"), "different-photo")
    window._after_edit()
    assert window.coach_text._feedback is None and not window._apply_coach_suggestion()
