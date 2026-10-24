"""Coach navigation is local; older advice is labeled and never crosses photos."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtCore import QUrl, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from photocoach import api_coach
from photocoach.api_coach import ApiConfig, GEMINI_ENDPOINT
from photocoach.app import MainWindow
from photocoach.budget import SessionBudget
from photocoach.coach import CoachFeedback
from photocoach.coach_ui import ApiSettingsDialog, budget_text
from photocoach.coach_view import CoachOutput, CoachResponsePanel, related_control, usage_summary
from photocoach.settings import ApiSettings, SettingsStore


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Presentation tests must not contact a provider")
    monkeypatch.setattr(api_coach, "_https_transport", forbidden)


class NoVault:
    def read(self, *args):
        pytest.fail("Presentation must not read saved secrets")

    write = delete = read


def feedback(**changes):
    return replace(CoachFeedback("photo-a", "session-a", 2, "No changes yet.",
        "Parameters are neutral.", "Increase exposure slightly.", "Stop when rock texture looks natural.",
        ("Observed: Dark rocks.", "Inference: Possibly underexposed.", "Preference: Natural light."),
        "AI · example-model"), **changes)


def test_feedback_priority_disclosure_and_safe_text(application):
    output = CoachOutput()
    payload = '<img src="https://example.invalid/tracker.png"><a href="https://example.invalid">Click</a>'
    output.show_feedback(feedback(next_step_or_stop=payload))
    text = output.toPlainText()
    assert text.index("NEXT STEP") < text.index("Watch / stop")
    assert "WHAT CHANGED" not in text and "Parameters are neutral." in text
    assert payload in text  # Provider markup is literal text, never an image or link.
    assert "tracker.png&quot;" in output.toHtml()
    assert '<img src="https://example.invalid' not in output.toHtml()
    assert "Observed: Dark rocks." not in text
    output.anchorClicked.emit(QUrl("https://example.invalid"))
    assert not output.evidence_expanded
    output.anchorClicked.emit(QUrl("photocoach:evidence"))
    assert output.evidence_expanded
    text = output.toPlainText()
    assert text.index("WHY THIS HELPS") < text.index("WHAT CHANGED")
    assert all(item in output.toPlainText() for item in feedback().evidence)
    output.anchorClicked.emit(QUrl("photocoach:evidence"))
    assert "Observed: Dark rocks." not in output.toPlainText()
    output.deleteLater()


def test_direct_answer_and_original_question_are_visible_and_escaped(application):
    output = CoachOutput()
    question = 'How can I adjust the RGB curve? <img src="https://example.invalid/question">'
    answer = 'The RGB curve and white balance gains affect the image differently.\n<img src="https://example.invalid/answer">is literal text.'
    steps = "1. Double-click the RGB curve to add a control point.\n2. Drag the midtone point; compare with Undo."
    output.set_question_draft(question)
    output.show_feedback(feedback(question=question, answer=answer, next_step_or_stop=steps))
    text = output.toPlainText()
    assert text.index(question) < text.index("ANSWER") < text.index(answer) < text.index("NEXT STEP")
    assert steps in text and "WHY THIS HELPS" in text
    assert "Parameters are neutral." in text and not output.evidence_expanded
    assert '<img src="https://example.invalid' not in output.toHtml()
    output.set_question_draft("What should I try next?")
    text = output.toPlainText()
    assert "Question changed." in text and question in text
    assert "What should I try next" not in text  # Keep the request's question, not the new draft, beside this answer.
    output.set_question_draft(question)
    assert "Question changed." not in output.toPlainText()
    output.deleteLater()


def test_question_edit_keeps_reply_as_labeled_reference_without_editing_or_sending(application, tmp_path):
    window = MainWindow(settings_store=SettingsStore(tmp_path / "settings.json"), credentials=NoVault())
    state = window.store.activate(Path("synthetic.ARW"), "photo-a")
    window._set_controls(True)
    window.question.setText("How should I use RGB curve?")
    window.coach_text.show_feedback(feedback(photo_id=state.photo_id, session_id=state.session_id,
        state_version=state.revision, question=window.question.text(), answer="Use the shared tonal curve."))
    events = list(window.store.events)
    token = window._coach_generation
    window.question.setText("How can I adjust the RGB curve?")
    assert window.store.current is state and window.store.events == events
    assert window._coach_generation > token
    assert "Question changed." in window.coach_text.toPlainText()
    assert "How should I use RGB curve?" in window.coach_text.toPlainText()
    assert "Earlier advice" in window.coach_summary.toolTip()
    assert window.api_budget.attempts == 0
    window.close()
    window.deleteLater()


def test_status_and_new_response_clear_previous_evidence(application):
    output = CoachOutput()
    available = []
    output.feedbackAvailable.connect(available.append)
    output.show_feedback(feedback())
    output.anchorClicked.emit(QUrl("photocoach:evidence"))
    output.setPlainText("The edit changed. Analyze again when ready.")
    output.anchorClicked.emit(QUrl("photocoach:evidence"))
    assert not output.evidence_expanded
    assert output.toPlainText() == "The edit changed. Analyze again when ready."
    assert available == [True, False]
    output.show_feedback(feedback(evidence=("Observed: New photo.",)))
    output.anchorClicked.emit(QUrl("photocoach:evidence"))
    assert "New photo" in output.toPlainText()
    assert "Dark rocks" not in output.toPlainText()
    output.deleteLater()


def test_bottom_coach_navigation_does_not_change_edits_or_send(application, tmp_path):
    window = MainWindow(settings_store=SettingsStore(tmp_path / "settings.json"), credentials=NoVault())
    window.store.activate(Path("synthetic.ARW"), "photo-a")
    state, events = window.store.current, list(window.store.events)
    window.tabs.setCurrentIndex(1)  # History can be open without hiding the coach.
    assert window.tabs.count() == 2 and window.tabs.tabText(1) == "History"
    assert window.coach_details.isHidden()
    window.api_settings = ApiSettings(ApiConfig(GEMINI_ENDPOINT, "gemini-example", "gemini"), {}, key_source="saved")
    window.coach_mode.setCurrentIndex(1)
    window._update_coach_status()
    assert "gemini-example" in window.coach_connection.text()
    assert "Saved key selected" in window.coach_connection.text()
    assert window.coach_connection.textFormat() == Qt.TextFormat.PlainText
    window.coach_text.show_feedback(feedback())
    window.coach_details_button.click()
    assert not window.coach_details.isHidden()
    window.coach_text.anchorClicked.emit(QUrl("photocoach:locate/exposure_ev"))
    assert window.tabs.currentIndex() == 0
    assert window.exposure_label.property("coachTarget") is True
    assert window.store.current == state and window.store.events == events
    assert window.api_budget.attempts == 0 and window._network_active is None
    window._invalidate_coach("Photo changed.")
    assert window.exposure_label.property("coachTarget") is False
    assert window.coach_text.toPlainText() == "Photo changed."
    window._close_approved = True
    window.close()
    window.deleteLater()


def test_collapse_restores_dragged_height_and_keeps_editing_visible(application, tmp_path):
    window = MainWindow(settings_store=SettingsStore(tmp_path / "settings.json"), credentials=NoVault())
    window.show()
    application.processEvents()
    assert window.coach_body.isHidden()
    assert window.coach_collapse.text() == "Open coach"
    window.coach_collapse.click()
    application.processEvents()
    assert window.coach_text.isVisible() and window.exposure_slider.isVisible()
    total = sum(window.workspace_splitter.sizes())
    window.workspace_splitter.setSizes([total - 320, 320])
    application.processEvents()
    before = window.workspace_splitter.sizes()
    window.coach_collapse.click()
    application.processEvents()
    assert window.coach_body.isHidden()
    assert window.coach_panel.height() < 85
    assert window.workspace_splitter.sizes()[0] > before[0]
    window.coach_text.show_feedback(feedback(next_step_or_stop="1. Increase exposure slightly.\n2. Compare with Undo."))
    application.processEvents()
    assert "Increase exposure" in window.coach_summary.toolTip()
    assert "\n" not in window.coach_summary.toolTip()
    assert window.coach_panel.height() < 85
    assert window.coach_body.isHidden()  # New feedback does not defeat the user's photo space choice.
    assert "New reply" in window.coach_collapse.text()
    window.coach_collapse.click()
    application.processEvents()
    assert window.coach_text.isVisible() and window.exposure_slider.isVisible()
    assert abs(window.workspace_splitter.sizes()[1] - before[1]) <= 4
    assert window.api_budget.attempts == 0 and not window.coach.paused
    window.close()
    window.deleteLater()


def test_reply_keeps_expanded_height_and_question_cursor(application, tmp_path):
    window = MainWindow(settings_store=SettingsStore(tmp_path / "settings.json"), credentials=NoVault())
    window.store.activate(Path("synthetic.ARW"), "photo-a")
    window._set_controls(True)
    window.show()
    window.coach_collapse.click()
    window.goal.setText("Natural light")
    window.question.setText("How should I adjust RGB?")
    window.question.setFocus()
    window.question.setCursorPosition(8)
    application.processEvents()
    sizes = window.workspace_splitter.sizes()
    window.coach_text.show_feedback(feedback())
    application.processEvents()
    assert window.question.hasFocus() and window.question.cursorPosition() == 8
    assert window.workspace_splitter.sizes() == sizes
    window.coach_collapse.click()
    window.coach_collapse.click()
    assert window.goal.text() == "Natural light"
    assert window.question.text() == "How should I adjust RGB?"
    assert window.coach_text._feedback is not None
    window._close_approved = True
    window.close()
    window.deleteLater()


def test_completed_advice_survives_edit_as_reference_but_not_photo_or_session_change(application, tmp_path):
    window = MainWindow(settings_store=SettingsStore(tmp_path / "settings.json"), credentials=NoVault())
    state = window.store.activate(Path("synthetic.ARW"), "photo-a")
    window._set_controls(True)
    window.coach_text.show_feedback(feedback(photo_id=state.photo_id, session_id=state.session_id, state_version=0))
    window.change_exposure(25)
    window._render_timer.stop()
    text = window.coach_text.toPlainText()
    assert "Increase exposure" in text and "Stop when rock texture" in text
    assert "Earlier advice · based on v0" in text and "Current edit is v1" in text
    assert "not been rechecked" in text
    assert window.store.current.exposure_ev == .25
    assert window.coach_text.reference_revision == 1
    window.undo()
    window._render_timer.stop()
    assert window.store.current.exposure_ev == 0
    assert window.coach_text.reference_revision == 2
    window.store.activate(Path("synthetic.ARW"), "photo-a")  # Same bytes, new session.
    window._after_edit()
    window._render_timer.stop()
    assert "Increase exposure" not in window.coach_text.toPlainText()
    state = window.store.current
    window.coach_text.show_feedback(feedback(photo_id=state.photo_id, session_id=state.session_id, state_version=state.revision))
    window.store.activate(Path("other.ARW"), "photo-b")
    window._after_edit()
    window._render_timer.stop()
    assert "Increase exposure" not in window.coach_text.toPlainText()
    window._close_approved = True
    window.close()
    window.deleteLater()


@pytest.mark.parametrize("suggestion,control", [
    ("Increase exposure_ev slightly.", "exposure_ev"), ("Compare the white balance.", "white_balance"),
    ("Reduce saturation.", "saturation"), ("Try a little contrast.", "contrast"),
    ("Lower the RGB curve black endpoint.", "rgb_curve"),
    ("Try exposure and contrast.", None), ("Stop; this already fits your goal.", None),
    ("Try sharpening.", None),
    ("Add a point to the RGB curve.", "rgb_curve"), ("Increase exposure a little.", "exposure_ev"),
    ("Adjust white balance.", "white_balance"), ("Increase contrast and saturation.", None),
])
def test_related_control_is_a_limited_unambiguous_navigation_hint(suggestion, control):
    assert related_control(suggestion) == control


def test_location_links_only_emit_the_referenced_control(application):
    output = CoachOutput()
    emitted = []
    summaries = []
    output.locateRequested.connect(emitted.append)
    output.summaryChanged.connect(summaries.append)
    output.show_feedback(feedback(next_step_or_stop="Increase exposure_ev slightly."))
    assert "Increase Exposure slightly" in output.toPlainText()
    assert summaries == ["State v2 · Increase Exposure slightly."]
    output.anchorClicked.emit(QUrl("photocoach:locate/saturation"))
    output.anchorClicked.emit(QUrl("https://example.invalid"))
    assert not emitted
    output.anchorClicked.emit(QUrl("photocoach:locate/exposure_ev"))
    assert emitted == ["exposure_ev"]
    output.setPlainText("Photo changed.")
    output.anchorClicked.emit(QUrl("photocoach:locate/exposure_ev"))
    assert emitted == ["exposure_ev"]
    output.deleteLater()


def test_reader_is_modeless_retains_size_and_follows_question_and_state(application):
    panel = CoachResponsePanel()
    assert not panel.reader_button.isEnabled() and not panel.copy_button.isEnabled()
    panel.open_reader()
    assert panel.reader.isHidden()
    question = "How should I adjust the curve?"
    panel.output.set_question_draft(question)
    panel.output.show_feedback(feedback(question=question, answer="Lift one midtone point."))
    panel.reader_button.click()
    application.processEvents()
    assert panel.reader.isVisible() and not panel.reader.isModal()
    assert "Lift one midtone point." in panel.reader_output.toPlainText()
    panel.reader.resize(520, 420)
    panel.reader.hide()
    panel.reader_button.click()
    assert panel.reader.size().width() == 520 and panel.reader.size().height() == 420
    panel.output.keep_as_reference("photo-a", "session-a", 3)
    panel.output.set_question_draft("A new question")
    assert "Current edit is v3" in panel.reader_output.toPlainText()
    assert "Question changed." in panel.reader_output.toPlainText()
    assert question in panel.reader_output.toPlainText()
    panel.output.setPlainText("Photo changed. Ask about the current edit.")
    assert panel.reader_output.toPlainText() == "Photo changed. Ask about the current edit."
    assert not panel.copy_button.isEnabled() and not panel.reader_copy.isEnabled()
    panel.reader.close()
    panel.deleteLater()


def test_reader_disclosure_scroll_and_navigation_are_local(application):
    panel = CoachResponsePanel()
    panel.output.show_feedback(feedback(answer="A long answer. " * 120, next_step_or_stop="Use the RGB curve."))
    panel.reader_button.click()
    application.processEvents()
    panel.reader_sections[2].click()
    assert panel.reader_output.verticalScrollBar().value() > 0
    panel.reader_output.anchorClicked.emit(QUrl("photocoach:evidence"))
    assert panel.reader_output.evidence_expanded and not panel.output.evidence_expanded
    assert "Observed: Dark rocks." in panel.reader_output.toPlainText()
    scrollbar = panel.reader_output.verticalScrollBar()
    scrollbar.setValue(scrollbar.maximum() // 2)
    position = scrollbar.value()
    panel.output.keep_as_reference("photo-a", "session-a", 8)
    assert panel.reader_output.evidence_expanded and scrollbar.value() == position
    controls = []
    panel.locateRequested.connect(controls.append)
    panel.reader_output.anchorClicked.emit(QUrl("https://example.invalid"))
    panel.reader_output.anchorClicked.emit(QUrl("photocoach:locate/saturation"))
    assert controls == [] and panel.reader.isVisible()
    panel.reader_output.anchorClicked.emit(QUrl("photocoach:locate/rgb_curve"))
    assert controls == ["rgb_curve"] and panel.reader.isHidden()
    panel.output.show_feedback(feedback(answer="New response."))
    panel.reader_button.click()
    assert "New response." in panel.reader_output.toPlainText()
    assert not panel.reader_output.evidence_expanded
    assert panel.reader_output.verticalScrollBar().value() == 0
    panel.deleteLater()


def test_hidden_reader_does_not_relayout_during_edits(application, monkeypatch):
    panel = CoachResponsePanel()
    renders = []
    original = panel.reader_output._render_feedback
    monkeypatch.setattr(panel.reader_output, "_render_feedback", lambda: (renders.append(True), original()))
    panel.output.show_feedback(feedback())
    for revision in range(3, 13):
        panel.output.keep_as_reference("photo-a", "session-a", revision)
    assert not renders
    panel.open_reader()
    assert renders and "Current edit is v12" in panel.reader_output.toPlainText()
    panel.reader.close()
    panel.deleteLater()


def test_closing_main_window_closes_its_reader(application, tmp_path):
    window = MainWindow(settings_store=SettingsStore(tmp_path / "settings.json"), credentials=NoVault())
    window.show()
    window.coach_text.show_feedback(feedback())
    window.coach_response.open_reader()
    application.processEvents()
    assert window.coach_response.reader.isVisible()
    window._close_approved = True
    window.close()
    application.processEvents()
    assert not window.coach_response.reader.isVisible()
    window.deleteLater()


def test_copy_includes_bound_question_hidden_evidence_and_reference_warning(application, monkeypatch):
    panel = CoachResponsePanel()
    copies = []
    class Clipboard:
        setText = staticmethod(copies.append)
    monkeypatch.setattr(QApplication, "clipboard", lambda: Clipboard())
    panel.output.set_question_draft("Original question")
    panel.output.show_feedback(feedback(question="Original question", answer='<img src="https://example.invalid">'))
    panel.output.keep_as_reference("photo-a", "session-a", 9)
    panel.output.set_question_draft("New draft")
    panel.copy_button.click()
    assert len(copies) == 1
    assert "Current edit is v9" in copies[0] and "Question changed." in copies[0]
    assert "Question sent: Original question" in copies[0] and "New draft" not in copies[0]
    assert "Observed: Dark rocks." in copies[0] and "WHAT CHANGED" in copies[0]
    assert '<img src="https://example.invalid">' in copies[0]  # Literal clipboard text.
    assert not panel.output.evidence_expanded
    assert panel.copy_button.text() == "Copied"
    panel.output.keep_as_reference("photo-a", "session-a", 10)
    assert panel.copy_button.text() == "Copy response"
    panel.output.setPlainText("Photo changed.")
    panel.copy_response()
    assert len(copies) == 1
    panel.deleteLater()


def test_reader_navigation_and_question_shortcut_do_not_edit_or_send(application, tmp_path, monkeypatch):
    window = MainWindow(settings_store=SettingsStore(tmp_path / "settings.json"), credentials=NoVault())
    state = window.store.activate(Path("synthetic.ARW"), "photo-a")
    window._set_controls(True)
    monkeypatch.setattr(window, "request_preview", lambda: None)
    window.show()
    window.coach_collapse.click()
    window.goal.setText("Natural light")
    window.question.setText("My original draft")
    application.processEvents()
    events = list(window.store.events)
    size = window.workspace_splitter.sizes()
    window.coach_question_shortcuts[1].click()
    assert "shared RGB curve" in window.question.text() and window.question.hasFocus()
    QTest.keyClick(window.question, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert window.question.text() == "My original draft"
    window.coach_text.show_feedback(feedback(photo_id=state.photo_id, session_id=state.session_id,
        state_version=state.revision, next_step_or_stop="Use the RGB curve."))
    window.coach_response.reader_button.click()
    application.processEvents()
    assert window.workspace_splitter.sizes() == size
    window.coach_response.reader_output.anchorClicked.emit(QUrl("photocoach:locate/rgb_curve"))
    application.processEvents()
    assert window.curve_section.toggle.isChecked() and window.tabs.currentIndex() == 0
    assert window.goal.text() == "Natural light"
    assert window.store.current is state and window.store.events == events
    assert window.api_budget.attempts == window.coach.calls == 0
    window._invalidate_coach("Photo changed.")
    assert window.coach_response.reader_output.toPlainText() == "Photo changed."
    window._close_approved = True
    window.close()
    window.deleteLater()


def test_success_without_prices_is_unknown_cost_not_failed():
    budget = SessionBudget(call_limit=5)
    ticket = budget.reserve()
    budget.settle(ticket, input_tokens=100, output_tokens=20, success=True)
    assert "1/5" in usage_summary(budget, 0, True)
    assert "Cost unknown" in usage_summary(budget, 0, True)
    text = budget_text(budget)
    assert "Requests with unknown cost: 1" in text
    assert "including successful requests without pricing" in text
    assert "Unresolved attempts" not in text


def test_wrong_official_provider_is_blocked_before_saving(application, tmp_path, monkeypatch):
    storage = SettingsStore(tmp_path / "settings.json")
    dialog = ApiSettingsDialog(None, None, SessionBudget(), settings_store=storage, credentials=NoVault())
    dialog.model.setText("gemini-example")
    assert not dialog.connection_issue.isHidden()
    messages = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, text: messages.append(text))
    dialog.validate_and_accept()
    assert messages and "pointing to OpenAI" in messages[0]
    assert dialog.result_settings is None and storage.load() is None
    dialog.protocol.setCurrentIndex(dialog.protocol.findData("gemini"))
    assert dialog.connection_issue.isHidden()
    assert dialog.endpoint.text() == GEMINI_ENDPOINT
    dialog.reject()
    dialog.deleteLater()


def test_connection_details_keep_custom_provider_and_values(application, tmp_path):
    config = ApiConfig("https://example.invalid/v1/chat/completions", "gemini-example", "chat_completions", timeout_seconds=75)
    settings = ApiSettings(config, {"call_limit": 3}, api_key="fake-test-key")
    storage = SettingsStore(tmp_path / "settings.json")
    dialog = ApiSettingsDialog(None, settings, SessionBudget(), settings_store=storage, credentials=NoVault())
    assert dialog.connection_details.isHidden() and dialog.connection_issue.isHidden()
    dialog.connection_details_button.click()
    assert not dialog.connection_details.isHidden()
    dialog.connection_details_button.click()
    dialog.validate_and_accept()
    assert dialog.result_settings.config == config
    assert dialog.result_settings.budget_settings["call_limit"] == 3
    assert dialog.result_settings.api_key == "fake-test-key"
    dialog.deleteLater()


def test_switching_known_interfaces_updates_default_endpoint_only(application, tmp_path):
    dialog = ApiSettingsDialog(None, None, SessionBudget(),
        settings_store=SettingsStore(tmp_path / "settings.json"), credentials=NoVault())
    dialog.protocol.setCurrentIndex(dialog.protocol.findData("chat_completions"))
    assert dialog.endpoint.text() == "https://api.openai.com/v1/chat/completions"
    dialog.protocol.setCurrentIndex(dialog.protocol.findData("responses"))
    assert dialog.endpoint.text() == "https://api.openai.com/v1/responses"
    dialog.endpoint.setText("https://example.invalid/custom")
    dialog.protocol.setCurrentIndex(dialog.protocol.findData("chat_completions"))
    assert dialog.endpoint.text() == "https://example.invalid/custom"
    dialog.reject()
    dialog.deleteLater()
