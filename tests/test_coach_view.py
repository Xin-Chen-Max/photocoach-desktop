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
from photocoach.coach_view import (
    CoachOutput,
    CoachResponsePanel,
    _display_text,
    agent_trace_summary,
    format_agent_step_line,
    format_live_agent_step,
    related_control,
    usage_summary,
)
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
    assert text.index("Suggestion") < text.index("Check after editing")
    assert "No changes yet." not in text and "Parameters are neutral." not in text
    assert payload in text  # Provider markup is literal text, never an image or link.
    assert "tracker.png&quot;" in output.toHtml()
    assert '<img src="https://example.invalid' not in output.toHtml()
    assert "Observed: Dark rocks." not in text
    output.anchorClicked.emit(QUrl("https://example.invalid"))
    assert not output.evidence_expanded
    output.anchorClicked.emit(QUrl("photocoach:evidence"))
    assert output.evidence_expanded
    text = output.toPlainText()
    assert text.index("Parameters are neutral.") < text.index("No changes yet.")
    assert all(item in output.toPlainText() for item in feedback().evidence)
    output.anchorClicked.emit(QUrl("photocoach:evidence"))
    assert "Observed: Dark rocks." not in output.toPlainText()
    output.deleteLater()


def test_direct_answer_and_original_question_are_visible_and_escaped(application):
    output = CoachOutput()
    question = '具体怎么调整rgb？ <img src="https://example.invalid/question">'
    answer = '先区分 RGB 总曲线和白平衡增益。\n<img src="https://example.invalid/answer">是文本。'
    steps = "1. 在 RGB curve 双击添加控制点。\n2. 拖动中间点；用 Undo 比较。"
    output.set_question_draft(question)
    output.show_feedback(feedback(question=question, answer=answer, next_step_or_stop=steps))
    text = output.toPlainText()
    assert text.index(question) < text.index(answer) < text.index("Suggestion")
    assert steps in text and "Evidence and reasoning" in text
    assert "Parameters are neutral." not in text and not output.evidence_expanded
    assert '<img src="https://example.invalid' not in output.toHtml()
    output.set_question_draft("接下来怎么做？")
    text = output.toPlainText()
    assert "Edit or question changed." in text and question in text
    assert "接下来怎么做" not in text  # Keep the request's question, not the new draft, beside this answer.
    output.set_question_draft(question)
    assert "Edit or question changed." in output.toPlainText()
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
    window.question.setText("具体怎么调整rgb？")
    assert window.store.current is state and window.store.events == events
    assert window._coach_generation > token
    assert "Edit or question changed." in window.coach_text.toPlainText()
    assert "How should I use RGB curve?" in window.coach_text.toPlainText()
    assert "changed" in window.coach_text.toPlainText()
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
    window.coach_usage_action.trigger()
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
    assert "Increase exposure" in window.coach_text.toPlainText()
    assert "New reply" in window.coach_collapse.text()
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
    cursor = window.question.textCursor()
    cursor.setPosition(8)
    window.question.setTextCursor(cursor)
    application.processEvents()
    sizes = window.workspace_splitter.sizes()
    window.coach_text.show_feedback(feedback())
    application.processEvents()
    assert window.question.hasFocus() and window.question.textCursor().position() == 8
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
    assert "Edit or question changed." in text
    assert "Current edit is v1" in window.coach_text.response_text()
    assert "not been rechecked" in window.coach_text.response_text()
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
    ("在 RGB 曲线中添加一个点。", "rgb_curve"), ("稍微增加曝光。", "exposure_ev"),
    ("调整白平衡。", "white_balance"), ("增加对比度和饱和度。", None),
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
    panel.reader_button.trigger()
    application.processEvents()
    assert panel.reader.isVisible() and not panel.reader.isModal()
    assert "Lift one midtone point." in panel.reader_output.toPlainText()
    panel.reader.resize(520, 420)
    panel.reader.hide()
    panel.reader_button.trigger()
    assert panel.reader.size().width() == 520 and panel.reader.size().height() == 420
    panel.output.keep_as_reference("photo-a", "session-a", 3)
    panel.output.set_question_draft("A new question")
    assert "Edit or question changed." in panel.reader_output.toPlainText()
    assert "Edit or question changed." in panel.reader_output.toPlainText()
    assert question in panel.reader_output.toPlainText()
    panel.output.setPlainText("Photo changed. Ask about the current edit.")
    assert panel.reader_output.toPlainText() == "Photo changed. Ask about the current edit."
    assert not panel.copy_button.isEnabled() and not panel.reader_copy.isEnabled()
    panel.reader.close()
    panel.deleteLater()


def test_reader_disclosure_scroll_and_navigation_are_local(application):
    panel = CoachResponsePanel()
    panel.output.show_feedback(feedback(answer="A long answer. " * 120, next_step_or_stop="Use the RGB curve."))
    panel.reader_button.trigger()
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
    panel.reader_button.trigger()
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
    assert renders and "Current edit is v12" in panel.reader_output.response_text()
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
    panel.copy_button.trigger()
    assert len(copies) == 1
    assert "Current edit is v9" in copies[0] and "Edit or question changed." in copies[0]
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
    window.coach_question_actions[1].trigger()
    assert "shared RGB curve" in window.question.text() and window.question.hasFocus()
    QTest.keyClick(window.question, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert window.question.text() == "My original draft"
    window.coach_text.show_feedback(feedback(photo_id=state.photo_id, session_id=state.session_id,
        state_version=state.revision, next_step_or_stop="Use the RGB curve."))
    window.coach_response.reader_button.trigger()
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


def test_display_text_collapses_duplicate_control_labels_and_keeps_notes():
    assert _display_text("Drag Exposure (exposure_ev) up to +2.20 EV") == "Drag Exposure up to +2.20 EV"
    assert _display_text("Drag Exposure (Exposure) up to +2.20 EV") == "Drag Exposure up to +2.20 EV"
    assert _display_text("Check White balance (white_balance) warmth") == "Check White balance warmth"
    assert _display_text("Use RGB curve (RGB curve) gently") == "Use RGB curve gently"
    assert _display_text("Adjust Exposure (about +0.3 EV)") == "Adjust Exposure (about +0.3 EV)"


def test_agent_v2_badge_visible_in_coach_output_and_trial_dialog(application):
    import numpy as np
    from photocoach.agent_runtime import TraceSpan
    from photocoach.coach_agent import AgentPlan, AgentReport, Finding

    spans = (
        TraceSpan(0, "observe", "context_ready", 0.0, 2.0, "ok"),
        TraceSpan(1, "reason", "turn_1_tool_plan", 2.0, 120.0, "ok"),
        TraceSpan(2, "tool_call", "inspect_region_100pct", 122.0, 45.0, "ok"),
        TraceSpan(3, "reason", "turn_2_tool_plan", 167.0, 110.0, "ok"),
        TraceSpan(4, "trial_render", "propose_and_preview_trial", 277.0, 35.0, "ok"),
        TraceSpan(5, "reflect", "trial_1_check", 312.0, 1.0, "blocked"),
        TraceSpan(6, "reason", "turn_3_tool_plan", 313.0, 115.0, "ok"),
        TraceSpan(7, "trial_render", "propose_and_preview_trial", 428.0, 32.0, "ok"),
        TraceSpan(8, "reflect", "trial_2_check", 460.0, 1.0, "ok"),
        TraceSpan(9, "reason", "turn_4_final", 461.0, 95.0, "ok"),
        TraceSpan(10, "finalize", "ready_for_user", 556.0, 1.0, "ok"),
    )
    summary_en = agent_trace_summary(spans, zh=False)
    assert "Agent v2" in summary_en and "repaired" in summary_en
    assert agent_trace_summary((), zh=False) == ""

    preview = np.full((32, 48, 3), 90, dtype=np.uint8)
    plan = AgentPlan(
        "recommend", "Lift underexposed seals carefully.",
        (Finding("center", "Midtones are dim.", "Lift exposure safely.", "technical", "medium"),),
        (), "Noise increases when lifting shadows.",
    )
    report = AgentReport("review", True, (), (("Trial checked.", "试调已校验。"),), ("observe", "trial"), preview, preview)

    panel = CoachResponsePanel()
    panel.output.show_feedback(feedback(
        next_step_or_stop="1. Drag Exposure (exposure_ev) up to +2.20 EV.",
        agent_plan=plan,
        agent_report=report,
        agent_trace=spans,
    ))
    text_en = panel.output.toPlainText()
    assert not panel.output.evidence_expanded
    assert "Agent v2" in text_en
    assert text_en.index("Agent v2") < text_en.index("Suggestion")
    assert "Drag Exposure up to +2.20 EV" in text_en
    assert "Exposure (Exposure)" not in text_en

    panel.open_trial()
    assert "Agent v2" in panel.trial_report.toPlainText()

    panel.set_language("zh")
    text_zh = panel.output.toPlainText()
    assert "智能体 v2" in text_zh and "已自修复" in text_zh
    assert "智能体 v2" in panel.trial_report.toPlainText()

    panel.output.show_feedback(feedback(agent_trace=()))
    assert "Agent v2" not in panel.output.toPlainText()
    panel.trial.hide()
    panel.deleteLater()


def test_agent_trace_tree_collapses_and_expands_on_click_in_en_and_zh(application):
    from photocoach.agent_runtime import TraceSpan

    spans = (
        TraceSpan(0, "observe", "context_ready", 0.0, 2.0, "ok", detail={"tools": 6}),
        TraceSpan(1, "reason", "turn_1_tool_plan", 2.0, 120.0, "ok", detail={"calls": ["inspect_region_100pct"]}),
        TraceSpan(2, "tool_call", "inspect_region_100pct", 122.0, 45.0, "ok", detail={"x_norm": 0.48, "y_norm": 0.52}),
        TraceSpan(3, "reason", "turn_2_tool_plan", 167.0, 110.0, "ok", detail={"calls": ["propose_and_preview_trial"]}),
        TraceSpan(4, "trial_render", "propose_and_preview_trial", 277.0, 35.0, "ok", detail={"params": ["exposure_ev", "shadows"]}),
        TraceSpan(5, "reflect", "trial_1_check", 312.0, 1.0, "ok", detail={"passed": True, "needs_repair": False}),
        TraceSpan(6, "reason", "turn_3_final", 313.0, 95.0, "ok"),
        TraceSpan(7, "finalize", "ready_for_user", 408.0, 2.0, "ok"),
    )
    assert "Inspected 100% native RAW crop at (0.48, 0.52)" in format_agent_step_line(spans[2], zh=False)
    assert "查验 100% 原生局部细节 (0.48, 0.52)" in format_agent_step_line(spans[2], zh=True)

    panel = CoachResponsePanel()
    panel.output.show_feedback(feedback(agent_trace=spans))
    assert not panel.output.trace_expanded
    collapsed_en = panel.output.toPlainText()
    assert "▸" in collapsed_en and "Worked for" in collapsed_en
    assert "Inspected 100% native RAW crop" not in collapsed_en

    # Click disclosure link to expand hierarchical timeline
    panel.output.anchorClicked.emit(QUrl("photocoach:trace"))
    assert panel.output.trace_expanded
    expanded_en = panel.output.toPlainText()
    assert "▾" in expanded_en
    assert "Prepared photo previews & editing context" in expanded_en
    assert "Inspected 100% native RAW crop at (0.48, 0.52)" in expanded_en
    assert "Rendered sandbox trial preview (Exposure, Shadows)" in expanded_en
    assert "Verified trial quality checks · Passed" in expanded_en
    assert "Finalized verified coaching advice" in expanded_en
    assert expanded_en.index("Finalized verified coaching advice") < expanded_en.index("Suggestion")

    # Language switch updates expanded steps in place
    panel.set_language("zh")
    expanded_zh = panel.output.toPlainText()
    assert "耗时" in expanded_zh
    assert "查验 100% 原生局部细节 (0.48, 0.52)" in expanded_zh
    assert "渲染沙盒试调预览（Exposure, Shadows）" in expanded_zh
    assert "核验试调质量指标 · 通过" in expanded_zh

    # Clicking again collapses cleanly
    panel.output.anchorClicked.emit(QUrl("photocoach:trace"))
    assert not panel.output.trace_expanded
    assert "查验 100% 原生局部细节" not in panel.output.toPlainText()
    panel.deleteLater()


def test_live_progress_renders_completed_spans_and_active_step_in_main_window(application, tmp_path):
    from photocoach.agent_runtime import AgentProgress, TraceSpan
    from photocoach.app import Job
    from photocoach.api_coach import ApiConfig, GEMINI_ENDPOINT
    from photocoach.settings import ApiSettings
    import threading

    active_step, hint = format_live_agent_step(2, 3, "propose_and_preview_trial", "exposure_ev, shadows", "waiting", zh=False)
    assert "Turn 2/3 · Running sandbox trial (Exposure, Shadows)" in active_step
    assert "Executing local diagnostic tool" in hint

    window = MainWindow(settings_store=SettingsStore(tmp_path / "settings.json"), credentials=NoVault())
    state = window.store.activate(Path("synthetic.ARW"), "photo-a")
    window._set_controls(True)
    window.api_settings = ApiSettings(ApiConfig(GEMINI_ENDPOINT, "gemini-3.5-flash", "gemini"), {}, key_source="saved")
    window.coach_mode.setCurrentIndex(1)

    progress = AgentProgress(max_turns=3)
    progress.advance("waiting")
    progress.record_spans((
        TraceSpan(0, "observe", "context_ready", 0.0, 2.0, "ok"),
        TraceSpan(1, "reason", "turn_1_tool_plan", 2.0, 90.0, "ok", detail={"calls": ["inspect_region_100pct"]}),
        TraceSpan(2, "tool_call", "inspect_region_100pct", 92.0, 42.0, "ok", detail={"x_norm": 0.45, "y_norm": 0.55}),
    ))
    progress.set_agent_step(2, 3, "propose_and_preview_trial", detail="exposure_ev, shadows")
    cancel = threading.Event()
    job = Job("api_prepare", window._generation, lambda: None, state, window._coach_generation)
    window._network_active = (None, job, cancel, None)
    window._network_progress = (progress, 60)

    window._update_network_progress()
    live_text = window.coach_text.toPlainText()
    assert "Analyzing selected previews" in live_text
    assert "Inspected 100% native RAW crop at (0.45, 0.55)" in live_text
    assert "Turn 2/3 · Running sandbox trial (Exposure, Shadows)" in live_text

    window._network_active = window._network_progress = None
    window._close_approved = True
    window.close()
    window.deleteLater()


def test_trial_compare_widget_and_apply_anyway_override_with_single_undo(application, tmp_path):
    import numpy as np
    from photocoach.coach_actions import resolve_actions
    from photocoach.coach_agent import AgentPlan, AgentReport, Finding, VerificationCheck
    from photocoach.coach_view import TrialCompareWidget

    # 1. TrialCompareWidget supports split/hold/side_by_side + zoom/pan and paints cleanly.
    widget = TrialCompareWidget()
    widget.resize(800, 500)
    before = np.full((64, 96, 3), 60, np.uint8)
    after = np.full((64, 96, 3), 140, np.uint8)
    widget.set_images(before, after, "Before", "Trial")
    for mode in ("split", "hold", "side_by_side"):
        widget.set_mode(mode)
        widget.set_zoom(2.0)
        widget.grab()
    widget.set_zoom(1.0)
    widget.deleteLater()

    # 2. Check-failed ("blocked") trial on recommend decision hides main Apply but enables Apply anyway in Preview trial.
    window = MainWindow(settings_store=SettingsStore(tmp_path / "settings.json"), credentials=NoVault())
    state = window.store.activate(Path("synthetic.ARW"), "photo-override")
    window._set_controls(True)
    actions = resolve_actions([{"operation": "add", "parameter": "exposure_ev", "value": 0.5}], state.values)
    plan = AgentPlan(
        decision="recommend",
        objective="Brighten the dark subject cleanly.",
        findings=(Finding("center", "Dim subject.", "Exposure +0.5 EV.", "technical", "high"),),
        checks=(VerificationCheck("center", "mean_luminance", "increase", "Raise subject brightness."),),
        uncertainty="Reduced preview only.",
    )
    blocked_report = AgentReport(
        status="blocked",
        can_apply=False,
        notices=(("Held for ceiling warning.", "请核对高光提示。"),),
        current_preview=before,
        trial_preview=after,
    )
    fb = CoachFeedback(
        photo_id=state.photo_id,
        session_id=state.session_id,
        state_version=state.revision,
        main_change="Exposure +0.5 EV",
        reason="Brightens dark subject.",
        next_step_or_stop="Compare in Preview trial.",
        observe_and_stop_when="Check highlight separation.",
        evidence=("Observed: dim.", "Inference: lift.", "Preference: natural."),
        actions=actions,
        agent_plan=plan,
        agent_report=blocked_report,
    )
    window.coach_text.show_feedback(fb)
    assert not window.coach_response.apply_button.isEnabled()
    assert window.coach_response.trial_button.isEnabled()
    window.coach_response.open_trial()
    assert window.coach_response.trial_apply.isVisible() and window.coach_response.trial_apply.isEnabled()
    assert "Apply anyway" in window.coach_response.trial_apply.text()

    window.coach_response.trial_apply.click()
    assert not window.coach_response.trial.isVisible()
    assert window.store.current.values.exposure_ev == 0.5
    assert len(window.store.undo_stack) == 1
    window.undo()
    assert window.store.current.values.exposure_ev == 0.0

    # 3. Stop / inspect / policy-blocked outputs must NEVER offer Apply anyway override.
    for blocked_decision, notice in (("inspect", ""), ("recommend", "Policy blocked.")):
        bad_plan = replace(plan, decision=blocked_decision)
        bad_fb = replace(fb, state_version=window.store.current.revision, agent_plan=bad_plan, action_notice=notice)
        window.coach_text.show_feedback(bad_fb)
        window.coach_response.open_trial()
        assert not window.coach_response.trial_apply.isVisible()
        assert window._apply_coach_suggestion(0, allow_previewed_override=True) is False
        window.coach_response.trial.hide()

    window._close_approved = True
    window.close()
    window.deleteLater()


def test_send_review_dialog_zero_jitter_on_turns_and_details(application):
    import numpy as np
    from photocoach.coach import CoachRequest
    from photocoach.coach_ui import SendReviewDialog
    from photocoach.state import EDIT_PIPELINE_VERSION, EditValues

    px = np.zeros((120, 160, 3), dtype=np.uint8)
    ev = EditValues().to_dict()
    req = CoachRequest(
        request_id="r1",
        photo_id="p1",
        session_id="s1",
        state_version=1,
        before_state_version=0,
        pipeline_version=EDIT_PIPELINE_VERSION,
        goal="",
        question="Fix lighting",
        parameters=ev,
        before_parameters=ev,
        recent_operations=(),
        before_preview=px,
        current_preview=px,
    )
    cfg = ApiConfig("http://127.0.0.1:11435/api/chat", "qwen3.6:27b", "ollama", 300, 0)
    dialog = SendReviewDialog(None, req, cfg, SessionBudget(call_limit=100))
    dialog.show()
    for _ in range(3):
        application.processEvents()

    for width in (480, 520, 600, 720):
        dialog.resize(width, 540)
        for _ in range(3):
            application.processEvents()
        baseline_vp = dialog.body_scroll.viewport().width()
        baseline_summary_h = dialog._summary_label.height()
        baseline_spin = dialog.agent_max_turns.geometry()
        baseline_details_btn = dialog.details_button.geometry()
        for turns in (1, 2, 6, 9, 10, 12):
            dialog.agent_max_turns.setValue(turns)
            for _ in range(2):
                application.processEvents()
            assert dialog.body_scroll.viewport().width() == baseline_vp
            assert dialog._summary_label.height() == baseline_summary_h
            assert dialog.agent_max_turns.geometry() == baseline_spin
            assert dialog.details_button.geometry() == baseline_details_btn
        dialog.agent_loop_enabled.click()
        for _ in range(2):
            application.processEvents()
        assert dialog.body_scroll.viewport().width() == baseline_vp
        assert dialog._summary_label.height() == baseline_summary_h
        assert dialog.agent_max_turns.geometry() == baseline_spin
        assert dialog.details_button.geometry() == baseline_details_btn
        dialog.agent_loop_enabled.click()
        for _ in range(2):
            application.processEvents()
        dialog.details_button.click()
        for _ in range(2):
            application.processEvents()
        assert dialog.body_scroll.viewport().width() == baseline_vp
        assert dialog._summary_label.height() == baseline_summary_h
        assert dialog.agent_max_turns.geometry() == baseline_spin
        assert dialog.details_button.geometry() == baseline_details_btn
        dialog.details_button.click()
        for _ in range(2):
            application.processEvents()

    dialog.reject()
    dialog.deleteLater()
