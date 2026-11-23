"""Tests for Roadmap #8: Coach Follow-Up Review ("Did my adjustment help?")."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication, QDialog

from photocoach import api_coach as api
from photocoach import app as desktop
from photocoach.agent_memory import (
    PlaybookRAG,
    SessionWorkingMemory,
    TurnSummary,
    summarize_parameter_deltas,
)
from photocoach.agent_runtime import AgentRuntime, RunBudget
from photocoach.agent_tools import build_default_tool_registry
from photocoach.app import MainWindow
from photocoach.budget import SessionBudget
from photocoach.coach import CoachFeedback, MockCoach, build_request, validate_follow_up_context
from photocoach.coach_actions import resolve_actions
from photocoach.coach_ui import SendReviewDialog
from photocoach.color_grading import ColorGrading, ColorWheel
from photocoach.composition import Composition
from photocoach.detail import DetailControls
from photocoach.gradients import LinearGradient
from photocoach.lens import LensCorrections
from photocoach.settings import ApiSettings, SettingsStore
from photocoach.spots import SpotRemoval
from photocoach.state import EditValues, StateStore


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def _stop_workflow(objective: str = "Evaluate adjustment impact.") -> dict:
    return {
        "version": 1,
        "decision": "stop",
        "objective": objective,
        "findings": [
            {
                "region": "whole",
                "observation": "Mean luminance rose smoothly without ceiling clipping.",
                "interpretation": "The lift moved tones toward the goal without degrading highlights.",
                "category": "technical",
                "confidence": "medium",
            }
        ],
        "checks": [
            {
                "region": "whole",
                "metric": "visual",
                "direction": "visual",
                "expectation": "Confirm subject contrast matches your preference.",
            }
        ],
        "uncertainty": "Reduced preview cannot verify full-resolution noise.",
    }


def test_working_memory_follow_up_lifecycle_undo_redo_and_photo_isolation():
    store = StateStore()
    s0 = store.activate(Path("E:/private/photo-1.ARW"), "hash-photo-1")
    store.set_exposure(0.25)
    s1 = store.current
    assert s1 is not None

    mem = SessionWorkingMemory()
    mem.record(
        s1.photo_id,
        s1.session_id,
        TurnSummary(
            state_version=s1.revision,
            question="Brighten foreground?",
            decision="recommend",
            objective="Lift shadows slightly and watch bright sky.",
            tools_used=("propose_and_preview_trial",),
            action_label="Exposure +0.60 EV",
            applied=False,
            parameters=s1.values.to_dict(),
            proposed_actions=("Exposure +0.60 EV",),
            next_step="Raise Exposure toward +0.60 EV.",
            watch_stop="Stop before highlights clip.",
            goal=s1.goal,
        ),
    )
    # Right after advice, no subsequent edit exists yet.
    assert mem.has_subsequent_edits(s1.photo_id, s1.session_id, s1.revision, store.project_events(), current_values=s1.values) is False

    # Pure display comparisons (Baseline / A/B Snapshot) must NOT count as subsequent edits.
    snap = store.save_snapshot("Before trial")
    store.set_compare(True)
    store.set_compare_snapshot(snap.id)
    store.set_compare_snapshot(None)
    assert mem.has_subsequent_edits(
        s1.photo_id, s1.session_id, store.current.revision, store.project_events(), current_values=store.current.values
    ) is False

    # Actual gesture edit -> undo -> redo -> snapshot_restore -> follow-up context
    store.begin_drag("exposure_ev")
    store.set_exposure(0.60)
    store.end_drag()
    mem.mark_last_applied("Exposure +0.60 EV", photo_id=s1.photo_id, session_id=s1.session_id)
    assert mem.has_subsequent_edits(
        s1.photo_id, s1.session_id, store.current.revision, store.project_events(), current_values=store.current.values
    ) is True

    store.undo()  # back to 0.25 EV
    reverted_ctx = mem.follow_up_context(store.project_events(), store.current)
    assert reverted_ctx is not None
    assert reverted_ctx["undo_count"] == 1
    assert reverted_ctx["redo_count"] == 0
    assert reverted_ctx["parameter_deltas"] == []
    assert reverted_ctx["parameters_changed_since_advice"] is False
    assert reverted_ctx["reverted_to_prior_parameters"] is True

    store.redo()  # back to 0.60 EV
    store.set_parameter("contrast", 0.20)
    ctx = mem.follow_up_context(store.project_events(), store.current)
    assert ctx is not None
    assert ctx["prior_advice"]["state_version"] == s1.revision
    assert ctx["prior_advice"]["applied"] is True
    assert ctx["current_state_version"] == store.current.revision
    assert ctx["undo_count"] == 1 and ctx["redo_count"] == 1 and ctx["reset_count"] == 0
    assert ctx["parameters_changed_since_advice"] is True
    assert ctx["reverted_to_prior_parameters"] is False
    delta_params = {d["parameter"]: d for d in ctx["parameter_deltas"]}
    assert delta_params["exposure_ev"]["delta"] == pytest.approx(0.35)
    assert delta_params["contrast"]["after"] == pytest.approx(0.20)
    assert any(a["type"] == "undo" for a in ctx["subsequent_actions"])
    assert any(a["type"] == "redo" for a in ctx["subsequent_actions"])

    # Photo switch or new session on same photo isolates working memory completely.
    s_new_session = store.activate(Path("E:/private/photo-1.ARW"), "hash-photo-1")
    assert mem.last_advice_for(s_new_session.photo_id, s_new_session.session_id) is None
    assert mem.follow_up_context(store.project_events(), s_new_session) is None
    mem.mark_last_applied("Should not touch old binding", photo_id=s_new_session.photo_id, session_id=s_new_session.session_id)
    assert len(mem.entries()) == 1  # untouched until new photo records

    s_other = store.activate(Path("E:/private/photo-2.ARW"), "hash-photo-2")
    assert mem.follow_up_context(store.project_events(), s_other) is None


def test_summarize_parameter_deltas_covers_complex_tools_and_caps_at_16():
    before = EditValues()
    after = EditValues(
        exposure_ev=0.5,
        contrast=0.2,
        highlights=-0.25,
        shadows=0.3,
        whites=0.1,
        blacks=-0.1,
        vibrance=0.15,
        saturation=1.1,
        base_rendering="natural_v1",
        composition=Composition((0.05, 0.05, 0.95, 0.95), 1.5),
        lens_corrections=LensCorrections(distortion=12.0, vignetting_ev=-0.5),
        detail=DetailControls(sharpening=40.0, masking=15.0, luminance_nr=10.0, chroma_nr=25.0),
        color_grading=ColorGrading(shadows=ColorWheel(210.0, 0.3, 0.0), blend=0.65),
        local_gradients=(LinearGradient("g1", "Sky", (0.5, 0.0), (0.5, 0.5), exposure_ev=-0.4),),
        spots=(SpotRemoval("sp1", (0.3, 0.3), (0.4, 0.3), mode="heal"),),
        rgb_curve=((0.0, 0.0), (0.5, 0.55), (1.0, 1.0)),
        goal="Crisp golden sunrise",
    )
    deltas = summarize_parameter_deltas(before.to_dict(), after.to_dict())
    assert len(deltas) == 16
    assert all("photo_id" not in d and "source_path" not in d for d in deltas)


def test_validate_follow_up_context_rejects_bad_versions_nonfinite_and_forbidden_keys():
    store = StateStore()
    s0 = store.activate(Path("E:/secret/raw.ARW"), "secret-photo-id")
    store.set_exposure(0.4)
    mem = SessionWorkingMemory()
    mem.record(s0.photo_id, s0.session_id, TurnSummary(0, "Q", "recommend", "Obj", parameters=s0.values.to_dict()))
    valid = mem.follow_up_context(store.project_events(), store.current)
    assert valid is not None

    with pytest.raises(ValueError):
        validate_follow_up_context(valid, state_version=99)

    bad_prior = json.loads(json.dumps(valid))
    bad_prior["prior_advice"]["state_version"] = 5
    with pytest.raises(ValueError):
        validate_follow_up_context(bad_prior, state_version=store.current.revision)

    bad_key = json.loads(json.dumps(valid))
    bad_key["parameter_deltas"].append({"source_path": "E:/secret/raw.ARW"})
    with pytest.raises(ValueError):
        validate_follow_up_context(bad_key, state_version=store.current.revision)

    bad_float = json.loads(json.dumps(valid))
    bad_float["parameter_deltas"][0]["delta"] = float("nan")
    with pytest.raises(ValueError):
        validate_follow_up_context(bad_float, state_version=store.current.revision)


@pytest.mark.parametrize("protocol", ["responses", "chat_completions", "gemini", "ollama"])
@pytest.mark.parametrize("include_before", [True, False])
def test_follow_up_payload_serialization_and_privacy_across_protocols(protocol, include_before):
    store = StateStore()
    s0 = store.activate(Path("E:/private-folder/7C204108.ARW"), "private-hash-123")
    mem = SessionWorkingMemory()
    mem.record(
        s0.photo_id,
        s0.session_id,
        TurnSummary(0, "Brighten foreground?", "recommend", "Lift shadows cleanly.", parameters=s0.values.to_dict()),
    )
    snap = store.save_snapshot("Initial")
    store.set_exposure(0.45)
    store.restore_snapshot(snap.id)
    store.set_exposure(0.45)
    follow_up = mem.follow_up_context(store.project_events(), store.current)
    assert follow_up is not None

    img_before = np.full((16, 24, 3), 70, dtype=np.uint8)
    img_curr = np.full((16, 24, 3), 110, dtype=np.uint8)
    req = build_request(
        store.current,
        s0,
        store.project_events(),
        img_before,
        img_curr,
        question="Did my adjustment help?",
        follow_up=follow_up,
    )

    endpoint = (
        "http://127.0.0.1:11435/api/chat"
        if protocol == "ollama"
        else (api.GEMINI_ENDPOINT if protocol == "gemini" else "https://api.openai.com/v1/responses")
    )
    model = "gemini-3.5-flash" if protocol == "gemini" else ("qwen3.6:27b" if protocol == "ollama" else "gpt-test")
    cfg = api.ApiConfig(endpoint, model, protocol)
    payload = api.build_payload(cfg, req, include_before=include_before)
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "private-folder" not in serialized and "private-hash-123" not in serialized and "7C204108.ARW" not in serialized

    ctx = api.request_context(req, include_before=include_before)
    assert ctx["follow_up"]["kind"] == "follow_up_review"
    assert ctx["follow_up"]["before_preview_included"] is include_before
    assert ctx["follow_up"]["prior_advice"]["state_version"] == 0
    assert any(op["type"] == "snapshot_restore" for op in ctx["recent_operations"])

    # Without follow_up, request_context preserves existing schema without a follow_up key.
    plain_req = build_request(store.current, s0, store.project_events(), img_before, img_curr, question="Analyze.")
    assert "follow_up" not in api.request_context(plain_req, include_before=include_before)

    # Tampered follow_up version fails cleanly with ApiError("input").
    bad_req = replace(req, follow_up={**follow_up, "current_state_version": 999})
    with pytest.raises(api.ApiError):
        api.request_context(bad_req)


def test_mock_coach_follow_up_verdicts_and_agent_runtime_integration():
    store = StateStore()
    s0 = store.activate(Path("synthetic.ARW"), "photo-mock")
    mem = SessionWorkingMemory()
    mem.record(s0.photo_id, s0.session_id, TurnSummary(0, "Too dark?", "recommend", "Lift exposure.", parameters=s0.values.to_dict()))
    store.set_exposure(0.45)

    dark = np.full((24, 36, 3), 65, dtype=np.uint8)
    bright = np.full((24, 36, 3), 120, dtype=np.uint8)
    clipped = np.full((24, 36, 3), 255, dtype=np.uint8)

    mock = MockCoach()
    # 1. Helped (EN)
    ctx_helped = mem.follow_up_context(store.project_events(), store.current)
    req_helped = build_request(store.current, s0, store.project_events(), dark, bright, "Did my adjustment help?", follow_up=ctx_helped)
    fb_helped = mock.analyze(req_helped)
    assert fb_helped.actions == ()
    assert fb_helped.agent_plan is not None and fb_helped.agent_plan.decision == "stop"
    assert "Helped" in fb_helped.answer and "preference" in fb_helped.answer.lower()
    assert fb_helped.follow_up_context == ctx_helped

    # 2. Overshoot / ceiling clipping spike (ZH)
    req_clip = build_request(store.current, s0, store.project_events(), dark, clipped, "这次调整有效吗？", follow_up=ctx_helped)
    fb_clip = mock.analyze(req_clip)
    assert "过度" in fb_clip.answer or "溢出" in fb_clip.answer
    assert fb_clip.agent_plan is not None and fb_clip.agent_plan.decision == "stop"

    # 3. Reverted via Undo -> Neutral
    store.undo()
    ctx_reverted = mem.follow_up_context(store.project_events(), store.current)
    req_reverted = build_request(store.current, s0, store.project_events(), dark, dark, "Did my adjustment help?", follow_up=ctx_reverted)
    fb_reverted = mock.analyze(req_reverted)
    assert "Neutral" in fb_reverted.answer and fb_reverted.agent_plan.decision == "stop"

    # 4. Simulated AgentRuntime multi-step follow-up run with PlaybookRAG
    rag = PlaybookRAG()
    hits = rag.search("Did my adjustment help? follow up review", top_k=2)
    assert any(h["id"] == "pb_follow_up_review" for h in hits)

    captured_payloads: list[dict] = []
    turn = {"n": 0}

    def transport(cfg, key, body_bytes, cancel):
        turn["n"] += 1
        captured_payloads.append(json.loads(body_bytes.decode("utf-8")))
        if turn["n"] == 1:
            return 200, json.dumps({
                "candidates": [{"finishReason": "STOP", "content": {"parts": [
                    {"functionCall": {"id": "c1", "name": "analyze_color_zones", "args": {"region": "whole"}}}
                ]}}]
            }).encode()
        ans = {
            "answer": "Helped objectively: mean luminance increased smoothly while highlight headroom stayed clean; confirm visually against your goal.",
            "main_change": "Exposure +0.45 EV since v0.",
            "reason": "The lift opened midtones without spiking display clipping.",
            "next_step_or_stop": "Stop here if the balanced midtone separation matches your preference.",
            "observe_and_stop_when": "Compare Edited vs Baseline; stop unless shadow noise appears at 100%.",
            "observed": "Mean luminance rose from ~0.053 to ~0.188 with 0% display ceiling.",
            "inferred": "Linear exposure lift cleanly separated midtones.",
            "preference": "Keeping this brighter mood depends on your goal.",
            "actions": [],
            "workflow": _stop_workflow("Verify exposure lift helped and decide whether to stop."),
        }
        return 200, json.dumps({"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(ans)}]}}]}).encode()

    budget = SessionBudget(call_limit=5)
    cfg = api.ApiConfig(api.GEMINI_ENDPOINT, "gemini-3.5-flash", "gemini")
    runtime = AgentRuntime(
        api.ApiCoach(cfg, "fake-key", transport),
        build_default_tool_registry(),
        run_budget=RunBudget(max_llm_turns=3),
        session_budget=budget,
        playbook=rag,
        working_memory_snippet=mem.prompt_snippet(),
    )
    res = runtime.run(req_helped, state=store.current, include_before=True)
    assert res.feedback.agent_plan is not None and res.feedback.agent_plan.decision == "stop"
    assert res.feedback.actions == ()
    assert res.feedback.follow_up_context == ctx_helped
    assert res.spans[0].detail.get("is_follow_up") is True
    assert budget.attempts == 2 and budget.in_flight == 0
    first_user_ctx = json.loads(captured_payloads[0]["contents"][0]["parts"][0]["text"])
    assert first_user_ctx["follow_up"]["prior_advice"]["state_version"] == 0
    assert any(c["id"] == "pb_follow_up_review" for c in first_user_ctx["playbook_hints"])


def test_main_window_follow_up_gating_no_auto_calls_and_bilingual_rendering(qt_app, tmp_path, monkeypatch):
    class NoVault:
        def read(self, *_):
            return "dummy-key"

    window = MainWindow(settings_store=SettingsStore(tmp_path / "settings.json"), credentials=NoVault(), dev_mode=True)
    monkeypatch.setattr(window, "request_preview", lambda **kwargs: None)
    monkeypatch.setattr(window, "request_detail", lambda: None)
    try:
        assert not window.followup_button.isEnabled()
        assert len(window.coach_question_actions) == 4
        assert window.coach_question_actions[3].text() == "Follow-up: Did it help?"

        window.store.activate(Path("E:/private/sample.ARW"), "photo-ui-1")
        window.preview.configure_photo((1200, 800))
        window.api_settings = ApiSettings(api.ApiConfig(api.GEMINI_ENDPOINT, "gemini-3.5-flash", "gemini"), {"call_limit": 8}, key_source="saved")
        window.coach_mode.setCurrentIndex(1)
        window._set_controls(True)
        window._sync_controls()
        window.coach_collapse.setChecked(False)
        qt_app.processEvents()

        # Still disabled before any advice is given.
        assert not window.followup_button.isEnabled()

        # Capture before-preview values used by PreviewEngine.coach_request
        captured_before_ev: list[float] = []
        captured_requests: list[object] = []

        def fake_coach_request(state, before, events, question, *, follow_up=None):
            captured_before_ev.append(before.exposure_ev)
            p_before = np.full((24, 36, 3), round(70 + before.exposure_ev * 40), dtype=np.uint8)
            p_curr = np.full((24, 36, 3), round(70 + state.exposure_ev * 40), dtype=np.uint8)
            req = build_request(state, before, events, p_before, p_curr, question, follow_up=follow_up)
            captured_requests.append(req)
            return req

        monkeypatch.setattr(window.engine, "coach_request", fake_coach_request)

        def fake_render_trial(state, values, *, source_size=None, cancel=None):
            return np.full((24, 36, 3), min(255, round(70 + values.exposure_ev * 40)), dtype=np.uint8)

        monkeypatch.setattr(window.engine, "render_trial_values", fake_render_trial)

        dialog_calls: list[object] = []

        class DialogController:
            next_accept = True

            def __init__(self, parent, request, config, budget):
                dialog_calls.append(request)
                self.real = SendReviewDialog(parent, request, config, budget)
                self.include_before = self.real.include_before
                self.skip_session = self.real.skip_session

            def selected_max_turns(self):
                return 1

            def exec(self):
                assert self.real.follow_up_note.isHidden() is (getattr(dialog_calls[-1], "follow_up", None) is None)
                return QDialog.DialogCode.Accepted if self.next_accept else QDialog.DialogCode.Rejected

            def deleteLater(self):
                self.real.deleteLater()

        monkeypatch.setattr(desktop, "SendReviewDialog", DialogController)

        # Turn 1: initial advice at v0 with an actionable Exposure +0.50 EV proposal
        def transport_round1(cfg, key, payload_bytes, cancel):
            ans = {
                "answer": "Midtones look dim; try +0.50 EV.",
                "main_change": "Neutral start.",
                "reason": "Opening midtones gently.",
                "next_step_or_stop": "1. Apply Exposure +0.50 EV.",
                "observe_and_stop_when": "Watch bright texture.",
                "observed": "Whole preview mean is dim.",
                "inferred": "Moderate lift may help.",
                "preference": "Brighter midtones match goal.",
                "actions": [{"operation": "set", "parameter": "exposure_ev", "value": 0.50}],
                "workflow": {
                    "version": 1,
                    "decision": "recommend",
                    "objective": "Lift midtones by +0.50 EV.",
                    "findings": [{"region": "whole", "observation": "Preview is dim.", "interpretation": "Try +0.50 EV.", "category": "technical", "confidence": "medium"}],
                    "checks": [{"region": "whole", "metric": "mean_luminance", "direction": "increase", "expectation": "Luminance rises."}],
                    "uncertainty": "Reduced preview only.",
                },
            }
            return 200, json.dumps({"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(ans)}]}}]}).encode()

        monkeypatch.setattr(desktop, "ApiCoach", lambda cfg, key: api.ApiCoach(cfg, key, transport_round1))
        window.question.setText("What next?")
        req1 = fake_coach_request(window.store.current, window.store.current, window.store.project_events(), window.question.text())
        window.review_api_request(req1, desktop.Job("api_prepare", window._generation, lambda: None, window.store.current, window._coach_generation))

        import time as _t
        deadline = _t.monotonic() + 10.0
        while (window._network_active is not None or window._active is not None or window.coach_text._feedback is None) and _t.monotonic() < deadline:
            qt_app.processEvents()
            _t.sleep(0.01)

        assert window.api_budget.attempts == 1
        assert len(window.agent_memory.entries()) == 1
        # Advice delivered at v0, no subsequent edit yet -> followup_button MUST stay disabled!
        assert not window.followup_button.isEnabled()

        # Apply the coach suggestion (+0.50 EV) and make another slider edit (+0.70 EV) -> zero automatic API calls!
        assert window._apply_coach_suggestion(0) is True
        assert window.store.current.exposure_ev == pytest.approx(0.50)
        window.change_exposure(70)
        assert window.store.current.exposure_ev == pytest.approx(0.70)
        assert window.api_budget.attempts == 1
        assert window.followup_button.isEnabled()

        # Clicking Common questions -> Follow-up inserts prompt only and NEVER sends automatically.
        window.coach_question_actions[3].trigger()
        assert "Did my adjustment help" in window.question.text()
        assert window._pending_follow_up is True
        assert window.api_budget.attempts == 1
        # Choosing another common question clears _pending_follow_up.
        window.coach_question_actions[0].trigger()
        assert window._pending_follow_up is False
        window.question.clear()

        # Cancelling SendReviewDialog during follow-up sends zero API calls.
        DialogController.next_accept = False
        window.ask_follow_up()
        deadline = _t.monotonic() + 5.0
        while (window._active is not None or window._reviewing) and _t.monotonic() < deadline:
            qt_app.processEvents()
            _t.sleep(0.01)
        assert window.api_budget.attempts == 1
        assert len(dialog_calls) == 2
        assert dialog_calls[-1].follow_up is not None
        assert captured_before_ev[-1] == pytest.approx(0.0)  # bound to prior v0 advice, not 0.50 EV!

        # Confirm follow-up request and verify bilingual rendering & Reader sync.
        DialogController.next_accept = True

        def transport_round2(cfg, key, payload_bytes, cancel):
            payload = json.loads(payload_bytes.decode("utf-8"))
            ctx_json = json.loads(payload["contents"][0]["parts"][0]["text"])
            assert ctx_json["follow_up"]["prior_advice"]["state_version"] == 0
            assert ctx_json["follow_up"]["prior_advice"]["applied"] is True
            assert any(d["parameter"] == "exposure_ev" for d in ctx_json["follow_up"]["parameter_deltas"])
            ans = {
                "answer": "Helped objectively: midtones lifted cleanly from v0 to v2 without clipping; stop here if balanced.",
                "main_change": "Exposure moved from +0.00 to +0.70 EV.",
                "reason": "The lift brightened shadows as intended.",
                "next_step_or_stop": "1. Compare Edited vs Baseline and stop.",
                "observe_and_stop_when": "Stop unless bright texture clips.",
                "observed": "Luminance rose cleanly across whole frame.",
                "inferred": "Exposure adjustment achieved the tonal lift.",
                "preference": "Final brightness remains your creative choice.",
                "actions": [],
                "workflow": _stop_workflow("Confirm +0.70 EV adjustment helped and stop."),
            }
            return 200, json.dumps({"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(ans)}]}}]}).encode()

        monkeypatch.setattr(desktop, "ApiCoach", lambda cfg, key: api.ApiCoach(cfg, key, transport_round2))
        window.ask_follow_up()
        deadline = _t.monotonic() + 10.0
        while (window._active is not None or window._network_active is not None or window.api_budget.attempts < 2) and _t.monotonic() < deadline:
            qt_app.processEvents()
            _t.sleep(0.01)
        # Wait until feedback finishes showing
        deadline = _t.monotonic() + 5.0
        while (window.coach_text._feedback is None or window.coach_text._feedback.follow_up_context is None) and _t.monotonic() < deadline:
            qt_app.processEvents()
            _t.sleep(0.01)

        assert window.api_budget.attempts == 2
        fb2 = window.coach_text._feedback
        assert fb2 is not None and fb2.follow_up_context is not None
        text_en = window.coach_text.toPlainText()
        assert "Follow-up review · v0 → v2" in text_en
        assert "Exposure" in text_en and "exposure_ev" not in text_en
        assert "Helped objectively" in text_en

        # Reader + clipboard + bilingual 中文 toggle
        window.coach_response.open_reader()
        qt_app.processEvents()
        assert "Follow-up review · v0 → v2" in window.coach_response.reader_output.toPlainText()
        window.set_language("zh")
        qt_app.processEvents()
        assert window.followup_button.text() == "效果复盘"
        assert window.coach_question_actions[3].text() == "效果复盘：调整有效吗？"
        text_zh = window.coach_text.toPlainText()
        assert "效果复盘 · v0 → v2" in text_zh
        assert "曝光 Exposure" in text_zh
        window.coach_text.anchorClicked.emit(QUrl("photocoach:evidence"))
        assert "上一轮建议 (v0)" in window.coach_text.toPlainText()
        assert "效果复盘 · v0 → v2" in window.coach_text.response_text()
        window.coach_response.reader.close()

        # Switching photos disables followup_button and clears prior session follow-up state.
        window.store.activate(Path("E:/private/other.ARW"), "photo-ui-2")
        window._after_edit()
        qt_app.processEvents()
        assert not window.followup_button.isEnabled()
        assert window.agent_memory.last_advice_for("photo-ui-2", window.store.current.session_id) is None

        # Dev MockCoach follow-up path works end-to-end inside MainWindow (zero network calls)
        window.coach_mode.setCurrentIndex(0)
        window._set_controls(True)
        window._sync_controls()
        window.question.clear()
        window.analyze()
        deadline = _t.monotonic() + 5.0
        while (window._active is not None or window.coach_text._feedback is None) and _t.monotonic() < deadline:
            qt_app.processEvents()
            _t.sleep(0.01)
        assert not window.followup_button.isEnabled()
        # Toggling Baseline must NOT enable followup_button or trigger any AI call.
        window.change_compare(True)
        window.change_compare(False)
        assert not window.followup_button.isEnabled()
        window.change_exposure(40)
        assert window.followup_button.isEnabled()
        window.ask_follow_up()
        deadline = _t.monotonic() + 5.0
        while (window._active is not None or window.coach_text._feedback is None or window.coach_text._feedback.follow_up_context is None) and _t.monotonic() < deadline:
            qt_app.processEvents()
            _t.sleep(0.01)
        assert window.coach_text._feedback.follow_up_context["prior_advice"]["state_version"] == 0
        assert window.api_budget.attempts == 2  # unchanged
    finally:
        window._network_active = window._active = None
        window._protected_jobs.clear()
        window._close_approved = True
        window.close()
        window.deleteLater()
        qt_app.processEvents()
