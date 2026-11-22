"""Evidence tools, one-trial verification and production/UI workflow boundaries."""
from dataclasses import replace
from concurrent.futures import CancelledError
import json
from pathlib import Path
import threading

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from shiboken6 import delete

from photocoach import api_coach as api
from photocoach.app import Job, MainWindow, PreviewEngine
from photocoach.coach import CoachFeedback, build_request
from photocoach.coach_actions import resolve_actions
from photocoach.coach_agent import check_trial, evidence_context, measure_preview, parse_plan
from photocoach.coach_view import CoachResponsePanel
from photocoach.imaging import DecodedPhoto, RawOptions, quantize_srgb
from photocoach.adjustments import render_edit
from photocoach.settings import SettingsStore
from photocoach.state import EditValues, StateStore


def workflow(**changes):
    data = dict(version=1, decision="recommend", objective="Lift the central subject while checking bright texture.",
        findings=[dict(region="center", observation="The central area is relatively dark.",
                       interpretation="A small lift is a reversible experiment, not a measured ideal exposure.",
                       category="technical", confidence="medium")],
        checks=[dict(region="center", metric="mean_luminance", direction="increase",
                     expectation="The central area becomes brighter.")],
        uncertainty="A reduced SDR preview cannot establish sensor headroom or aesthetic improvement.")
    return dict(data, **changes)


def setup_request():
    store = StateStore()
    store.activate(Path("private-photo.ARW"), "private-hash")
    pixels = np.full((96, 144, 3), 80, dtype=np.uint8)
    request = build_request(store.current, store.current, store.events, pixels, pixels,
                            source_size=(144, 96))
    return store, request


def feedback_for(request, *, plan=None, items=None):
    actions = resolve_actions(items if items is not None else [dict(operation="add", parameter="exposure_ev", value=.35)],
                              EditValues.from_dict(request.parameters),
                              request.source_size)
    return CoachFeedback(request.photo_id, request.session_id, request.state_version,
        "No edit has been applied.", "One reversible trial.", "Compare a small exposure lift.", "Undo if texture disappears.",
        ("Observed: the central area is relatively dark.",), answer="Compare the trial before deciding.",
        question=request.question, actions=actions, agent_plan=parse_plan(plan or workflow()))


def renderer(values):
    linear = np.full((96, 144, 3), .08, dtype=np.float32)
    return quantize_srgb(render_edit(linear, values))


def test_measurements_are_linear_display_evidence_with_correct_regions():
    image = np.full((9, 9, 3), 128, dtype=np.uint8)
    image[:3, :3] = [255, 0, 0]
    image[6:, 6:] = 0
    measured = measure_preview(image)
    assert measured["center"]["mean_luminance"] == pytest.approx(.21586, abs=1e-4)
    assert measured["top_left"]["mean_luminance"] == pytest.approx(.2126, abs=1e-5)
    assert measured["top_left"]["display_ceiling_fraction"] == 1
    assert measured["bottom_right"]["display_floor_fraction"] == 1
    assert measured["whole"]["display_floor_fraction"] == pytest.approx(1/9)
    assert "diagnosis" not in measured
    with pytest.raises(ValueError):
        measure_preview(image.astype(np.float32))


def test_current_only_context_does_not_measure_or_disclose_before_preview():
    _, request = setup_request()
    request = replace(request, before_preview=np.zeros((8, 8, 3), dtype=np.uint8))
    context = evidence_context(request, False)
    assert "before" not in context and "geometry_aligned" not in context
    assert "sensor" in " ".join(context["limitations"])
    assert evidence_context(request)["before"]["whole"]["display_floor_fraction"] == 1


@pytest.mark.parametrize("change", [dict(version=True), dict(version=2), dict(objective=""),
    dict(findings=[]), dict(checks=[]), dict(findings=[dict(region="face")]),
    dict(checks=[dict(region="whole", metric="visual", direction="increase", expectation="Wrong claim.")])])
def test_invalid_workflow_is_rejected(change):
    with pytest.raises(ValueError):
        parse_plan(workflow(**change))


def encoded_response(protocol, fields):
    answer = json.dumps(fields)
    if protocol == "responses":
        document = dict(status="completed", output=[dict(type="message", content=[dict(type="output_text", text=answer)])])
    elif protocol == "gemini":
        document = dict(candidates=[dict(finishReason="STOP", content=dict(parts=[dict(text=answer)]))])
    elif protocol == "ollama":
        document = dict(done=True, done_reason="stop", message=dict(role="assistant", content=answer))
    else:
        document = dict(choices=[dict(finish_reason="stop", message=dict(content=answer))])
    return json.dumps(document).encode()


@pytest.mark.parametrize("protocol", ["responses", "chat_completions", "gemini", "ollama"])
def test_workflow_contract_shared_by_all_transports_without_key_or_path_disclosure(protocol):
    _, request = setup_request()
    endpoint = "http://127.0.0.1:11435/api/chat" if protocol == "ollama" else (
        api.GEMINI_ENDPOINT if protocol == "gemini" else "https://api.openai.com/v1")
    model = "gemini-test-model" if protocol == "gemini" else "qwen3.6:27b" if protocol == "ollama" else "test-model"
    config = api.ApiConfig(endpoint, model, protocol)
    fields = {name: "A bounded photographic observation." for name in api.FEEDBACK_FIELDS}
    fields.update(actions=[dict(operation="add", parameter="exposure_ev", value=.35)], workflow=workflow())
    payload = api.build_payload(config, request, include_before=False)
    serialized = json.dumps(payload)
    assert 'application_evidence' in serialized
    assert 'private-photo' not in serialized and 'private-hash' not in serialized
    assert 'workflow' in serialized
    result = api._parse_response(config, request, encoded_response(protocol, fields))
    assert result.feedback.agent_plan.decision == "recommend"
    assert result.feedback.actions[0].new_value == .35
    assert result.feedback.agent_report is None  # Transport cannot claim local checks happened.


@pytest.mark.parametrize("decision", ["stop", "inspect", "clarify"])
def test_non_edit_decision_cannot_smuggle_an_action(decision):
    _, request = setup_request()
    fields = {name: "No change is justified." for name in api.FEEDBACK_FIELDS}
    fields.update(actions=[dict(operation="add", parameter="exposure_ev", value=.35)],
                  workflow=workflow(decision=decision))
    feedback = api._parse_response(api.ApiConfig(model="test"), request, encoded_response("responses", fields)).feedback
    assert not feedback.actions and feedback.action_notice


def test_legacy_reply_cannot_bypass_the_trial_contract():
    _, request = setup_request()
    fields = {name: "A legacy explanation remains readable." for name in api.FEEDBACK_FIELDS}
    fields["actions"] = [dict(operation="add", parameter="exposure_ev", value=.35)]
    feedback = api._parse_response(api.ApiConfig(model="test"), request, encoded_response("responses", fields)).feedback
    assert not feedback.actions and feedback.agent_plan is None
    assert "Legacy" in feedback.action_notice and "legacy explanation" in feedback.answer


@pytest.mark.parametrize("category,confidence", [("technical", "low"), ("preference", "high")])
def test_uncertainty_or_missing_style_goal_holds_the_proposal(category, confidence):
    _, request = setup_request()
    plan = workflow()
    plan["findings"][0].update(category=category, confidence=confidence)
    feedback = feedback_for(request, plan=plan)
    report = check_trial(request, feedback, lambda _: pytest.fail("Evidence policy must run before rendering")).agent_report
    assert not report.can_apply and report.status == "blocked"


def test_trial_is_one_transaction_no_state_change_and_no_mutable_images():
    store, request = setup_request()
    state, history = store.current, tuple(store.undo_stack)
    calls = []
    def render(values):
        calls.append(values)
        return renderer(values)
    feedback = check_trial(request, feedback_for(request), render)
    assert store.current == state and tuple(store.undo_stack) == history
    report = feedback.agent_report
    assert len(calls) == 2 and report.can_apply and report.status == "review"
    assert report.checks[0].verdict == "observed" and report.checks[0].after > report.checks[0].before
    assert report.trace[-1] == "await_user"
    assert not report.trial_preview.flags.writeable
    assert "not aesthetic" in " ".join(report.lines())


@pytest.mark.parametrize("mode", ["opposite", "no_change", "render_error", "detail", "geometry"])
def test_failed_unobservable_or_incomparable_trials_do_not_enable_apply(mode):
    _, request = setup_request()
    feedback = feedback_for(request)
    render = renderer
    if mode == "opposite":
        plan = workflow()
        plan["checks"][0]["direction"] = "decrease"
        feedback = feedback_for(request, plan=plan)
    elif mode == "no_change":
        render = lambda _: request.current_preview
    elif mode == "render_error":
        def render(_):
            raise MemoryError("Do not expose internal paths.")
    elif mode == "detail":
        feedback = feedback_for(request, items=[dict(operation="set", parameter="detail.sharpening", value=40)])
        render = lambda _: pytest.fail("Reduced previews must abstain on native-detail quality")
    else:
        feedback = feedback_for(request, items=[dict(operation="set", parameter="lens_corrections.distortion", value=10)])
    report = check_trial(request, feedback, render).agent_report
    assert report.status == "blocked" and not report.can_apply
    assert "internal paths" not in " ".join(report.lines())


def test_visual_check_remains_a_human_judgment_even_when_pixels_change():
    _, request = setup_request()
    plan = workflow(checks=[dict(region="center", metric="visual", direction="visual", expectation="Does the separation serve the goal?")])
    report = check_trial(request, feedback_for(request, plan=plan), renderer).agent_report
    assert report.can_apply and report.checks[0].verdict == "visual"


def test_cancellation_and_wrong_snapshot_never_return_a_verified_trial():
    _, request = setup_request()
    cancel = threading.Event()
    def render(values):
        cancel.set()
        return renderer(values)
    with pytest.raises(CancelledError):
        check_trial(request, feedback_for(request), render, cancel=cancel)
    with pytest.raises(ValueError):
        check_trial(request, replace(feedback_for(request), session_id="another"), renderer)


def test_engine_uses_the_same_formal_renderer_for_both_trial_images(monkeypatch):
    store, request = setup_request()
    engine = PreviewEngine()
    linear = np.full((96, 144, 3), .08, dtype=np.float32)
    photo = DecodedPhoto(linear, dict(sha256=store.current.photo_id, output_size=[144,96], visible_size=[144,96], half_size=False),
                         RawOptions(), False, store.current.source_path)
    monkeypatch.setattr(engine.cache, "get", lambda *a, **k: photo)
    report = engine.check_coach_trial(store.current, request, feedback_for(request)).agent_report
    assert report.can_apply
    np.testing.assert_array_equal(report.current_preview, renderer(store.current.values))
    np.testing.assert_array_equal(report.trial_preview, renderer(replace(store.current.values, exposure_ev=.35)))


def test_ui_requires_checks_shares_report_and_invalidates_trial_without_reanalysis():
    _, request = setup_request()
    panel = CoachResponsePanel()
    panel.output.set_question_draft("")
    feedback = feedback_for(request)
    panel.output.show_feedback(feedback)
    assert not panel.apply_button.isEnabled()
    checked = check_trial(request, feedback, renderer)
    panel.output.show_feedback(checked)
    assert panel.apply_button.isEnabled() and panel.trial_button.isEnabled()
    panel.open_trial()
    assert panel.trial.isVisible() and not panel.trial.isModal()
    panel.open_reader()
    assert panel.reader_output._feedback.agent_report is checked.agent_report
    panel.set_language("zh")
    assert panel.trial_button.text() == "预览试调"
    assert "需要" in " ".join(checked.agent_report.lines(True)) or "试调" in panel.trial_notice.text()
    panel.output.set_question_draft("A different question")
    assert not panel.apply_button.isEnabled() and not panel.trial.isVisible()
    assert "WORKFLOW" in panel.output.response_text()
    panel.reader.hide()
    panel.close()
    delete(panel)
    QApplication.instance().processEvents()


@pytest.fixture
def window(tmp_path, monkeypatch):
    view = MainWindow(settings_store=SettingsStore(tmp_path / "settings.json"))
    monkeypatch.setattr(view, "request_preview", lambda **kw: None)
    monkeypatch.setattr(view, "request_detail", lambda: None)
    monkeypatch.setattr(view.network_pool, "start", lambda _: pytest.fail("No real model request"))
    view.store.activate(Path("private-photo.ARW"), "private-hash")
    view._set_controls(True)
    view._sync_controls()
    yield view
    view._active = view._network_active = None
    view._protected_jobs.clear()
    view._close_approved = True
    view._render_timer.stop()
    view._settle_timer.stop()
    view.close()
    delete(view)
    QApplication.instance().processEvents()


def test_apply_boundary_requires_checked_trial_and_is_idempotent(window):
    state = window.store.current
    pixels = np.full((96,144,3), 80, np.uint8)
    request = build_request(state, state, window.store.events, pixels, pixels)
    feedback = feedback_for(request)
    window.coach_text.show_feedback(feedback)
    assert not window._apply_coach_suggestion()
    window.coach_text.show_feedback(check_trial(request, feedback, renderer))
    assert window._apply_coach_suggestion()
    assert window.store.current.exposure_ev == .35 and len(window.store.undo_stack) == 1
    assert not window._apply_coach_suggestion()
    window.undo()
    assert window.store.current.exposure_ev == 0


def test_provider_response_enters_serial_trial_before_feedback_becomes_actionable(window, monkeypatch):
    state = window.store.current
    pixels = np.full((96,144,3), 80, np.uint8)
    request = build_request(state, state, window.store.events, pixels, pixels)
    feedback = feedback_for(request)
    job = Job("api_prepare", window._generation, lambda: None, state, window._coach_generation,
              coach_request=request)
    ticket = window.api_budget.reserve()
    window._network_active = (None, job, threading.Event(), ticket)
    queued = []
    monkeypatch.setattr(window, "_queue", queued.append)
    monkeypatch.setattr(window.engine, "check_coach_trial", lambda state, req, fb, **kw: check_trial(req, fb, renderer))
    window._network_done(api.ApiResult(feedback, 100, 200), None)
    assert window.coach_text._feedback is None
    assert len(queued) == 1 and queued[0].kind == "coach_check"
    assert window.api_budget.attempts == 1
    trial = queued[0]
    window._active = (None, trial)
    assert window._coach_busy()
    window._done(trial.function(), None)
    assert window.coach_text._feedback.agent_report.can_apply
    assert window.coach_response.apply_button.isEnabled()
    assert window.api_budget.attempts == 1 and state == window.store.current


def test_close_cancels_active_trial_and_waits_for_worker(window):
    from PySide6.QtGui import QCloseEvent
    cancel = threading.Event()
    window._active = (None, Job("coach_check", window._generation, lambda: None,
                                window.store.current, window._coach_generation, cancel=cancel))
    window._close_approved = True
    event = QCloseEvent()
    window.closeEvent(event)
    assert cancel.is_set() and window._closing and not event.isAccepted()
    window._done(None, CancelledError())
    QApplication.instance().processEvents()


@pytest.mark.parametrize("invalidate", ["question", "edit", "coach", "photo", "cancel", "close"])
def test_late_local_verification_cannot_override_new_context(window, invalidate, monkeypatch):
    state = window.store.current
    pixels = np.full((96,144,3), 80, np.uint8)
    request = build_request(state, state, window.store.events, pixels, pixels)
    checked = check_trial(request, feedback_for(request), renderer)
    cancel = threading.Event()
    job = Job("coach_check", window._generation, lambda: None, state, window._coach_generation, cancel=cancel)
    window._active = (None, job)
    if invalidate == "edit":
        window.change_parameter("exposure_ev", -.2)
    elif invalidate == "photo":
        window.store.activate(Path("another.ARW"), "another")
    elif invalidate == "question":
        window.question.setText("Something else")
    elif invalidate == "close":
        window._closing = True
        monkeypatch.setattr(window, "close", lambda: None)
        cancel.set()
    else:
        window._invalidate_coach("Cancelled or coach changed.", coach_changed=invalidate == "coach")
    window._done(checked, CancelledError() if cancel.is_set() else None)
    assert window.coach_text._feedback is None
    window._closing = False


def test_explicit_user_question_authorizes_preference_recommendation_without_goal():
    _, base_request = setup_request()
    assert not base_request.goal.strip()
    question_req = replace(base_request, question="Add a warm spotlight mask on the sea lion.")
    plan_dict = workflow()
    plan_dict["findings"][0].update(category="preference", confidence="high")
    fields = {name: "Try a radial key-light mask." for name in api.FEEDBACK_FIELDS}
    fields.update(
        actions=[dict(operation="add", parameter="exposure_ev", value=.35)],
        workflow=plan_dict,
    )
    parsed = api._parse_response(api.ApiConfig(model="test"), question_req, encoded_response("responses", fields))
    assert parsed.feedback.actions and not parsed.feedback.action_notice
    checked = check_trial(question_req, parsed.feedback, renderer)
    assert checked.agent_report and checked.agent_report.can_apply


def test_pinhole_hard_radial_spotlight_triggers_reflexion_repair_warning():
    from photocoach.coach_agent import build_reflexion_prompt_summary, evaluate_trial_candidate
    from photocoach.composition import Composition
    from photocoach.gradients import RenderGeometry

    _, base_request = setup_request()
    req = replace(base_request, question="Add a warm spotlight in the center.")
    pinhole = [{
        "operation": "create_mask",
        "kind": "radial",
        "name": "Tiny Dot",
        "start": [0.45, 0.45],
        "end": [0.54, 0.54],  # 0.09 x 0.09 pinhole spot!
        "exposure_ev": 0.85,
        "warmth": 0.45,
        "highlights": 0.2,
        "shadows": 0.0,
        "inverted": False,
        "angle": 0.0,
        "feather": 0.30,  # Hard edge < 0.45!
    }]
    plan = parse_plan(workflow(decision="recommend"))
    acts = resolve_actions(pinhole, EditValues(), req.source_size)

    def render_with_geom(values):
        linear = np.full((96, 144, 3), .08, dtype=np.float32)
        return quantize_srgb(render_edit(
            linear, values, geometry=RenderGeometry.full((144, 96), Composition(), (144, 96)),
        ))

    report, delta = evaluate_trial_candidate(req, plan, acts, render_with_geom)
    reflex = build_reflexion_prompt_summary(report, delta)
    assert reflex["needs_repair"] is True
    joined = " ".join(delta["warnings"]).lower()
    assert "small" in joined and "feather" in joined and "shadows" in joined
