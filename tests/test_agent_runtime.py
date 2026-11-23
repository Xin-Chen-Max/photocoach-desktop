"""Tests for AgentRuntime state graph, RunBudget, 4-protocol tool calling, reflexion repair, and replay."""
from __future__ import annotations

import json
from pathlib import Path
import threading

import numpy as np
import pytest

from photocoach import api_coach as api
from photocoach.adjustments import render_edit
from photocoach.agent_runtime import (
    AgentProgress,
    AgentRuntime,
    RunBudget,
    replay_run_record,
)
from photocoach.agent_tools import (
    ToolCall,
    ToolResult,
    build_default_tool_registry,
)
from photocoach.budget import SessionBudget
from photocoach.coach import build_request
from photocoach.imaging import quantize_srgb
from photocoach.state import StateStore


def _make_request():
    store = StateStore()
    state = store.activate(Path("C:/secret-dir/private-raw.ARW"), "secret-hash-1234")
    pixels = np.full((64, 96, 3), 72, dtype=np.uint8)
    request = build_request(state, state, store.events, pixels, pixels,
                            question="Lift the midtones and check detail.", source_size=(96, 64))
    return store, state, request


def _sample_workflow(decision="recommend", metric="mean_luminance", direction="increase"):
    return {
        "version": 1,
        "decision": decision,
        "objective": "Lift the midtones carefully without clipping highlights.",
        "findings": [{
            "region": "center",
            "observation": "The center preview appears dim.",
            "interpretation": "A moderate exposure lift is a reversible trial.",
            "category": "technical",
            "confidence": "medium",
        }],
        "checks": [{
            "region": "center",
            "metric": metric,
            "direction": direction,
            "expectation": "Center luminance moves in the intended direction.",
        }],
        "uncertainty": "Preview luminance does not measure aesthetic preference.",
    }


def _final_json(actions=None, workflow=None):
    payload = {field: "Measured photographic guidance." for field in api.FEEDBACK_FIELDS}
    payload["actions"] = actions if actions is not None else []
    payload["workflow"] = workflow if workflow is not None else _sample_workflow()
    return payload


def _tool_response(protocol: str, call_name: str, args: dict, *, thought_sig: str = "sig-xyz"):
    if protocol == "gemini":
        return json.dumps({
            "candidates": [{
                "finishReason": "STOP",
                "content": {
                    "parts": [
                        {"thought": True, "text": "Checking crop first."},
                        {"functionCall": {"id": "fc-1", "name": call_name, "args": args},
                         "thoughtSignature": thought_sig},
                    ]
                },
            }],
            "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 30, "thoughtsTokenCount": 12},
        }).encode()
    if protocol == "responses":
        return json.dumps({
            "status": "completed",
            "output": [
                {"type": "reasoning", "id": "rs_1", "summary": []},
                {"type": "function_call", "call_id": "fc-1", "name": call_name, "arguments": json.dumps(args)},
            ],
            "usage": {"input_tokens": 100, "output_tokens": 40},
        }).encode()
    if protocol == "ollama":
        return json.dumps({
            "done": True,
            "done_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "fc-1", "function": {"name": call_name, "arguments": args}}],
            },
            "prompt_eval_count": 100,
            "eval_count": 40,
        }).encode()
    return json.dumps({
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "fc-1", "type": "function",
                                "function": {"name": call_name, "arguments": json.dumps(args)}}],
            },
        }],
        "usage": {"prompt_tokens": 100, "completion_tokens": 40},
    }).encode()


def _final_response(protocol: str, body_dict: dict):
    text = json.dumps(body_dict)
    if protocol == "gemini":
        return json.dumps({
            "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": text}]}}],
            "usageMetadata": {"promptTokenCount": 150, "candidatesTokenCount": 80, "thoughtsTokenCount": 10},
        }).encode()
    if protocol == "responses":
        return json.dumps({
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
            "usage": {"input_tokens": 150, "output_tokens": 80},
        }).encode()
    if protocol == "ollama":
        return json.dumps({
            "done": True,
            "done_reason": "stop",
            "message": {"role": "assistant", "content": text},
            "prompt_eval_count": 150,
            "eval_count": 80,
        }).encode()
    return json.dumps({
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 150, "completion_tokens": 80},
    }).encode()


def test_session_budget_can_authorize_turns():
    budget = SessionBudget(call_limit=5, budget_usd="1.00", reserve_usd="0.30")
    assert budget.can_authorize_turns(3) is None
    assert budget.can_authorize_turns(4) is not None
    t = budget.reserve()
    budget.settle(t, 100, 100, success=False)
    assert budget.can_authorize_turns(2) is None
    assert budget.can_authorize_turns(3) is not None


@pytest.mark.parametrize("protocol", ["gemini", "responses", "chat_completions", "ollama"])
def test_four_protocols_tool_roundtrip_and_multimodal_crop(protocol):
    _, state, request = _make_request()
    endpoint = ("http://127.0.0.1:11435/api/chat" if protocol == "ollama"
                else api.GEMINI_ENDPOINT if protocol == "gemini"
                else "https://api.openai.com/v1/responses")
    model = ("qwen3.6:27b" if protocol == "ollama"
             else "gemini-3.8-flash" if protocol == "gemini"
             else "gpt-6-astra")
    cfg = api.ApiConfig(endpoint, model, protocol)
    captured_payloads: list[dict] = []

    def transport(c, key, payload_bytes, cancel):
        doc = json.loads(payload_bytes.decode("utf-8"))
        captured_payloads.append(doc)
        serialized = payload_bytes.decode("utf-8")
        assert "secret-dir" not in serialized and "secret-hash" not in serialized
        if len(captured_payloads) == 1:
            return 200, _tool_response(protocol, "inspect_region_100pct", {"x_norm": 0.5, "y_norm": 0.5, "size_px": 256})
        return 200, _final_response(protocol, _final_json(
            actions=[{"operation": "add", "parameter": "exposure_ev", "value": 0.4}],
        ))

    adapter = api.ApiCoach(cfg, "" if protocol == "ollama" else "test-api-key", transport=transport)
    session_budget = SessionBudget(call_limit=5)
    runtime = AgentRuntime(adapter, build_default_tool_registry(), run_budget=RunBudget(max_llm_turns=3),
                           session_budget=session_budget)
    crop_patch = np.full((128, 128, 3), 90, dtype=np.uint8)

    def render_crop(x_norm, y_norm, size_px, apply_current_edits=True):
        return crop_patch, {"edge_energy_mean": 0.12, "rect_px": [0, 0, 128, 128]}

    def render_trial(values):
        return quantize_srgb(render_edit(np.full((64, 96, 3), 0.10, dtype=np.float32), values))

    result = runtime.run(request, state=state, render_trial=render_trial, render_crop_100pct=render_crop)
    assert result.turns_used == 2 and result.tool_calls_used == 1
    assert session_budget.attempts == 2 and session_budget.in_flight == 0
    assert result.feedback.agent_report and result.feedback.agent_report.can_apply
    assert len(result.feedback.inspected_crops) == 1
    assert len(captured_payloads) == 2

    second_payload = captured_payloads[1]
    if protocol == "gemini":
        # Preserved thoughtSignature from turn 1 and attached multimodal FunctionResponse.parts with $ref
        model_turn = second_payload["contents"][1]
        assert model_turn["role"] == "model"
        assert model_turn["parts"][1]["thoughtSignature"] == "sig-xyz"
        tool_turn = second_payload["contents"][2]
        fr = tool_turn["parts"][0]["functionResponse"]
        assert fr["name"] == "inspect_region_100pct" and len(fr["parts"]) == 1
        assert "inlineData" in fr["parts"][0]
        assert fr["response"]["attached_images"][0]["image"]["$ref"] == fr["parts"][0]["inlineData"]["displayName"]
    elif protocol == "responses":
        assert any(item.get("type") == "function_call_output" for item in second_payload["input"])
        assert any(item.get("role") == "user" and any(p.get("type") == "input_image" for p in item["content"])
                   for item in second_payload["input"][1:])
    elif protocol == "ollama":
        assert any(m.get("role") == "tool" and m.get("tool_name") == "inspect_region_100pct"
                   for m in second_payload["messages"])
        assert any(m.get("role") == "user" and m.get("images") for m in second_payload["messages"][2:])
    else:
        assert any(m.get("role") == "tool" for m in second_payload["messages"])
        assert any(m.get("role") == "user" and isinstance(m.get("content"), list)
                   and any(p.get("type") == "image_url" for p in m["content"])
                   for m in second_payload["messages"][2:])


def test_visual_reflexion_self_repair_and_detail_gate():
    store, state, request = _make_request()
    before_history = tuple(store.undo_stack)
    cfg = api.ApiConfig(api.GEMINI_ENDPOINT, "gemini-3.8-flash", "gemini")
    turn_idx = 0
    render_calls = 0

    def transport(c, key, payload_bytes, cancel):
        nonlocal turn_idx
        turn_idx += 1
        doc = json.loads(payload_bytes.decode("utf-8"))
        if turn_idx == 1:
            # Turn 1: propose Detail sharpening WITHOUT 100% crop -> blocked by policy gate!
            return 200, _tool_response("gemini", "propose_and_preview_trial", {
                "actions": [{"operation": "set", "parameter": "detail.sharpening", "value": 45}],
                "workflow": _sample_workflow(metric="visual", direction="visual"),
            })
        if turn_idx == 2:
            # Verify tool error instructed 100% inspection; now inspect 100% crop + wrong-direction exposure trial
            text = json.dumps(doc)
            assert "100%" in text
            return 200, _tool_response("gemini", "inspect_region_100pct", {"x_norm": 0.5, "y_norm": 0.5, "size_px": 256})
        if turn_idx == 3:
            # Turn 3: propose excessive darkening when check expects increase -> fails numeric check, triggers reflexion
            return 200, _tool_response("gemini", "propose_and_preview_trial", {
                "actions": [{"operation": "add", "parameter": "exposure_ev", "value": -0.8}],
                "workflow": _sample_workflow(metric="mean_luminance", direction="increase"),
            })
        if turn_idx == 4:
            # Turn 4: repair with positive lift + sharpening (now unlocked!)
            text = json.dumps(doc)
            assert "FAILED" in text
            return 200, _tool_response("gemini", "propose_and_preview_trial", {
                "actions": [
                    {"operation": "add", "parameter": "exposure_ev", "value": 0.45},
                    {"operation": "set", "parameter": "detail.sharpening", "value": 35},
                ],
                "workflow": _sample_workflow(metric="mean_luminance", direction="increase"),
            })
        # Turn 5: finalize matching repaired trial (must reuse cached trial report without extra render!)
        return 200, _final_response("gemini", _final_json(
            actions=[
                {"operation": "add", "parameter": "exposure_ev", "value": 0.45},
                {"operation": "set", "parameter": "detail.sharpening", "value": 35},
            ],
            workflow=_sample_workflow(metric="mean_luminance", direction="increase"),
        ))

    def render_trial(values):
        nonlocal render_calls
        render_calls += 1
        return quantize_srgb(render_edit(np.full((64, 96, 3), 0.10, dtype=np.float32), values))

    adapter = api.ApiCoach(cfg, "test-key", transport=transport)
    runtime = AgentRuntime(adapter, build_default_tool_registry(), run_budget=RunBudget(max_llm_turns=5, max_trial_renders=3))
    result = runtime.run(
        request, state=state, render_trial=render_trial,
        render_crop_100pct=lambda x, y, s, apply_current_edits=True: (np.full((128, 128, 3), 85, np.uint8), {"ok": True}),
    )

    assert result.turns_used == 5
    assert result.trial_renders_used == 2  # Turn 1 blocked before rendering; Turn 3 + Turn 4 rendered; Turn 5 reused!
    assert render_calls == 4  # 2 trials * (before + after) = 4, zero redundant render on finalize!
    assert result.feedback.agent_report.can_apply is True
    assert store.current is state and tuple(store.undo_stack) == before_history

    record = result.to_record(request)
    replayed = replay_run_record(record)
    assert replayed["valid"] and replayed["can_apply"]
    assert "reflect" in replayed["nodes"] and "trial_render" in replayed["nodes"]


def test_cancellation_and_stale_guard_abort_cleanly_and_settle_budget():
    _, state, request = _make_request()
    cfg = api.ApiConfig(api.GEMINI_ENDPOINT, "gemini-3.8-flash", "gemini")
    cancel = threading.Event()
    calls = 0

    def transport(c, key, payload_bytes, ev):
        nonlocal calls
        calls += 1
        cancel.set()  # Cancel immediately after turn 1
        return 200, _tool_response("gemini", "analyze_color_zones", {"region": "whole"})

    budget = SessionBudget(call_limit=5)
    runtime = AgentRuntime(api.ApiCoach(cfg, "k", transport), build_default_tool_registry(),
                           session_budget=budget, run_budget=RunBudget(max_llm_turns=3))
    with pytest.raises(api.ApiError, match="cancelled"):
        runtime.run(request, state=state, cancel=cancel)
    assert calls == 1 and budget.attempts == 1 and budget.in_flight == 0

    # Stale state check aborts before sending turn 1
    budget2 = SessionBudget(call_limit=5)
    runtime2 = AgentRuntime(api.ApiCoach(cfg, "k", transport), build_default_tool_registry(), session_budget=budget2)
    with pytest.raises(api.ApiError, match="cancelled"):
        runtime2.run(request, state=state, stale_check=lambda: True)
    assert budget2.attempts == 0


def test_agent_progress_preserves_transport_progress_contract():
    progress = AgentProgress(max_turns=3)
    progress.advance("waiting")
    progress.set_agent_step(2, 3, "inspect_region_100pct")
    assert isinstance(progress, api.TransportProgress)
    assert progress.snapshot()[0] == "waiting"
    assert progress.agent_snapshot() == (2, 3, "inspect_region_100pct")


def test_verified_trial_triggers_force_final_schema_on_next_turn():
    _, state, request = _make_request()
    cfg = api.ApiConfig(api.GEMINI_ENDPOINT, "gemini-3.5-flash", "gemini")
    captured: list[dict] = []

    def transport(c, key, payload_bytes, cancel):
        doc = json.loads(payload_bytes.decode("utf-8"))
        captured.append(doc)
        if len(captured) == 1:
            return 200, _tool_response("gemini", "propose_and_preview_trial", {
                "actions": [{"operation": "add", "parameter": "exposure_ev", "value": 0.4}],
                "workflow": _sample_workflow(metric="mean_luminance", direction="increase"),
            })
        return 200, _final_response("gemini", _final_json(
            actions=[{"operation": "add", "parameter": "exposure_ev", "value": 0.4}],
            workflow=_sample_workflow(metric="mean_luminance", direction="increase"),
        ))

    def render_trial(values):
        return quantize_srgb(render_edit(np.full((64, 96, 3), 0.10, dtype=np.float32), values))

    runtime = AgentRuntime(
        api.ApiCoach(cfg, "test-key", transport=transport),
        build_default_tool_registry(),
        run_budget=RunBudget(max_llm_turns=4),
    )
    res = runtime.run(request, state=state, render_trial=render_trial)
    assert res.turns_used == 2
    assert res.feedback.agent_report and res.feedback.agent_report.can_apply
    # Turn 2 forced strict schema and omitted tools because Turn 1 produced a verified clean trial
    turn2_doc = captured[1]
    assert "tools" not in turn2_doc
    assert turn2_doc["generationConfig"].get("responseMimeType") == "application/json"
    assert "responseJsonSchema" in turn2_doc["generationConfig"]


@pytest.mark.parametrize("error_code", ["unavailable", "timeout"])
def test_transient_mid_run_api_error_recovers_verified_best_trial(error_code):
    _, state, request = _make_request()
    cfg = api.ApiConfig(api.GEMINI_ENDPOINT, "gemini-3.5-flash", "gemini")
    calls = 0

    def transport(c, key, payload_bytes, cancel):
        nonlocal calls
        calls += 1
        if calls == 1:
            return 200, _tool_response("gemini", "propose_and_preview_trial", {
                "actions": [{"operation": "add", "parameter": "exposure_ev", "value": 0.4}],
                "workflow": _sample_workflow(metric="mean_luminance", direction="increase"),
            })
        raise api.ApiError(error_code)

    def render_trial(values):
        return quantize_srgb(render_edit(np.full((64, 96, 3), 0.10, dtype=np.float32), values))

    budget = SessionBudget(call_limit=5)
    runtime = AgentRuntime(
        api.ApiCoach(cfg, "test-key", transport=transport),
        build_default_tool_registry(),
        run_budget=RunBudget(max_llm_turns=3),
        session_budget=budget,
    )
    res = runtime.run(request, state=state, render_trial=render_trial)
    assert res.feedback.agent_report and res.feedback.agent_report.can_apply
    assert len(res.feedback.actions) == 1
    assert res.aborted_reason and budget.in_flight == 0 and budget.attempts == 2


def test_agent_progress_streams_live_spans_and_sanitized_tool_details():
    _, state, request = _make_request()
    cfg = api.ApiConfig(api.GEMINI_ENDPOINT, "gemini-3.5-flash", "gemini")
    snapshots: list[tuple[int, int, str, str, int]] = []
    progress = AgentProgress(max_turns=3)
    calls = 0

    def transport(c, key, payload_bytes, cancel):
        nonlocal calls
        calls += 1
        turn, max_t, label, detail, spans = progress.live_trace_snapshot()
        snapshots.append((turn, max_t, label, detail, len(spans)))
        if calls == 1:
            return 200, _tool_response("gemini", "inspect_region_100pct", {"x_norm": 0.48, "y_norm": 0.52, "size_px": 256})
        if calls == 2:
            return 200, _tool_response("gemini", "propose_and_preview_trial", {
                "actions": [
                    {"operation": "add", "parameter": "exposure_ev", "value": 0.4},
                    {"operation": "add", "parameter": "shadows", "value": 0.15},
                ],
                "workflow": _sample_workflow(metric="mean_luminance", direction="increase"),
            })
        return 200, _final_response("gemini", _final_json(
            actions=[
                {"operation": "add", "parameter": "exposure_ev", "value": 0.4},
                {"operation": "add", "parameter": "shadows", "value": 0.15},
            ],
            workflow=_sample_workflow(metric="mean_luminance", direction="increase"),
        ))

    def render_crop(x_norm, y_norm, size_px, apply_current_edits=True):
        turn, max_t, label, detail, spans = progress.live_trace_snapshot()
        snapshots.append((turn, max_t, label, detail, len(spans)))
        return np.full((64, 64, 3), 80, dtype=np.uint8), {"rect_px": [0, 0, 64, 64]}

    def render_trial(values):
        turn, max_t, label, detail, spans = progress.live_trace_snapshot()
        snapshots.append((turn, max_t, label, detail, len(spans)))
        return quantize_srgb(render_edit(np.full((64, 96, 3), 0.10, dtype=np.float32), values))

    runtime = AgentRuntime(
        api.ApiCoach(cfg, "test-key", transport=transport),
        build_default_tool_registry(),
        run_budget=RunBudget(max_llm_turns=3),
    )
    res = runtime.run(request, state=state, render_trial=render_trial, render_crop_100pct=render_crop, progress=progress)
    _, _, _, _, final_spans = progress.live_trace_snapshot()
    assert final_spans == res.spans
    # During crop inspection, active detail showed coordinates; during trial render, parameter names
    assert any(lbl == "inspect_region_100pct" and detail == "(0.48, 0.52)" for _, _, lbl, detail, _ in snapshots)
    assert any(lbl == "propose_and_preview_trial" and "exposure_ev" in detail for _, _, lbl, detail, _ in snapshots)
    # Recorded TraceSpan details carry sanitized metadata for UI timeline rendering
    crop_span = next(s for s in res.spans if s.name == "inspect_region_100pct")
    assert crop_span.detail["x_norm"] == 0.48 and crop_span.detail["y_norm"] == 0.52
    trial_span = next(s for s in res.spans if s.node == "trial_render")
    assert trial_span.detail["params"] == ["exposure_ev", "shadows"]
    reflect_span = next(s for s in res.spans if s.node == "reflect")
    assert reflect_span.detail["passed"] is True


def test_agent_runtime_precomputes_lighting_anchors_and_retries_turn2_rate_limit(monkeypatch):
    import photocoach.agent_runtime as rt_mod
    monkeypatch.setattr(rt_mod.time, "sleep", lambda *_: None)

    store = StateStore()
    state = store.activate(Path("C:/secret-dir/beach.ARW"), "secret-hash-beach")
    pixels = np.full((64, 96, 3), 135, dtype=np.uint8)
    pixels[20:42, 32:64] = 48
    request = build_request(state, state, store.events, pixels, pixels,
                            question="Add an overhead spotlight on the sea lion", source_size=(96, 64))
    cfg = api.ApiConfig(api.GEMINI_ENDPOINT, "gemini-3.5-flash", "gemini")
    turn1_payloads: list[dict] = []
    calls = 0

    def transport(c, key, payload_bytes, cancel):
        nonlocal calls
        calls += 1
        doc = json.loads(payload_bytes.decode("utf-8"))
        if calls == 1:
            turn1_payloads.append(doc)
            return 200, _tool_response("gemini", "build_lighting_rig", {
                "recipe": "spotlight",
                "light_direction": "overhead",
                "strength": "moderate",
            })
        if calls == 2:
            raise api.ApiError("rate_limit", http_status=429)
        return 200, _final_response("gemini", _final_json(
            actions=[{
                "operation": "create_mask",
                "kind": "radial",
                "name": "Overhead Key Light",
                "start": [0.25, 0.20],
                "end": [0.70, 0.65],
                "exposure_ev": 0.45,
                "warmth": 0.35,
                "highlights": 0.20,
                "shadows": 0.35,
                "inverted": False,
                "angle": 0.0,
                "feather": 0.80,
            }],
            workflow=_sample_workflow(metric="visual", direction="visual"),
        ))

    def render_trial(values):
        from photocoach.composition import Composition
        from photocoach.gradients import RenderGeometry
        return quantize_srgb(render_edit(
            np.full((64, 96, 3), 0.14, dtype=np.float32), values,
            geometry=RenderGeometry.full((96, 64), Composition(), (96, 64)),
        ))

    budget = SessionBudget(call_limit=5)
    runtime = AgentRuntime(
        api.ApiCoach(cfg, "test-key", transport=transport),
        build_default_tool_registry(),
        run_budget=RunBudget(max_llm_turns=3),
        session_budget=budget,
    )
    res = runtime.run(request, state=state, render_trial=render_trial)
    # Observe pre-populated compact lighting anchors directly in Turn 1 prompt
    prompt_text = json.dumps(turn1_payloads[0])
    assert "precomputed_lighting_anchors" in prompt_text
    # Rate limit on Turn 2 retried once and completed Turn 2 final guidance cleanly
    assert calls == 3 and budget.attempts == 3 and budget.in_flight == 0
    assert res.feedback.agent_report and res.feedback.agent_report.can_apply


def test_recommend_with_omitted_actions_recovers_verified_best_trial():
    _, state, request = _make_request()
    cfg = api.ApiConfig(api.GEMINI_ENDPOINT, "gemini-3.5-flash", "gemini")
    calls = 0

    def transport(c, key, payload_bytes, cancel):
        nonlocal calls
        calls += 1
        if calls == 1:
            return 200, _tool_response("gemini", "propose_and_preview_trial", {
                "actions": [{"operation": "add", "parameter": "exposure_ev", "value": 0.5}],
                "workflow": _sample_workflow(metric="mean_luminance", direction="increase"),
            })
        return 200, _final_response("gemini", _final_json(
            actions=[],
            workflow=_sample_workflow(decision="recommend", metric="mean_luminance", direction="increase"),
        ))

    def render_trial(values):
        return quantize_srgb(render_edit(np.full((64, 96, 3), 0.10, dtype=np.float32), values))

    runtime = AgentRuntime(
        api.ApiCoach(cfg, "test-key", transport=transport),
        build_default_tool_registry(),
        run_budget=RunBudget(max_llm_turns=3),
    )
    res = runtime.run(request, state=state, render_trial=render_trial)
    assert res.turns_used == 2
    assert res.feedback.agent_report and res.feedback.agent_report.can_apply is True
    assert len(res.feedback.actions) == 1
    assert res.feedback.actions[0].new_value == 0.5


@pytest.mark.parametrize("decision", ["stop", "inspect"])
def test_stop_or_inspect_after_trial_does_not_force_apply(decision):
    _, state, request = _make_request()
    cfg = api.ApiConfig(api.GEMINI_ENDPOINT, "gemini-3.5-flash", "gemini")
    calls = 0

    def transport(c, key, payload_bytes, cancel):
        nonlocal calls
        calls += 1
        if calls == 1:
            return 200, _tool_response("gemini", "propose_and_preview_trial", {
                "actions": [{"operation": "add", "parameter": "exposure_ev", "value": 0.5}],
                "workflow": _sample_workflow(metric="mean_luminance", direction="increase"),
            })
        return 200, _final_response("gemini", _final_json(
            actions=[],
            workflow=_sample_workflow(decision=decision, metric="visual", direction="visual"),
        ))

    def render_trial(values):
        return quantize_srgb(render_edit(np.full((64, 96, 3), 0.10, dtype=np.float32), values))

    runtime = AgentRuntime(
        api.ApiCoach(cfg, "test-key", transport=transport),
        build_default_tool_registry(),
        run_budget=RunBudget(max_llm_turns=3),
    )
    res = runtime.run(request, state=state, render_trial=render_trial)
    assert res.feedback.actions == ()
    assert res.feedback.agent_report is not None
    assert res.feedback.agent_report.status == "no_edit"
    assert res.feedback.agent_report.can_apply is False


def test_policy_blocked_final_actions_are_not_overridden_by_best_trial():
    store = StateStore()
    state = store.activate(Path("C:/secret-dir/private-raw.ARW"), "secret-hash-pref")
    pixels = np.full((64, 96, 3), 72, dtype=np.uint8)
    request = build_request(state, state, store.events, pixels, pixels, question="", source_size=(96, 64))
    cfg = api.ApiConfig(api.GEMINI_ENDPOINT, "gemini-3.5-flash", "gemini")
    calls = 0

    def transport(c, key, payload_bytes, cancel):
        nonlocal calls
        calls += 1
        if calls == 1:
            return 200, _tool_response("gemini", "propose_and_preview_trial", {
                "actions": [{"operation": "add", "parameter": "exposure_ev", "value": 0.4}],
                "workflow": _sample_workflow(metric="mean_luminance", direction="increase"),
            })
        pref_wf = _sample_workflow(decision="recommend", metric="visual", direction="visual")
        pref_wf["findings"][0]["category"] = "preference"
        return 200, _final_response("gemini", _final_json(
            actions=[{"operation": "add", "parameter": "saturation", "value": 0.1}],
            workflow=pref_wf,
        ))

    def render_trial(values):
        return quantize_srgb(render_edit(np.full((64, 96, 3), 0.10, dtype=np.float32), values))

    runtime = AgentRuntime(
        api.ApiCoach(cfg, "test-key", transport=transport),
        build_default_tool_registry(),
        run_budget=RunBudget(max_llm_turns=3),
    )
    res = runtime.run(request, state=state, render_trial=render_trial)
    assert res.feedback.actions == ()
    assert res.feedback.action_notice != ""
    assert res.feedback.agent_report is not None and res.feedback.agent_report.can_apply is False


def test_full_retouch_mode_allows_multipass_trial_refinement_without_early_exit():
    _, state, request = _make_request()
    cfg = api.ApiConfig(api.GEMINI_ENDPOINT, "gemini-3.5-flash", "gemini")
    seen_docs: list[dict] = []

    def transport(c, key, payload_bytes, cancel):
        doc = json.loads(payload_bytes.decode("utf-8"))
        seen_docs.append(doc)
        step_n = len(seen_docs)
        if step_n == 1:
            return 200, _tool_response("gemini", "propose_and_preview_trial", {
                "actions": [{"operation": "set", "parameter": "exposure_ev", "value": 0.6}],
                "workflow": _sample_workflow(metric="mean_luminance", direction="increase"),
            })
        if step_n == 2:
            # In full_retouch mode, tools must STILL remain enabled on Turn 2 after clean Trial #1!
            assert "tools" in doc and doc["tools"]
            return 200, _tool_response("gemini", "propose_and_preview_trial", {
                "actions": [
                    {"operation": "set", "parameter": "exposure_ev", "value": 0.6},
                    {"operation": "set", "parameter": "contrast", "value": 0.25},
                ],
                "workflow": _sample_workflow(metric="mean_luminance", direction="increase"),
            })
        return 200, _final_response("gemini", _final_json(
            actions=[
                {"operation": "set", "parameter": "exposure_ev", "value": 0.6},
                {"operation": "set", "parameter": "contrast", "value": 0.25},
            ],
            workflow=_sample_workflow(decision="recommend", metric="mean_luminance", direction="increase"),
        ))

    def render_trial(values):
        return quantize_srgb(render_edit(np.full((64, 96, 3), 0.12, dtype=np.float32), values))

    runtime = AgentRuntime(
        api.ApiCoach(cfg, "test-key", transport=transport),
        build_default_tool_registry(),
        run_budget=RunBudget(max_llm_turns=5, max_trial_renders=4, retouch_mode="full_retouch"),
    )
    res = runtime.run(request, state=state, render_trial=render_trial)
    assert res.turns_used == 3
    assert res.trial_renders_used == 2
    assert res.feedback.agent_report and res.feedback.agent_report.can_apply is True
    assert "Multi-Pass Full Retouch Workflow" in seen_docs[0]["systemInstruction"]["parts"][0]["text"]
    assert res.feedback.actions[0].changes == (("exposure_ev", 0.6), ("contrast", 0.25))

