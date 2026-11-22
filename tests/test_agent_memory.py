"""Tests for Context Engineering: working memory, compaction, style memory privacy, and Playbook RAG."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from photocoach import api_coach as api
from photocoach.agent_memory import (
    PlaybookRAG,
    SessionWorkingMemory,
    StyleMemoryStore,
    TurnSummary,
    compact_conversation,
)
from photocoach.agent_runtime import AgentRuntime, RunBudget
from photocoach.agent_tools import ToolImage, ToolResult, build_default_tool_registry
from photocoach.coach import build_request
from photocoach.gradients import LinearGradient
from photocoach.state import EditValues, StateStore


def test_session_working_memory_binds_and_caps_at_six_turns():
    mem = SessionWorkingMemory(max_turns=6)
    for i in range(8):
        mem.record("photo-1", "sess-1", TurnSummary(i, f"Q{i}", "recommend", f"Objective {i}", ("inspect_region_100pct",), f"Exposure +{i}"))
    assert len(mem.entries()) == 6
    assert mem.entries()[0].state_version == 2
    mem.mark_last_applied("Exposure +7 (Applied)")
    assert mem.entries()[-1].applied is True
    assert "Applied" in mem.prompt_snippet() or "applied" in mem.prompt_snippet()

    # Photo/session switch clears working memory automatically
    mem.record("photo-2", "sess-2", TurnSummary(0, "New photo", "stop", "Check highlights"))
    assert len(mem.entries()) == 1 and mem.entries()[0].question == "New photo"


def test_compact_conversation_evicts_older_tool_images_but_pins_initial_and_latest():
    preview = np.full((32, 32, 3), 100, dtype=np.uint8)
    crop_old = ToolImage("Old 100% crop", preview)
    crop_new = ToolImage("Latest trial render", preview)
    messages = [
        {"role": "user", "text": "x" * 2000, "images": [("Current", preview)]},
        {"role": "assistant", "raw_parts": [{"functionCall": {"name": "inspect_region_100pct"}}]},
        {"role": "tool_results", "results": [ToolResult("c1", "inspect_region_100pct", True, {"ok": True}, images=(crop_old,))]},
        {"role": "assistant", "raw_parts": [{"functionCall": {"name": "propose_and_preview_trial"}}]},
        {"role": "tool_results", "results": [ToolResult("c2", "propose_and_preview_trial", True, {"ok": True}, images=(crop_new,))]},
    ]
    compacted = compact_conversation(messages, max_est_tokens=100)
    # Initial preview pinned
    assert len(compacted[0]["images"]) == 1
    # Older tool image evicted with compact marker
    old_res = compacted[2]["results"][0]
    assert old_res.images == ()
    assert old_res.payload["compacted_images"] == ["Old 100% crop"]
    # Latest tool image pinned
    assert len(compacted[4]["results"][0].images) == 1
    # Original list untouched
    assert len(messages[2]["results"][0].images) == 1


def test_style_memory_opt_in_persistence_and_privacy_filter(tmp_path):
    store = StyleMemoryStore(tmp_path / "style.json")
    assert store.enabled is False and store.preferences() == () and store.prompt_snippet() == ""

    # When disabled, record_applied_edit is a no-op
    before = EditValues()
    after = EditValues(base_rendering="natural_v1", highlights=-0.25, shadows=0.20,
                       local_gradients=(LinearGradient("g1", "Warm", (0.1, 0.1), (0.8, 0.8), warmth=0.4),))
    store.record_applied_edit(before, after, "Warm golden mountain")
    assert store.preferences() == ()

    # Opt in and record
    store.set_enabled(True)
    store.record_applied_edit(before, after, "Warm golden mountain")
    prefs = store.preferences()
    assert len(prefs) >= 1 and "natural_v1" in " ".join(prefs).lower()

    # Privacy filter rejects paths, raw extensions, keys, and hex hashes
    assert store.add_preference("Photo C:\\Users\\test\\7C204108.ARW") is False
    assert store.add_preference("Use key sk-1234567890abcdef") is False
    assert store.add_preference("Hash " + ("a" * 64)) is False
    assert store.add_preference("Prefers subtle contrast.") is True

    # Reload from disk verifies persistence & atomic schema
    reloaded = StyleMemoryStore(tmp_path / "style.json")
    assert reloaded.enabled is True
    assert "Prefers subtle contrast." in reloaded.preferences()

    reloaded.clear()
    assert reloaded.preferences() == ()


def test_playbook_rag_bilingual_retrieval_and_runtime_injection():
    rag = PlaybookRAG()
    hits_en = rag.search("sharpening halos edge masking noise 100%", top_k=3)
    assert any("detail" in h["id"] for h in hits_en)

    hits_zh = rag.search("金色山脊 丁达尔 斜射光", top_k=3)
    assert any("mountain_beam" in h["id"] for h in hits_zh)

    hits_hl = rag.search("高光过曝 天空 传感器 宽容度", top_k=3)
    assert any(h["id"] in ("pb_highlights_sky", "pb_cfa_headroom") for h in hits_hl)

    # Verify PlaybookRAG integrates into AgentRuntime observe context
    store = StateStore()
    state = store.activate(Path("test.ARW"), "test-id")
    px = np.full((32, 48, 3), 80, dtype=np.uint8)
    req = build_request(state, state, [], px, px, question="How do I sharpen without sky halos?")
    captured = []

    def transport(cfg, key, payload_bytes, cancel):
        captured.append(json.loads(payload_bytes.decode("utf-8")))
        ans = {f: "Check 100% crop first." for f in api.FEEDBACK_FIELDS}
        ans["actions"] = []
        ans["workflow"] = {
            "version": 1, "decision": "inspect", "objective": "Inspect at 100% first.",
            "findings": [{"region": "whole", "observation": "Reduced preview.", "interpretation": "Needs 100% check.",
                          "category": "technical", "confidence": "medium"}],
            "checks": [{"region": "whole", "metric": "visual", "direction": "visual", "expectation": "Check 100% edges."}],
            "uncertainty": "Reduced preview cannot verify sharpening.",
        }
        return 200, json.dumps({"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(ans)}]}}]}).encode()

    cfg = api.ApiConfig(api.GEMINI_ENDPOINT, "gemini-3.8-flash", "gemini")
    runtime = AgentRuntime(
        api.ApiCoach(cfg, "k", transport),
        build_default_tool_registry(),
        run_budget=RunBudget(max_llm_turns=2),
        playbook=rag,
        style_memory_snippet="Prefers subtle highlight protection.",
        working_memory_snippet="- v0 Q='initial' -> stop",
    )
    res = runtime.run(req, state=state)
    assert res.feedback.agent_report.status == "no_edit"
    user_payload = captured[0]["contents"][0]["parts"][0]["text"]
    assert "playbook_hints" in user_payload
    assert "user_style_preferences" in user_payload
    assert "session_working_memory" in user_payload
