"""Tests for PhotoCoach telemetry, OTel fallback, and 25-scenario v1 vs v2 evaluation harness."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PySide6.QtWidgets import QApplication, QDialog

from photocoach import app as desktop
from photocoach.agent_eval import EVAL_SCENARIOS, run_benchmark, write_eval_report
from photocoach.agent_runtime import TraceSpan
from photocoach.api_coach import ApiCoach, ApiConfig, FEEDBACK_FIELDS
from photocoach.coach import build_request
from photocoach.coach_ui import ApiSettings, SendReviewDialog
from photocoach.imaging import DecodedPhoto, RawOptions
from photocoach.settings import SettingsStore
from photocoach.telemetry import TraceStore, export_otel_spans


def test_trace_store_sanitizes_rotates_and_otel_falls_back(tmp_path):
    store = TraceStore(tmp_path / "traces", max_files=3, max_total_bytes=16 * 1024)
    for i in range(5):
        store.record_run(f"run-{i}", {
            "schema_version": 1,
            "note": f"Path C:\\secret\\photo.ARW key sk-abcdef1234567890 run {i}",
        })
    recent = store.load_recent(limit=10)
    assert len(recent) == 3
    joined = json.dumps(recent)
    assert "secret" not in joined and "sk-abcdef" not in joined
    spans = (TraceSpan(0, "observe", "context_ready", 0.0, 1.2, "ok"),)
    # Should return bool cleanly whether opentelemetry is installed or not
    assert isinstance(export_otel_spans("run-0", spans, {"turns": 1}), bool)


def test_eval_benchmark_v2_beats_v1_and_writes_artifacts(tmp_path):
    assert len(EVAL_SCENARIOS) == 25
    report = write_eval_report(tmp_path)
    v1 = report["v1_baseline"]
    v2 = report["v2_industrial_agent"]
    assert v2["scenario_pass_rate"] > v1["scenario_pass_rate"]
    assert v2["scenario_pass_rate"] == 1.0
    assert v2["self_repair_recovery_rate"] == 1.0 and v1["self_repair_recovery_rate"] == 0.0
    assert v2["detail_evidence_compliance_rate"] == 1.0
    assert v2["abstention_accuracy"] == 1.0
    assert v2["fault_resilience_rate"] == 1.0
    assert (tmp_path / "benchmark.json").is_file() and (tmp_path / "benchmark.md").is_file()


def test_mainwindow_multi_turn_agent_loop_end_to_end(tmp_path, monkeypatch):
    qt_app = QApplication.instance() or QApplication([])
    settings_path = tmp_path / "settings.json"
    window = desktop.MainWindow(settings_store=SettingsStore(settings_path))
    try:
        linear = np.full((160, 240, 3), 0.12, dtype=np.float32)
        linear[:, 120:, :] = 0.25
        linear.setflags(write=False)
        meta = {
            "sha256": "e2e-photo-hash",
            "photo_id": "e2e-photo-hash",
            "half_size": False,
            "output_size": (240, 160),
            "visible_size": (240, 160),
            "orientation_flip": 0,
        }
        decoded = DecodedPhoto(linear, meta, RawOptions(), False, tmp_path / "sample.ARW")
        monkeypatch.setattr(window.engine.cache, "get", lambda *a, **kw: decoded)
        window.store.activate(tmp_path / "sample.ARW", "e2e-photo-hash")
        window._set_controls(True)
        window.style_memory_enabled.setChecked(True)
        window.api_settings = ApiSettings(
            ApiConfig("https://generativelanguage.googleapis.com/v1beta/models", "gemini-3.8-flash", "gemini"),
            {"call_limit": 8},
            "dummy-key",
        )
        window.coach_mode.setCurrentIndex(1)

        preview = np.full((64, 96, 3), 85, dtype=np.uint8)
        window.question.setText("Sharpen slightly and brighten center.")
        request = build_request(window.store.current, window.store.current, [], preview, preview,
                                question=window.question.text(), source_size=(240, 160))

        # Verify SendReviewDialog exposes multi-step controls and defaults to 3 turns
        dlg = SendReviewDialog(window, request, window.api_settings.config, window.api_budget)
        assert dlg.agent_loop_enabled.isChecked() and dlg.selected_max_turns() == 3
        dlg.deleteLater()

        class AcceptReview:
            include_before = SimpleNamespace(isChecked=lambda: False)
            skip_session = SimpleNamespace(isChecked=lambda: False)
            def __init__(self, *args):
                pass
            def exec(self):
                return QDialog.DialogCode.Accepted
            def selected_max_turns(self):
                return 3
            def deleteLater(self):
                pass

        turn = {"n": 0}
        def transport(cfg, key, payload_bytes, cancel):
            turn["n"] += 1
            if turn["n"] == 1:
                return 200, json.dumps({
                    "candidates": [{"finishReason": "STOP", "content": {"parts": [
                        {"functionCall": {"id": "c1", "name": "inspect_region_100pct",
                                          "args": {"x_norm": 0.5, "y_norm": 0.5, "size_px": 128}},
                         "thoughtSignature": "sig-1"}
                    ]}}],
                    "usageMetadata": {"promptTokenCount": 90, "candidatesTokenCount": 20},
                }).encode()
            if turn["n"] == 2:
                return 200, json.dumps({
                    "candidates": [{"finishReason": "STOP", "content": {"parts": [
                        {"functionCall": {"id": "c2", "name": "propose_and_preview_trial", "args": {
                            "actions": [
                                {"operation": "add", "parameter": "exposure_ev", "value": 0.35},
                                {"operation": "set", "parameter": "detail.sharpening", "value": 30},
                            ],
                            "workflow": {
                                "version": 1, "decision": "recommend",
                                "objective": "Brighten midtones and add light sharpening after 100% check.",
                                "findings": [{"region": "center", "observation": "Dim edge.",
                                              "interpretation": "Moderate EV and sharpening trial.",
                                              "category": "technical", "confidence": "medium"}],
                                "checks": [{"region": "center", "metric": "mean_luminance", "direction": "increase",
                                            "expectation": "Center luminance rises cleanly."}],
                                "uncertainty": "Confirm visual edge quality in 100% crop.",
                            },
                        }}, "thoughtSignature": "sig-2"}
                    ]}}],
                    "usageMetadata": {"promptTokenCount": 140, "candidatesTokenCount": 35},
                }).encode()
            ans = {f: "Verified after 100% crop and trial preview." for f in FEEDBACK_FIELDS}
            ans["actions"] = [
                {"operation": "add", "parameter": "exposure_ev", "value": 0.35},
                {"operation": "set", "parameter": "detail.sharpening", "value": 30},
            ]
            ans["workflow"] = {
                "version": 1, "decision": "recommend",
                "objective": "Brighten midtones and add light sharpening after 100% check.",
                "findings": [{"region": "center", "observation": "Dim edge.",
                              "interpretation": "Moderate EV and sharpening trial.",
                              "category": "technical", "confidence": "medium"}],
                "checks": [{"region": "center", "metric": "mean_luminance", "direction": "increase",
                            "expectation": "Center luminance rises cleanly."}],
                "uncertainty": "Confirm visual edge quality in 100% crop.",
            }
            return 200, json.dumps({
                "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(ans)}]}}],
                "usageMetadata": {"promptTokenCount": 190, "candidatesTokenCount": 60},
            }).encode()

        monkeypatch.setattr(desktop, "SendReviewDialog", AcceptReview)
        monkeypatch.setattr(desktop, "ApiCoach", lambda cfg, key: ApiCoach(cfg, key, transport=transport))
        job = desktop.Job("api_prepare", window._generation, lambda: None, window.store.current, window._coach_generation)
        window.review_api_request(request, job)

        import time as _t
        deadline = _t.monotonic() + 10.0
        while (window._network_active is not None or window._active is not None or window.coach_text._feedback is None) and _t.monotonic() < deadline:
            qt_app.processEvents()
            _t.sleep(0.01)

        fb = window.coach_text._feedback
        assert fb is not None
        assert fb.agent_report is not None and fb.agent_report.can_apply is True
        assert len(fb.inspected_crops) == 1 and len(fb.agent_trace) >= 5
        assert window.api_budget.attempts == 3 and window.api_budget.in_flight == 0
        assert len(window.trace_store.load_recent(5)) == 1
        assert len(window.agent_memory.entries()) == 1

        # Open trial dialog and verify 100% crop widget is visible
        window.coach_response.open_trial()
        assert not window.coach_response.trial_crop_box.isHidden()
        window.coach_response.trial.hide()

        # Explicit HITL Apply -> 1 undoable transaction -> Undo restores cleanly
        assert window._apply_coach_suggestion(0) is True
        assert window.store.current.exposure_ev == 0.35
        assert window.store.current.detail.sharpening == 30.0
        assert window.agent_memory.entries()[-1].applied is True
        assert len(window.style_memory.preferences()) >= 1
        window.undo()
        assert window.store.current.exposure_ev == 0.0 and window.store.current.detail.sharpening == 0.0
    finally:
        window._close_approved = True
        window.close()
        window.deleteLater()
        qt_app.processEvents()
