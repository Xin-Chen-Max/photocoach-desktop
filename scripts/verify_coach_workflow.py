"""Real ARW + simulated API integration. No network requests or real credentials."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QTimer
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication, QDialog
from photocoach import app as desktop
from photocoach.api_coach import ApiConfig, ApiCoach, ApiError, FEEDBACK_FIELDS, GEMINI_ENDPOINT
from photocoach.coach_ui import ApiSettings, ApiSettingsDialog, SendReviewDialog
from photocoach.safe_io import sha256_file
from photocoach.cli import _make_output_directory
from photocoach.imaging import dependency_versions
from photocoach.settings import SettingsStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", choices=("responses", "gemini"), default="responses")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    output = _make_output_directory(args.output)
    before_hash = sha256_file(source)
    app = QApplication([])
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
    desktop.configure_appearance(app)
    window = desktop.MainWindow(settings_store=SettingsStore(output / "unused-settings.json"))
    window.show()
    captured = []
    wait_for_release = threading.Event()
    wait_for_release.set()
    request_started = threading.Event()
    transport_calls = []
    transport_mode = ["success"]

    def transport(config, key, body, cancel):
        # .invalid endpoint is never resolved: this function replaces networking.
        transport_calls.append(1)
        captured[:] = [json.loads(body)]
        if transport_mode[0] == "timeout":
            raise ApiError("timeout")  # Match the guarded transport's public contract.
        request_started.set()
        wait_for_release.wait(10)
        feedback = {name: "Simulated provider response for integration verification." for name in FEEDBACK_FIELDS}
        response = {"status": "completed", "output": [{"type": "message", "content": [{
            "type": "output_text", "text": json.dumps(feedback)}]}],
            "usage": {"input_tokens": 1400, "output_tokens": 200}}
        if config.protocol == "gemini":
            response = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{
                "text": json.dumps(feedback)}]}}],
                "usageMetadata": {"promptTokenCount": 1400, "candidatesTokenCount": 200, "thoughtsTokenCount": 40}}
        return 200, json.dumps(response).encode()

    original_adapter = desktop.ApiCoach
    original_review = desktop.SendReviewDialog
    desktop.ApiCoach = lambda config, key: ApiCoach(config, key, transport)
    review_count = [0]
    class Reviewed(SendReviewDialog):
        def exec(self):
            self.show()
            app.processEvents()
            review_count[0] += 1
            if review_count[0] == 1:
                self.grab().save(str(output / "request-review.png"))
            QTimer.singleShot(20, self.accept)
            return super().exec()
    desktop.SendReviewDialog = Reviewed

    def until(condition, label):
        deadline = time.monotonic() + 45
        while not condition():
            app.processEvents()
            if time.monotonic() > deadline:
                raise TimeoutError(label)
            time.sleep(.005)
        app.processEvents()
    def idle():
        return (window._active is None and window._pending is None and window._pending_detail is None
                and not window._protected_jobs and not window._render_timer.isActive()
                and not window._detail_timer.isActive())
    try:
        window.open_raw(source)
        until(lambda: idle() and window.store.current is not None, "RAW open")
        window.goal.setText("Natural light with believable color.")
        window.change_goal()
        window.exposure_slider.setValue(20)
        until(idle, "preview")
        config = (ApiConfig(GEMINI_ENDPOINT, "gemini-simulated-vision-model", "gemini") if args.protocol == "gemini"
                  else ApiConfig(endpoint="https://provider.invalid/v1/responses", model="simulated-vision-model"))
        window.api_settings = ApiSettings(config, {"call_limit": 3, "budget_usd": "0.30", "reserve_usd": "0.10"}, "dummy-test-only")
        window.api_budget.configure(**window.api_settings.budget_settings)
        window.coach_mode.setCurrentIndex(1)
        settings = ApiSettingsDialog(window, window.api_settings, window.api_budget)
        settings.show()
        app.processEvents()
        settings.grab().save(str(output / "settings.png"))
        settings.reject()
        settings.deleteLater()
        window.analyze()
        until(lambda: idle() and window.api_budget.attempts == 1 and window._network_active is None, "first simulated analysis")
        assert "AI · " + config.model in window.coach_text.toPlainText()
        context = json.loads(captured[0]["contents"][0]["parts"][0]["text"] if config.protocol == "gemini"
                             else captured[0]["input"][0]["content"][0]["text"])
        assert context["parameters"] == window.store.current.values.to_dict()
        assert source.name not in json.dumps(captured[0]) and str(source) not in json.dumps(captured[0])
        assert before_hash not in json.dumps(captured[0])
        window.grab().save(str(output / "simulated-feedback.png"))
        request_started.clear()
        wait_for_release.clear()
        window.analyze()
        until(request_started.is_set, "second analysis started")
        window.exposure_slider.setValue(35)
        until(lambda: idle() and window.store.preview_revision == window.store.current.revision, "preview while network blocked")
        assert window._network_active is not None
        wait_for_release.set()
        until(lambda: window._network_active is None, "stale result discarded")
        # Starting a new analysis clears the previous answer. Its now-stale
        # in-flight response must not be displayed after the edit.
        assert window.coach_text._feedback is None
        assert "AI · " + config.model not in window.coach_text.toPlainText()
        assert window.api_budget.attempts == 2
        transport_mode[0] = "timeout"
        window.analyze()
        until(lambda: idle() and window.api_budget.attempts == 3 and window._network_active is None,
              "simulated timeout")
        assert "timed out" in window.coach_text.toPlainText().lower()
        assert len(transport_calls) == 3  # One attempt, no automatic retry.
        window.exposure_slider.setValue(40)
        until(lambda: idle() and window.store.preview_revision == window.store.current.revision,
              "editing after timeout")
        after_hash = sha256_file(source)
        assert before_hash == after_hash
        report = {"phase": 2, "protocol": config.protocol, "live_api_tested": False, "remote_requests": 0,
                  "simulated_attempts": len(transport_calls), "source_name": source.name,
                  "source_sha256": before_hash, "original_unchanged": True,
                  "context_exact": True, "preview_while_network_waits": True, "stale_advice_discarded": True,
                  "timeout_no_retry": True, "editing_after_timeout": True,
                  "review_dialog_exercised": review_count[0], "dependencies": dependency_versions(),
                  "remaining": "Choose provider/model and run an explicitly authorized live test; manual visual acceptance."}
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        wait_for_release.set()
        desktop.ApiCoach, desktop.SendReviewDialog = original_adapter, original_review
        if window.store.current:
            window.store.mark_saved(window.store.current.revision)
        window.close()
        until(lambda: window._active is None and window._network_active is None, "closing workers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
