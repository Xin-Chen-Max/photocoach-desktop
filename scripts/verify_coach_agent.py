"""Native real-RAW Coach agent acceptance with one explicitly simulated model reply.

No network, credential reads or changes to the original photograph. Evidence and
screenshots remain local. Run with RAW and a new output directory.
"""
from dataclasses import replace
import argparse
import json
from pathlib import Path
import threading
import time

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from photocoach import imaging
from photocoach.api_coach import ApiCoach, ApiConfig, ApiResult
from photocoach.app import Job, MainWindow, configure_appearance
from photocoach.safe_io import sha256_file
from photocoach.settings import ApiSettings, SettingsStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    digest = sha256_file(args.raw)
    app = QApplication([])
    assert app.platformName() == "windows", "Native Windows fonts required"
    configure_appearance(app)
    class NoCredentials:
        def read(self, *_):
            raise AssertionError("No credential access permitted")
    window = MainWindow(settings_store=SettingsStore(args.output/"unused-settings.json"), credentials=NoCredentials())
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    for dialog in (window.coach_response.trial, window.coach_response.reader):
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    window.network_pool.start = lambda *_: (_ for _ in ()).throw(AssertionError("No real network worker"))
    window.show()
    original_decode = imaging.decode_raw
    decodes = []
    def decode(*a, **kw):
        decodes.append(kw.get("half_size", False))
        return original_decode(*a, **kw)
    imaging.decode_raw = decode
    transport_calls = []
    def wait_idle():
        deadline = time.monotonic()+180
        while (window._active or window._pending or window._pending_detail or window._protected_jobs
               or window._render_timer.isActive() or window._settle_timer.isActive() or window._detail_timer.isActive()):
            app.processEvents()
            if time.monotonic() > deadline:
                raise TimeoutError(window.status.text())
            time.sleep(.005)
        app.processEvents()
    shots = []
    def shot(name, widget):
        app.processEvents()
        path = args.output/(name+".png")
        assert widget.grab().save(str(path))
        shots.append(path.name)
    try:
        window.open_raw(args.raw)
        wait_idle()
        state = window.store.current
        window.question.setText("Try a modest central brightness lift; compare bright texture before keeping it.")
        request = window.engine.coach_request(state, state, window.store.events, window.question.text())
        fields = dict(answer="This simulated reply proposes a reversible exposure trial. Compare the two images before deciding.",
            main_change="No edit has been applied.", reason="The trial tests brightness separation, not an ideal numerical exposure.",
            next_step_or_stop="Preview the trial. Apply only if it serves your intention, then compare with Undo.",
            observe_and_stop_when="Inspect the central subject and the bright upper region. Undo if texture or natural separation is lost.",
            observed="The supplied preview contains brighter and darker regions.",
            inferred="A small lift may help separation; this is a hypothesis.", preference="The desired look requires the user's judgment.",
            actions=[dict(operation="add", parameter="exposure_ev", value=.15)],
            workflow=dict(version=1, decision="recommend", objective="Test central brightness without assuming aesthetic improvement.",
                findings=[dict(region="center", observation="The center is darker than the bright upper region.",
                               interpretation="A small exposure trial can test whether separation better serves the goal.",
                               category="technical", confidence="medium")],
                checks=[dict(region="center", metric="mean_luminance", direction="increase", expectation="Central brightness increases."),
                        dict(region="top_center", metric="visual", direction="visual", expectation="Compare the bright texture; undo if separation is lost.")],
                uncertainty="Simulated proposal; neither semantic understanding nor aesthetic quality is established by this test."))
        def transport(config, key, payload, cancel):
            transport_calls.append(1)
            serialized = payload.decode("utf-8")
            assert digest not in serialized and str(args.raw) not in serialized
            return 200, json.dumps(dict(status="completed", output=[dict(type="message", content=[
                dict(type="output_text", text=json.dumps(fields))])], usage=dict(input_tokens=100, output_tokens=200))).encode()
        config = ApiConfig(model="simulated-vision-model", max_output_tokens=4096)
        result = ApiCoach(config, "dummy-test-key", transport).analyze(request)
        result = replace(result, feedback=replace(result.feedback, source="SIMULATED · Workflow verification"))
        window.api_settings = ApiSettings(config, {}, key_source="saved")
        job = Job("api_prepare", window._generation, lambda: None, state, window._coach_generation, coach_request=request)
        ticket = window.api_budget.reserve()
        window._network_active = (None, job, threading.Event(), ticket)
        history_count = len(window.store.undo_stack)
        started = time.perf_counter()
        window._network_done(result, None)
        wait_idle()
        elapsed = time.perf_counter()-started
        feedback = window.coach_text._feedback
        assert feedback.agent_report.can_apply
        assert window.store.current == state and len(window.store.undo_stack) == history_count
        assert len(transport_calls) == window.api_budget.attempts == 1
        assert max(feedback.agent_report.trial_preview.shape[:2]) <= 768
        assert window.engine.retained_bytes <= 512*1024*1024
        for language in ("en", "zh"):
            window.set_language(language)
            for width, height in ((1280,800),(1920,1080)):
                window.resize(width,height)
                wait_idle()
                window.coach_collapse.setChecked(False)
                window.coach_text.show_feedback(feedback)
                app.processEvents()
                assert window.coach_response.apply_button.isEnabled()
                for button in (window.coach_response.apply_button, window.coach_response.trial_button):
                    assert button.isVisible() and window.rect().contains(button.mapTo(window, button.rect().center()))
                prefix = f"{language}-{width}"
                shot(prefix+"-ready", window)
                window.coach_response.open_trial()
                app.processEvents()
                assert window.coach_response.trial.isVisible()
                assert all(not label.pixmap().isNull() for label in window.coach_response.trial_images)
                shot(prefix+"-trial", window.coach_response.trial)
                window.coach_response.trial.hide()
                window.coach_response.open_reader()
                shot(prefix+"-reader", window.coach_response.reader)
                window.coach_response.reader.hide()
                blocked = replace(feedback, agent_report=replace(feedback.agent_report, status="blocked", can_apply=False))
                window.coach_text.show_feedback(blocked)
                assert not window.coach_response.apply_button.isEnabled()
                shot(prefix+"-held", window)
        window.coach_text.show_feedback(feedback)
        assert window._apply_coach_suggestion()
        wait_idle()
        assert window.store.current.exposure_ev == .15
        assert len(window.store.undo_stack) == history_count+1
        assert not window._apply_coach_suggestion()
        # Re-render through the SAME full-resolution formal path at the trial's
        # resolution to compare actual applied pixels, not a fast preview.
        from photocoach.composition import output_size
        from photocoach.preview import DetailRequest
        full = output_size(request.source_size, window.store.current.composition)
        detail = DetailRequest(0, full, (0,0,*full), max(1, int(np.ceil(max(full)/768))), request.source_size)
        actual = window.engine.detail(window.store.current, detail, include_metadata=False)[0]
        assert np.array_equal(actual, feedback.agent_report.trial_preview)
        window.undo()
        wait_idle()
        assert window.store.current.values == state.values
        original = window.engine.detail(window.store.current, detail, include_metadata=False)[0]
        assert np.array_equal(original, feedback.agent_report.current_preview)
        assert sha256_file(args.raw) == digest
        report = dict(source_unchanged=True, remote_requests=0, credential_reads=0,
            simulated_model_requests=len(transport_calls), model_budget_attempts=window.api_budget.attempts,
            one_undo_step=True, duplicate_apply_blocked=True, trial_matches_applied_pixels=True,
            undo_restores_current_pixels=True, pretrial_state_unchanged=True,
            trial_seconds=round(elapsed,3), decode_calls=len(decodes), retained_bytes=window.engine.retained_bytes,
            screenshots=shots, limitations="Simulated model response; no live semantic or aesthetic quality validation.")
        (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(report))
    finally:
        window._close_approved = True
        window.close()
        window.pool.waitForDone(10000)
        app.processEvents()
        imaging.decode_raw = original_decode


if __name__ == "__main__":
    main()
