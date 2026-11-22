"""Local engine protocol, privacy boundaries, download lifecycle and UI state."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import json
from dataclasses import replace
from pathlib import Path
import socket
import threading
import time
from types import SimpleNamespace
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from photocoach import ollama_client as client
from photocoach import local_runtime as runtime
from photocoach.api_coach import ApiConfig, ApiCoach, ApiError, FEEDBACK_FIELDS, build_payload, _parse_response
from photocoach.settings import ApiSettings, SettingsStore, resolve_api_key
from photocoach.connections import ConnectionBook, connection_from_settings, ConnectionStore
from photocoach.coach import build_request
from photocoach.state import StateStore


@pytest.fixture
def photo_request():
    store = StateStore()
    before = store.activate(Path("local-test.ARW"), "local-photo")
    pixels = np.full((32, 48, 3), 100, np.uint8)
    return build_request(before, before, [], pixels, pixels)


def config():
    return ApiConfig(client.ENDPOINT, client.MODEL, "ollama", 300, 4096)


class Vault:
    def read(self, *args):
        pytest.fail("Local request must not read credentials")


def test_native_protocol_payload_and_feedback(photo_request, agent_workflow):
    request = photo_request
    cfg = config()
    assert resolve_api_key(ApiSettings(cfg, {}, key_source="saved"), Vault()) == ""
    payload = build_payload(cfg, request)
    assert payload["think"] is False and payload["options"]["num_ctx"] == 8192
    assert len(payload["messages"][1]["images"]) == 2
    assert "local-test" not in json.dumps(payload) and "Authorization" not in json.dumps(payload)
    answer = {field: "Measured trial" for field in FEEDBACK_FIELDS}
    answer["actions"] = [{"operation": "set", "parameter": "exposure_ev", "value": .35}]
    answer["workflow"] = agent_workflow
    doc = {"done": True, "done_reason": "stop", "message": {"role": "assistant", "content": json.dumps(answer)},
           "prompt_eval_count": 30, "eval_count": 90}
    result = ApiCoach(cfg, "", transport=lambda *args: (200, json.dumps(doc).encode())).analyze(request)
    assert result.feedback.photo_id == request.photo_id and result.input_tokens == 30
    action, = result.feedback.actions
    assert action.new_value == .35 and action.parameter == "exposure_ev"
    assert result.feedback.state_version == request.state_version
    with pytest.raises(ApiError):
        ApiCoach(cfg, "fake-cloud-sentinel")
    doc["done_reason"] = "length"
    with pytest.raises(ApiError):
        _parse_response(cfg, request, json.dumps(doc).encode())


def test_local_inference_marks_model_loading_stage_without_extra_requests(photo_request, agent_workflow, monkeypatch):
    from photocoach.api_coach import TransportProgress
    progress = TransportProgress()
    calls = []
    def exchange(path, payload, **kwargs):
        calls.append(path)
        assert progress.snapshot()[0] == "local_running"
        answer = {field: "Local test reply." for field in FEEDBACK_FIELDS}
        answer.update(actions=[], workflow=agent_workflow)
        return dict(done=True, done_reason="stop", message=dict(role="assistant", content=json.dumps(answer)))
    monkeypatch.setattr(client, "exchange", exchange)
    ApiCoach(config(), "").analyze(photo_request, progress=progress)
    assert calls == ["/api/chat"]


@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.parametrize("stage", ["local_preparing", "local_running"])
def test_local_loading_notice_is_bilingual_cancellable_and_stale_safe(tmp_path, language, stage):
    from photocoach.app import Job, MainWindow
    from photocoach.api_coach import TransportProgress
    app = QApplication.instance() or QApplication([])
    w = MainWindow(settings_store=SettingsStore(tmp_path/"settings.json"), credentials=Vault())
    cfg = runtime.LocalConfig(model_dir=str(tmp_path/"models"), verified_digest="test", engine_version="test")
    w._use_local_config(cfg)
    state = w.store.activate(Path("loading-test.ARW"), "loading-test")
    w._set_controls(True)
    w.set_language(language)
    progress, cancel = TransportProgress(), threading.Event()
    progress.advance(stage)
    job = Job("api_prepare", w._generation, lambda: None, state, w._coach_generation)
    w._network_active = (None, job, cancel, None)
    w._network_progress = (progress, 300)
    w._update_network_progress()
    expected = ("Preparing local model" if stage == "local_preparing" else "Loading local model") if language == "en" else (
        "正在准备本地模型" if stage == "local_preparing" else "正在加载本地模型")
    assert expected in w.coach_text.toPlainText()
    assert ("First use" if language == "en" else "首次使用") in w.coach_hint.text()
    assert ("First use" if language == "en" else "首次使用") in w.coach_text.toPlainText()
    assert ("Cancel analysis" if language == "en" else "取消分析") == w.analyze_button.text()
    assert w.analyze_button.isEnabled()
    original = w.coach_text.toPlainText()
    cancel.set()
    progress.advance("local_running" if stage == "local_preparing" else "local_preparing")
    w._update_network_progress()
    assert w.coach_text.toPlainText() == original
    cancel.clear()
    w._coach_generation += 1
    w._update_network_progress()
    assert w.coach_text.toPlainText() == original
    w._network_active = w._network_progress = None
    w._close_approved = True
    w.close()
    w.deleteLater()
    app.processEvents()


@pytest.mark.parametrize("endpoint", ["http://127.0.0.1:11434/api/chat", "http://localhost:11435/api/chat",
    "http://192.168.1.2:11435/api/chat", "https://example.com/api/chat", client.ENDPOINT + "?redirect=1"])
def test_local_endpoint_is_fixed(endpoint):
    with pytest.raises(ApiError):
        ApiConfig(endpoint, client.MODEL, "ollama")
    with pytest.raises(ApiError):
        ApiConfig(client.ENDPOINT, "test", "chat_completions")


def test_local_connection_and_separate_config_roundtrip(tmp_path):
    store = ConnectionStore(tmp_path / "settings.json")
    connection = connection_from_settings(ApiSettings(config(), {}, key_environment="", remember_settings=True))
    store.save(ConnectionBook().with_connection(connection))
    assert store.load().active_connection.provider == "ollama"
    local = runtime.LocalSettingsStore(tmp_path / "local_ai.json")
    cfg = runtime.LocalConfig(model_dir=str(tmp_path / "models"), verified_digest="digest", engine_version="0.40")
    local.save(cfg)
    assert local.load() == cfg
    local.path.write_text('{"schema_version":99}')
    with pytest.raises(ValueError):
        local.load()


class Response:
    def __init__(self, lines=(), data=None, status=200):
        self.lines = iter(lines)
        self.status = status
        self.data = json.dumps(data or {}).encode()
    def readline(self, limit):
        return next(self.lines, b"")
    def read(self, limit):
        return self.data


def fake_exchange(monkeypatch, response):
    seen = []
    class Connection:
        sock = None
        def __init__(self, host, port, **kwargs):
            assert host == "127.0.0.1" and port == 11435
        def request(self, method, path, body, headers):
            seen.append((method, path, body, headers))
        def getresponse(self):
            return response
        def close(self):
            pass
    monkeypatch.setattr(client.http.client, "HTTPConnection", Connection)
    return seen


def test_pull_requires_success_and_no_auth(monkeypatch):
    seen = fake_exchange(monkeypatch, Response([b'{"status":"pulling","total":100,"completed":50}\n', b'{"status":"success"}\n']))
    progress = []
    assert client.exchange("/api/pull", {"model": client.MODEL}, progress=progress.append)["status"] == "success"
    assert progress[0]["completed"] == 50 and set(seen[0][3]) == {"Content-Type"}


@pytest.mark.parametrize("lines,status,code", [([], 200, "response"), ([b'not JSON'], 200, "connection"),
    ([b'{"error":"requires a newer version"}'], 200, "version"), ([], 302, "http"), ([], 412, "version")])
def test_pull_failures_are_not_ready(monkeypatch, lines, status, code):
    fake_exchange(monkeypatch, Response(lines, status=status))
    with pytest.raises(client.LocalError) as error:
        client.exchange("/api/pull", {"model": client.MODEL})
    assert error.value.code == code


def test_exchange_cancel_timeout_and_busy(monkeypatch):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(client.LocalError, match="cancelled"):
        client.exchange("/api/tags", cancel=cancel)
    release = threading.Event()
    class Slow(Response):
        def read(self, limit):
            release.wait(2)
            return b'{}'
    fake_exchange(monkeypatch, Slow())
    try:
        with pytest.raises(client.LocalError, match="timed out"):
            client.exchange("/api/tags", timeout=.05)
    finally:
        release.set()
        deadline = time.monotonic() + 2
        while client._LOCK.locked() and time.monotonic() < deadline:
            time.sleep(.01)
    client._LOCK.acquire()
    try:
        with pytest.raises(client.LocalError, match="Another"):
            client.exchange("/api/tags")
    finally:
        client._LOCK.release()


def test_disk_space_and_model_verification(monkeypatch, tmp_path):
    engine = runtime.LocalRuntime(runtime.LocalSettingsStore(tmp_path / "local.json"))
    cfg = runtime.LocalConfig(model_dir=str(tmp_path))
    monkeypatch.setattr(engine, "start", lambda *args: None)
    monkeypatch.setattr(runtime.shutil, "disk_usage", lambda *args: SimpleNamespace(free=100))
    monkeypatch.setattr(runtime, "exchange", lambda *a, **k: pytest.fail("No download when disk full"))
    monkeypatch.setattr(runtime, "model_identity", lambda *a, **k: (_ for _ in ()).throw(client.LocalError("model")))
    with pytest.raises(RuntimeError, match="24 GB"):
        engine.download(cfg, threading.Event(), lambda *args: None)
    monkeypatch.setattr(runtime, "model_identity", lambda *a, **k: "digest")
    monkeypatch.setattr(runtime, "verify_vision", lambda *a, **k: None)
    engine.version = "test-version"
    state, verified = engine.verify(cfg, threading.Event())
    assert state == "ready" and engine.store.load() == verified
    assert engine.inspect(verified, threading.Event())[0] == "ready"
    assert engine.inspect(replace(verified, engine_version="old"), threading.Event())[0] == "unverified"


def test_model_requires_vision_and_probe_content(monkeypatch):
    replies = iter([{"models": [{"name": client.MODEL, "digest": "abc"}]}, {"capabilities": ["completion"]}])
    monkeypatch.setattr(client, "exchange", lambda *a, **k: next(replies))
    with pytest.raises(client.LocalError, match="image"):
        client.model_identity()
    monkeypatch.setattr(client, "exchange", lambda *a, **k: {"done": True, "done_reason": "stop",
        "message": {"role": "assistant", "content": '{"left":"blue","right":"red"}'}})
    with pytest.raises(client.LocalError):
        client.verify_vision()


def test_missing_engine_and_port_conflict(monkeypatch, tmp_path):
    engine = runtime.LocalRuntime(runtime.LocalSettingsStore(tmp_path / "local.json"))
    cfg = runtime.LocalConfig(model_dir=str(tmp_path / "models"))
    monkeypatch.setattr(runtime, "find_ollama", lambda: None)
    with pytest.raises(RuntimeError, match="not installed"):
        engine.start(cfg, threading.Event())
    monkeypatch.setattr(runtime, "find_ollama", lambda: str(Path(__file__)))
    class Occupied:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def bind(self, *args): raise OSError()
    monkeypatch.setattr(runtime.socket, "socket", Occupied)
    with pytest.raises(RuntimeError, match="11435"):
        engine.start(cfg, threading.Event())
    assert engine.process is None


def test_local_ui_selection_budget_and_no_automatic_network(tmp_path, monkeypatch):
    from photocoach.app import MainWindow
    from photocoach.local_models_ui import LocalModelsDialog
    from photocoach.i18n import set_global_language
    app = QApplication.instance() or QApplication([])
    w = MainWindow(settings_store=SettingsStore(tmp_path / "settings.json"), credentials=Vault())
    monkeypatch.setattr(w.local_runtime, "inspect", lambda *args: pytest.fail("No automatic network"))
    w.open_local_models()
    dialog = w._local_dialog
    assert dialog.isVisible() and not dialog.use.isEnabled()
    cfg = runtime.LocalConfig(model_dir=str(tmp_path / "models"), verified_digest="test", engine_version="test")
    dialog.config, dialog.state = cfg, "ready"
    dialog.refresh()
    dialog.use.click()
    assert w.api_settings.config.protocol == "ollama" and w.api_budget.attempts == 0
    assert w._coach_budget() is w.local_budget and w.local_budget.call_limit == 20
    assert resolve_api_key(w.api_settings, Vault()) == ""
    assert "Needs verification" not in w.coach_model_picker.currentText()
    assert "Account access" not in w.coach_catalog_status.text()
    assert w.connection_store.load().active_connection.provider == "ollama"
    dialog.config = replace(cfg, call_limit=31)
    w._use_local_config(dialog.config)
    assert w.local_budget.call_limit == 31
    for lang in ("zh", "en"):
        set_global_language(lang)
        d = LocalModelsDialog(w, w.local_runtime, cfg)
        assert bool(d.windowTitle()) and not d.busy
        d.done(0)
    w._close_approved = True
    w.close()
    w.deleteLater()
    app.processEvents()


def test_local_request_uses_own_budget_and_discards_stale_reply(tmp_path, monkeypatch, photo_request):
    import photocoach.app as module
    from photocoach.api_coach import ApiResult
    from photocoach.coach import CoachFeedback
    from PySide6.QtWidgets import QDialog
    app = QApplication.instance() or QApplication([])
    w = module.MainWindow(settings_store=SettingsStore(tmp_path / "settings.json"), credentials=Vault())
    cfg = runtime.LocalConfig(model_dir=str(tmp_path / "models"), verified_digest="test", engine_version="test")
    w._use_local_config(cfg)
    current = w.store.activate(Path("local-test.ARW"), "local-photo")
    pixels = np.full((32, 48, 3), 100, np.uint8)
    req = build_request(current, current, [], pixels, pixels)
    w.api_budget.configure(call_limit=0)
    workers = []
    monkeypatch.setattr(w.network_pool, "start", workers.append)
    monkeypatch.setattr(w.local_runtime, "ensure_ready", lambda *a: None)
    class Review:
        def __init__(self, *a):
            self.include_before = SimpleNamespace(isChecked=lambda: True)
            self.skip_session = SimpleNamespace(isChecked=lambda: False)
        def exec(self): return QDialog.DialogCode.Accepted
        def deleteLater(self): pass
    monkeypatch.setattr(module, "SendReviewDialog", Review)
    class Adapter:
        def __init__(self, config, key):
            assert config.protocol == "ollama" and key == ""
        def analyze(self, request, **kwargs):
            return ApiResult(CoachFeedback(current.photo_id, current.session_id, current.revision,
                "Local result", "Reason", "Stop", "Observe", ()), 10, 20)
    monkeypatch.setattr(module, "ApiCoach", Adapter)
    job = module.Job("api_prepare", w._generation, lambda: None, current, w._coach_generation)
    w.review_api_request(req, job)
    assert len(workers) == 1 and w.local_budget.attempts == 1 and w.api_budget.attempts == 0
    result = workers[0].function()
    w.store.set_parameter("exposure_ev", .3)
    w._network_done(result, None)
    assert "Local result" not in w.coach_text.toPlainText()
    assert w._network_active is None and w.local_budget.attempts == 1
    w._close_approved = True
    w.close()
    app.processEvents()


def test_dialog_cancel_close_waits_and_never_marks_ready(tmp_path):
    from photocoach.local_models_ui import LocalModelsDialog
    app = QApplication.instance() or QApplication([])
    engine = runtime.LocalRuntime(runtime.LocalSettingsStore(tmp_path / "local.json"))
    dialog = LocalModelsDialog(None, engine, runtime.LocalConfig(model_dir=str(tmp_path)))
    dialog.show()
    dialog.worker = object()
    dialog.cancel = threading.Event()
    dialog.reject()
    assert dialog.cancel.is_set() and dialog.isVisible()
    dialog.finished_operation(("ready", dialog.config), None)
    assert dialog.state == "failed" and not dialog.use.isEnabled() and not dialog.isVisible()
    app.processEvents()


def test_equivalent_windows_folder_spelling_keeps_verification(tmp_path, monkeypatch):
    from photocoach.local_models_ui import LocalModelsDialog
    app = QApplication.instance() or QApplication([])
    engine = runtime.LocalRuntime(runtime.LocalSettingsStore(tmp_path / "local.json"))
    cfg = runtime.LocalConfig(model_dir=(tmp_path / "models").as_posix(), verified_digest="digest", engine_version="version")
    seen = []
    def inspect(config, cancel):
        seen.append(config)
        return "ready", config
    monkeypatch.setattr(engine, "inspect", inspect)
    dialog = LocalModelsDialog(None, engine, cfg)
    dialog.detect.click()
    deadline = time.monotonic()+3
    while dialog.busy and time.monotonic()<deadline:
        app.processEvents()
        time.sleep(.01)
    assert seen[0].verified_digest == "digest" and dialog.use.isEnabled()
    dialog.reject()
    app.processEvents()
