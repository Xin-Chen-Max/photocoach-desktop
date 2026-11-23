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
    assert payload["think"] is False and 16384 <= payload["options"]["num_ctx"] <= client.MAX_NUM_CTX
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


@pytest.mark.parametrize("endpoint", ["http://127.0.0.1:80/api/chat", "http://localhost:11435/api/chat",
    "http://192.168.1.2:11435/api/chat", "https://example.com/api/chat", "http://127.0.0.1:11435/api/tags",
    client.ENDPOINT + "?redirect=1"])
def test_local_endpoint_is_fixed(endpoint):
    with pytest.raises(ApiError):
        ApiConfig(endpoint, client.MODEL, "ollama")
    with pytest.raises(ApiError):
        ApiConfig(client.ENDPOINT, "test", "chat_completions")
    assert ApiConfig("http://127.0.0.1:11435/api/chat", client.MODEL, "ollama").endpoint.endswith(":11435/api/chat")
    assert ApiConfig("http://127.0.0.1:11434/api/chat", client.MODEL, "ollama").endpoint.endswith(":11434/api/chat")


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
    assert w._coach_budget() is w.local_budget and w.local_budget.call_limit == 100
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


def test_curated_model_catalog_verified_tags_and_8192_ctx():
    ids = [m.id for m in client.LOCAL_MODEL_CATALOG]
    assert ids == [
        "qwen3.5:4b",
        "qwen3-vl:8b",
        "qwen3.5:9b",
        "gemma4:12b",
        "gemma4:26b",
        "qwen3.6:27b",
    ]
    assert "qwen3-vl:14b" not in ids
    for entry in client.LOCAL_MODEL_CATALOG:
        assert entry.default_num_ctx >= 16384
        assert entry.min_ollama_version != "0.6.0"
        assert runtime.parse_ollama_version(entry.min_ollama_version) >= (0, 32, 0)
        assert entry.kv_vision_headroom_bytes >= 2_500_000_000


def test_conditional_think_false_only_when_thinking_capability_reported(monkeypatch):
    client.clear_observed_capabilities()
    # When /api/show reports capabilities WITHOUT "thinking", chat_payload and verify_vision must omit "think".
    def fake_show_no_think(path, payload=None, **kwargs):
        if path == "/api/show":
            return {"capabilities": ["completion", "vision"]}
        if path == "/api/chat":
            assert "think" not in payload
            assert payload["options"]["num_ctx"] == 16384
            return {"done": True, "done_reason": "stop", "message": {"role": "assistant", "content": '{"left":"red","right":"blue"}'}}
        raise AssertionError(path)

    monkeypatch.setattr(client, "exchange", fake_show_no_think)
    cap_info = client.inspect_model_capabilities("qwen3-vl:8b")
    assert cap_info["has_vision"] is True and cap_info["has_thinking"] is False and cap_info["supports_tools"] is False
    payload_no_think = client.chat_payload("qwen3-vl:8b", [{"role": "user", "content": "hi"}])
    assert "think" not in payload_no_think
    assert payload_no_think["options"]["num_ctx"] == 16384
    client.verify_vision("qwen3-vl:8b")

    # Even if a catalog model defaults to has_thinking=True, if /api/show says no "thinking", omit "think".
    client.inspect_model_capabilities("qwen3.5:4b")
    assert "think" not in client.chat_payload("qwen3.5:4b", [{"role": "user", "content": "hi"}])

    # When /api/show reports "thinking" in capabilities, "think": False is included.
    def fake_show_with_think(path, payload=None, **kwargs):
        if path == "/api/show":
            return {"capabilities": ["completion", "vision", "thinking", "tools"]}
        if path == "/api/chat":
            assert payload.get("think") is False
            return {"done": True, "done_reason": "stop", "message": {"role": "assistant", "content": '{"left":"red","right":"blue"}'}}
        raise AssertionError(path)

    monkeypatch.setattr(client, "exchange", fake_show_with_think)
    cap_think = client.inspect_model_capabilities("gemma4:12b")
    assert cap_think["has_thinking"] is True and cap_think["supports_tools"] is True
    payload_think = client.chat_payload("gemma4:12b", [{"role": "user", "content": "hi"}])
    assert payload_think["think"] is False
    client.verify_vision("gemma4:12b")
    client.clear_observed_capabilities()


def test_windows_vram_detection_nvidia_smi_winreg_and_headroom_fit(monkeypatch):
    # 1. Multi-GPU nvidia-smi picks max dedicated VRAM across adapters.
    class FakeProc:
        returncode = 0
        stdout = "Intel UHD Graphics, 1024\nNVIDIA GeForce RTX 4080, 16384\nNVIDIA GeForce RTX 3060, 12288\n"

    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: FakeProc())
    vram_b, gpu_name = runtime._probe_nvidia_smi_vram()
    assert vram_b == 16384 * 1024 * 1024
    assert "RTX 4080" in gpu_name

    # 2. Winreg qwMemorySize supports both REG_QWORD (int) and 8-byte REG_BINARY (>4 GB) and skips Basic Render Driver.
    assert runtime._parse_winreg_qw_memory_size(16 * 1024 ** 3) == 16 * 1024 ** 3
    binary_24gb = (24 * 1024 ** 3).to_bytes(8, "little", signed=False)
    assert runtime._parse_winreg_qw_memory_size(binary_24gb) == 24 * 1024 ** 3

    # 3. classify_hardware_fit accounts for +2.5-3.0 GB headroom and skips disk warning when is_installed=True.
    hw_20gb_low_disk = runtime.LocalHardwareInfo(
        system_ram_bytes=64 * 1024 ** 3,
        vram_bytes=20 * 1000 ** 3,
        gpu_name="Test 20GB GPU",
        free_disk_bytes=500_000_000,  # 0.5 GB free disk
    )
    # qwen3.6:27b is 19 GB + 3 GB headroom = 22 GB working memory -> exceeds 20 GB VRAM (cpu_offload).
    # When not installed, 0.5 GB free disk triggers disk_low; when is_installed=True, disk warning is skipped!
    fit_uninstalled = runtime.classify_hardware_fit("qwen3.6:27b", hw_20gb_low_disk, is_installed=False)
    assert fit_uninstalled.status == "disk_low" and not fit_uninstalled.disk_ok
    fit_installed = runtime.classify_hardware_fit("qwen3.6:27b", hw_20gb_low_disk, is_installed=True)
    assert fit_installed.disk_ok is True and fit_installed.status == "cpu_offload"

    # gemma4:12b is 8.2 GB + 2.7 GB headroom = 10.9 GB working memory -> fits in 20 GB VRAM!
    fit_12b = runtime.classify_hardware_fit("gemma4:12b", hw_20gb_low_disk, is_installed=True)
    assert fit_12b.status == "fits_vram"


def test_pull_progress_tracker_monotonic_multi_layer_and_success_100():
    tracker = client.PullProgressTracker(expected_model_bytes=19_000_000_000)
    e1 = tracker.update({"status": "pulling", "digest": "sha256:layer1", "total": 10_000_000_000, "completed": 5_000_000_000})
    e2 = tracker.update({"status": "pulling", "digest": "sha256:layer1", "total": 10_000_000_000, "completed": 10_000_000_000})
    # Starting layer 2 with completed=0 must never move aggregate progress backward.
    e3 = tracker.update({"status": "pulling", "digest": "sha256:layer2", "total": 2_000_000_000, "completed": 0})
    assert e1["percent"] <= e2["percent"] <= e3["percent"]
    assert e3["completed"] >= e2["completed"]
    # Even if actual layers sum to 12 GB (< 19 GB advisory expected_model_bytes), "status": "success" snaps to 100%.
    e4 = tracker.update({"status": "success"})
    assert e4["percent"] == 100 and e4["status"] == "success"


def test_existing_ollama_discovery_filters_cloud_and_hides_non_vision(monkeypatch, tmp_path):
    engine = runtime.LocalRuntime(runtime.LocalSettingsStore(tmp_path / "local.json"))
    cfg = runtime.LocalConfig(service_mode="existing", port=11434)

    def fake_ex(path, payload=None, **kwargs):
        assert kwargs.get("port") == 11434
        if path == "/api/version":
            return {"version": "0.40.16"}
        if path == "/api/tags":
            return {
                "models": [
                    {"name": "gemma4:12b", "digest": "dig-gemma12", "size": 8_200_000_000},
                    {"name": "qwen3:8b", "digest": "dig-textonly", "size": 5_000_000_000},
                    {"name": "gemma4:31b:cloud", "digest": "dig-cloud", "size": 0},
                    {"name": "remote-vlm:latest", "digest": "dig-remote", "remote_host": "https://cloud.ollama.com"},
                    {"name": "custom-vlm:7b", "digest": "dig-customvlm", "size": 4_500_000_000},
                ]
            }
        if path == "/api/show":
            m = payload["model"]
            if m == "gemma4:12b":
                return {"capabilities": ["completion", "vision", "tools", "thinking"]}
            if m == "qwen3:8b":
                return {"capabilities": ["completion", "tools"]}
            if m == "custom-vlm:7b":
                return {"capabilities": ["completion", "vision"]}
            raise AssertionError(f"Should not query /api/show for filtered cloud model {m}")
        raise AssertionError(path)

    monkeypatch.setattr(client, "exchange", fake_ex)
    disc = engine.discover_models(cfg, threading.Event())
    assert disc.engine_version == "0.40.16"
    # qwen3:8b (non-vision), gemma4:31b:cloud (:cloud), and remote-vlm:latest (remote_host) are hidden (3 total).
    assert disc.hidden_non_vision_count == 3
    row_ids = [r.id for r in disc.models]
    assert "custom-vlm:7b" in row_ids
    assert "qwen3:8b" not in row_ids
    assert "gemma4:31b:cloud" not in row_ids
    assert "remote-vlm:latest" not in row_ids


def test_verified_models_capability_persistence_and_single_turn_clamping(tmp_path, photo_request):
    from photocoach.coach_ui import SendReviewDialog
    from photocoach.budget import SessionBudget
    app = QApplication.instance() or QApplication([])
    store = runtime.LocalSettingsStore(tmp_path / "local.json")
    cfg = runtime.LocalConfig(
        model_dir=str(tmp_path / "models"),
        model="qwen3.6:27b",
        verified_digest="dig-27b",
        engine_version="0.40.16",
        supports_tools=True,
        has_thinking=True,
        verified_models={
            "qwen3.6:27b": {"digest": "dig-27b", "engine_version": "0.40.16", "supports_tools": True, "has_thinking": True},
            "qwen3-vl:8b": {"digest": "dig-8b", "engine_version": "0.40.16", "supports_tools": False, "has_thinking": False},
        },
    )
    store.save(cfg)
    loaded = store.load()
    assert loaded.supports_tools is True and loaded.has_thinking is True

    switched = loaded.with_active_model("qwen3-vl:8b")
    assert switched.model == "qwen3-vl:8b"
    assert switched.verified_digest == "dig-8b"
    assert switched.supports_tools is False and switched.has_thinking is False

    # ApiCoach and SendReviewDialog must clamp supports_tools=False models to max_turns=1.
    api_cfg_no_tools = ApiConfig(switched.endpoint, switched.model, "ollama", 300, 4096)
    coach_no_tools = ApiCoach(api_cfg_no_tools, "")
    assert coach_no_tools.supports_tools is False
    from PySide6.QtWidgets import QWidget
    parent_widget = QWidget()
    parent_widget.local_config = switched
    dlg = SendReviewDialog(parent_widget, photo_request, api_cfg_no_tools, SessionBudget(call_limit=10))
    assert not dlg.agent_loop_enabled.isEnabled()
    assert dlg.selected_max_turns() == 1
    dlg.deleteLater()
    parent_widget.deleteLater()

    switched_back = switched.with_active_model("qwen3.6:27b")
    assert switched_back.verified_digest == "dig-27b"
    assert switched_back.supports_tools is True and switched_back.has_thinking is True


def test_delete_model_and_version_update_hint_in_dialog(tmp_path, monkeypatch):
    from photocoach.local_models_ui import LocalModelsDialog
    app = QApplication.instance() or QApplication([])
    engine = runtime.LocalRuntime(runtime.LocalSettingsStore(tmp_path / "local.json"))
    cfg = runtime.LocalConfig(
        model_dir=str(tmp_path / "models"),
        model="qwen3.6:27b",
        verified_digest="dig-27b",
        engine_version="0.40.16",
    )
    deleted_calls = []

    def fake_ex(path, payload=None, **kwargs):
        if path == "/api/version":
            return {"version": "0.40.16"}
        if path == "/api/delete":
            deleted_calls.append(payload["model"])
            return {"status": "deleted"}
        raise AssertionError(path)

    monkeypatch.setattr(engine, "start", lambda *a: setattr(engine, "version", "0.40.16"))
    monkeypatch.setattr(client, "exchange", fake_ex)
    state, after_del = engine.delete_model(cfg, "qwen3.6:27b", threading.Event())
    assert state == "not_downloaded" and after_del.verified_digest == ""
    assert "qwen3.6:27b" not in after_del.verified_models
    assert deleted_calls == ["qwen3.6:27b"]

    # Existing mode disables Stop engine; old version shows update warning banner.
    engine.version = "0.30.0"
    existing_cfg = runtime.LocalConfig(service_mode="existing", port=11434, model="qwen3.6:27b")
    dialog = LocalModelsDialog(None, engine, existing_cfg)
    assert not dialog.stop.isEnabled()
    assert not dialog.version_warning_label.isHidden()
    assert "0.40.0" in dialog.version_warning_label.text()
    dialog.reject()
    app.processEvents()


def test_local_api_errors_preserve_http_status_and_connection_detail(monkeypatch, photo_request):
    # HTTP 500 / 400 from local Ollama must surface http_error + sanitized Ollama error body, never "secure connection".
    for status in (400, 500):
        class ErrResp:
            def __init__(self, code):
                self.status = code
            def read(self, limit):
                return json.dumps({"error": "model requires more system memory (22.0 GiB)"}).encode()
            def readline(self, limit):
                return b""

        fake_exchange(monkeypatch, ErrResp(status))
        with pytest.raises(ApiError) as exc_info:
            ApiCoach(config(), "").analyze(photo_request)
        err = exc_info.value
        assert err.code == "http_error"
        assert err.http_status == status
        assert "model requires more system memory" in str(err)
        assert "secure connection" not in str(err).lower()

    # Loopback socket failure must raise local_connection with 127.0.0.1 detail, never "secure connection".
    def broken_exchange(*args, **kwargs):
        raise client.LocalError("connection", detail="127.0.0.1:11435 (ConnectionRefusedError)")

    monkeypatch.setattr(client, "exchange", broken_exchange)
    with pytest.raises(ApiError) as conn_exc:
        ApiCoach(config(), "").analyze(photo_request)
    assert conn_exc.value.code == "local_connection"
    assert "127.0.0.1:11435" in str(conn_exc.value)
    assert "secure connection" not in str(conn_exc.value).lower()


def test_review_api_request_syncs_local_endpoint_and_falls_back_when_no_tools(tmp_path, monkeypatch):
    import photocoach.app as module
    from photocoach.api_coach import ApiResult
    from photocoach.coach import CoachFeedback
    from PySide6.QtWidgets import QDialog

    client.clear_observed_capabilities()
    app = QApplication.instance() or QApplication([])
    w = module.MainWindow(settings_store=SettingsStore(tmp_path / "settings.json"), credentials=Vault())
    # Activate local coach first on default port 11435, then simulate switching local_config to port 11434
    # (as happens when user edits service mode/port in LocalModelsDialog without clicking Use local coach again).
    cfg_11435 = runtime.LocalConfig(
        model_dir=str(tmp_path / "models"),
        model="qwen3.6:27b",
        verified_digest="dig-1",
        engine_version="0.40.16",
        port=11435,
        supports_tools=True,
    )
    w._use_local_config(cfg_11435)
    assert w.api_settings.config.endpoint == "http://127.0.0.1:11435/api/chat"

    cfg_11434 = runtime.LocalConfig(
        service_mode="existing",
        port=11434,
        model="qwen3.6:27b",
        verified_digest="dig-1",
        engine_version="0.40.16",
        supports_tools=True,
    )
    w.local_config = cfg_11434

    current = w.store.activate(Path("sync-test.ARW"), "sync-photo")
    pixels = np.full((32, 48, 3), 100, np.uint8)
    req = build_request(current, current, [], pixels, pixels)

    workers = []
    monkeypatch.setattr(w.network_pool, "start", workers.append)

    # Simulate ensure_ready updating live /api/show capabilities on worker thread to supports_tools=False.
    def fake_ensure_ready(local_cfg, cancel):
        client.register_model_capabilities(
            local_cfg.model,
            has_vision=True,
            supports_tools=False,
            has_thinking=False,
            port=local_cfg.port,
        )

    monkeypatch.setattr(w.local_runtime, "ensure_ready", fake_ensure_ready)

    class Review:
        def __init__(self, *a):
            self.include_before = SimpleNamespace(isChecked=lambda: True)
            self.skip_session = SimpleNamespace(isChecked=lambda: False)
        def selected_max_turns(self):
            return 3
        def exec(self):
            return QDialog.DialogCode.Accepted
        def deleteLater(self):
            pass

    monkeypatch.setattr(module, "SendReviewDialog", Review)

    observed = {"endpoints": [], "analyze_calls": 0, "step_calls": 0}

    class Adapter:
        def __init__(self, cfg, key):
            observed["endpoints"].append(cfg.endpoint)
            self.config = cfg
            self.supports_tools = True
        def analyze(self, request, **kwargs):
            observed["analyze_calls"] += 1
            return ApiResult(
                CoachFeedback(current.photo_id, current.session_id, current.revision,
                              "Fallback single-turn OK", "Reason", "Stop", "Observe", ()),
                12, 24,
            )
        def step(self, *args, **kwargs):
            observed["step_calls"] += 1
            raise AssertionError("Should fall back to single-turn analyze when /api/show lacks tools")

    monkeypatch.setattr(module, "ApiCoach", Adapter)
    job = module.Job("api_prepare", w._generation, lambda: None, current, w._coach_generation)
    w.review_api_request(req, job)
    assert len(workers) == 1
    assert observed["endpoints"] == ["http://127.0.0.1:11434/api/chat"]

    result = workers[0].function()
    assert observed["analyze_calls"] == 1
    assert observed["step_calls"] == 0
    assert result.feedback.main_change == "Fallback single-turn OK"

    w._network_active = None
    w._close_approved = True
    w.close()
    w.deleteLater()
    app.processEvents()
    client.clear_observed_capabilities()


def test_ollama_num_ctx_scales_for_large_prompts_and_tools(photo_request):
    from photocoach.agent_tools import ToolImage, ToolResult, build_default_tool_registry
    from photocoach.api_coach import AGENT_INSTRUCTIONS, build_tool_turn_payload, request_context

    # 1. Tiny prompt with 0 images uses DEFAULT_NUM_CTX (16384).
    tiny_payload = client.chat_payload(client.MODEL, [{"role": "user", "content": "hi"}])
    assert tiny_payload["options"]["num_ctx"] == 16384

    # 2. Full 2-image single-shot Coach payload steps up cleanly to 24576 (prevents >8192 HTTP 400 errors).
    cfg = config()
    single_payload = build_payload(cfg, photo_request)
    assert single_payload["options"]["num_ctx"] == 24576

    # 3. Heavy multi-turn Agent v2 payload (7 tool specs + 4 images) steps up to MAX_NUM_CTX (32768).
    tools = build_default_tool_registry().specs()
    assert len(tools) >= 7
    px = np.full((32, 48, 3), 120, np.uint8)
    ctx_json = json.dumps(request_context(photo_request, include_before=True), ensure_ascii=False)
    messages = [
        {
            "role": "user",
            "text": ctx_json,
            "images": [("Before this editing round", photo_request.before_preview), ("Current effect", photo_request.current_preview)],
        },
        {
            "role": "assistant",
            "raw_message": {"role": "assistant", "content": "Checking details.", "tool_calls": [{"function": {"name": "inspect_region_100pct", "arguments": {"x_norm": 0.5, "y_norm": 0.5}}}]},
        },
        {
            "role": "tool_results",
            "results": [
                ToolResult(
                    "call_0",
                    "inspect_region_100pct",
                    True,
                    {"note": "inspected " * 400},
                    images=(ToolImage("crop_1", px), ToolImage("crop_2", px)),
                )
            ],
        },
    ]
    heavy_payload = build_tool_turn_payload(cfg, AGENT_INSTRUCTIONS, messages, tools)
    assert heavy_payload["options"]["num_ctx"] == 32768


def test_ollama_think_fenced_json_output_limit_and_local_structural_diagnostics(photo_request, agent_workflow):
    from photocoach.api_coach import parse_tool_turn_response

    cfg = config()
    valid_answer = {field: "Measured trial." for field in FEEDBACK_FIELDS}
    valid_answer["actions"] = []
    valid_answer["workflow"] = agent_workflow

    # 1. Single-turn _parse_response must unwrap <think>...</think> and ```json ... ``` fences.
    wrapped_content = f"<think>internal note</think>\n```json\n{json.dumps(valid_answer)}\n```"
    fenced_doc = {
        "done": True,
        "done_reason": "stop",
        "message": {"role": "assistant", "content": wrapped_content},
        "prompt_eval_count": 100,
        "eval_count": 200,
    }
    parsed = _parse_response(cfg, photo_request, json.dumps(fenced_doc).encode())
    assert parsed.feedback.main_change == "Measured trial."

    # 2. Ollama done_reason == "length" maps cleanly to ApiError("output_limit") in both parsers.
    length_doc = {
        "done": True,
        "done_reason": "length",
        "message": {"role": "assistant", "content": '{"answer":"truncated'},
        "eval_count": 4096,
    }
    with pytest.raises(ApiError) as single_len_err:
        _parse_response(cfg, photo_request, json.dumps(length_doc).encode())
    assert single_len_err.value.code == "output_limit"
    assert "done_reason=length" in str(single_len_err.value)

    with pytest.raises(ApiError) as turn_len_err:
        parse_tool_turn_response(cfg, photo_request, json.dumps(length_doc).encode(), allow_tools=True, require_schema=True)
    assert turn_len_err.value.code == "output_limit"
    assert "done_reason=length" in str(turn_len_err.value)

    # 3. Ollama invalid_response includes actionable structural detail without breaking cloud zero-leakage.
    empty_pref = {**valid_answer, "preference": "   "}
    doc_empty_pref = {"done": True, "done_reason": "stop", "message": {"role": "assistant", "content": json.dumps(empty_pref)}}
    with pytest.raises(ApiError) as empty_err:
        _parse_response(cfg, photo_request, json.dumps(doc_empty_pref).encode())
    assert empty_err.value.code == "invalid_response"
    assert "empty feedback field 'preference'" in str(empty_err.value)

    too_long = {**valid_answer, "answer": "x" * 2450}
    doc_long = {"done": True, "done_reason": "stop", "message": {"role": "assistant", "content": json.dumps(too_long)}}
    with pytest.raises(ApiError) as long_err:
        _parse_response(cfg, photo_request, json.dumps(doc_long).encode())
    assert long_err.value.code == "invalid_response"
    assert "feedback field 'answer' length 2450 > 2400" in str(long_err.value)

    bad_wf = {**valid_answer, "workflow": {**agent_workflow, "findings": []}}
    doc_bad_wf = {"done": True, "done_reason": "stop", "message": {"role": "assistant", "content": json.dumps(bad_wf)}}
    with pytest.raises(ApiError) as wf_err:
        parse_tool_turn_response(cfg, photo_request, json.dumps(doc_bad_wf).encode(), allow_tools=False, require_schema=True)
    assert wf_err.value.code == "invalid_response"
    assert "workflow: A workflow needs one to three findings and checks." in str(wf_err.value)

    # Cloud protocols keep local_detail strictly empty on invalid_response.
    cloud_cfg = ApiConfig("https://api.openai.com/v1/responses", "gpt-4o", "responses")
    cloud_bad = {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(empty_pref)}]}]}
    with pytest.raises(ApiError) as cloud_err:
        _parse_response(cloud_cfg, photo_request, json.dumps(cloud_bad).encode())
    assert cloud_err.value.code == "invalid_response"
    assert cloud_err.value.local_detail == ""


def test_ollama_feedback_schema_enforces_workflow_and_field_bounds(photo_request):
    from photocoach.api_coach import (
        AGENT_INSTRUCTIONS,
        FEEDBACK_SCHEMA,
        GEMINI_ENDPOINT,
        build_tool_turn_payload,
        request_context,
    )

    cfg = config()
    single_payload = build_payload(cfg, photo_request)
    msgs = [{"role": "user", "text": json.dumps(request_context(photo_request)), "images": [("Current", photo_request.current_preview)]}]
    turn_payload = build_tool_turn_payload(cfg, AGENT_INSTRUCTIONS, msgs, (), force_final_schema=True)

    for schema in (single_payload["format"], turn_payload["format"]):
        props = schema["properties"]
        assert props["actions"]["maxItems"] == 18
        for name in FEEDBACK_FIELDS:
            assert props[name]["minLength"] == 1
            assert props[name]["maxLength"] == 2400
        wf_props = props["workflow"]["properties"]
        assert wf_props["objective"]["minLength"] == 1 and wf_props["objective"]["maxLength"] == 400
        assert wf_props["uncertainty"]["minLength"] == 1 and wf_props["uncertainty"]["maxLength"] == 400
        findings = wf_props["findings"]
        assert findings["minItems"] == 1 and findings["maxItems"] == 3
        assert findings["items"]["properties"]["observation"]["minLength"] == 1
        assert findings["items"]["properties"]["interpretation"]["maxLength"] == 400
        checks = wf_props["checks"]
        assert checks["minItems"] == 1 and checks["maxItems"] == 3
        variants = checks["items"]["anyOf"]
        assert len(variants) == 2
        assert variants[0]["properties"]["metric"]["enum"] == ["mean_luminance", "display_ceiling_fraction", "display_floor_fraction"]
        assert variants[0]["properties"]["direction"]["enum"] == ["increase", "decrease", "preserve"]
        assert variants[1]["properties"]["metric"]["enum"] == ["visual"]
        assert variants[1]["properties"]["direction"]["enum"] == ["visual"]

    # Shared FEEDBACK_SCHEMA and cloud payloads must remain completely untouched.
    assert "minItems" not in FEEDBACK_SCHEMA["properties"]["workflow"]["properties"]["findings"]
    assert "minLength" not in FEEDBACK_SCHEMA["properties"]["answer"]
    for cloud_cfg in (
        ApiConfig("https://api.openai.com/v1/responses", "gpt-4o", "responses"),
        ApiConfig("https://api.openai.com/v1/chat/completions", "gpt-4o", "chat_completions"),
        ApiConfig(GEMINI_ENDPOINT, "gemini-2.5-flash", "gemini"),
    ):
        cloud_payload = build_payload(cloud_cfg, photo_request)
        if cloud_cfg.protocol == "responses":
            cloud_schema = cloud_payload["text"]["format"]["schema"]
        elif cloud_cfg.protocol == "chat_completions":
            cloud_schema = cloud_payload["response_format"]["json_schema"]["schema"]
        else:
            cloud_schema = cloud_payload["generationConfig"]["responseJsonSchema"]
        assert "minItems" not in cloud_schema["properties"]["workflow"]["properties"]["findings"]
        assert "minLength" not in cloud_schema["properties"]["answer"]


def test_local_ai_five_turns_clamps_wall_budget_without_crashing(tmp_path, monkeypatch):
    import photocoach.app as module
    from photocoach.api_coach import ModelTurnResponse
    from photocoach.coach import CoachFeedback
    from PySide6.QtWidgets import QDialog

    client.clear_observed_capabilities()
    app = QApplication.instance() or QApplication([])
    w = module.MainWindow(settings_store=SettingsStore(tmp_path / "settings.json"), credentials=Vault())
    cfg = runtime.LocalConfig(
        model_dir=str(tmp_path / "models"),
        model="qwen3.6:27b",
        verified_digest="dig-5",
        engine_version="0.40.16",
        port=11435,
        supports_tools=True,
        has_thinking=True,
    )
    w._use_local_config(cfg)
    assert w.api_settings.config.timeout_seconds == 300
    assert w.api_settings.config.max_output_tokens == 0

    current = w.store.activate(Path("five-turns.ARW"), "five-turns-photo")
    pixels = np.full((32, 48, 3), 100, np.uint8)
    req = build_request(current, current, [], pixels, pixels)

    workers = []
    monkeypatch.setattr(w.network_pool, "start", workers.append)

    def fake_ensure_ready(local_cfg, cancel):
        client.register_model_capabilities(
            local_cfg.model,
            has_vision=True,
            supports_tools=True,
            has_thinking=True,
            port=local_cfg.port,
        )

    monkeypatch.setattr(w.local_runtime, "ensure_ready", fake_ensure_ready)

    class Review5Turns:
        def __init__(self, *a):
            self.include_before = SimpleNamespace(isChecked=lambda: True)
            self.skip_session = SimpleNamespace(isChecked=lambda: False)
        def selected_max_turns(self):
            return 5
        def exec(self):
            return QDialog.DialogCode.Accepted
        def deleteLater(self):
            pass

    monkeypatch.setattr(module, "SendReviewDialog", Review5Turns)

    captured = {}
    RealAgentRuntime = module.AgentRuntime

    class SpyRuntime(RealAgentRuntime):
        def __init__(self, *args, **kwargs):
            captured["run_budget"] = kwargs.get("run_budget")
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(module, "AgentRuntime", SpyRuntime)

    fb = CoachFeedback(current.photo_id, current.session_id, current.revision,
                       "Five-turn local check OK", "Reason", "Stop", "Observe", ())

    class Adapter5:
        def __init__(self, api_cfg, key):
            self.config = api_cfg
            self.supports_tools = True
        def analyze(self, *args, **kwargs):
            raise AssertionError("Should run multi-turn AgentRuntime when tools are supported")
        def step(self, *args, **kwargs):
            return ModelTurnResponse((), fb, {"role": "assistant", "content": "{}"}, 50, 40)

    monkeypatch.setattr(module, "ApiCoach", Adapter5)
    job = module.Job("api_prepare", w._generation, lambda: None, current, w._coach_generation)
    w.review_api_request(req, job)
    assert len(workers) == 1

    result = workers[0].function()
    assert captured["run_budget"].max_llm_turns == 5
    assert captured["run_budget"].max_wall_seconds == 1500.0
    assert captured["run_budget"].retouch_mode == "full_retouch"
    assert result.feedback.main_change == "Five-turn local check OK"

    w._network_active = None
    w._close_approved = True
    w.close()
    w.deleteLater()
    app.processEvents()
    client.clear_observed_capabilities()


def test_ollama_unlimited_output_64k_ctx_and_twelve_turn_clamp(photo_request):
    from photocoach.agent_runtime import RunBudget

    client.clear_observed_capabilities()
    unlimited = client.chat_payload(client.MODEL, [{"role": "user", "content": "hi"}], output_tokens=0)
    assert unlimited["options"]["num_predict"] == -1
    assert client.MAX_NUM_CTX == 65536
    assert 65536 in client._NUM_CTX_STEPS

    # RunBudget supports up to 16 turns, 10 trials, 48 tool calls, and 3600s wall budget.
    b12 = RunBudget(
        max_llm_turns=12,
        max_tool_calls=48,
        max_trial_renders=10,
        max_wall_seconds=min(3600.0, 600.0 * 12),
        retouch_mode="full_retouch",
    )
    assert b12.max_llm_turns == 12 and b12.max_wall_seconds == 3600.0
    assert b12.retouch_mode == "full_retouch"

    with pytest.raises(ValueError):
        RunBudget(max_llm_turns=17)
    with pytest.raises(ValueError):
        RunBudget(max_wall_seconds=3601.0)
    with pytest.raises(ValueError):
        RunBudget(retouch_mode="invalid")


def test_ollama_tool_turns_keep_native_thinking_strip_history_and_use_8192_budget(photo_request):
    from photocoach.agent_tools import build_default_tool_registry
    from photocoach.api_coach import AGENT_INSTRUCTIONS, build_tool_turn_payload, request_context

    client.clear_observed_capabilities()
    tools = build_default_tool_registry().specs()
    ollama_tools = [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}} for t in tools[:1]]

    # chat_payload keeps native thinking on tool turns with anti-loop sampling, disables think on schema/non-tool turns.
    with_tools = client.chat_payload(client.MODEL, [{"role": "user", "content": "hi"}], tools=ollama_tools)
    assert "think" not in with_tools
    assert with_tools["options"]["temperature"] == 0.6
    assert with_tools["options"]["top_p"] == 0.95
    assert with_tools["options"]["repeat_penalty"] == 1.05
    with_schema = client.chat_payload(client.MODEL, [{"role": "user", "content": "hi"}], schema={"type": "object"})
    assert with_schema["think"] is False
    assert with_schema["options"]["temperature"] == 0.2
    assert "repeat_penalty" not in with_schema["options"]

    # build_tool_turn_payload omits think with tools (force_final_schema=False), sets think=False on final schema,
    # and strips prior-turn "thinking" traces from replayed assistant messages.
    cfg = ApiConfig(client.ENDPOINT, client.MODEL, "ollama", 300, 8192)
    ctx_text = json.dumps(request_context(photo_request, include_before=True), ensure_ascii=False)
    messages = [
        {"role": "user", "text": ctx_text, "images": [("Before", photo_request.before_preview), ("Current", photo_request.current_preview)]},
        {
            "role": "assistant",
            "raw_message": {
                "role": "assistant",
                "content": "",
                "thinking": "long internal reasoning chain that must not accumulate in prompt history",
                "tool_calls": [{"function": {"name": "analyze_color_zones", "arguments": {"region": "whole"}}}],
            },
        },
    ]
    tool_turn = build_tool_turn_payload(cfg, AGENT_INSTRUCTIONS, messages, tools, force_final_schema=False)
    assert "think" not in tool_turn
    assert tool_turn["options"]["num_predict"] == 8192
    replayed = [m for m in tool_turn["messages"] if m.get("role") == "assistant"]
    assert len(replayed) == 1 and "thinking" not in replayed[0] and replayed[0]["tool_calls"]

    final_turn = build_tool_turn_payload(cfg, AGENT_INSTRUCTIONS, messages, tools, force_final_schema=True)
    assert final_turn["think"] is False
    assert "format" in final_turn and "tools" not in final_turn






