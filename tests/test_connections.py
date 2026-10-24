"""Multiple connections: schema-2 store, migration, backup safety and key isolation."""
import json

import pytest

from photocoach.api_coach import ApiConfig, GEMINI_ENDPOINT
from photocoach.connections import (Connection, ConnectionBook, ConnectionStore, MAX_RECENT, connection_from_settings,
                                    infer_provider, settings_for)
from photocoach.settings import ApiSettings, SettingsError, SettingsStore, credential_target

BUDGET = {"call_limit": 5, "budget_usd": None, "reserve_usd": None,
          "input_usd_per_million": None, "output_usd_per_million": None}
GEMINI = ApiConfig(GEMINI_ENDPOINT, "gemini-test-model", "gemini")
OPENAI = ApiConfig("https://api.openai.com/v1/responses", "gpt-test", "responses")
CUSTOM = ApiConfig("https://llm.example.org/v1/chat/completions", "local-model", "chat_completions")


class Vault:
    def __init__(self, values=None):
        self.values = dict(values or {})
        self.reads = []

    def read(self, target):
        self.reads.append(target)
        return self.values.get(target)


def book(*configs):
    result = ConnectionBook(budget_settings=BUDGET)
    for index, config in enumerate(configs):
        result = result.with_connection(connection_from_settings(
            ApiSettings(config, BUDGET, key_source="saved"), connection_id=f"c{index}"))
    return result


def test_provider_inference():
    assert [infer_provider(c) for c in (GEMINI, OPENAI, CUSTOM)] == ["gemini", "openai", "custom"]


def test_round_trip_keeps_connections_active_models_and_no_secrets(tmp_path):
    store = ConnectionStore(tmp_path / "settings.json")
    saved = book(GEMINI, OPENAI).with_model("c1", "gpt-other").toggle_favorite("c0", "gemini-fav").with_active("c1")
    store.save(saved)
    text = (tmp_path / "settings.json").read_text(encoding="utf-8")
    assert "key" not in json.loads(text)["connections"][0] or json.loads(text)["connections"][0]["key_source"] == "saved"
    assert "api_key" not in text and "credential_target" not in text
    loaded = store.load()
    assert [c.id for c in loaded.connections] == ["c0", "c1"] and loaded.active == "c1"
    assert loaded.get("c1").config.model == "gpt-other" and loaded.get("c1").recent == ("gpt-other",)
    assert loaded.get("c0").favorites == ("gemini-fav",)
    assert loaded.budget_settings["call_limit"] == 5


def test_recent_models_are_most_recent_first_unique_and_bounded():
    b = book(OPENAI)
    for index in range(MAX_RECENT + 4):
        b = b.with_model("c0", f"gpt-{index}")
    b = b.with_model("c0", "gpt-5")
    recent = b.get("c0").recent
    assert recent[0] == "gpt-5" and len(recent) == MAX_RECENT and len(set(recent)) == MAX_RECENT


def test_keys_are_isolated_per_connection():
    b = book(GEMINI, OPENAI)
    gemini, openai = b.get("c0"), b.get("c1")
    assert gemini.credential_target != openai.credential_target
    vault = Vault({gemini.credential_target: "gemini-secret", openai.credential_target: "openai-secret"})
    assert openai.resolve_key("", vault) == "openai-secret"
    assert gemini.resolve_key("", vault) == "gemini-secret"
    assert vault.reads == [openai.credential_target, gemini.credential_target]
    # A connection without its own saved key never receives another provider's key.
    assert openai.resolve_key("", Vault({gemini.credential_target: "gemini-secret"})) == ""


def test_settings_adapter_preserves_runtime_type():
    connection = book(GEMINI).get("c0")
    adapted = settings_for(connection, BUDGET)
    assert adapted.config == GEMINI and adapted.key_source == "saved" and adapted.remember_settings


def test_invalid_documents_are_rejected(tmp_path):
    store = ConnectionStore(tmp_path / "settings.json")
    for text in ("{", "[]", json.dumps({"schema_version": 2}),
                 json.dumps({"schema_version": 3, "connections": [], "active": None, "budget": {}})):
        (tmp_path / "settings.json").write_text(text, encoding="utf-8")
        with pytest.raises(SettingsError):
            store.load()
    with pytest.raises(SettingsError):
        ConnectionBook((), "missing", BUDGET)
    with pytest.raises(SettingsError):
        Connection("bad id", "gemini", "x", GEMINI)


def legacy_file(tmp_path, config=GEMINI, source="saved"):
    path = tmp_path / "settings.json"
    SettingsStore(path).save(ApiSettings(config, BUDGET, key_source=source))
    return path


def test_schema1_migrates_in_memory_without_writing_and_keeps_key_target(tmp_path):
    path = legacy_file(tmp_path)
    before = path.read_bytes()
    loaded = ConnectionStore(path).load()
    assert path.read_bytes() == before and not (tmp_path / "settings.json.bak").exists()
    connection = loaded.active_connection
    assert connection.provider == "gemini" and connection.config == GEMINI and connection.key_source == "saved"
    # The saved key's vault target is unchanged, so the user need not re-enter it.
    assert connection.credential_target == credential_target(GEMINI)


def test_first_save_creates_backup_then_upgrades_and_old_backup_is_never_overwritten(tmp_path):
    path = legacy_file(tmp_path)
    original = path.read_bytes()
    store = ConnectionStore(path)
    store.save(store.load().with_model("migrated", "gemini-newer"))
    backup = tmp_path / "settings.json.bak"
    assert backup.read_bytes() == original
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 2
    # A later save, even over a (fake) schema-1 file, must keep the valid backup.
    path.write_bytes(original)
    store.save(store.load().with_model("migrated", "gemini-third"))
    assert backup.read_bytes() == original
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 2


def test_failed_main_write_leaves_old_settings_and_backup_recoverable(tmp_path, monkeypatch):
    path = legacy_file(tmp_path)
    original = path.read_bytes()
    store = ConnectionStore(path)
    loaded = store.load()
    import photocoach.connections as module
    real = module.os.replace

    def fail_main(source, destination):
        if str(destination) == str(path):
            raise OSError("simulated replace failure")
        return real(source, destination)

    monkeypatch.setattr(module.os, "replace", fail_main)
    with pytest.raises(SettingsError):
        store.save(loaded.with_model("migrated", "gemini-newer"))
    monkeypatch.undo()
    assert path.read_bytes() == original  # old valid settings untouched
    assert not list(tmp_path.glob(".photocoach-settings-*.tmp"))
    assert store.load().active_connection.config == GEMINI


def test_backup_failure_aborts_before_changing_main_file(tmp_path, monkeypatch):
    path = legacy_file(tmp_path)
    original = path.read_bytes()
    store = ConnectionStore(path)
    loaded = store.load()
    import photocoach.connections as module
    real = module.os.replace

    def fail_backup(source, destination):
        if str(destination).endswith(".bak"):
            raise OSError("simulated backup failure")
        return real(source, destination)

    monkeypatch.setattr(module.os, "replace", fail_backup)
    with pytest.raises(SettingsError):
        store.save(loaded)
    monkeypatch.undo()
    assert path.read_bytes() == original and not (tmp_path / "settings.json.bak").exists()


def test_unreadable_settings_recover_from_valid_backup_but_not_invalid_one(tmp_path):
    path = legacy_file(tmp_path)
    store = ConnectionStore(path)
    store.save(store.load())
    path.write_text("corrupt", encoding="utf-8")
    recovered = store.load()
    assert recovered.active_connection.config == GEMINI
    assert path.read_text(encoding="utf-8") == "corrupt"  # recovery is read-only
    store.save(recovered)
    assert next(tmp_path.glob("settings.json.damaged-*")).read_text() == "corrupt"
    path.write_text("corrupt", encoding="utf-8")
    (tmp_path / "settings.json.bak").write_text("also corrupt", encoding="utf-8")
    with pytest.raises(SettingsError):
        store.load()


def test_invalid_book_never_touches_existing_file(tmp_path):
    path = legacy_file(tmp_path)
    original = path.read_bytes()
    store = ConnectionStore(path)
    with pytest.raises(SettingsError):
        store.save(ConnectionBook((), None, {"call_limit": -1}))
    assert path.read_bytes() == original


def test_oversized_existing_settings_are_not_replaced_or_truncated(tmp_path):
    from photocoach.settings import MAX_SETTINGS_BYTES
    path=tmp_path/'settings.json';raw=b'x'*(MAX_SETTINGS_BYTES+100)
    path.write_bytes(raw)
    with pytest.raises(SettingsError):ConnectionStore(path).save(book(GEMINI))
    assert path.read_bytes()==raw and not list(tmp_path.glob('settings.json.damaged-*'))
