from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path

import pytest

from photocoach.api_coach import ApiConfig, GEMINI_ENDPOINT
from photocoach.settings import (ApiSettings, MAX_SETTINGS_BYTES, SettingsError,
                                SettingsStore, credential_target, resolve_api_key)


def settings(**changes):
    value = ApiSettings(ApiConfig(endpoint=GEMINI_ENDPOINT, model="gemini-test", protocol="gemini"),
                        {"call_limit": 7, "budget_usd": Decimal("0.2000"),
                         "reserve_usd": Decimal("0.025"), "input_usd_per_million": None,
                         "output_usd_per_million": None}, api_key="synthetic-secret",
                        key_environment="GEMINI_API_KEY", key_source="saved", remember_settings=True)
    return replace(value, **changes)


class Credentials:
    def __init__(self, result=None):
        self.result = result
        self.reads = []

    def read(self, target):
        self.reads.append(target)
        return self.result


def test_settings_round_trip_preserves_decimal_config_and_source_without_key(tmp_path):
    store = SettingsStore(tmp_path / "PhotoCoach" / "settings.json")
    original = settings()
    store.save(original)
    text = store.path.read_text(encoding="utf-8")
    document = json.loads(text)
    assert set(document) == {"schema_version", "config", "budget", "key_source", "key_environment"}
    assert "synthetic-secret" not in text
    assert "api_key" not in text
    assert "credential_target" not in text
    assert document["budget"]["budget_usd"] == "0.2000"
    assert store.load() == replace(original, api_key="")
    assert "synthetic-secret" not in repr(original)


def test_missing_read_and_clear_do_not_create_directories(tmp_path):
    store = SettingsStore(tmp_path / "missing" / "settings.json")
    assert store.load() is None
    store.clear()
    assert not store.path.parent.exists()


def test_default_location_uses_user_local_app_data_without_writing(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    store = SettingsStore()
    assert store.path == tmp_path / "PhotoCoach" / "settings.json"
    assert store.load() is None
    assert not store.path.parent.exists()


@pytest.mark.parametrize("value", [None, "", "relative-folder"])
def test_missing_or_relative_local_folder_has_safe_error(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("LOCALAPPDATA", raising=False)
    else:
        monkeypatch.setenv("LOCALAPPDATA", value)
    with pytest.raises(SettingsError) as caught:
        SettingsStore()
    assert caught.value.code == "location"
    assert "relative-folder" not in str(caught.value)


@pytest.mark.parametrize("location, field, value", [
    (None, "schema_version", 2),
    (None, "schema_version", True),
    (None, "credential_target", "arbitrary/other-app"),
    (None, "api_key", "must-not-load"),
    (None, "key_source", "fallback"),
    (None, "key_environment", "invalid name"),
    ("config", "timeout_seconds", True),
    ("config", "max_output_tokens", 8193),
    ("config", "endpoint", "http://unsafe.invalid"),
    ("config", "endpoint", "https://user:secret@unsafe.invalid/path"),
    ("config", "endpoint", "https://api.invalid/path?key=secret"),
    ("config", "unknown", "extra"),
    ("budget", "call_limit", True),
    ("budget", "reserve_usd", "NaN"),
    ("budget", "budget_usd", "-1"),
    ("budget", "reserve_usd", None),
    ("budget", "input_usd_per_million", "2"),
    ("budget", "budget_usd", 0.2),
])
def test_strict_schema_rejects_unsafe_or_invalid_saved_values(tmp_path, location, field, value):
    store = SettingsStore(tmp_path / "settings.json")
    store.save(settings())
    document = json.loads(store.path.read_text(encoding="utf-8"))
    target = document if location is None else document[location]
    target[field] = value
    store.path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(SettingsError):
        store.load()


@pytest.mark.parametrize("raw", [b'{"schema_version":1,"schema_version":1}', b'{"value":NaN}',
                                  b'[]', b'\xff', b'{' * 2000, b' ' * (MAX_SETTINGS_BYTES + 1)],
                         ids=["duplicate", "nonfinite", "array", "encoding", "nesting", "oversize"])
def test_malformed_oversized_or_duplicate_json_is_rejected(tmp_path, raw):
    store = SettingsStore(tmp_path / "settings.json")
    store.path.write_bytes(raw)
    with pytest.raises(SettingsError):
        store.load()


def test_invalid_save_leaves_existing_settings_and_does_not_create_new_directory(tmp_path):
    existing = SettingsStore(tmp_path / "settings.json")
    existing.save(settings())
    before = existing.path.read_bytes()
    invalid = settings(budget_settings={"call_limit": -1})
    with pytest.raises(SettingsError):
        existing.save(invalid)
    assert existing.path.read_bytes() == before
    missing = SettingsStore(tmp_path / "missing" / "settings.json")
    with pytest.raises(SettingsError):
        missing.save(invalid)
    assert not missing.path.parent.exists()


def test_atomic_replace_failure_preserves_old_file_and_removes_temporary_file(tmp_path, monkeypatch):
    store = SettingsStore(tmp_path / "settings.json")
    store.save(settings())
    before = store.path.read_bytes()

    def fail_replace(*_args):
        raise PermissionError("private-path/synthetic-secret")

    monkeypatch.setattr("photocoach.settings.os.replace", fail_replace)
    with pytest.raises(SettingsError) as caught:
        store.save(settings(budget_settings={"call_limit": 2}))
    assert caught.value.code == "save"
    assert "private-path" not in str(caught.value)
    assert store.path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [store.path]


def test_clear_removes_only_own_file(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    store.save(settings())
    unrelated = tmp_path / "keep.txt"
    unrelated.write_text("keep", encoding="utf-8")
    store.clear()
    store.clear()
    assert unrelated.read_text(encoding="utf-8") == "keep"
    assert not store.path.exists()


def test_read_and_clear_failures_are_safe_and_clear_never_recurses(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    store.path.mkdir()
    child = store.path / "keep.txt"
    child.write_text("keep", encoding="utf-8")
    for method, code in ((store.load, "read"), (store.clear, "clear")):
        with pytest.raises(SettingsError) as caught:
            method()
        assert caught.value.code == code
        assert str(tmp_path) not in str(caught.value)
    assert child.read_text(encoding="utf-8") == "keep"


def test_read_does_not_resolve_environment_or_credentials(tmp_path, monkeypatch):
    store = SettingsStore(tmp_path / "settings.json")
    store.save(settings(key_source="environment"))
    monkeypatch.setenv("GEMINI_API_KEY", "not-loaded-from-environment")
    loaded = store.load()
    assert loaded.api_key == ""
    assert loaded.key_source == "environment"
    assert loaded.remember_settings is True


def test_credential_target_normalizes_only_equivalent_endpoint_forms_and_excludes_model():
    config = ApiConfig(endpoint="https://API.EXAMPLE.invalid:443/v1/respond/", model="model-one")
    target = credential_target(config)
    assert target == credential_target(replace(config, endpoint="HTTPS://api.example.invalid/v1/respond", model="model-two"))
    assert target.startswith("PhotoCoach/api/") and len(target.rsplit("/", 1)[1]) == 64
    for changed in (
        replace(config, endpoint="https://another.invalid/v1/respond"),
        replace(config, endpoint="https://api.example.invalid:444/v1/respond"),
        replace(config, endpoint="https://api.example.invalid/v2/respond"),
        replace(config, endpoint="https://api.example.invalid/v1/Respond"),
        replace(config, protocol="chat_completions"),
    ):
        assert credential_target(changed) != target
    ipv6 = replace(config, endpoint="https://[::1]:443/")
    assert credential_target(ipv6) == credential_target(replace(ipv6, endpoint="https://[::1]"))


def test_resolution_uses_exactly_selected_source_without_fallback(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "environment-key")
    vault = Credentials("vault-key")
    assert resolve_api_key(settings(key_source="session"), vault) == "synthetic-secret"
    assert resolve_api_key(settings(key_source="session", api_key=""), vault) == ""
    assert resolve_api_key(settings(key_source="environment"), vault) == "environment-key"
    monkeypatch.delenv("GEMINI_API_KEY")
    assert resolve_api_key(settings(key_source="environment"), vault) == ""
    assert resolve_api_key(settings(), vault) == "synthetic-secret"
    assert vault.reads == []
    blank = settings(api_key="")
    assert resolve_api_key(blank, vault) == "vault-key"
    assert vault.reads == [credential_target(blank.config)]
    assert resolve_api_key(blank, Credentials()) == ""
