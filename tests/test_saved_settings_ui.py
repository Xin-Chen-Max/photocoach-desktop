"""Restart and saved-key workflows with an isolated filesystem and fake vault."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
from pathlib import Path
import time
import threading

import pytest
from PySide6.QtWidgets import QApplication, QDialog

from photocoach import api_coach, app as desktop
from photocoach.api_coach import ApiConfig, GEMINI_ENDPOINT
from photocoach.budget import SessionBudget
from photocoach.coach_ui import ApiSettingsDialog
from photocoach.credentials import CredentialError
from photocoach.settings import ApiSettings, SettingsError, SettingsStore, credential_target, resolve_api_key


class Vault:
    def __init__(self):
        self.values = {}
        self.calls = []
        self.failure = None

    def read(self, target):
        self.calls.append(("read", target))
        if self.failure == "read":
            raise CredentialError("read")
        return self.values.get(target)

    def write(self, target, key):
        self.calls.append(("write", target))
        if self.failure == "write":
            raise CredentialError("write")
        self.values[target] = key

    def delete(self, target):
        self.calls.append(("delete", target))
        if self.failure == "delete":
            raise CredentialError("delete")
        self.values.pop(target, None)


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Saved-settings tests must not send a network request")
    monkeypatch.setattr(api_coach, "_https_transport", forbidden)


@pytest.fixture
def storage(tmp_path):
    return SettingsStore(tmp_path / "settings.json"), Vault()


def configured_dialog(application, storage, previous=None, budget=None):
    store, vault = storage
    dialog = ApiSettingsDialog(None, previous, budget or SessionBudget(),
                               settings_store=store, credentials=vault)
    if previous is None:
        dialog.protocol.setCurrentIndex(dialog.protocol.findData("gemini"))
        dialog.model.setText("gemini-test-model")
        dialog.key.setText("fake-ui-test-key")
    return dialog


def save_key(application, storage):
    dialog = configured_dialog(application, storage)
    dialog.remember_key.setChecked(True)
    dialog.calls.setValue(7)
    dialog.validate_and_accept()
    assert dialog.result() == QDialog.DialogCode.Accepted
    result = dialog.result_settings
    dialog.deleteLater()
    return result


def wait_metadata(application, dialog):
    deadline = time.monotonic() + 5
    while dialog._check_worker is not None:
        application.processEvents()
        assert time.monotonic() < deadline
        time.sleep(.002)


@pytest.mark.parametrize("source", ["saved", "session", "environment"])
def test_model_catalog_empty_id_sources_pagination_and_explicit_selection(application, storage, monkeypatch, source):
    from photocoach import coach_ui
    from photocoach.api_coach import GeminiModel, GeminiModelsPage
    original = save_key(application, storage)
    budget = SessionBudget(call_limit=1)
    budget.settle(budget.reserve(), success=False)
    dialog = configured_dialog(application, storage, original, budget)
    dialog.key_source.setCurrentIndex(dialog.key_source.findData(source))
    if source == "session":
        dialog.key.setText("fake-ui-test-key")
    elif source == "environment":
        dialog.environment.setText("CATALOG_TEST_KEY")
        monkeypatch.setenv("CATALOG_TEST_KEY", "fake-ui-test-key")
    dialog.model.clear()
    calls = []
    gui_thread = threading.get_ident()
    def load(config, key, cancel, token):
        assert threading.get_ident() != gui_thread
        assert key == "fake-ui-test-key" and not hasattr(config, "model")
        calls.append(token)
        if not token:
            return GeminiModelsPage((GeminiModel("gemini-first", "First"),), "page-two")
        return GeminiModelsPage((GeminiModel("gemini-second", "Second"), GeminiModel("gemini-first", "First")))
    monkeypatch.setattr(coach_ui, "list_gemini_models", load)
    assert not calls
    dialog.load_models()
    wait_metadata(application, dialog)
    assert calls == [""] and dialog.model.text() == ""
    assert not dialog.more_models_button.isHidden()
    assert dialog.result_settings is None  # Loading never applies settings.
    dialog.model_choices.activated.emit(dialog.model_choices.findData("gemini-first"))
    assert dialog.model.text() == "gemini-first"
    assert dialog.key_source.currentData() == source
    assert dialog.key.text() == ("fake-ui-test-key" if source == "session" else "")
    dialog.load_models(more=True)
    wait_metadata(application, dialog)
    assert calls == ["", "page-two"] and dialog.model.text() == "gemini-first"
    assert dialog.model_choices.count() == 3 and dialog.more_models_button.isHidden()
    dialog.model_choices.activated.emit(dialog.model_choices.findData("gemini-second"))
    assert dialog.model.text() == "gemini-second"
    assert not dialog.connection_details.isVisible()
    dialog.validate_and_accept()
    assert dialog.result_settings.config.model == "gemini-second"
    assert resolve_api_key(dialog.result_settings, storage[1]) == "fake-ui-test-key"
    assert budget.attempts == 1 and budget.in_flight == 0 and budget.unresolved == 1
    dialog.deleteLater()


def test_catalog_missing_saved_key_never_falls_back_or_starts_worker(application, storage, monkeypatch):
    original = save_key(application, storage)
    storage[1].values.clear()
    monkeypatch.setenv("GEMINI_API_KEY", "must-not-fall-back")
    dialog = configured_dialog(application, storage, original)
    dialog.model.clear()
    dialog.load_models()
    assert dialog._check_worker is None and dialog.model.text() == ""
    assert "No photo" in dialog.models_status.text()
    assert "must-not-fall-back" not in dialog.models_status.text()
    dialog.reject()
    dialog.deleteLater()


@pytest.mark.parametrize("outcome", ["failure", "timeout", "empty", "close", "key", "endpoint", "model"])
def test_catalog_failures_and_stale_results_preserve_configuration(application, storage, monkeypatch, outcome):
    from photocoach import coach_ui
    from photocoach.api_coach import GeminiModel, GeminiModelsPage, ModelCheckError, ApiError
    dialog = configured_dialog(application, storage)
    started, release = threading.Event(), threading.Event()
    calls = []
    def load(config, key, cancel, token):
        calls.append(cancel)
        started.set()
        assert release.wait(5)
        if outcome == "failure":
            raise ModelCheckError(ApiError("authentication", http_status=401))
        if outcome == "timeout":
            raise ModelCheckError(ApiError("timeout"))
        return GeminiModelsPage(() if outcome == "empty" else (GeminiModel("gemini-late", "Late"),))
    monkeypatch.setattr(coach_ui, "list_gemini_models", load)
    try:
        dialog.load_models()
        assert started.wait(5)
        dialog.load_models()
        dialog.check_model()
        assert len(calls) == 1
        assert not dialog.model.isEnabled() and not dialog.save_button.isEnabled()
        if outcome == "close":
            dialog.reject()
        elif outcome == "key":
            dialog.key.setText("replacement-test-key")
        elif outcome == "endpoint":
            dialog.endpoint.setText("https://generativelanguage.googleapis.com/v1/models")
        elif outcome == "model":
            dialog.model.setText("gemini-manually-changed")
    finally:
        release.set()
    wait_metadata(application, dialog)
    assert not dialog._catalog and dialog.model_choices.count() == 1
    assert dialog.model.text() == ("gemini-manually-changed" if outcome == "model" else "gemini-test-model")
    assert dialog.result_settings is None and "fake-ui-test-key" not in dialog.models_status.text()
    if outcome == "failure":
        assert "HTTP 401" in dialog.models_status.text()
    elif outcome == "timeout":
        assert "timed out" in dialog.models_status.text()
    elif outcome == "empty":
        assert "No matching" in dialog.models_status.text()
    else:
        assert calls[0].is_set()
    dialog.close()
    dialog.deleteLater()


def test_catalog_failed_refresh_keeps_existing_options_and_model(application, storage, monkeypatch):
    from photocoach import coach_ui
    from photocoach.api_coach import GeminiModel, GeminiModelsPage, ModelCheckError, ApiError
    dialog = configured_dialog(application, storage)
    monkeypatch.setattr(coach_ui, "list_gemini_models", lambda *args: GeminiModelsPage((GeminiModel("gemini-kept", "Kept"),)))
    dialog.load_models()
    wait_metadata(application, dialog)
    def fail(*args):
        raise ModelCheckError(ApiError("unavailable", http_status=503))
    monkeypatch.setattr(coach_ui, "list_gemini_models", fail)
    dialog.load_models()
    wait_metadata(application, dialog)
    assert "gemini-kept" in dialog._catalog and dialog.model_choices.isEnabled()
    assert dialog.model.text() == "gemini-test-model"
    dialog.reject()
    dialog.deleteLater()


def test_restart_restores_settings_without_key_read_or_network(application, storage):
    store, vault = storage
    original = save_key(application, storage)
    raw = store.path.read_text()
    assert "fake-ui-test-key" not in raw and '"api_key"' not in raw
    assert original.api_key == "" and original.key_source == "saved"
    vault.calls.clear()
    window = desktop.MainWindow(settings_store=store, credentials=vault, dev_mode=True)
    try:
        assert window.api_settings.config == original.config
        assert window.api_budget.call_limit == 7 and window.api_budget.attempts == 0
        assert window.coach_mode.currentIndex() == 0
        assert window._network_active is None and not vault.calls
        assert "new local session count" in window.coach_text.toPlainText()
        reopened = configured_dialog(application, storage, window.api_settings)
        assert reopened.key.text() == "" and not reopened.key.isEnabled()
        assert reopened.key_source.currentData() == "saved"
        assert not vault.calls
        assert resolve_api_key(window.api_settings, vault) == "fake-ui-test-key"
        reopened.reject()
        reopened.deleteLater()
    finally:
        window.close()
        window.deleteLater()


@pytest.mark.parametrize("action", ["delete", "replace", "forget_settings"])
def test_cancel_keeps_settings_and_key(application, storage, action):
    store, vault = storage
    previous = save_key(application, storage)
    original = store.path.read_bytes()
    vault.calls.clear()
    dialog = configured_dialog(application, storage, previous)
    if action == "delete":
        dialog.delete_key_button.click()
    elif action == "replace":
        dialog.replace_key_button.click()
        dialog.key.setText("replacement-fake-key")
    else:
        dialog.remember_settings.setChecked(False)
    dialog.reject()
    assert not vault.calls and store.path.read_bytes() == original
    assert resolve_api_key(store.load(), vault) == "fake-ui-test-key"
    dialog.deleteLater()


def test_delete_survives_restart_without_environment_fallback(application, storage, monkeypatch):
    store, vault = storage
    previous = save_key(application, storage)
    monkeypatch.setenv("GEMINI_API_KEY", "must-not-resurrect-key")
    dialog = configured_dialog(application, storage, previous)
    dialog.delete_key_button.click()
    assert not any(call[0] == "delete" for call in vault.calls)
    dialog.validate_and_accept()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert not vault.values
    restored = store.load()
    assert restored.key_source == "session"
    assert resolve_api_key(restored, vault) == ""
    assert "must-not-resurrect-key" not in store.path.read_text()
    dialog.deleteLater()


def test_replace_saved_key_never_serializes_secret(application, storage):
    store, vault = storage
    previous = save_key(application, storage)
    dialog = configured_dialog(application, storage, previous)
    dialog.replace_key_button.click()
    dialog.key.setText("replacement-fake-key")
    dialog.validate_and_accept()
    assert resolve_api_key(store.load(), vault) == "replacement-fake-key"
    assert "replacement-fake-key" not in store.path.read_text()
    assert dialog.result_settings.api_key == ""
    dialog.deleteLater()


@pytest.mark.parametrize("source", ["session", "saved", "environment"])
def test_destination_change_clears_key_source_but_model_change_keeps_it(application, storage, source):
    previous = save_key(application, storage)
    previous = replace(previous, key_source=source, api_key="fake-memory-key" if source == "session" else "")
    dialog = configured_dialog(application, storage, previous)
    dialog.model.setText("gemini-another-model")
    assert dialog.key_source.currentData() == source
    assert dialog.key.text() == previous.api_key
    dialog.protocol.setCurrentIndex(dialog.protocol.findData("chat_completions"))
    assert dialog.key_source.currentData() == "session" and dialog.key.text() == ""
    assert not dialog.remember_key.isChecked()
    dialog.reject()
    dialog.deleteLater()


def test_explicit_environment_source_is_remembered_without_value(application, storage, monkeypatch):
    store, vault = storage
    monkeypatch.setenv("CUSTOM_TEST_KEY", "fake-environment-key")
    dialog = configured_dialog(application, storage)
    dialog.key_source.setCurrentIndex(dialog.key_source.findData("environment"))
    dialog.environment.setText("CUSTOM_TEST_KEY")
    dialog.validate_and_accept()
    assert resolve_api_key(store.load(), vault) == "fake-environment-key"
    assert "fake-environment-key" not in store.path.read_text()
    assert not vault.calls
    dialog.deleteLater()


def test_vault_write_failure_keeps_entries_and_allows_session_use(application, storage):
    store, vault = storage
    vault.failure = "write"
    dialog = configured_dialog(application, storage)
    dialog.remember_key.setChecked(True)
    dialog.validate_and_accept()
    assert dialog.result_settings is None and dialog.key.text() == "fake-ui-test-key"
    assert "could not be saved" in dialog.storage_status.text()
    assert not store.path.exists()
    dialog.remember_key.setChecked(False)
    dialog.validate_and_accept()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert resolve_api_key(dialog.result_settings, vault) == "fake-ui-test-key"
    assert resolve_api_key(store.load(), vault) == ""
    dialog.deleteLater()


def test_failed_delete_is_not_reported_as_success(application, storage):
    store, vault = storage
    previous = save_key(application, storage)
    before = store.path.read_bytes()
    vault.failure = "delete"
    dialog = configured_dialog(application, storage, previous)
    dialog.delete_key_button.click()
    dialog.validate_and_accept()
    assert dialog.result_settings is None
    assert "could not be deleted" in dialog.storage_status.text()
    assert store.path.read_bytes() == before and vault.values
    dialog.reject()
    dialog.deleteLater()


def test_settings_write_failure_keeps_entries(application, storage, monkeypatch):
    store, vault = storage
    def fail(*args):
        raise SettingsError("save")
    dialog = configured_dialog(application, storage)
    monkeypatch.setattr(dialog.connection_store, "save", fail)
    dialog.validate_and_accept()
    assert dialog.result_settings is None
    assert dialog.model.text() == "gemini-test-model" and dialog.key.text() == "fake-ui-test-key"
    assert "could not be saved" in dialog.storage_status.text()
    dialog.remember_settings.setChecked(False)
    dialog.validate_and_accept()
    assert dialog.result() == QDialog.DialogCode.Accepted
    dialog.deleteLater()


def test_unwritable_settings_allow_session_use_with_clear_notice(application, storage, monkeypatch):
    store, vault = storage
    save_key(application, storage)
    original = store.path.read_bytes()
    def fail(*args):
        raise SettingsError("clear")
    dialog = configured_dialog(application, storage)
    monkeypatch.setattr(dialog.connection_store, "clear", fail)
    dialog.remember_settings.setChecked(False)
    dialog.validate_and_accept()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert resolve_api_key(dialog.result_settings, vault) == "fake-ui-test-key"
    assert "may return on restart" in dialog.result_notice
    assert store.path.read_bytes() == original
    dialog.deleteLater()


def test_corrupt_saved_settings_do_not_block_editor(application, storage):
    store, vault = storage
    store.path.write_text('{"key":"private-value-not-for-display"}')
    window = desktop.MainWindow(settings_store=store, credentials=vault, dev_mode=True)
    try:
        assert window.api_settings is None and window.open_button.isEnabled()
        assert window.coach_mode.currentIndex() == 0
        assert "private-value" not in window.coach_text.toPlainText()
        assert "invalid" in window.coach_text.toPlainText()
        assert not vault.calls
    finally:
        window.close()
        window.deleteLater()


def test_apply_preserves_current_session_attempts(application, storage, monkeypatch):
    store, vault = storage
    window = desktop.MainWindow(settings_store=store, credentials=vault)
    ticket = window.api_budget.reserve()
    window.api_budget.settle(ticket, success=False)
    def apply(dialog):
        dialog.protocol.setCurrentIndex(dialog.protocol.findData("gemini"))
        dialog.model.setText("gemini-test-model")
        dialog.calls.setValue(8)
        dialog.validate_and_accept()
        return dialog.result()
    monkeypatch.setattr(ApiSettingsDialog, "exec", apply)
    try:
        window._review_settings = object()
        window.configure_api()
        assert window._review_settings is None
        assert window.api_budget.attempts == 1 and window.api_budget.unresolved == 1
        assert window.api_budget.call_limit == 8
        assert "No request was sent" in window.coach_text.toPlainText()
    finally:
        window.close()
        window.deleteLater()


def test_advanced_costs_collapsed_and_retained(application, storage):
    previous = save_key(application, storage)
    previous = replace(previous, budget_settings={"call_limit": 5, "budget_usd": "2", "reserve_usd": "0.1"})
    dialog = configured_dialog(application, storage, previous)
    assert dialog.advanced_costs.isHidden()
    dialog.advanced_button.click()
    assert not dialog.advanced_costs.isHidden()
    assert dialog.amounts["budget_usd"].text() == "2"
    dialog.advanced_button.click()
    dialog.validate_and_accept()
    assert dialog.result_settings.budget_settings["budget_usd"] == 2
    dialog.deleteLater()


def test_reviewed_text_request_uses_saved_key_and_503_keeps_connection(application, storage, monkeypatch):
    store, vault = storage
    save_key(application, storage)
    original = store.path.read_bytes()
    window = desktop.MainWindow(settings_store=store, credentials=vault)
    window.coach_mode.setCurrentIndex(1)
    calls = []
    class Review:
        def __init__(self, *args):
            pass
        def exec(self):
            return QDialog.DialogCode.Accepted
        def deleteLater(self):
            pass
    def fail(config, key, cancel, **kwargs):
        calls.append(config.model)
        assert key == "fake-ui-test-key"
        raise api_coach.ApiError("unavailable", http_status=503)
    monkeypatch.setattr(desktop, "TextProbeReviewDialog", Review)
    monkeypatch.setattr(desktop, "run_text_probe", fail)
    settings = window.api_settings
    try:
        window.test_text_only()
        deadline = time.monotonic() + 5
        while window._network_active is not None:
            application.processEvents()
            assert time.monotonic() < deadline
            time.sleep(0.002)
        assert calls == ["gemini-test-model"]
        assert window.api_settings is settings and store.path.read_bytes() == original
        assert window.api_budget.attempts == 1 and window.api_budget.unresolved == 1
        assert "HTTP 503" in window.coach_text.toPlainText()
        assert "settings are unchanged" in window.coach_text.toPlainText()
    finally:
        window.close()
        window.deleteLater()

def test_visible_coach_mode_and_model_switch_are_passive(application, storage):
    store, vault = storage
    original = save_key(application, storage)
    vault.calls.clear()
    window = desktop.MainWindow(settings_store=store, credentials=vault, dev_mode=True)
    try:
        window.show()
        window.coach_collapse.setChecked(False)
        application.processEvents()
        assert window.coach_mode_picker.isVisible()
        assert window.coach_mode_picker.currentIndex() == 0
        assert window.analyze_button.text() == 'Try local demo'
        window.goal.setText('Natural light')
        window.question.setText('How should I lift the hillside?')
        window._remember_coach_models(['gemini-first', 'gemini-second'])
        window.coach_mode_picker.setCurrentIndex(1)
        assert window.coach_mode.currentIndex() == 1
        assert window.analyze_button.text() == 'Review and ask coach…'
        assert window.coach_model_picker.findData('gemini-second') >= 0
        window._review_settings = window.api_settings
        attempts = window.api_budget.attempts
        window.coach_model_picker.activated.emit(window.coach_model_picker.findData('gemini-second'))
        assert window.api_settings.config.model == 'gemini-second'
        assert window.api_settings.key_source == original.key_source
        assert window.api_settings.config.endpoint == original.config.endpoint
        assert window._review_settings is None
        assert window.api_budget.attempts == attempts == 0
        assert window._network_active is None and not vault.calls
        assert store.load().config.model == "gemini-second"  # remembered connection restores the selection
        assert window.goal.text() == 'Natural light'
        assert window.question.text() == 'How should I lift the hillside?'
        window.coach_mode.setCurrentIndex(0)
        assert window.coach_mode_picker.currentIndex() == 0
        assert not window.coach_model_picker.isEnabled()
        window.api_settings = replace(original, config=ApiConfig('https://example.com/v1/chat/completions', 'other-model', 'chat_completions'))
        window._update_coach_status()
        assert window.coach_model_picker.findData('gemini-second') == -1
        assert window.coach_model_picker.findData('other-model') >= 0
    finally:
        window._close_approved = True
        window.close()
        window.deleteLater()


def test_model_picker_open_connection_and_cancel_preserves_settings(application, storage, monkeypatch):
    store, vault = storage
    save_key(application, storage)
    window = desktop.MainWindow(settings_store=store, credentials=vault)
    try:
        window.coach_mode_picker.setCurrentIndex(1)
        previous = window.api_settings
        monkeypatch.setattr(ApiSettingsDialog, 'exec', lambda self: QDialog.DialogCode.Rejected)
        window.coach_model_picker.activated.emit(window.coach_model_picker.count()-1)
        assert window.api_settings is previous
        assert window.coach_model_picker.currentData() == previous.config.model
        assert window.api_budget.attempts == 0
    finally:
        window.close()
        window.deleteLater()

def test_loaded_catalog_reaches_main_model_dropdown(application, storage, monkeypatch):
    store, vault = storage
    original = save_key(application, storage)
    window = desktop.MainWindow(settings_store=store, credentials=vault, dev_mode=True)
    def accept(dialog):
        dialog._catalog = {'gemini-option-a': object(), 'gemini-option-b': object()}
        dialog.result_settings = original
        dialog.result_notice = ''
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(ApiSettingsDialog, 'exec', accept)
    try:
        window.configure_api()
        assert window.coach_model_picker.findData('gemini-option-a') >= 0
        assert window.coach_model_picker.findData('gemini-option-b') >= 0
        assert window.coach_model_picker.currentData() == original.config.model
        assert window.coach_mode_picker.currentIndex() == 0
        assert window.api_budget.attempts == 0
    finally:
        window.close()
        window.deleteLater()


def test_reopening_connection_preserves_id_favorites_and_other_destinations(application, storage):
    from photocoach.connections import ConnectionStore
    store, vault = storage
    previous = save_key(application, storage)
    multi = ConnectionStore(store.path)
    book = multi.load()
    original_id = book.active
    book = book.toggle_favorite(original_id, previous.config.model)
    multi.save(book)
    for _ in range(3):
        dialog = configured_dialog(application, storage, previous)
        dialog.validate_and_accept()
        previous = dialog.result_settings
        assert previous is not None
        dialog.deleteLater()
    restored = multi.load()
    assert len(restored.connections) == 1 and restored.active == original_id
    assert previous.config.model in restored.active_connection.favorites
