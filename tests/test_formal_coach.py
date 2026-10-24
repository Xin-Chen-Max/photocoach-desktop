"""Formal AI Coach: the local mock is developer-only and normal launches use the real coach."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QCheckBox, QComboBox
import pytest

from photocoach import api_coach, app as desktop
from photocoach.api_coach import ApiConfig, GEMINI_ENDPOINT
from photocoach.devmode import dev_mode_enabled
from photocoach.settings import ApiSettings, SettingsStore


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Formal-coach UI tests must not send a network request")
    monkeypatch.setattr(api_coach, "_https_transport", forbidden)


def window(tmp_path, *, dev_mode, settings=None):
    store = SettingsStore(tmp_path / "settings.json")
    if settings is not None:
        store.save(settings)
    w = desktop.MainWindow(settings_store=store, dev_mode=dev_mode)
    w.network_pool.start = lambda *_: pytest.fail("No request may start automatically")
    w.coach_collapse.setChecked(False)
    return w


def saved_settings():
    return ApiSettings(ApiConfig(GEMINI_ENDPOINT, "gemini-test-model", "gemini"),
                       {"call_limit": 5, "budget_usd": None, "reserve_usd": None,
                        "input_usd_per_million": None, "output_usd_per_million": None},
                       key_source="saved")


def visible_texts(root):
    texts = []
    for widget in root.findChildren(QWidget):
        if not widget.isVisibleTo(root):
            continue
        if isinstance(widget, (QLabel, QPushButton, QCheckBox)):
            texts.append(widget.text())
        if isinstance(widget, QComboBox):
            texts.extend(widget.itemText(i) for i in range(widget.count()))
    return texts


def test_dev_mode_switch_defaults_off_and_honours_env(monkeypatch):
    monkeypatch.delenv("PHOTOCOACH_DEV", raising=False)
    assert dev_mode_enabled() is False
    monkeypatch.setenv("PHOTOCOACH_DEV", "1")
    assert dev_mode_enabled() is True
    assert dev_mode_enabled(False) is False
    monkeypatch.setenv("PHOTOCOACH_DEV", "0")
    assert dev_mode_enabled() is False and dev_mode_enabled(True) is True


@pytest.mark.parametrize("argv, expected", [([], None), (["--dev"], True)])
def test_command_line_dev_flag(application, monkeypatch, argv, expected):
    seen = {}

    class FakeWindow:
        def __init__(self, **kwargs):
            seen.update(kwargs)
        def show(self):
            pass

    monkeypatch.setattr(desktop, "MainWindow", FakeWindow)
    monkeypatch.setattr(QApplication, "exec", lambda *a: 0)
    assert desktop.main(argv) == 0
    assert seen["dev_mode"] is expected


def test_normal_ui_has_no_mock_and_first_run_shows_setup(application, tmp_path):
    w = window(tmp_path, dev_mode=False)
    assert w.dev_mode is False and w.coach_mode.currentIndex() == 1
    assert w.analyze_button.text() == "Set up AI coach"
    assert w.coach_model_status.text() == "Set up AI coach"
    assert w.coach.calls == 0
    for root in (w, w.coach_details):
        for text in visible_texts(root):
            assert "mock" not in text.lower() and "demo" not in text.lower(), text
    assert w.coach_mode_picker.isHidden() and w.auto_mock.isHidden() and w.mock_limit_row.isHidden()
    w.close()


def test_saved_connection_restores_and_shows_ask_coach_without_sending(application, tmp_path):
    w = window(tmp_path, dev_mode=False, settings=saved_settings())
    assert w.api_settings is not None and w.coach_mode.currentIndex() == 1
    assert w.analyze_button.text() == "Ask coach"
    assert "gemini-test-model" in w.coach_model_status.text()
    assert w.api_budget.attempts == 0 and w.coach.calls == 0
    assert "mock" not in w.coach_text.toPlainText().lower()
    w.close()


def test_developer_mode_keeps_local_mock_available(application, tmp_path):
    w = window(tmp_path, dev_mode=True)
    assert w.coach_mode.currentIndex() == 0
    assert w.analyze_button.text() == "Try local demo"
    assert not w.coach_mode_picker.isHidden()
    w.close()


def test_setup_is_available_without_photo_and_sends_nothing(application, tmp_path, monkeypatch):
    w = window(tmp_path, dev_mode=False)
    seen = []
    monkeypatch.setattr(w, 'configure_api', lambda: seen.append(True))
    assert w.store.current is None and w.analyze_button.isEnabled()
    w.analyze_button.click()
    assert seen == [True] and w.api_budget.attempts == w.coach.calls == 0
    w.set_coach_paused(True)
    assert not w.analyze_button.isEnabled()
    w.close()
