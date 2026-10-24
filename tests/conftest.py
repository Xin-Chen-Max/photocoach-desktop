"""Keep automatic tests away from the user's real connection settings."""

import pytest


@pytest.fixture(autouse=True)
def isolated_user_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local-app-data"))
    monkeypatch.delenv("PHOTOCOACH_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PHOTOCOACH_DEV", raising=False)
