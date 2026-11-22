"""Keep automatic tests away from the user's real connection settings."""

import pytest


@pytest.fixture
def agent_workflow():
    """Minimal valid provider plan for transport tests; no live model judgment."""
    return dict(version=1, decision="recommend", objective="Compare one reversible brightness trial.",
        findings=[dict(region="whole", observation="The supplied preview is dark.",
                       interpretation="Test a modest lift and let the user judge.", category="technical", confidence="medium")],
        checks=[dict(region="whole", metric="visual", direction="visual", expectation="Compare texture and tonal separation.")],
        uncertainty="A reduced preview cannot establish RAW recoverability or aesthetic quality.")


@pytest.fixture(scope="session", autouse=True)
def qt_application_session():
    """Own Qt until every GUI fixture has finished, then destroy it explicitly.

    Leaving QWidget/Python wrappers alive until interpreter finalization can
    abort Windows ucrtbase after pytest has already printed its result.
    """
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication
    from shiboken6 import delete, isValid
    app = QApplication.instance() or QApplication([])
    yield app
    windows = list(app.topLevelWidgets())
    for window in windows:
        if not isValid(window):
            continue
        if hasattr(window, "_close_approved"):
            window._close_approved = True
        window.close()
        for name in ("pool", "network_pool"):
            pool = getattr(window, name, None)
            if pool is not None:
                assert pool.waitForDone(5000), "Test left a background worker running"
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    for window in windows:
        if isValid(window):
            delete(window)
    delete(app)


@pytest.fixture(autouse=True)
def isolated_user_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local-app-data"))
    monkeypatch.delenv("PHOTOCOACH_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PHOTOCOACH_DEV", raising=False)
