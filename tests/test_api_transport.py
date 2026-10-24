"""Offline transport failure coverage: no real connections, keys, or provider calls."""
import socket
import threading
import time

import pytest

from photocoach import api_coach as api


PRIVATE_ERROR = "synthetic-secret-and-private-photo-path"


@pytest.fixture(autouse=True)
def isolated_transport_guard(monkeypatch):
    # A regression that leaks the guard must not turn later tests into busy failures.
    monkeypatch.setattr(api, "_TRANSPORT_LOCK", threading.Lock())


def wait_for_transport_release():
    deadline = time.monotonic() + 2
    while api._TRANSPORT_LOCK.locked() and time.monotonic() < deadline:
        time.sleep(.01)
    assert not api._TRANSPORT_LOCK.locked()


def fake_connection(*, fail_stage=None, close_fails=False, status=200):
    calls = []

    class Connection:
        sock = None

        def __init__(self, *args, **kwargs):
            calls.append("initialize")

        def connect(self):
            calls.append("connect")
            if fail_stage == "connecting":
                raise socket.timeout(PRIVATE_ERROR)

        def request(self, *args, **kwargs):
            calls.append("request")
            if fail_stage == "sending":
                raise socket.timeout(PRIVATE_ERROR)

        def getresponse(self):
            calls.append("getresponse")
            if fail_stage == "waiting":
                raise socket.timeout(PRIVATE_ERROR)

            class Response:
                def read(self, *args):
                    calls.append("read")
                    if fail_stage == "receiving":
                        raise socket.timeout(PRIVATE_ERROR)
                    return b"{}"

            response = Response()
            response.status = status
            return response

        def close(self):
            calls.append("close")
            if close_fails:
                raise OSError(PRIVATE_ERROR)

    return Connection, calls


@pytest.mark.parametrize("stage", ["connecting", "sending", "waiting", "receiving"])
def test_socket_timeout_identifies_phase_without_retry_or_private_details(monkeypatch, stage):
    connection, calls = fake_connection(fail_stage=stage)
    monkeypatch.setattr(api.http.client, "HTTPSConnection", connection)
    config = api.ApiConfig(model="offline-test", timeout_seconds=1)

    with pytest.raises(api.ApiError) as failure:
        api._https_transport(config, "dummy-test-key", b"{}", threading.Event())

    error = failure.value
    assert error.code == "timeout"
    assert error.transport_stage == stage
    assert 0 <= error.elapsed_seconds < 2
    assert error.timeout_seconds == 1
    assert error.request_may_have_been_sent is (stage != "connecting")
    assert PRIVATE_ERROR not in str(error)
    assert "dummy-test-key" not in str(error)
    assert calls.count("connect") == 1
    assert calls.count("request") == (0 if stage == "connecting" else 1)
    wait_for_transport_release()


@pytest.mark.parametrize("cancel", [False, True], ids=["deadline", "cancelled"])
def test_late_connection_never_sends_after_abandonment(monkeypatch, cancel):
    released = threading.Event()
    cancellation = threading.Event()
    calls = []

    class Connection:
        sock = None

        def __init__(self, *args, **kwargs):
            pass

        def connect(self):
            calls.append("connect")
            if cancel:
                cancellation.set()
            released.wait(3)

        def request(self, *args, **kwargs):
            calls.append("request")

        def getresponse(self):
            pytest.fail("An abandoned connection must not send a request or wait for a response")

        def close(self):
            calls.append("close")

    monkeypatch.setattr(api.http.client, "HTTPSConnection", Connection)
    started = time.monotonic()
    try:
        with pytest.raises(api.ApiError) as failure:
            api._https_transport(api.ApiConfig(model="offline-test", timeout_seconds=1),
                                 "dummy-test-key", b"{}", cancellation)
        assert time.monotonic() - started < 2.5
        assert failure.value.code == ("cancelled" if cancel else "timeout")
        assert failure.value.transport_stage == "connecting"
        assert failure.value.request_may_have_been_sent is False
        assert "request" not in calls
        # A late DNS/TLS worker must retain the guard until it actually exits.
        with pytest.raises(api.ApiError) as busy:
            api._https_transport(api.ApiConfig(model="offline-test"),
                                 "dummy-test-key", b"{}", threading.Event())
        assert busy.value.code == "busy"
    finally:
        released.set()
        wait_for_transport_release()
    assert calls.count("connect") == 1
    assert "request" not in calls


@pytest.mark.parametrize("failure_point", ["constructor", "thread_start"])
def test_setup_failure_releases_guard_and_sanitizes_error(monkeypatch, failure_point):
    connection, _ = fake_connection()
    monkeypatch.setattr(api.http.client, "HTTPSConnection", connection)

    def fail(*args, **kwargs):
        raise OSError(PRIVATE_ERROR)

    with monkeypatch.context() as failure_patch:
        if failure_point == "constructor":
            failure_patch.setattr(connection, "__init__", fail)
        else:
            failure_patch.setattr(api.threading.Thread, "start", fail)
        with pytest.raises(api.ApiError) as failure:
            api._https_transport(api.ApiConfig(model="offline-test"),
                                 "dummy-test-key", b"{}", threading.Event())
        assert PRIVATE_ERROR not in str(failure.value)
        wait_for_transport_release()

    assert api._https_transport(api.ApiConfig(model="offline-test"),
                                "dummy-test-key", b"{}", threading.Event()) == (200, b"{}")
    wait_for_transport_release()


@pytest.mark.parametrize("fail_stage", [None, "waiting"])
def test_cleanup_error_cannot_leak_guard_or_override_timeout(monkeypatch, fail_stage):
    connection, calls = fake_connection(fail_stage=fail_stage, close_fails=True)
    monkeypatch.setattr(api.http.client, "HTTPSConnection", connection)
    if fail_stage:
        with pytest.raises(api.ApiError) as failure:
            api._https_transport(api.ApiConfig(model="offline-test"),
                                 "dummy-test-key", b"{}", threading.Event())
        assert failure.value.code == "timeout"
        assert PRIVATE_ERROR not in str(failure.value)
    else:
        assert api._https_transport(api.ApiConfig(model="offline-test"),
                                    "dummy-test-key", b"{}", threading.Event()) == (200, b"{}")
    wait_for_transport_release()
    assert calls.count("request") == 1


@pytest.mark.parametrize("status", [302, 307, 400, 401, 429, 503])
def test_non_success_response_never_reads_private_body_or_retries(monkeypatch, status):
    connection, calls = fake_connection(status=status)
    monkeypatch.setattr(api.http.client, "HTTPSConnection", connection)

    result = api._https_transport(api.ApiConfig(model="offline-test"),
                                  "dummy-test-key", b"{}", threading.Event())

    assert result == (status, b"")
    assert "read" not in calls
    assert calls.count("connect") == calls.count("request") == 1
    wait_for_transport_release()


def test_shared_progress_reports_the_actual_transport_phase(monkeypatch):
    progress = api.TransportProgress()
    observed = []

    class Connection:
        sock = None

        def __init__(self, *args, **kwargs):
            pass

        def connect(self):
            observed.append(progress.snapshot()[0])

        def request(self, *args, **kwargs):
            observed.append(progress.snapshot()[0])

        def getresponse(self):
            observed.append(progress.snapshot()[0])

            class Response:
                status = 200

                def read(self, *args):
                    observed.append(progress.snapshot()[0])
                    return b"{}"

            return Response()

        def close(self):
            pass

    monkeypatch.setattr(api.http.client, "HTTPSConnection", Connection)
    assert api._https_transport(api.ApiConfig(model="offline-test"),
                                "dummy-test-key", b"{}", threading.Event(),
                                progress=progress) == (200, b"{}")
    assert observed == ["connecting", "sending", "waiting", "receiving"]
    stage, elapsed = progress.snapshot()
    assert stage == "receiving" and elapsed >= 0
    wait_for_transport_release()


def test_error_diagnostics_only_expose_known_stage_and_valid_local_times():
    error = api.ApiError("timeout", transport_stage=PRIVATE_ERROR,
                         elapsed_seconds=PRIVATE_ERROR, timeout_seconds=float("nan"))
    assert error.transport_stage is None
    assert error.elapsed_seconds is None and error.timeout_seconds is None
    assert error.request_may_have_been_sent is True
    assert PRIVATE_ERROR not in str(error)
