"""Credential/model diagnostics use metadata only, with fake local transports."""
import json
import threading

import pytest

from photocoach import api_coach as api


def config():
    return api.ApiConfig(api.GEMINI_ENDPOINT, "gemini-test-model", "gemini", 120)


def test_model_check_get_has_no_generation_body_and_key_only_in_header(monkeypatch):
    calls = []

    class Connection:
        sock = None

        def __init__(self, host, port, timeout):
            assert host == "generativelanguage.googleapis.com" and timeout == 15

        def connect(self):
            pass

        def request(self, method, path, *, body, headers):
            calls.append((method, path))
            assert method == "GET" and path == "/v1beta/models/gemini-test-model"
            assert body is None and "?" not in path and ":generateContent" not in path
            assert headers["x-goog-api-key"] == "dummy-test-key"

        def getresponse(self):
            class Response:
                status = 200

                def read(self, limit):
                    return json.dumps({"name": "models/gemini-test-model-version",
                                       "supportedGenerationMethods": ["generateContent"]}).encode()
            return Response()

        def close(self):
            pass

    monkeypatch.setattr(api.http.client, "HTTPSConnection", Connection)
    result = api.check_gemini_model(config(), "dummy-test-key", threading.Event())
    assert result.supports_generation and len(calls) == 1
    assert "not generation quota" in result.text() and "dummy-test-key" not in result.text()
    assert not api._TRANSPORT_LOCK.locked()


@pytest.mark.parametrize("status", [302, 400, 401, 403, 404, 429, 500])
def test_model_check_failure_sanitized_and_never_retried(monkeypatch, status):
    calls = []

    def transport(*args, **kwargs):
        calls.append(kwargs)
        return status, b"secret-key-and-private-path"

    monkeypatch.setattr(api, "_https_transport", transport)
    with pytest.raises(api.ModelCheckError) as failure:
        api.check_gemini_model(config(), "dummy", threading.Event())
    assert len(calls) == 1 and calls[0]["metadata_only"]
    assert f"HTTP {status}" in str(failure.value)
    assert "secret-key" not in str(failure.value) and "No retry" in str(failure.value)
    assert "generation request" in str(failure.value)


@pytest.mark.parametrize("body", [b"not json", b"[]", b"{}", b'{"name":"models/test","supportedGenerationMethods":1}'])
def test_model_check_invalid_metadata(monkeypatch, body):
    monkeypatch.setattr(api, "_https_transport", lambda *a, **kw: (200, body))
    with pytest.raises(api.ModelCheckError, match="unexpected model metadata"):
        api.check_gemini_model(config(), "dummy", threading.Event())


def test_model_without_generate_content_is_not_reported_as_supported(monkeypatch):
    body = json.dumps({"name": "models/test", "supportedGenerationMethods": ["embedContent"]}).encode()
    monkeypatch.setattr(api, "_https_transport", lambda *a, **kw: (200, body))
    result = api.check_gemini_model(config(), "dummy", threading.Event())
    assert not result.supports_generation


def test_model_check_validates_native_protocol_key_and_cancellation_before_network(monkeypatch):
    class NoConnection:
        def __init__(self, *args, **kwargs):
            pytest.fail("must not connect")
    monkeypatch.setattr(api.http.client, "HTTPSConnection", NoConnection)
    cancelled = threading.Event()
    cancelled.set()
    for options, key, event in ((api.ApiConfig(model="test"), "dummy", threading.Event()),
                                (config(), "", threading.Event()), (config(), "dummy", cancelled)):
        with pytest.raises(api.ModelCheckError):
            api.check_gemini_model(options, key, event)
    assert not api._TRANSPORT_LOCK.locked()


def test_model_check_timeout_does_not_claim_generation_or_billing(monkeypatch):
    def transport(*a, **kw):
        raise api.ApiError("timeout", transport_stage="waiting", elapsed_seconds=15, timeout_seconds=15)
    monkeypatch.setattr(api, "_https_transport", transport)
    with pytest.raises(api.ModelCheckError) as failure:
        api.check_gemini_model(config(), "dummy", threading.Event())
    assert "waiting for response headers" in str(failure.value)
    assert "charges" not in str(failure.value)
