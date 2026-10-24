"""Text generation diagnostics use deterministic fake transports, never paid calls."""
import json
import threading

import pytest

from photocoach import api_coach as api
from photocoach import text_probe as probe


def config(**changes):
    values = {"endpoint": api.GEMINI_ENDPOINT, "model": "gemini-test-model",
              "protocol": "gemini", "timeout_seconds": 120, "max_output_tokens": 1500}
    values.update(changes)
    return api.ApiConfig(**values)


def response(text="OK", finish="STOP", **changes):
    document = {
        "candidates": [{"finishReason": finish, "content": {"parts": [{"text": text}]}}],
        "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 1, "thoughtsTokenCount": 19},
    }
    document.update(changes)
    return json.dumps(document).encode()


def run(monkeypatch, body, status=200):
    monkeypatch.setattr(probe, "_https_transport", lambda *args, **kwargs: (status, body))
    return probe.run_text_probe(config(), "dummy-key", threading.Event())


def test_minimal_payload_and_one_post_use_existing_transport(monkeypatch):
    calls = []
    progress = api.TransportProgress()
    cancelled = threading.Event()
    options = config(max_output_tokens=256)

    class Connection:
        sock = None

        def __init__(self, host, port, timeout):
            assert host == "generativelanguage.googleapis.com" and timeout == 120

        def connect(self):
            pass

        def request(self, method, path, *, body, headers):
            calls.append((method, path))
            assert method == "POST" and path == "/v1beta/models/gemini-test-model:generateContent"
            assert "?" not in path and headers["x-goog-api-key"] == "dummy-key"
            assert json.loads(body) == {
                "contents": [{"role": "user", "parts": [{"text": "Reply with exactly OK."}]}],
                "store": False, "generationConfig": {"maxOutputTokens": 256},
            }
            assert "dummy-key" not in body.decode()

        def getresponse(self):
            class Response:
                status = 200

                def read(self, limit):
                    assert limit == api.MAX_RESPONSE_BYTES + 1
                    return response()
            return Response()

        def close(self):
            pass

    monkeypatch.setattr(api.http.client, "HTTPSConnection", Connection)
    result = probe.run_text_probe(options, "dummy-key", cancelled, progress)
    assert result.outcome == "ok" and result.input_tokens == 8 and result.output_tokens == 20
    assert len(calls) == 1 and not api._TRANSPORT_LOCK.locked()
    assert progress.snapshot()[0] == "receiving"
    assert "does not verify image input" in result.text()


@pytest.mark.parametrize("text, outcome", [("OK", "ok"), ("  OK\n", "ok"),
    ("ok", "other_text"), ("private-provider-output", "other_text")])
def test_response_classification_does_not_echo_output(monkeypatch, text, outcome):
    result = run(monkeypatch, response(text))
    assert result.outcome == outcome
    assert "private-provider-output" not in result.text() and "private-provider-output" not in repr(result)


def test_thought_text_not_exposed_or_used_as_visible_reply(monkeypatch):
    candidates = [{"finishReason": "STOP", "content": {"parts": [
        {"thought": True, "text": "private-thought"}, {"text": "O"}, {"text": "K"}]}}]
    result = run(monkeypatch, response(candidates=candidates))
    assert result.outcome == "ok" and result.output_tokens == 20
    assert "private-thought" not in result.text() and "private-thought" not in repr(result)


@pytest.mark.parametrize("content", [None, {}, {"parts": []},
    {"parts": [{"thought": True, "text": "private-thought"}]}, {"parts": [{"text": "OK"}]}])
def test_output_limit_preserves_usage_without_claiming_success(monkeypatch, content):
    candidate = {"finishReason": "MAX_TOKENS"}
    if content is not None:
        candidate["content"] = content
    result = run(monkeypatch, response(candidates=[candidate]))
    assert result.outcome == "output_limit" and result.input_tokens == 8 and result.output_tokens == 20
    assert "before normal completion" in result.text() and "private-thought" not in repr(result)


@pytest.mark.parametrize("usage, expected", [
    ({}, (None, None)), (None, (None, None)),
    ({"promptTokenCount": 8, "candidatesTokenCount": 2}, (8, 2)),
    ({"promptTokenCount": 0, "candidatesTokenCount": 0, "thoughtsTokenCount": 5}, (0, 5)),
    ({"promptTokenCount": True, "candidatesTokenCount": -1}, (None, None)),
    ({"promptTokenCount": "8", "candidatesTokenCount": 1, "thoughtsTokenCount": False}, (None, None)),
    ({"promptTokenCount": 8, "thoughtsTokenCount": 9}, (8, None)),
])
def test_usage_unknown_is_not_assumed_zero(monkeypatch, usage, expected):
    result = run(monkeypatch, response(usageMetadata=usage))
    assert (result.input_tokens, result.output_tokens) == expected


@pytest.mark.parametrize("status, code", [(302, "http_error"), (400, "http_error"),
    (401, "authentication"), (403, "authentication"), (404, "http_error"),
    (429, "rate_limit"), (500, "unavailable")])
def test_errors_sanitized_never_retried(monkeypatch, status, code):
    calls = []

    def transport(*args, **kwargs):
        calls.append(1)
        return status, b"private-provider-output dummy-key"

    monkeypatch.setattr(probe, "_https_transport", transport)
    with pytest.raises(api.ApiError) as failure:
        probe.run_text_probe(config(), "dummy-key", threading.Event())
    assert len(calls) == 1 and failure.value.code == code and failure.value.http_status == status
    assert "dummy-key" not in str(failure.value) and "private-provider-output" not in str(failure.value)


@pytest.mark.parametrize("body", [b"not-json", b"[]", b"{}", b'{"candidates":[],"candidates":[]}',
    b'\xff', b"x" * (api.MAX_RESPONSE_BYTES + 1), response(candidates=[]),
    response(candidates=[{}, {}]), response(finish="SAFETY"), response(text=""),
    response(error={"message": "private-provider-output"}),
    response(promptFeedback={"blockReason": "SAFETY"}), response(promptFeedback="invalid"),
    response(candidates=[{"finishReason": "STOP", "content": {"parts": [{"functionCall": {}}]}}]),
    response(candidates=[{"finishReason": "STOP", "content": {"parts": [{"thought": True, "text": "OK"}]}}]),
    response(candidates=[{"finishReason": "STOP", "content": {"parts": "invalid"}}]),
    response(candidates=[{"finishReason": "MAX_TOKENS", "content": None}]),
    response(usageMetadata="invalid"), b'{"candidates":NaN}', "not bytes",
], ids=["not-json", "array", "empty", "duplicates", "encoding", "oversized", "no-candidates",
        "multiple-candidates", "refusal", "empty-text", "provider-error", "blocked", "bad-block",
        "tool-call", "thought-only", "bad-parts", "null-content", "bad-usage", "nan", "not-bytes"])
def test_invalid_refused_and_malformed_responses_are_safe(monkeypatch, body):
    with pytest.raises(api.ApiError) as failure:
        run(monkeypatch, body)
    assert failure.value.code == "invalid_response" and "private-provider-output" not in str(failure.value)


@pytest.mark.parametrize("status", [True, None, "200", 99, 600])
def test_invalid_http_status_is_not_accepted(monkeypatch, status):
    with pytest.raises(api.ApiError) as failure:
        run(monkeypatch, response(), status)
    assert failure.value.code == "invalid_response"


def test_invalid_config_key_or_precancel_never_dispatches(monkeypatch):
    monkeypatch.setattr(probe, "_https_transport", lambda *a, **kw: pytest.fail("must not dispatch"))
    cancelled = threading.Event()
    cancelled.set()
    for options, key, event in ((api.ApiConfig(model="test"), "dummy", threading.Event()),
                               (config(), "", threading.Event()),
                               (config(), "key\n", threading.Event()),
                               (config(), "dummy", cancelled)):
        with pytest.raises(api.ApiError):
            probe.run_text_probe(options, key, event)
    with pytest.raises(api.ApiError):
        probe.build_text_probe_payload(None)


def test_endpoint_validation_is_shared_before_network():
    for endpoint in ("http://generativelanguage.googleapis.com/v1beta/models",
                     "https://other.example/v1beta/models",
                     api.GEMINI_ENDPOINT + "?key=dummy"):
        with pytest.raises(api.ApiError):
            config(endpoint=endpoint)


def test_preserves_safe_transport_error_and_progress(monkeypatch):
    progress = api.TransportProgress()
    expected = api.ApiError("timeout", transport_stage="waiting", elapsed_seconds=120, timeout_seconds=120)
    calls = []

    def transport(*args, **kwargs):
        assert kwargs == {"progress": progress}
        calls.append(1)
        raise expected

    monkeypatch.setattr(probe, "_https_transport", transport)
    with pytest.raises(api.ApiError) as failure:
        probe.run_text_probe(config(), "dummy", threading.Event(), progress)
    assert failure.value is expected and len(calls) == 1


def test_arbitrary_transport_error_is_sanitized(monkeypatch):
    def transport(*args, **kwargs):
        raise OSError("private-provider-output dummy-key")

    monkeypatch.setattr(probe, "_https_transport", transport)
    with pytest.raises(api.ApiError) as failure:
        probe.run_text_probe(config(), "dummy-key", threading.Event())
    assert failure.value.code == "connection" and "dummy-key" not in str(failure.value)


def test_cancellation_after_transport_discards_reply(monkeypatch):
    event = threading.Event()

    def transport(*args, **kwargs):
        event.set()
        return 200, response()

    monkeypatch.setattr(probe, "_https_transport", transport)
    with pytest.raises(api.ApiError) as failure:
        probe.run_text_probe(config(), "dummy", event)
    assert failure.value.code == "cancelled"
