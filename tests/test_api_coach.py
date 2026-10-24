"""Offline provider-contract tests. No real keys, providers, or remote requests."""
import base64
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image
import pytest

from photocoach import api_coach as api
from photocoach.coach import build_request
from photocoach.state import StateStore


@pytest.fixture
def coach_request():
    store = StateStore()
    store.activate(Path("E:/private-not-sent/photo.ARW"), "private-content-hash")
    store.set_exposure(.3)
    image = np.full((12, 20, 3), 80, dtype=np.uint8)
    return build_request(store.current, store.before_last_edit(), store.events, image, image, "What next?")


def response_body(protocol="responses", *, feedback=None, **changes):
    answer = {field: "A short photographic observation." for field in api.FEEDBACK_FIELDS}
    if feedback is not None:
        answer = feedback
    if protocol == "gemini":
        response = {"candidates": [{"finishReason": "STOP", "content": {"parts": [
            {"text": json.dumps(answer, ensure_ascii=False)}]}}]}
    elif protocol == "responses":
        response = {"status": "completed", "output": [{"type": "message", "status": "completed",
            "content": [{"type": "output_text", "text": json.dumps(answer)}]}],
            "usage": {"input_tokens": 1200, "output_tokens": 180}}
    else:
        response = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answer)}}],
                    "usage": {"prompt_tokens": 1200, "completion_tokens": 180}}
    response.update(changes)
    return json.dumps(response).encode()


@pytest.mark.parametrize("protocol", ["responses", "chat_completions"])
@pytest.mark.parametrize("include_before", [True, False])
def test_payload_images_context_and_result_binding(coach_request, protocol, include_before):
    config = api.ApiConfig(model="chosen-test-model", protocol=protocol)
    captured = []
    def transport(config, key, payload, cancel):
        captured.append(json.loads(payload))
        return 200, response_body(protocol)
    client = api.ApiCoach(config, "dummy-test-key", transport)
    result = client.analyze(coach_request, include_before=include_before)
    assert "dummy-test-key" not in repr(client)
    payload = captured[0]
    serialized = json.dumps(payload)
    assert "private-content-hash" not in serialized and "private-not-sent" not in serialized
    assert payload["store"] is False
    content = payload["input"][0]["content"] if protocol == "responses" else payload["messages"][1]["content"]
    context = json.loads(content[0]["text"])
    assert context == api.request_context(coach_request, include_before=include_before)
    assert context["parameters"] == coach_request.parameters
    assert context["before_state_version"] == coach_request.before_state_version
    image_items = [part for part in content if part["type"] in ("input_image", "image_url")]
    assert len(image_items) == (2 if include_before else 1)
    for item in image_items:
        uri = item["image_url"] if protocol == "responses" else item["image_url"]["url"]
        with Image.open(BytesIO(base64.b64decode(uri.split(",")[1]))) as image:
            assert image.size == (20, 12) and not image.getexif()
    assert result.feedback.photo_id == coach_request.photo_id
    assert result.feedback.state_version == coach_request.state_version
    assert result.feedback.source == "AI · chosen-test-model"
    assert result.input_tokens == 1200 and result.output_tokens == 180


@pytest.mark.parametrize("protocol", ["responses", "chat_completions", "gemini"])
@pytest.mark.parametrize("question", ["How can I adjust the RGB curve?", "How do I adjust the RGB curve?"])
def test_question_and_actionable_answer_survive_all_adapters(coach_request, protocol, question):
    request = replace(coach_request, question=question)
    answer = {name: "No new visible comparison is established." for name in api.FEEDBACK_FIELDS}
    answer["answer"] = "The shared RGB curve affects all three channels.\nKeep the black and white endpoints, then try a small midtone lift."
    answer["next_step_or_stop"] = "1. Double-click the RGB curve to add a midtone point.\n2. Drag it upward slightly; use Undo if tonal separation is lost."
    captured = []
    config = (api.ApiConfig(api.GEMINI_ENDPOINT, "gemini-test", "gemini") if protocol == "gemini"
              else api.ApiConfig(model="test", protocol=protocol))
    def transport(config, key, body, cancel):
        captured.append(json.loads(body))
        return 200, response_body(protocol, feedback=answer)
    result = api.ApiCoach(config, "fake", transport).analyze(request)
    payload = captured[0]
    if protocol == "gemini":
        context = json.loads(payload["contents"][0]["parts"][0]["text"])
        schema = payload["generationConfig"]["responseJsonSchema"]
    elif protocol == "responses":
        context = json.loads(payload["input"][0]["content"][0]["text"])
        schema = payload["text"]["format"]["schema"]
    else:
        context = json.loads(payload["messages"][1]["content"][0]["text"])
        schema = payload["response_format"]["json_schema"]["schema"]
    assert context["question"] == question
    assert "answer" in schema["required"]
    assert result.feedback.answer == answer["answer"]
    assert result.feedback.next_step_or_stop == answer["next_step_or_stop"]
    assert result.feedback.question == question
    assert answer["answer"] in result.feedback.text()


@pytest.mark.parametrize("protocol", ["responses", "chat_completions", "gemini"])
@pytest.mark.parametrize("problem", ["missing", "empty", "oversized", "spoofed_question"])
def test_direct_answer_is_required_bounded_and_cannot_replace_question(coach_request, protocol, problem):
    answer = {name: "A valid observation." for name in api.FEEDBACK_FIELDS}
    if problem == "missing":
        del answer["answer"]
    elif problem == "empty":
        answer["answer"] = "  "
    elif problem == "oversized":
        answer["answer"] = "x" * 1201
    else:
        answer["question"] = "A different question invented by the model"
    config = (api.ApiConfig(api.GEMINI_ENDPOINT, "gemini-test", "gemini") if protocol == "gemini"
              else api.ApiConfig(model="test", protocol=protocol))
    with pytest.raises(api.ApiError, match="invalid structured"):
        api.ApiCoach(config, "fake", lambda *a: (200, response_body(protocol, feedback=answer))).analyze(coach_request)


@pytest.mark.parametrize("kwargs", [{"endpoint": "http://example.test/v1"}, {"model": ""},
    {"endpoint": "https://user:pass@example.test/v1"}, {"endpoint": "https://example.test/v1?key=x"},
    {"endpoint": "https://example.test/v1#fragment"}, {"timeout_seconds": float("nan")},
    {"timeout_seconds": 121}, {"max_output_tokens": True}, {"protocol": "unsupported"}])
def test_invalid_configuration(kwargs):
    values = {"model": "test"}
    values.update(kwargs)
    with pytest.raises(api.ApiError):
        api.ApiConfig(**values)


@pytest.mark.parametrize("status, code", [(401, "authentication"), (403, "authentication"),
    (429, "rate_limit"), (500, "unavailable"), (302, "http_error"), (400, "http_error")])
def test_failure_sanitized_no_retry(coach_request, status, code):
    calls = []
    def transport(*args):
        calls.append(1)
        return status, b"provider echoed secret-key and private-photo-path"
    with pytest.raises(api.ApiError) as failure:
        api.ApiCoach(api.ApiConfig(model="test"), "dummy", transport).analyze(coach_request)
    assert failure.value.code == code
    assert failure.value.http_status == status
    assert f"HTTP {status}" in str(failure.value)
    assert "secret-key" not in str(failure.value) and len(calls) == 1


@pytest.mark.parametrize("body", [b"not-json", b'{"status":"completed","status":"failed"}',
    response_body(status="incomplete"), response_body(output=[]),
    response_body(output=[{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}]),
    response_body(output=[{"type": "message", "content": [{"type": "output_text", "text": "{\"bad\":true}"}]}]),
    b"x" * (api.MAX_RESPONSE_BYTES + 1)], ids=["json", "duplicates", "incomplete", "empty", "refusal", "schema", "oversized"])
def test_malformed_refused_oversized_and_incomplete_responses(coach_request, body):
    with pytest.raises(api.ApiError, match="invalid structured"):
        api.ApiCoach(api.ApiConfig(model="test"), "dummy", lambda *a: (200, body)).analyze(coach_request)


def test_cancellation_before_send_and_secret_exception(coach_request):
    event = threading.Event()
    event.set()
    with pytest.raises(api.ApiError, match="cancelled"):
        api.ApiCoach(api.ApiConfig(model="test"), "dummy", lambda *a: pytest.fail("must not send")).analyze(coach_request, cancel_event=event)
    def failure(*args):
        raise OSError("secret-dummy-key")
    with pytest.raises(api.ApiError) as result:
        api.ApiCoach(api.ApiConfig(model="test"), "dummy", failure).analyze(coach_request)
    assert "secret-dummy-key" not in str(result.value)


def test_default_transport_timeout_is_bounded_and_does_not_follow_redirects(monkeypatch):
    released = threading.Event()
    closed = threading.Event()
    calls = []
    class Connection:
        sock = None
        def __init__(self, *args, **kwargs):
            pass
        def connect(self):
            pass
        def request(self, *args, **kwargs):
            calls.append(1)
            released.wait(4)
        def getresponse(self):
            class Response:
                status = 302
                def read(self, *args):
                    pytest.fail("redirect body must not be read")
            return Response()
        def close(self):
            closed.set()
    monkeypatch.setattr(api.http.client, "HTTPSConnection", Connection)
    start = time.monotonic()
    try:
        with pytest.raises(api.ApiError, match="timed out"):
            api._https_transport(api.ApiConfig(model="test", timeout_seconds=1), "dummy", b"{}", threading.Event())
        assert time.monotonic() - start < 2.5
        with pytest.raises(api.ApiError, match="previous connection"):
            api._https_transport(api.ApiConfig(model="test"), "dummy", b"{}", threading.Event())
    finally:
        released.set()
    deadline = time.monotonic() + 2
    while api._TRANSPORT_LOCK.locked() and time.monotonic() < deadline:
        time.sleep(.01)
    assert not api._TRANSPORT_LOCK.locked() and closed.is_set() and calls == [1]
    assert api._https_transport(api.ApiConfig(model="test"), "dummy", b"{}", threading.Event()) == (302, b"")


@pytest.mark.parametrize("include_before", [True, False])
def test_gemini_native_images_schema_auth_and_usage(coach_request, include_before):
    config = api.ApiConfig(endpoint=api.GEMINI_ENDPOINT, model="gemini-test-model", protocol="gemini")
    assert api.request_endpoint(config) == api.GEMINI_ENDPOINT + "/gemini-test-model:generateContent"
    headers = api.request_headers(config, "dummy-test-key")
    assert headers["x-goog-api-key"] == "dummy-test-key" and "Authorization" not in headers
    captured = []
    def transport(config, key, body, cancel):
        captured.append(json.loads(body))
        answer = {field: "A concise photographic observation." for field in api.FEEDBACK_FIELDS}
        response = {"candidates": [{"finishReason": "STOP", "content": {"parts": [
            {"text": "ignored internal thought", "thought": True}, {"text": json.dumps(answer)}]}}],
            "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 120, "thoughtsTokenCount": 40}}
        return 200, json.dumps(response).encode()
    result = api.ApiCoach(config, "dummy-test-key", transport).analyze(coach_request, include_before=include_before)
    payload = captured[0]
    assert "dummy-test-key" not in json.dumps(payload) and "private-content-hash" not in json.dumps(payload)
    assert "messages" not in payload and "max_completion_tokens" not in payload
    assert payload["generationConfig"]["maxOutputTokens"] == config.max_output_tokens
    assert payload["generationConfig"]["responseJsonSchema"] == api.FEEDBACK_SCHEMA
    assert payload["store"] is False
    parts = payload["contents"][0]["parts"]
    assert json.loads(parts[0]["text"]) == api.request_context(coach_request, include_before=include_before)
    images = [part["inlineData"] for part in parts if "inlineData" in part]
    assert len(images) == (2 if include_before else 1)
    for image in images:
        assert image["mimeType"] == "image/png"
        with Image.open(BytesIO(base64.b64decode(image["data"]))) as pixels:
            assert pixels.size == (20, 12) and not pixels.getexif()
    assert result.feedback.state_version == coach_request.state_version
    assert "internal thought" not in result.feedback.text()
    assert result.input_tokens == 1000 and result.output_tokens == 160


@pytest.mark.parametrize("change", ["blocked", "truncated", "no_candidate", "malformed", "tool_call"])
def test_gemini_native_rejects_incomplete_and_unsafe_structure(coach_request, change):
    document = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "{}"}]}}]}
    if change == "blocked":
        document["promptFeedback"] = {"blockReason": "SAFETY"}
    elif change == "truncated":
        document["candidates"][0]["finishReason"] = "MAX_TOKENS"
    elif change == "no_candidate":
        document["candidates"] = []
    elif change == "tool_call":
        document["candidates"][0]["content"]["parts"] = [{"functionCall": {"name": "change_photo"}}]
    config = api.ApiConfig(api.GEMINI_ENDPOINT, "gemini-test-model", "gemini")
    with pytest.raises(api.ApiError) as error:
        api.ApiCoach(config, "dummy", lambda *a: (200, json.dumps(document).encode())).analyze(coach_request)
    assert error.value.code == ("output_limit" if change == "truncated" else "invalid_response")


@pytest.mark.parametrize("model", ["Gemini API Key", "gemini-test/name", "test", "gemini-test\n"])
def test_gemini_rejects_display_names_and_unsafe_model_paths(model):
    with pytest.raises(api.ApiError, match="exact model ID"):
        api.ApiConfig(api.GEMINI_ENDPOINT, model, "gemini")


@pytest.mark.parametrize("endpoint", ["https://example.test/models", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"])
def test_native_gemini_requires_native_base(endpoint):
    with pytest.raises(api.ApiError):
        api.ApiConfig(endpoint, "gemini-test", "gemini")


def test_error_diagnostics_distinguish_http400_and_http404_without_body(coach_request):
    for status, expected in ((400, "request parameters"), (404, "not found")):
        with pytest.raises(api.ApiError) as error:
            api.ApiCoach(api.ApiConfig(model="test"), "dummy", lambda *a: (status, b"secret-provider-error")).analyze(coach_request)
        assert expected in str(error.value) and "secret-provider-error" not in str(error.value)
