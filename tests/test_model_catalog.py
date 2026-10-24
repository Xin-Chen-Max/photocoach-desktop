"""Model catalog metadata only; all HTTPS and credentials are simulated."""
import json
import threading
from urllib.parse import parse_qs, urlsplit

import pytest
from photocoach import api_coach as api
from photocoach.settings import credential_target


def row(name="gemini-test", methods=None, display="Test model"):
    return {"name": "models/" + name, "displayName": display,
            "supportedGenerationMethods": methods if methods is not None else ["generateContent"]}


def test_metadata_catalog_get_encodes_page_token_and_never_sends_generation(monkeypatch):
    calls = []
    class Connection:
        sock = None
        def __init__(self, host, port, timeout):
            assert host == "generativelanguage.googleapis.com" and timeout == 15
        def connect(self):
            pass
        def request(self, method, path, *, body, headers):
            assert method == "GET" and body is None
            assert headers["x-goog-api-key"] == "fake-catalog-key"
            assert "fake-catalog-key" not in path
            target = urlsplit(path)
            assert target.path == "/v1beta/models"
            assert parse_qs(target.query) == {"pageSize": ["1000"], "pageToken": ["abc+/=&?next"]}
            calls.append(path)
        def getresponse(self):
            class Response:
                status = 200
                def read(self, limit):
                    assert limit == 2 * 1024 * 1024 + 1
                    return json.dumps({"models": [row()], "nextPageToken": "second-page"}).encode()
            return Response()
        def close(self):
            pass
    monkeypatch.setattr(api.http.client, "HTTPSConnection", Connection)
    result = api.list_gemini_models(api.GeminiModelsConfig(), "fake-catalog-key", threading.Event(), "abc+/=&?next")
    assert result.models == (api.GeminiModel("gemini-test", "Test model"),)
    assert result.next_page_token == "second-page" and len(calls) == 1
    assert "second-page" not in repr(result) and not api._TRANSPORT_LOCK.locked()


def test_catalog_filters_deduplicates_and_does_not_invent_image_support(monkeypatch):
    rows = [row("gemini-z"), row("gemini-a"), row("gemini-z"), row("text-embedding-004", ["embedContent"]),
            row("gemini-no-generation", ["countTokens"]), row("../bad"), row("gemini-unsafe name"),
            row("gemini-no-label", display="bad\nlabel")]
    monkeypatch.setattr(api, "_https_transport", lambda *a, **k: (200, json.dumps({"models": rows}).encode()))
    result = api.list_gemini_models(api.GeminiModelsConfig(), "fake-catalog-key", threading.Event())
    assert [model.model_id for model in result.models] == ["gemini-a", "gemini-no-label", "gemini-z"]
    assert result.models[1].display_name == "gemini-no-label" and result.next_page_token == ""


@pytest.mark.parametrize("status", [302, 400, 401, 403, 404, 429, 503])
def test_catalog_http_errors_are_sanitized_and_not_retried(monkeypatch, status):
    calls = []
    def transport(*args, **kwargs):
        calls.append(kwargs)
        return status, b"fake-catalog-key private provider error"
    monkeypatch.setattr(api, "_https_transport", transport)
    with pytest.raises(api.ModelCheckError) as error:
        api.list_gemini_models(api.GeminiModelsConfig(), "fake-catalog-key", threading.Event())
    assert str(status) in str(error.value) and "fake-catalog-key" not in str(error.value)
    assert calls == [{"models_page": ""}]


@pytest.mark.parametrize("body", [b"[]", b"bad-json", b'{"models": {}}', b'{"models": [null]}',
    b'{"models": [], "nextPageToken": 5}', b'{"models": [{"name":"models/gemini-test"}]}',
    json.dumps({"models": [row(display="fake-catalog-key")]}).encode(),
    b'{"nextPageToken":"fake-catalog-key"}', b'{"nextPageToken":"repeated"}'])
def test_catalog_invalid_response_never_leaks_key(monkeypatch, body):
    monkeypatch.setattr(api, "_https_transport", lambda *a, **k: (200, body))
    with pytest.raises(api.ModelCheckError) as error:
        api.list_gemini_models(api.GeminiModelsConfig(), "fake-catalog-key", threading.Event(), "repeated")
    assert "fake-catalog-key" not in str(error.value)


@pytest.mark.parametrize("endpoint", ["https://example.invalid/v1beta/models", api.GEMINI_ENDPOINT + "?key=private",
    api.GEMINI_ENDPOINT + "/gemini-test", "http://generativelanguage.googleapis.com/v1beta/models",
    "https://private@generativelanguage.googleapis.com/v1beta/models"])
def test_catalog_rejects_nonofficial_destinations(endpoint):
    with pytest.raises(api.ApiError):
        api.GeminiModelsConfig(endpoint)


def test_catalog_timeout_and_cancellation_do_not_retry(monkeypatch):
    calls = []
    def transport(*args, **kwargs):
        calls.append(True)
        raise api.ApiError("timeout", transport_stage="waiting")
    monkeypatch.setattr(api, "_https_transport", transport)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(api.ModelCheckError, match="cancelled"):
        api.list_gemini_models(api.GeminiModelsConfig(), "fake-key", cancel)
    assert not calls
    with pytest.raises(api.ModelCheckError, match="timed out"):
        api.list_gemini_models(api.GeminiModelsConfig(), "fake-key", threading.Event())
    assert len(calls) == 1


def test_catalog_credentials_share_destination_identity_without_a_model():
    assert credential_target(api.GeminiModelsConfig()) == credential_target(
        api.ApiConfig(api.GEMINI_ENDPOINT, "gemini-user-model", "gemini"))
    assert not hasattr(api.GeminiModelsConfig(), "model")


def test_catalog_connection_cannot_accidentally_generate(monkeypatch):
    monkeypatch.setattr(api.http.client, "HTTPSConnection", lambda *a, **k: pytest.fail("must not connect"))
    with pytest.raises(api.ApiError):
        api._https_transport(api.GeminiModelsConfig(), "fake-key", b"private", threading.Event())


def test_empty_catalog_is_valid(monkeypatch):
    monkeypatch.setattr(api, "_https_transport", lambda *a, **k: (200, b"{}"))
    assert api.list_gemini_models(api.GeminiModelsConfig(), "fake-key", threading.Event()).models == ()
