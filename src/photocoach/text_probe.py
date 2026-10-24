"""One manually requested Gemini text generation, isolated from photo coaching."""
from __future__ import annotations

from dataclasses import dataclass
import json
import threading
from typing import Literal

from .api_coach import (ApiCoach, ApiConfig, ApiError, MAX_RESPONSE_BYTES,
                        TransportProgress, _https_transport, _json_load, _token_count)


TEXT_PROBE_PROMPT = "Reply with exactly OK."


@dataclass(frozen=True)
class TextProbeResult:
    outcome: Literal["ok", "other_text", "output_limit"]
    input_tokens: int | None
    output_tokens: int | None

    def text(self) -> str:
        messages = {
            "ok": "Text-only generation succeeded; the provider returned exactly OK.",
            "other_text": "Text-only generation completed, but the reply was not exactly OK. The reply is not displayed.",
            "output_limit": "Text-only generation reached the output-token limit before normal completion. This does not establish a successful OK reply.",
        }
        return (messages[self.outcome] + " No photo or editing information was sent. "
                "This does not verify image input, structured feedback or photo coaching. "
                "No retry was made.")


def build_text_probe_payload(config: ApiConfig) -> dict:
    """The exact request disclosed by the review UI; no user or photo context."""
    if not isinstance(config, ApiConfig) or config.protocol != "gemini":
        raise ApiError("configuration")
    return {
        "contents": [{"role": "user", "parts": [{"text": TEXT_PROBE_PROMPT}]}],
        "store": False,
        "generationConfig": {"maxOutputTokens": config.max_output_tokens},
    }


def _parse_text_probe(body: bytes) -> TextProbeResult:
    try:
        if not isinstance(body, bytes) or len(body) > MAX_RESPONSE_BYTES:
            raise ValueError
        document = _json_load(body)
        if not isinstance(document, dict) or document.get("error"):
            raise ValueError
        prompt_feedback = document.get("promptFeedback") or {}
        if not isinstance(prompt_feedback, dict) or prompt_feedback.get("blockReason"):
            raise ValueError
        candidates = document["candidates"]
        if not isinstance(candidates, list) or len(candidates) != 1 or not isinstance(candidates[0], dict):
            raise ValueError
        candidate = candidates[0]
        finish = candidate.get("finishReason")
        if finish not in ("STOP", "MAX_TOKENS"):
            raise ValueError
        texts = []
        # Thinking may consume the entire output allocation without visible text.
        content = candidate.get("content", {})
        if not isinstance(content, dict):
            raise ValueError
        parts = content.get("parts", [])
        if not isinstance(parts, list):
            raise ValueError
        for part in parts:
            if not isinstance(part, dict):
                raise ValueError
            if part.get("thought") is True:
                continue
            if not isinstance(part.get("text"), str):
                raise ValueError
            texts.append(part["text"])
        answer = "".join(texts).strip()
        if finish == "STOP" and not answer:
            raise ValueError
        usage = document.get("usageMetadata") or {}
        if not isinstance(usage, dict):
            raise ValueError
        input_tokens = _token_count(usage, "promptTokenCount")
        candidate_tokens = _token_count(usage, "candidatesTokenCount")
        thought_tokens = _token_count(usage, "thoughtsTokenCount") if "thoughtsTokenCount" in usage else 0
        output_tokens = (candidate_tokens + thought_tokens
                         if candidate_tokens is not None and thought_tokens is not None else None)
        outcome = "output_limit" if finish == "MAX_TOKENS" else "ok" if answer == "OK" else "other_text"
        return TextProbeResult(outcome, input_tokens, output_tokens)
    except (TypeError, ValueError, KeyError, AttributeError, IndexError, UnicodeError):
        raise ApiError("invalid_response") from None


def run_text_probe(config: ApiConfig, api_key: str, cancel_event: threading.Event,
                   progress: TransportProgress | None = None) -> TextProbeResult:
    """One native generation POST, using existing bounded transport and safe errors."""
    ApiCoach(config, api_key)  # Validate the key without accessing any photo state.
    payload = build_text_probe_payload(config)
    if cancel_event.is_set():
        raise ApiError("cancelled")
    try:
        status, body = _https_transport(
            config, api_key, json.dumps(payload, allow_nan=False).encode("utf-8"),
            cancel_event, progress=progress)
    except ApiError:
        raise
    except Exception:
        raise ApiError("connection") from None
    if cancel_event.is_set():
        raise ApiError("cancelled")
    if type(status) is not int or not 100 <= status <= 599:
        raise ApiError("invalid_response")
    if not 200 <= status < 300:
        code = ("authentication" if status in (401, 403) else "rate_limit" if status == 429 else
                "unavailable" if 500 <= status < 600 else "http_error")
        raise ApiError(code, http_status=status)
    return _parse_text_probe(body)
