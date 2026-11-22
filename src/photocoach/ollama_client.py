"""Bounded, cancellable native Ollama requests to the owned loopback instance."""
from __future__ import annotations
import http.client
import json
import socket
import threading
import time

ENDPOINT = "http://127.0.0.1:11435/api/chat"
MODEL = "qwen3.6:27b"
MODEL_BYTES = 19_000_000_000  # Advisory registry size, not an exact pull manifest.
_LOCK = threading.Lock()
MAX_JSON = 2 * 1024 * 1024


class LocalError(RuntimeError):
    def __init__(self, code):
        messages = {
            "connection": "The local engine is unavailable. Prepare Local AI and try again.",
            "cancelled": "Local operation cancelled.", "timeout": "The local operation timed out. No retry was made.",
            "busy": "Another local model operation is still running.",
            "response": "The local engine returned an incomplete or invalid response.",
            "vision": "This model did not pass the image and structured-output check.",
            "model": "The recommended local model is not installed.",
            "http": "The local engine rejected the operation. Check its version and model installation.",
            "version": "This model requires a newer Ollama version. Install or select an updated engine, then detect again.",
        }
        self.code = code if code in messages else "connection"
        super().__init__(messages[self.code])


def _decode(raw):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError("duplicate field")
            result[k] = v
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))


def exchange(path, payload=None, *, cancel=None, timeout=15, progress=None):
    """No DNS, proxies, authentication, redirects, or provider-controlled URLs."""
    if path not in ("/api/version", "/api/tags", "/api/show", "/api/chat", "/api/pull"):
        raise LocalError("http")
    cancel = cancel if cancel is not None else threading.Event()
    if cancel.is_set():
        raise LocalError("cancelled")
    if not _LOCK.acquire(blocking=False):
        raise LocalError("busy")
    connection = http.client.HTTPConnection("127.0.0.1", 11435, timeout=timeout)
    done, result = threading.Event(), []
    def run():
        try:
            body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
            connection.request("GET" if payload is None else "POST", path, body,
                               {"Content-Type": "application/json"})
            response = connection.getresponse()
            if response.status != 200:
                raise LocalError("version" if response.status == 412 else "http")
            if path == "/api/pull":
                success = False
                while True:
                    line = response.readline(65537)
                    if not line:
                        break
                    if cancel.is_set():
                        raise LocalError("cancelled")
                    if len(line) > 65536:
                        raise LocalError("response")
                    event = _decode(line)
                    if isinstance(event, dict) and event.get("error"):
                        reason = str(event["error"]).lower()
                        raise LocalError("version" if "newer version" in reason or "412" in reason else "response")
                    if not isinstance(event, dict):
                        raise LocalError("response")
                    if progress:
                        progress(event)
                    if event.get("status") == "success":
                        success = True
                        break
                if not success:
                    raise LocalError("response")
                result.append({"status": "success"})
            else:
                raw = response.read(MAX_JSON + 1)
                if len(raw) > MAX_JSON:
                    raise LocalError("response")
                value = _decode(raw)
                if not isinstance(value, dict) or value.get("error"):
                    raise LocalError("response")
                result.append(value)
        except LocalError as error:
            result.append(error)
        except Exception:
            result.append(LocalError("connection"))
        finally:
            connection.close()
            _LOCK.release()
            done.set()
    threading.Thread(target=run, name="PhotoCoach-Ollama", daemon=True).start()
    deadline = time.monotonic() + timeout
    while not done.wait(.05):
        code = "cancelled" if cancel.is_set() else "timeout" if time.monotonic() >= deadline else None
        if code:
            try:
                if connection.sock:
                    connection.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
            done.wait(2)
            raise LocalError(code)
    if cancel.is_set():
        raise LocalError("cancelled")
    if not result or isinstance(result[0], Exception):
        raise result[0] if result else LocalError("response")
    return result[0]


def model_identity(model=MODEL, *, cancel=None):
    if model != MODEL:
        raise LocalError("model")
    tags = exchange("/api/tags", cancel=cancel).get("models", [])
    matches = [item for item in tags if item.get("name") == model or item.get("model") == model]
    if len(matches) != 1 or not isinstance(matches[0].get("digest"), str):
        raise LocalError("model")
    details = exchange("/api/show", {"model": model}, cancel=cancel)
    if "vision" not in details.get("capabilities", []):
        raise LocalError("vision")
    return matches[0]["digest"]


def chat_payload(model, messages, schema=None, *, output_tokens=4096, tools=None):
    if model != MODEL:
        raise LocalError("model")
    payload = {"model": model, "messages": messages, "stream": False,
               "think": False, "options": {"num_ctx": 8192, "num_predict": output_tokens, "temperature": .2},
               "keep_alive": "5m"}
    if schema is not None:
        payload["format"] = schema
    if tools:
        payload["tools"] = tools
    return payload


def chat_result(document, *, allow_tool_calls: bool = False):
    reasons = ("stop", "tool_calls") if allow_tool_calls else ("stop",)
    if document.get("done") is not True or document.get("done_reason") not in reasons:
        raise LocalError("response")
    message = document.get("message", {})
    if message.get("role") != "assistant":
        raise LocalError("response")
    tool_calls = message.get("tool_calls")
    content = message.get("content")
    if not allow_tool_calls:
        if not isinstance(content, str) or tool_calls:
            raise LocalError("response")
        return document
    if content is not None and not isinstance(content, str):
        raise LocalError("response")
    if tool_calls:
        if not isinstance(tool_calls, list):
            raise LocalError("response")
        for call in tool_calls:
            fn = call.get("function") if isinstance(call, dict) else None
            if (not isinstance(fn, dict) or not isinstance(fn.get("name"), str)
                    or not fn["name"].strip() or not isinstance(fn.get("arguments"), (dict, str))):
                raise LocalError("response")
    elif not isinstance(content, str) or not content.strip():
        raise LocalError("response")
    return document


def verify_vision(model=MODEL, *, cancel=None):
    from PIL import Image
    from io import BytesIO
    import base64
    image = Image.new("RGB", (256, 128), "red")
    image.paste((0, 0, 255), (128, 0, 256, 128))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    schema = {"type": "object", "properties": {name: {"type": "string", "enum": ["red", "blue"]}
              for name in ("left", "right")}, "required": ["left", "right"], "additionalProperties": False}
    payload = chat_payload(model, [{"role": "user", "content":
        "Inspect the image. Return the color of its left half and right half as JSON. Use English red or blue.",
        "images": [base64.b64encode(buffer.getvalue()).decode()]}], schema, output_tokens=128)
    document = chat_result(exchange("/api/chat", payload, cancel=cancel, timeout=300))
    try:
        if _decode(document["message"]["content"]) != {"left": "red", "right": "blue"}:
            raise ValueError
    except (ValueError, TypeError, KeyError):
        raise LocalError("vision") from None
    return document
