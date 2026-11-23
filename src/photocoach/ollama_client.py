"""Bounded, cancellable native Ollama requests to a loopback instance (127.0.0.1)."""
from __future__ import annotations

from dataclasses import dataclass
import http.client
import json
import re
import socket
import threading
import time
from typing import Any
from urllib.parse import urlsplit

DEFAULT_PORT = 11435
EXISTING_DEFAULT_PORT = 11434
ENDPOINT = f"http://127.0.0.1:{DEFAULT_PORT}/api/chat"
MODEL = "qwen3.6:27b"
MODEL_BYTES = 19_000_000_000  # Advisory registry size, not an exact pull manifest.
DEFAULT_NUM_CTX = 16384
MAX_NUM_CTX = 65536
_NUM_CTX_STEPS = (16384, 24576, 32768, 49152, 65536)
_LOCK = threading.Lock()
MAX_JSON = 2 * 1024 * 1024

_MODEL_TAG_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._/-]{0,120}(?::[A-Za-z0-9._-]{1,64})?\Z")


@dataclass(frozen=True)
class CuratedLocalModel:
    """Verified multimodal Ollama model recommendation with hardware & version metadata."""

    id: str
    title_en: str
    title_zh: str
    tier: str
    approx_download_bytes: int
    kv_vision_headroom_bytes: int
    recommended_vram_gb: int
    min_ram_gb: int
    default_num_ctx: int = DEFAULT_NUM_CTX
    supports_agent_tools: bool = False
    has_thinking: bool = False
    min_ollama_version: str = "0.32.0"
    use_case_en: str = ""
    use_case_zh: str = ""


LOCAL_MODEL_CATALOG: tuple[CuratedLocalModel, ...] = (
    CuratedLocalModel(
        id="qwen3.5:4b",
        title_en="Qwen3.5 4B",
        title_zh="Qwen3.5 4B（超轻量）",
        tier="ultra_compact",
        approx_download_bytes=3_600_000_000,
        kv_vision_headroom_bytes=2_500_000_000,
        recommended_vram_gb=6,
        min_ram_gb=8,
        default_num_ctx=DEFAULT_NUM_CTX,
        supports_agent_tools=False,
        has_thinking=True,
        min_ollama_version="0.32.12",
        use_case_en="Fast single-turn exposure, histogram & crop Q&A on entry laptops / 6 GB GPUs.",
        use_case_zh="轻量单轮曝光、直方图与构图问答（适合入门笔记本 / 6 GB 显存）。",
    ),
    CuratedLocalModel(
        id="qwen3-vl:8b",
        title_en="Qwen3-VL 8B",
        title_zh="Qwen3-VL 8B（视觉紧凑）",
        tier="compact",
        approx_download_bytes=6_200_000_000,
        kv_vision_headroom_bytes=2_500_000_000,
        recommended_vram_gb=8,
        min_ram_gb=16,
        default_num_ctx=DEFAULT_NUM_CTX,
        supports_agent_tools=False,
        has_thinking=False,
        min_ollama_version="0.32.0",
        use_case_en="Single-turn regional observation & tonal/color review on 8 GB GPUs.",
        use_case_zh="单轮画面区域观察与影调色彩诊断（适合主流 8 GB 显存）。",
    ),
    CuratedLocalModel(
        id="qwen3.5:9b",
        title_en="Qwen3.5 9B",
        title_zh="Qwen3.5 9B（紧凑全能）",
        tier="compact",
        approx_download_bytes=6_800_000_000,
        kv_vision_headroom_bytes=2_500_000_000,
        recommended_vram_gb=10,
        min_ram_gb=16,
        default_num_ctx=DEFAULT_NUM_CTX,
        supports_agent_tools=True,
        has_thinking=True,
        min_ollama_version="0.32.12",
        use_case_en="Compact multimodal coach with multi-step tool support (10 GB VRAM / 16 GB RAM).",
        use_case_zh="支持多步诊断工具的紧凑多模态教练（建议 10 GB 显存 / 16 GB 内存）。",
    ),
    CuratedLocalModel(
        id="gemma4:12b",
        title_en="Gemma 4 12B",
        title_zh="Gemma 4 12B（均衡）",
        tier="balanced",
        approx_download_bytes=8_200_000_000,
        kv_vision_headroom_bytes=2_700_000_000,
        recommended_vram_gb=12,
        min_ram_gb=24,
        default_num_ctx=DEFAULT_NUM_CTX,
        supports_agent_tools=True,
        has_thinking=True,
        min_ollama_version="0.36.0",
        use_case_en="Balanced 3x3 regional diagnosis, 3-way Color Grading & HSL Color Mixer (12 GB VRAM).",
        use_case_zh="均衡九宫格区域诊断、三向色彩分级与 HSL 混色指导（建议 12 GB 显存）。",
    ),
    CuratedLocalModel(
        id="gemma4:26b",
        title_en="Gemma 4 26B MoE",
        title_zh="Gemma 4 26B MoE（发烧级）",
        tier="enthusiast",
        approx_download_bytes=17_000_000_000,
        kv_vision_headroom_bytes=3_000_000_000,
        recommended_vram_gb=20,
        min_ram_gb=32,
        default_num_ctx=DEFAULT_NUM_CTX,
        supports_agent_tools=True,
        has_thinking=True,
        min_ollama_version="0.36.0",
        use_case_en="High-capacity MoE visual reasoning & local lighting/mask diagnosis (20 GB VRAM).",
        use_case_zh="高容量 MoE 视觉推理与局部光效蒙版诊断（建议 20 GB 显存）。",
    ),
    CuratedLocalModel(
        id="qwen3.6:27b",
        title_en="Qwen3.6 27B",
        title_zh="Qwen3.6 27B（旗舰工作站）",
        tier="workstation",
        approx_download_bytes=MODEL_BYTES,
        kv_vision_headroom_bytes=3_000_000_000,
        recommended_vram_gb=24,
        min_ram_gb=48,
        default_num_ctx=DEFAULT_NUM_CTX,
        supports_agent_tools=True,
        has_thinking=True,
        min_ollama_version="0.40.0",
        use_case_en="Flagship reference coach for full Agent v2 multi-turn tool calling & trial checks (24+ GB VRAM).",
        use_case_zh="完整支持智能体 v2 多轮工具调用与试调核验的旗舰教练（建议 24+ GB 显存）。",
    ),
)

CATALOG_BY_ID: dict[str, CuratedLocalModel] = {item.id: item for item in LOCAL_MODEL_CATALOG}

# Observed capabilities from POST /api/show or restored LocalConfig.verified_models.
_OBSERVED_CAPABILITIES: dict[tuple[int, str], frozenset[str]] = {}


def get_curated_model(model: str) -> CuratedLocalModel | None:
    return CATALOG_BY_ID.get(str(model or "").strip())


def model_download_bytes(model: str) -> int:
    curated = get_curated_model(model)
    return curated.approx_download_bytes if curated is not None else MODEL_BYTES


def valid_loopback_port(port: Any) -> bool:
    return type(port) is int and 1024 <= port <= 65535


def endpoint_for_port(port: int = DEFAULT_PORT) -> str:
    if not valid_loopback_port(port):
        raise ValueError("Invalid local Ollama port.")
    return f"http://127.0.0.1:{port}/api/chat"


def parse_ollama_endpoint_port(address: str) -> int | None:
    """Validate a loopback-only Ollama /api/chat endpoint and return its port."""
    if not isinstance(address, str) or any(c.isspace() or ord(c) < 32 for c in address):
        return None
    try:
        parsed = urlsplit(address)
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path != "/api/chat"
        or not valid_loopback_port(port)
    ):
        return None
    return int(port)


def is_cloud_proxied_model(model: Any, metadata: dict[str, Any] | None = None) -> bool:
    """Return True if a model tag or manifest indicates a cloud-proxied / remote model."""
    if not isinstance(model, str):
        return True
    lower = model.strip().lower()
    if (
        lower.endswith(":cloud")
        or "-cloud:" in lower
        or lower.endswith("-cloud")
        or "@cloud" in lower
        or "cloud/" in lower
    ):
        return True
    if isinstance(metadata, dict):
        if metadata.get("remote_host") or metadata.get("remote_model"):
            return True
    return False


def valid_local_model_tag(model: Any) -> bool:
    if not isinstance(model, str):
        return False
    cleaned = model.strip()
    if not cleaned or len(cleaned) > 160 or _MODEL_TAG_RE.fullmatch(cleaned) is None:
        return False
    return not is_cloud_proxied_model(cleaned)


def register_model_capabilities(
    model: str,
    *,
    has_vision: bool = True,
    supports_tools: bool = False,
    has_thinking: bool = False,
    port: int | None = None,
) -> None:
    """Cache verified model capabilities so payload construction and tool clamping stay accurate."""
    if not valid_local_model_tag(model):
        return
    caps = {"completion"}
    if has_vision:
        caps.add("vision")
    if supports_tools:
        caps.add("tools")
    if has_thinking:
        caps.add("thinking")
    frozen = frozenset(caps)
    _OBSERVED_CAPABILITIES[(-1, model)] = frozen
    if valid_loopback_port(port):
        _OBSERVED_CAPABILITIES[(int(port), model)] = frozen


def clear_observed_capabilities(port: int | None = None) -> None:
    if port is None:
        _OBSERVED_CAPABILITIES.clear()
        return
    for key in [k for k in _OBSERVED_CAPABILITIES if k[0] == port]:
        _OBSERVED_CAPABILITIES.pop(key, None)


def _lookup_observed_capabilities(model: str, port: int = DEFAULT_PORT) -> frozenset[str] | None:
    if (port, model) in _OBSERVED_CAPABILITIES:
        return _OBSERVED_CAPABILITIES[(port, model)]
    if (-1, model) in _OBSERVED_CAPABILITIES:
        return _OBSERVED_CAPABILITIES[(-1, model)]
    return None


def model_has_thinking(model: str, *, port: int = DEFAULT_PORT, has_thinking: bool | None = None) -> bool:
    """Return True only when /api/show capabilities (or catalog fallback before /api/show) report thinking."""
    if has_thinking is not None:
        return bool(has_thinking)
    observed = _lookup_observed_capabilities(model, port)
    if observed is not None:
        return "thinking" in observed
    curated = get_curated_model(model)
    return bool(curated.has_thinking) if curated is not None else False


def model_supports_tools(model: str, *, port: int = DEFAULT_PORT, supports_tools: bool | None = None) -> bool:
    """Return True when the local model supports Coach v2 multi-turn tool calling."""
    curated = get_curated_model(model)
    if curated is not None and not curated.supports_agent_tools:
        return False
    if supports_tools is not None:
        return bool(supports_tools)
    observed = _lookup_observed_capabilities(model, port)
    if observed is not None:
        return "tools" in observed
    return bool(curated.supports_agent_tools) if curated is not None else False


class PullProgressTracker:
    """Aggregate multi-layer /api/pull events monotonically across blob digests."""

    def __init__(self, expected_model_bytes: int = 0) -> None:
        self.expected_model_bytes = max(0, int(expected_model_bytes))
        self.layer_totals: dict[str, int] = {}
        self.layer_completed: dict[str, int] = {}
        self.high_water_completed: int = 0
        self.high_water_percent: int = 0
        self.started_at: float = time.monotonic()

    def update(self, event: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(event, dict):
            return {}
        status = str(event.get("status") or "")
        digest = str(event.get("digest") or "")
        total = event.get("total")
        completed = event.get("completed")
        if status == "success":
            known_total = sum(self.layer_totals.values())
            final_total = known_total or self.high_water_completed or self.expected_model_bytes
            self.high_water_completed = max(self.high_water_completed, final_total)
            self.high_water_percent = 100
            return {
                **event,
                "status": "success",
                "completed": self.high_water_completed,
                "total": final_total,
                "percent": 100,
                "layers": len(self.layer_totals),
            }
        if type(total) is int and total > 0:
            key = digest or "_default"
            prev_total = self.layer_totals.get(key, 0)
            self.layer_totals[key] = max(prev_total, total)
            if type(completed) is int and completed >= 0:
                prev_comp = self.layer_completed.get(key, 0)
                self.layer_completed[key] = min(self.layer_totals[key], max(prev_comp, completed))
            agg_completed = max(self.high_water_completed, sum(self.layer_completed.values()))
            self.high_water_completed = agg_completed
            known_total = sum(self.layer_totals.values())
            agg_total = max(known_total, self.expected_model_bytes, agg_completed, 1)
            raw_pct = min(99, max(0, int(100 * agg_completed / agg_total)))
            self.high_water_percent = max(self.high_water_percent, raw_pct)
            elapsed = max(0.001, time.monotonic() - self.started_at)
            return {
                **event,
                "completed": agg_completed,
                "total": agg_total,
                "layer_completed": completed if type(completed) is int else 0,
                "layer_total": total,
                "percent": self.high_water_percent,
                "layers": len(self.layer_totals),
                "speed_bps": agg_completed / elapsed,
            }
        return {
            **event,
            "percent": self.high_water_percent,
            "layers": len(self.layer_totals),
        }


def _sanitize_local_detail(detail: Any) -> str:
    if not isinstance(detail, str):
        return ""
    cleaned = re.sub(r"[\x00-\x1f\x7f]+", " ", detail).strip()
    cleaned = " ".join(cleaned.split())
    return cleaned[:240]


class LocalError(RuntimeError):
    def __init__(self, code: str, *, http_status: int | None = None, detail: str = ""):
        messages = {
            "connection": "The local engine is unavailable. Prepare Local AI and try again.",
            "cancelled": "Local operation cancelled.",
            "timeout": "The local operation timed out. No retry was made.",
            "busy": "Another local model operation is still running.",
            "response": "The local engine returned an incomplete or invalid response.",
            "vision": "This model did not pass the image and structured-output check.",
            "model": "The selected local model is not installed or is not a valid local model.",
            "http": "The local engine rejected the operation. Check its version and model installation.",
            "version": "This model requires a newer Ollama version. Install or select an updated engine, then detect again.",
        }
        self.code = code if code in messages else "connection"
        self.http_status = http_status if type(http_status) is int and 100 <= http_status <= 599 else None
        self.detail = _sanitize_local_detail(detail)
        message = messages[self.code]
        if self.http_status is not None:
            message = f"HTTP {self.http_status}: " + message
        if self.detail:
            message = f"{message} ({self.detail})"
        super().__init__(message)


def _decode(raw: bytes | str) -> Any:
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError("duplicate field")
            result[k] = v
        return result

    return json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")),
    )


def exchange(
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    cancel: threading.Event | None = None,
    timeout: float = 15,
    progress=None,
    port: int = DEFAULT_PORT,
    expected_bytes: int = 0,
):
    """No DNS, proxies, authentication, redirects, or provider-controlled URLs."""
    if path not in ("/api/version", "/api/tags", "/api/show", "/api/chat", "/api/pull", "/api/delete"):
        raise LocalError("http")
    if not valid_loopback_port(port):
        raise LocalError("connection")
    cancel = cancel if cancel is not None else threading.Event()
    if cancel.is_set():
        raise LocalError("cancelled")
    if not _LOCK.acquire(blocking=False):
        raise LocalError("busy")
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    done, result = threading.Event(), []

    def run():
        try:
            body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
            method = "DELETE" if path == "/api/delete" else ("GET" if payload is None else "POST")
            connection.request(method, path, body, {"Content-Type": "application/json"})
            response = connection.getresponse()
            if response.status != 200:
                err_detail = ""
                try:
                    raw_err = response.read(4096)
                    if raw_err:
                        try:
                            parsed_err = _decode(raw_err)
                            if isinstance(parsed_err, dict) and parsed_err.get("error"):
                                err_detail = _sanitize_local_detail(str(parsed_err["error"]))
                        except Exception:
                            err_detail = _sanitize_local_detail(raw_err.decode("utf-8", errors="replace"))
                except Exception:
                    pass
                if path == "/api/delete" and response.status == 404:
                    raise LocalError("model", http_status=404, detail=err_detail)
                is_ver = response.status == 412 or "newer version" in err_detail.lower()
                raise LocalError("version" if is_ver else "http", http_status=response.status, detail=err_detail)
            if path == "/api/pull":
                success = False
                tracker = PullProgressTracker(expected_model_bytes=expected_bytes)
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
                        raise LocalError(
                            "version" if "newer version" in reason or "412" in reason else "response",
                            detail=str(event["error"]),
                        )
                    if not isinstance(event, dict):
                        raise LocalError("response")
                    tracked = tracker.update(event)
                    if progress:
                        progress(tracked)
                    if event.get("status") == "success":
                        success = True
                        break
                if not success:
                    raise LocalError("response")
                result.append({"status": "success", "percent": 100})
            else:
                raw = response.read(MAX_JSON + 1)
                if len(raw) > MAX_JSON:
                    raise LocalError("response", detail="response exceeded 2 MiB limit")
                if path == "/api/delete" and not raw.strip():
                    result.append({"status": "deleted"})
                else:
                    value = _decode(raw)
                    if not isinstance(value, dict) or value.get("error"):
                        err_text = str(value.get("error")) if isinstance(value, dict) and value.get("error") else ""
                        raise LocalError("response", detail=err_text)
                    result.append(value)
        except LocalError as error:
            result.append(error)
        except Exception as exc:
            result.append(LocalError("connection", detail=f"127.0.0.1:{port} ({type(exc).__name__})"))
        finally:
            connection.close()
            _LOCK.release()
            done.set()

    threading.Thread(target=run, name="PhotoCoach-Ollama", daemon=True).start()
    deadline = time.monotonic() + timeout
    while not done.wait(0.05):
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


def _call_exchange(path: str, payload: dict[str, Any] | None = None, *, port: int = DEFAULT_PORT, **kwargs):
    """Preserve single-port call signature for existing monkeypatched exchange tests when port == DEFAULT_PORT."""
    if port != DEFAULT_PORT:
        kwargs["port"] = port
    return exchange(path, payload, **kwargs)


def inspect_model_capabilities(
    model: str = MODEL,
    *,
    cancel: threading.Event | None = None,
    port: int = DEFAULT_PORT,
) -> dict[str, Any]:
    """Query POST /api/show for one installed model and cache its capabilities."""
    if not valid_local_model_tag(model):
        raise LocalError("model")
    details = _call_exchange("/api/show", {"model": model}, cancel=cancel, port=port)
    if not isinstance(details, dict) or is_cloud_proxied_model(model, details):
        raise LocalError("model")
    raw_caps = details.get("capabilities")
    if not isinstance(raw_caps, list):
        raw_caps = []
    caps = frozenset(str(c).strip().lower() for c in raw_caps if isinstance(c, str))
    _OBSERVED_CAPABILITIES[(port, model)] = caps
    _OBSERVED_CAPABILITIES[(-1, model)] = caps
    curated = get_curated_model(model)
    has_vision = "vision" in caps
    has_thinking = "thinking" in caps
    supports_tools = ("tools" in caps) and (curated.supports_agent_tools if curated is not None else True)
    return {
        "capabilities": tuple(sorted(caps)),
        "has_vision": has_vision,
        "supports_tools": supports_tools,
        "has_thinking": has_thinking,
    }


def model_identity(model: str = MODEL, *, cancel: threading.Event | None = None, port: int = DEFAULT_PORT) -> str:
    if not valid_local_model_tag(model):
        raise LocalError("model")
    tags = _call_exchange("/api/tags", cancel=cancel, port=port).get("models", [])
    matches = [
        item for item in tags
        if isinstance(item, dict)
        and (item.get("name") == model or item.get("model") == model)
        and not is_cloud_proxied_model(model, item)
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("digest"), str) or not matches[0]["digest"].strip():
        raise LocalError("model")
    cap_info = inspect_model_capabilities(model, cancel=cancel, port=port)
    if not cap_info["has_vision"]:
        raise LocalError("vision")
    return matches[0]["digest"]


def delete_installed_model(model: str, *, cancel: threading.Event | None = None, port: int = DEFAULT_PORT) -> dict[str, Any]:
    if not valid_local_model_tag(model):
        raise LocalError("model")
    res = _call_exchange("/api/delete", {"model": model}, cancel=cancel, port=port)
    _OBSERVED_CAPABILITIES.pop((port, model), None)
    _OBSERVED_CAPABILITIES.pop((-1, model), None)
    return res


def _estimate_required_num_ctx(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    output_tokens: int,
    base_ctx: int,
) -> int:
    text_chars = 0
    image_count = 0
    for msg in messages or ():
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            text_chars += len(content)
        tool_name = msg.get("tool_name")
        if isinstance(tool_name, str):
            text_chars += len(tool_name)
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            try:
                text_chars += len(json.dumps(tool_calls, ensure_ascii=False))
            except TypeError:
                text_chars += len(str(tool_calls))
        images = msg.get("images")
        if isinstance(images, (list, tuple)):
            image_count += len(images)
    if tools:
        try:
            text_chars += len(json.dumps(tools, ensure_ascii=False))
        except TypeError:
            text_chars += len(str(tools))
    reserved_out = min(8192, max(512, int(output_tokens) if int(output_tokens) > 0 else 4096))
    needed = (text_chars // 3) + (image_count * 2200) + reserved_out + 1024
    floor_ctx = max(DEFAULT_NUM_CTX, int(base_ctx or DEFAULT_NUM_CTX), needed)
    for step in _NUM_CTX_STEPS:
        if floor_ctx <= step:
            return step
    return MAX_NUM_CTX


def chat_payload(
    model: str,
    messages: list[dict[str, Any]],
    schema: dict[str, Any] | None = None,
    *,
    output_tokens: int = 0,
    tools: list[dict[str, Any]] | None = None,
    num_ctx: int = DEFAULT_NUM_CTX,
    has_thinking: bool | None = None,
    port: int = DEFAULT_PORT,
) -> dict[str, Any]:
    if not valid_local_model_tag(model):
        raise LocalError("model")
    curated = get_curated_model(model)
    base_ctx = int(num_ctx or (curated.default_num_ctx if curated else DEFAULT_NUM_CTX))
    effective_ctx = _estimate_required_num_ctx(messages, tools, output_tokens, base_ctx)
    thinking_model = model_has_thinking(model, port=port, has_thinking=has_thinking)
    thinking_active = bool(thinking_model and tools)
    num_predict = -1 if int(output_tokens) <= 0 else int(output_tokens)
    options: dict[str, Any] = {
        "num_ctx": effective_ctx,
        "num_predict": num_predict,
        "temperature": 0.6 if thinking_active else 0.2,
    }
    if thinking_active:
        options["top_p"] = 0.95
        options["repeat_penalty"] = 1.05
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
        "keep_alive": "5m",
    }
    if thinking_model and not tools:
        payload["think"] = False
    if schema is not None:
        payload["format"] = schema
    if tools:
        payload["tools"] = tools
    return payload


def chat_result(document: dict[str, Any], *, allow_tool_calls: bool = False) -> dict[str, Any]:
    reasons = ("stop", "tool_calls") if allow_tool_calls else ("stop",)
    if document.get("done") is not True or document.get("done_reason") not in reasons:
        raise LocalError(
            "response",
            detail=f"done={document.get('done')}, done_reason={document.get('done_reason')!r}",
        )
    message = document.get("message", {})
    if message.get("role") != "assistant":
        raise LocalError("response", detail=f"unexpected role {message.get('role')!r}")
    tool_calls = message.get("tool_calls")
    content = message.get("content")
    if not allow_tool_calls:
        if not isinstance(content, str) or tool_calls:
            raise LocalError(
                "response",
                detail="unexpected tool_calls in single-shot reply" if tool_calls else "non-string assistant content",
            )
        return document
    if content is not None and not isinstance(content, str):
        raise LocalError("response", detail="non-string assistant content")
    if tool_calls:
        if not isinstance(tool_calls, list):
            raise LocalError("response", detail="invalid tool_calls payload")
        for call in tool_calls:
            fn = call.get("function") if isinstance(call, dict) else None
            if (
                not isinstance(fn, dict)
                or not isinstance(fn.get("name"), str)
                or not fn["name"].strip()
                or not isinstance(fn.get("arguments"), (dict, str))
            ):
                raise LocalError("response", detail="invalid tool_calls payload")
    elif not isinstance(content, str) or not content.strip():
        raise LocalError("response", detail="empty assistant content")
    return document


def verify_vision(
    model: str = MODEL,
    *,
    cancel: threading.Event | None = None,
    port: int = DEFAULT_PORT,
    has_thinking: bool | None = None,
) -> dict[str, Any]:
    from PIL import Image
    from io import BytesIO
    import base64

    image = Image.new("RGB", (256, 128), "red")
    image.paste((0, 0, 255), (128, 0, 256, 128))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    schema = {
        "type": "object",
        "properties": {name: {"type": "string", "enum": ["red", "blue"]} for name in ("left", "right")},
        "required": ["left", "right"],
        "additionalProperties": False,
    }
    payload = chat_payload(
        model,
        [
            {
                "role": "user",
                "content": "Inspect the image. Return the color of its left half and right half as JSON. Use English red or blue.",
                "images": [base64.b64encode(buffer.getvalue()).decode()],
            }
        ],
        schema,
        output_tokens=128,
        has_thinking=has_thinking,
        port=port,
    )
    document = chat_result(_call_exchange("/api/chat", payload, cancel=cancel, timeout=300, port=port))
    try:
        if _decode(document["message"]["content"]) != {"left": "red", "right": "blue"}:
            raise ValueError
    except (ValueError, TypeError, KeyError):
        raise LocalError("vision") from None
    return document
