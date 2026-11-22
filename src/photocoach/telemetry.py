"""Bounded local trace logging and optional OpenTelemetry span export."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        cleaned = re.sub(r"[A-Za-z]:\\[^\s\"']+|/[^\s\"']+\.(?:arw|nef|photocoach)", "[redacted-path]", value, flags=re.I)
        cleaned = re.sub(r"\b(?:sk-[A-Za-z0-9_-]{8,}|AIza[A-Za-z0-9_-]{12,})\b", "[redacted-key]", cleaned)
        return cleaned[:500]
    if isinstance(value, dict):
        return {str(k): _sanitize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(v) for v in value]
    return value


class TraceStore:
    """Bounded local JSON trace store with automatic oldest-file rotation."""

    def __init__(self, directory: Path | None = None, *, max_files: int = 50,
                 max_total_bytes: int = 5 * 1024 * 1024) -> None:
        if directory is None:
            from .settings import SettingsStore
            directory = SettingsStore().path.parent / "traces"
        self.directory = Path(directory)
        self.max_files = max(1, int(max_files))
        self.max_total_bytes = max(1024, int(max_total_bytes))

    def record_run(self, run_id: str, record: dict[str, Any]) -> Path | None:
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "", str(run_id))[:48] or "run"
        sanitized = _sanitize_value(dict(record))
        sanitized["run_id"] = safe_id
        raw = (json.dumps(sanitized, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
        if len(raw) > self.max_total_bytes // 2:
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        existing = sorted(self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime)
        idx = len(existing) + 1
        target = self.directory / f"trace-{idx:05d}-{safe_id}.json"
        while target.exists():
            idx += 1
            target = self.directory / f"trace-{idx:05d}-{safe_id}.json"
        fd, name = tempfile.mkstemp(prefix=".photocoach-trace-", suffix=".tmp", dir=self.directory)
        tmp = Path(name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
        self._prune()
        return target

    def _prune(self) -> None:
        files = sorted(self.directory.glob("*.json"), key=lambda p: (p.stat().st_mtime, p.name))
        total = sum(p.stat().st_size for p in files)
        while files and (len(files) > self.max_files or total > self.max_total_bytes):
            victim = files.pop(0)
            try:
                size = victim.stat().st_size
                victim.unlink(missing_ok=True)
                total -= size
            except OSError:
                break

    def load_recent(self, limit: int = 10) -> list[dict[str, Any]]:
        if not self.directory.is_dir():
            return []
        files = sorted(self.directory.glob("*.json"), key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
        records: list[dict[str, Any]] = []
        for path in files[: max(0, int(limit))]:
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(doc, dict):
                    records.append(doc)
            except (OSError, ValueError):
                continue
        return records


def export_otel_spans(run_id: str, spans: tuple[Any, ...], attributes: dict[str, Any] | None = None) -> bool:
    """Export agent TraceSpans via OpenTelemetry if the optional `[otel]` extra is installed."""
    try:
        from opentelemetry import trace as otel_trace  # type: ignore[import-not-found]
    except ImportError:
        return False
    tracer = otel_trace.get_tracer("photocoach.agent")
    root_attrs = {"photocoach.run_id": str(run_id)[:64]}
    if attributes:
        for k, v in attributes.items():
            if isinstance(v, (str, int, float, bool)):
                root_attrs[f"photocoach.{k}"] = v
    with tracer.start_as_current_span("photocoach.agent.run", attributes=root_attrs):
        for span in spans:
            span_attrs: dict[str, Any] = {
                "agent.node": getattr(span, "node", ""),
                "agent.status": getattr(span, "status", ""),
                "agent.duration_ms": float(getattr(span, "duration_ms", 0.0)),
            }
            if getattr(span, "input_tokens", None) is not None:
                span_attrs["llm.input_tokens"] = int(span.input_tokens)
            if getattr(span, "output_tokens", None) is not None:
                span_attrs["llm.output_tokens"] = int(span.output_tokens)
            with tracer.start_as_current_span(
                f"photocoach.agent.{getattr(span, 'node', 'step')}.{getattr(span, 'name', 'op')}",
                attributes=span_attrs,
            ):
                pass
    return True
