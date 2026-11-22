"""Read-only Model Context Protocol (MCP) server exposing PhotoCoach's validated tools."""
from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import sys
from typing import Any, TextIO

import numpy as np
from PIL import Image

from . import __version__
from .agent_memory import PlaybookRAG
from .agent_tools import (ToolCall, ToolContext, ToolRegistry, ToolSpec,
                          build_default_tool_registry)
from .coach import CoachRequest, build_request
from .coach_agent import evidence_context
from .imaging import RawOptions
from .state import EditState, StateStore


MAX_RPC_LINE_BYTES = 1024 * 1024


def _png_base64(pixels: np.ndarray) -> str:
    buf = BytesIO()
    Image.fromarray(pixels).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _default_synthetic_context() -> tuple[EditState, CoachRequest]:
    store = StateStore()
    state = store.activate(Path("photocoach-session.ARW"), "photocoach-mcp-session")
    preview = np.full((64, 96, 3), 96, dtype=np.uint8)
    request = build_request(state, state, [], preview, preview, source_size=(96, 64))
    return state, request


class PhotoCoachMCPServer:
    """Stdlib JSON-RPC 2.0 MCP server sharing ToolRegistry with AgentRuntime."""

    PROTOCOL_VERSION = "2025-06-18"

    def __init__(
        self,
        *,
        registry: ToolRegistry | None = None,
        playbook: PlaybookRAG | None = None,
        state: EditState | None = None,
        request: CoachRequest | None = None,
        engine: Any | None = None,
    ) -> None:
        self.registry = registry if registry is not None else build_default_tool_registry()
        self.playbook = playbook if playbook is not None else PlaybookRAG()
        if state is None or request is None:
            state, request = _default_synthetic_context()
        self.state = state
        self.request = request
        self.engine = engine

        self._evidence_spec = ToolSpec(
            name="inspect_photo_evidence",
            description="Return 3x3 linear luminance measurements and active editing parameters (read-only).",
            parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            read_only=True,
        )

    def tools(self) -> tuple[ToolSpec, ...]:
        return (self._evidence_spec, *self.registry.specs())

    def tool_manifests(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "inputSchema": spec.parameters,
                "annotations": {"readOnlyHint": True, "destructiveHint": False},
            }
            for spec in self.tools()
        ]

    def _make_context(self) -> ToolContext:
        engine = self.engine
        state = self.state
        request = self.request
        return ToolContext(
            request=request,
            state=state,
            render_trial=(
                (lambda values: engine.render_trial_values(state, values, source_size=request.source_size))
                if engine is not None
                else None
            ),
            render_crop_100pct=(
                (lambda x, y, s, apply_current_edits=True:
                 engine.inspect_region_100pct(state, x, y, s, apply_current_edits=apply_current_edits))
                if engine is not None
                else None
            ),
            inspect_cfa=(
                (lambda x, y, p: engine.inspect_raw_headroom(state, x, y, p))
                if engine is not None
                else None
            ),
            playbook=self.playbook,
        )

    def handle_request(self, message: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request"}}
        method = message.get("method")
        req_id = message.get("id")
        if method == "notifications/initialized" or req_id is None:
            return None
        params = message.get("params") or {}
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": self.PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "photocoach-mcp", "version": __version__},
                },
            }
        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": self.tool_manifests()}}
        if method == "tools/call":
            if not isinstance(params, dict) or not isinstance(params.get("name"), str):
                return {"jsonrpc": "2.0", "id": req_id,
                        "error": {"code": -32602, "message": "tools/call requires string tool name."}}
            name = params["name"]
            arguments = params.get("arguments")
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                return {"jsonrpc": "2.0", "id": req_id,
                        "error": {"code": -32602, "message": "Tool arguments must be an object."}}
            if name == self._evidence_spec.name:
                payload = evidence_context(self.request, include_before=False)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
                        "isError": False,
                    },
                }
            ctx = self._make_context()
            result = self.registry.execute(ToolCall(str(req_id), name, arguments), ctx)
            content: list[dict[str, Any]] = [
                {"type": "text", "text": json.dumps(result.payload, ensure_ascii=False, allow_nan=False)}
            ]
            for img in result.images:
                content.append({
                    "type": "image",
                    "data": _png_base64(img.pixels),
                    "mimeType": "image/png",
                })
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": content, "isError": not result.ok},
            }
        return {"jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}}

    def serve_stdio(self, instream: TextIO = sys.stdin, outstream: TextIO = sys.stdout) -> None:
        while True:
            line = instream.readline(MAX_RPC_LINE_BYTES + 1)
            if not line:
                break
            if len(line) > MAX_RPC_LINE_BYTES:
                err = {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Line too long"}}
                outstream.write(json.dumps(err) + "\n")
                outstream.flush()
                continue
            stripped = line.strip()
            if not stripped:
                continue
            try:
                msg = json.loads(stripped)
            except ValueError:
                err = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
                outstream.write(json.dumps(err) + "\n")
                outstream.flush()
                continue
            reply = self.handle_request(msg)
            if reply is not None:
                outstream.write(json.dumps(reply, ensure_ascii=False, allow_nan=False) + "\n")
                outstream.flush()


def create_mcp_sdk_server(photo_server: PhotoCoachMCPServer | None = None):
    """Optional bridge into the official `mcp` Python SDK when `[mcp]` extra is installed."""
    import mcp.types as mcp_types  # type: ignore[import-not-found]
    from mcp.server.lowlevel import Server  # type: ignore[import-not-found]

    backend = photo_server if photo_server is not None else PhotoCoachMCPServer()
    server = Server("photocoach-mcp")

    @server.list_tools()
    async def _list_tools():
        return [
            mcp_types.Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"])
            for t in backend.tool_manifests()
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]):
        reply = backend.handle_request({
            "jsonrpc": "2.0", "id": "sdk", "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        })
        items = []
        for block in (reply or {}).get("result", {}).get("content", []):
            if block["type"] == "text":
                items.append(mcp_types.TextContent(type="text", text=block["text"]))
            elif block["type"] == "image":
                items.append(mcp_types.ImageContent(type="image", data=block["data"], mimeType=block["mimeType"]))
        return items

    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PhotoCoach read-only MCP tool server")
    parser.add_argument("--photo", type=Path, default=None, help="Optional read-only RAW photo path")
    parser.add_argument("--self-check", action="store_true", help="Print MCP tool manifest JSON and exit")
    args = parser.parse_args(argv)

    engine = None
    state = None
    request = None
    if args.photo is not None:
        from .app import PreviewEngine
        engine = PreviewEngine()
        photo, small = engine.photo(args.photo, RawOptions())
        store = StateStore()
        state = store.activate(args.photo, photo.metadata["sha256"])
        preview = engine._render_small(photo, small, state.values, 768)
        request = build_request(state, state, [], preview, preview, source_size=photo.metadata.get("output_size"))

    server = PhotoCoachMCPServer(state=state, request=request, engine=engine)
    if args.self_check:
        print(json.dumps({
            "server": "photocoach-mcp",
            "version": __version__,
            "protocol": server.PROTOCOL_VERSION,
            "tools": [t["name"] for t in server.tool_manifests()],
        }, indent=2))
        return 0
    server.serve_stdio()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
