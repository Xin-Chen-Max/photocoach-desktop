"""Tests for PhotoCoach read-only MCP stdio JSON-RPC server."""
from __future__ import annotations

from io import StringIO
import json

import numpy as np

from photocoach.mcp_server import PhotoCoachMCPServer, main
from photocoach.state import StateStore


def test_mcp_server_initialize_list_and_call_tools():
    server = PhotoCoachMCPServer()

    init = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init["result"]["serverInfo"]["name"] == "photocoach-mcp"
    assert server.handle_request({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert server.handle_request({"jsonrpc": "2.0", "id": 2, "method": "ping"}) == {"jsonrpc": "2.0", "id": 2, "result": {}}

    listed = server.handle_request({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
    names = {t["name"] for t in listed["result"]["tools"]}
    assert {
        "inspect_photo_evidence",
        "inspect_region_100pct",
        "inspect_raw_headroom",
        "analyze_color_zones",
        "detect_scene_lighting_anchors",
        "build_lighting_rig",
        "search_editing_playbook",
        "propose_and_preview_trial",
    } <= names

    # Call search_editing_playbook
    pb_res = server.handle_request({
        "jsonrpc": "2.0", "id": 4, "method": "tools/call",
        "params": {"name": "search_editing_playbook", "arguments": {"query": "sharpening edge masking", "top_k": 2}},
    })
    assert pb_res["result"]["isError"] is False
    data = json.loads(pb_res["result"]["content"][0]["text"])
    assert len(data["matches"]) >= 1

    # Call inspect_photo_evidence
    ev_res = server.handle_request({
        "jsonrpc": "2.0", "id": 5, "method": "tools/call",
        "params": {"name": "inspect_photo_evidence", "arguments": {}},
    })
    assert ev_res["result"]["isError"] is False
    ev_data = json.loads(ev_res["result"]["content"][0]["text"])
    assert "current" in ev_data and "whole" in ev_data["current"]

    # Unknown tool returns clean isError=True without crashing
    bad = server.handle_request({
        "jsonrpc": "2.0", "id": 6, "method": "tools/call",
        "params": {"name": "mutate_raw_file", "arguments": {}},
    })
    assert bad["result"]["isError"] is True


def test_mcp_stdio_loop_and_image_block_and_self_check(capsys):
    crop = np.full((64, 64, 3), 120, dtype=np.uint8)

    class FakeEngine:
        def inspect_region_100pct(self, state, x, y, size_px, apply_current_edits=True):
            return crop, {"edge_energy_mean": 0.05, "rect_px": [0, 0, 64, 64]}

    server = PhotoCoachMCPServer(engine=FakeEngine())
    in_lines = "\n".join([
        json.dumps({"jsonrpc": "2.0", "id": 10, "method": "tools/call",
                    "params": {"name": "inspect_region_100pct", "arguments": {"x_norm": 0.5, "y_norm": 0.5, "size_px": 128}}}),
        "{broken-json",
    ]) + "\n"
    out = StringIO()
    server.serve_stdio(StringIO(in_lines), out)
    responses = [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]
    assert len(responses) == 2
    blocks = responses[0]["result"]["content"]
    assert blocks[0]["type"] == "text"
    assert blocks[1]["type"] == "image" and blocks[1]["mimeType"] == "image/png"
    assert responses[1]["error"]["code"] == -32700

    # CLI self-check exits 0 and emits JSON manifest
    rc = main(["--self-check"])
    assert rc == 0
    captured = json.loads(capsys.readouterr().out)
    assert captured["server"] == "photocoach-mcp"
    assert "propose_and_preview_trial" in captured["tools"]
