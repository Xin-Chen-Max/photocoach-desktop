# Release notes

## 0.12.1

- Added Agent v2 bounded ReAct state-graph runtime (`AgentRuntime`: observe -> reason -> tool_call -> trial_render -> reflect -> finalize) with turn/tool/trial/timeout caps and deterministic checkpoint replay.
- Added Active Vision tools: 1:1 demosaiced crop inspection (`inspect_region_100pct`), normalized Bayer CFA sensor headroom probing (`probe_sensor_headroom`), and 8-band HSL/CIELAB zoning (`analyze_hsl_zones`).
- Added Visual Reflexion trial criticism loop (`propose_and_preview_trial`): renders trial edits through the live linear pipeline to self-repair overexposure, clipping, or no-op adjustments before final suggestions.
- Unified multi-protocol tool calling across Gemini (`thoughtSignature` + multimodal `$ref` `FunctionResponse.parts`), OpenAI Responses, Chat Completions, and local Ollama (`qwen3.6:27b`).
- Added Model Context Protocol (MCP `2025-06-18`) stdio server (`photocoach-mcp`) exposing editing, sensor diagnostics, and playbook tools to external MCP agents.
- Added 25-scenario evaluation and benchmark harness (`photocoach-eval`) testing reasoning, trial self-repair recovery, prompt-injection resilience, and tool robustness.
- Added zero-dependency bilingual BM25 Playbook RAG (24 pipeline-calibrated recipe cards) and privacy-preserving user style memory.
- Added 3D spotlight rig lighting masks, calibrated Detail engine, seamless zoom pyramid, and darkroom UI polish.

## 0.12.0

- Fixed sharpening and NR previews that used a larger effective radius during dragging than after refinement. Spatial filters now run before display reduction; inspect detail at 100%.
- Added manual lens correction, detail controls, Natural/Neutral base modes and schema 9 migration that keeps old project rendering in Legacy.
- Added explicit structured Coach actions, fixed target values, single-step application, stale-reply protection and a simpler collapsible Coach panel.
- Added optional local Ollama model preparation and English/Chinese controls.
- Added a standalone unsigned Windows AMD64 ZIP with licenses, dependency/source metadata and isolated runtime verification.
- Updated real-image verification scripts for settled detail frames, safe timeout messages and source-only public documentation.

## 0.10.2

Initial public source release.

- Non-destructive Sony ARW editing with tone controls, white balance, RGB curves, and HSL mixing.
- Linear/radial local adjustments, crop/straighten, grouped undo/redo, and schema 7 projects.
- Progressive adjustment previews with retained images during refinement.
- Full-resolution JPEG and 16-bit TIFF export.
- Optional manual AI coaching, saved connections, model selection, credential integration, and request limits.
- English documentation, offline regression tests, and local workflow/build verification tools.

Lens correction and other planned editing tools are not part of this release.
