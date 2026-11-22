# Testing and verification

## 0.12.1 Agent & Pipeline checks

The test suite covers the complete application, rendering pipeline, and Agent v2 subsystem (over 1,290 unit and integration tests). It requires no camera files or network API calls.

```powershell
& .\.venv\Scripts\python.exe -m pytest -q -rs
& .\.venv\Scripts\python.exe scripts\verify_wheel.py --output verification-runs\wheel-check
& .\.venv\Scripts\python.exe scripts\run_agent_eval.py
```

The Agent evaluation suite (`scripts/run_agent_eval.py`) verifies:
- State-graph ReAct execution (`observe -> reason -> tool_call -> trial_render -> reflect -> finalize`).
- Active Vision demosaiced 1:1 inspection and Bayer CFA headroom probing.
- Visual Reflexion self-repair: sandboxed trial candidate rendering that detects and corrects overexposure, contrast clipping, or no-op edits.
- Multi-protocol wire normalization across Gemini, OpenAI, and Ollama formats.
- Prompt injection and out-of-bounds parameter rejection.

## 0.12.0 release checks

The public source completed **1,238 tests with 2 skips** on Windows / Python 3.12.14, exit 0. The skips are the opt-in real credential test and a symlink operation unavailable to this Windows account. The isolated wheel build and package smoke test also passed.

The default suite isolates settings, removes common key variables and injects fake provider transports. It covers image operators, state/history, migration, file protection, request handling, cancellation, response escaping and Qt interactions. It requires no camera file or paid API call.

```powershell
& .\.venv\Scripts\python.exe -m pytest -q -rs
& .\.venv\Scripts\python.exe scripts\verify_wheel.py --output verification-runs\wheel-check
```

A test run passes only if its process exits normally. Qt is owned for the test session, and test windows/workers are closed before interpreter teardown.

## Native-scale detail regression

Before 0.12.0, dragging applied sharpening/NR to an already reduced preview. Refinement filtered native pixels before reducing, so Fit could show an exaggerated temporary effect. At 100% the two paths were much closer.

Active spatial detail now uses native sampling during dragging and refinement, including a detail gesture that reaches zero. Tests cover sharpening, edge masking with sharpening enabled, both NR controls, the preset, rotation, cold/warm decodes, zero endpoints, cancellation and one-gesture history. Active-detail project refreshes also use native filters.

`verify_parameter_previews.py` passed **113 cases on synthetic hue/texture data and 113 on a native Sony RAW region**. It covers global tone, curves, all 24 HSL controls in both directions, base modes, lens correction, detail, crop/straighten, linear/radial mask fields and combined edits. Formal 8-bit previews matched the export path; actual 16-bit TIFF files were written/read with at most one 8-bit level of quantization difference. Five full RAW re-decodes checked warmth, tint and each relative RGB gain.

`verify_parameter_ui.py` passed **113 native-window cases** on a real Sony ARW. Scalar, HSL, lens and individual detail cases used slider signals. Discrete composition/base changes and compound mask snapshots used their state commit interfaces. Displayed final pixels, Undo/Redo and stale-frame guards were checked. English/Chinese 1280x800 and 1920x1080 captures covered Fit/100% and dragging/refinement. No claim is made about every possible crop/mask pointer trajectory.

```powershell
& .\.venv\Scripts\python.exe scripts\verify_parameter_previews.py C:\photos\sample.ARW verification-runs\parameters
& .\.venv\Scripts\python.exe scripts\verify_parameter_ui.py C:\photos\sample.ARW verification-runs\native-ui
```

Use a fresh output directory. Reports and images can contain photographs and local paths; they stay local and are ignored by Git.

## Editing and request workflows

The two-source real-RAW workflow passed ten edits, exact Undo/Redo pixels, project save/reopen with redo, aligned comparison, photo switching and full JPEG/TIFF export. An independent full decode/render matched every 16-bit TIFF value; the native preview comparison had zero 8-bit difference. Original source hashes were unchanged.

The simulated cloud workflow passed request review, exact parameter context, editing while a request waited, stale reply rejection, timeout without retry and continued editing. It made three simulated attempts and zero remote calls. A regression also covers a timeout exception without duration metadata, which must produce a safe message rather than a formatting error.

```powershell
& .\.venv\Scripts\python.exe scripts\verify_workflow.py C:\photos\first.ARW --second C:\photos\second.ARW --output verification-runs\workflow
& .\.venv\Scripts\python.exe scripts\verify_coach_workflow.py C:\photos\first.ARW --output verification-runs\coach
```

## Performance observations

Five warmed repeats on one Windows workstation and one Sony ARW measured:

| Work | Median | Slowest |
| --- | ---: | ---: |
| Fit, Sharpening 80, native filtering | 1,549.70 ms | 1,783.73 ms |
| 100%, 960x640 region, Sharpening 80 | 83.95 ms | 91.72 ms |
| Fit, exposure-only fast approximation | 43.01 ms | 112.08 ms |

Cold quick import was 1.199 seconds. The first formal Fit render including full decode was 2.582 seconds. Repeated edits caused no additional RAW decodes. Retained linear data stayed below 512 MiB; measured process peak working set was about 944 MiB. This is a local reference, not a guarantee for another camera or machine. Fit detail is intentionally more expensive than the old reduced-grid filter, which did different work.

```powershell
& .\.venv\Scripts\python.exe scripts\benchmark_detail_preview.py C:\photos\sample.ARW verification-runs\detail-timing
```

## Win64 package

The final unsigned AMD64 ZIP was extracted into a new path containing spaces and Unicode. With project Python and the checkout removed from PATH, it opened the real RAW, applied Sharpening 80, undid it, loaded its icon and verified TIFF output. No cloud request or credential read was used. The builder records dependency versions and LF-normalized source hashes. The package includes the MIT license and component notices.

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
& .\.venv\Scripts\python.exe scripts\build_windows.py --output dist\win64
& .\.venv\Scripts\python.exe scripts\verify_windows_bundle.py dist\win64\PhotoCoach-0.12.0-win64.zip verification-runs\bundle --raw C:\photos\sample.ARW
```

The SHA-256 checksum for the downloadable file is in the release's `SHA256SUMS.txt`. Rebuilds can have different archive hashes. Another physical machine, ARM64, code signing and SmartScreen reputation remain unverified.

## Evidence limits

Real-photo verification used developer-supplied Sony files that are not redistributed. Synthetic colors exercise hue bands absent from a particular scene. The README screenshot uses a credited public-domain JPEG and demonstrates the UI, not RAW decoding. Live-provider quality, calibrated monitor color, camera-JPEG/Lightroom matching and aesthetic accuracy of local models were not established by these checks.

The earlier October 24, 2026 source publication recorded 968 passed and 2 skipped. Those historical results are separate from the 0.12.0 checks above.
