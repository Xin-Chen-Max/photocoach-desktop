# Testing and verification

The October 24, 2026 publication check on Windows / Python 3.12 completed with **968 passed, 2 skipped**. The isolated wheel build and headless package smoke check also passed, with zero API attempts. Real-RAW and live-provider checks were not rerun for this publication.

## Offline suite

From the repository root:

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
```

The suite isolates `%LOCALAPPDATA%` in temporary directories and removes common key variables from each test environment. GUI checks use Qt's offscreen platform. Provider checks inject fake transports; they do not call a live model service.

Coverage includes processing invariants, exact preview/export agreement on synthetic arrays, grouped undo/redo, project migration and replay, source overwrite protection, connection recovery, credential selection, cancellation, response escaping, and preview scheduling.

Real Windows Credential Manager smoke testing is opt-in. Filesystem tests may skip operations that require unavailable Windows privileges. Test output identifies each skip; a skip is not a passed manual check.

## Real-RAW workflow

Use your own camera files and a new output directory. The parent output directory must already exist. These commands create local projects, exports, screenshots, and reports; those outputs may contain your photographs and source paths and are ignored by Git.

```powershell
New-Item -ItemType Directory -Force verification-runs | Out-Null
& .\.venv\Scripts\python.exe scripts\verify_workflow.py C:\photos\first.ARW --second C:\photos\second.ARW --output verification-runs\workflow-check
& .\.venv\Scripts\python.exe scripts\verify_coach_workflow.py C:\photos\first.ARW --output verification-runs\coach-check
```

The coach workflow uses a fake provider transport, including success, cancellation, and failure paths. It checks request construction without a paid request. Source hashes are checked to detect modification of the originals.

## Build check

```powershell
& .\.venv\Scripts\python.exe scripts\verify_wheel.py --output verification-runs\wheel-check
```

This builds a wheel without dependency downloads, extracts it outside the checkout, verifies packaged modules/assets, and opens an offscreen window using the extracted package.

## Preview benchmark

```powershell
& .\.venv\Scripts\python.exe scripts\benchmark_preview.py C:\photos\large.ARW --output verification-runs\preview-benchmark
```

This measures synchronous worker rendering and records output hashes. It excludes GUI input latency, painting, and physical display scaling. Comparisons require the same source file, parameters, dependencies, and hardware. A benchmark result on one machine does not establish general desktop responsiveness.

## Manual checks

Before treating a release as suitable for a particular workflow, inspect orientation, color, crop/mask geometry, zoom and pan, drag behavior, project reopening, and exported files on the target Windows display. Evaluate each live provider/model separately when cloud coaching is needed.

The repository's demonstration screenshot uses a public-domain JPEG preview fixture. It provides no real-RAW decoding, color-calibration, or live-model evidence.
