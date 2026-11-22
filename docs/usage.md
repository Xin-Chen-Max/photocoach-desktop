# Usage and privacy

## Editing a photo

Open a Sony ARW or Nikon NEF. New imports use Natural / As Shot base rendering and neutral creative sliders. Natural includes a fixed baseline tone transform; Neutral omits that shaping. Old projects use Legacy to preserve their earlier appearance. A base-mode change is undoable and does not re-decode RAW.

White balance starts from camera metadata when valid. Warmth, Tint and relative RGB gains are drafts until **Apply white balance** is clicked. Applying white balance can decode RAW again. The values are relative sensor gains, not calibrated Kelvin. Use the neutral picker on an evenly lit matte neutral surface; unsuitable dark, clipped or uneven samples are rejected.

Use Light, Color, Curve, Crop and straighten, Lens corrections, Detail and Local adjustments. Slider drags group into one undo step. Crop and masks follow source coordinates. Lens corrections are manual; no lens profile is inferred. A mask can adjust exposure, highlights, shadows and warmth. Up to four masks are supported.

Baseline retains the current base rendering and geometry for comparison, with creative adjustments neutral. It is not the embedded camera JPEG. Reset edits keeps the selected base rendering. Save preserves the source reference, edits and undo/redo history; it does not embed RAW pixels. Opening an older project migrates it in memory without rewriting that file.

## Preview quality

**Adjusting** identifies a temporary visible-region update. **Refining** keeps the last image visible while formal processing runs. **Full detail** marks the settled rendering path.

At 100%, source pixels map to physical display pixels. Inspect sharpening and NR there. Native filtering occurs before display reduction, including during detail gestures; Fit can update more slowly. The first native render may require one full-resolution decode. Camera/WB changes may require another. Source, state and viewport checks reject obsolete frames.

## Export

Export writes the current edit as JPEG or 16-bit TIFF with an sRGB ICC profile. It omits source EXIF/GPS. Export requires a new output filename and protects source RAWs, hard links, symlinks and alternate data streams. Keep a project file separately if you need editable history.

## Manual coaching

Expand the bottom Coach pane and open **Choose coach**. Select a saved connection/model, configure **Cloud connections**, or open **Local models**. Opening these controls does not send a photograph.

Type a question or choose a common question. Enter adds a line; Ctrl+Enter or **Analyze photo** starts the manual flow. **More / Edit goal** reveals an optional goal. **Usage and diagnostics** contains budgets and troubleshooting.

The cloud review shows the destination, model, matching preview images and editing context before sending. The original RAW, source path and source EXIF/GPS are not attached. Text entered into the goal or question is part of the displayed request. Provider retention rules and possible charges still apply. Session-only direct sending requires an explicit choice and is reset by connection changes or restart.

Replies separate the answer, suggestion and checks, with evidence, reasoning, the multi-step tool execution trace timeline, and 1:1 crop inspection thumbnails expandable. When a trial candidate has been rendered through the visual reflexion loop, **Preview trial** opens a non-modal side-by-side comparison before you commit. Validated target values appear beside **Apply suggestion**. Long lists are available through **Reply options / View changes**. Applying submits the complete action list in one undo step and cannot repeat the same action. Advice remains reference-only after the photo, edit, question or coach changes. Cancel discards late replies and halts active agent turns, though it cannot retract a request already received by a provider.

## Local vision models

Local models use an installed Ollama executable and a separately selected PhotoCoach model directory. Downloads begin only after a user click. The recommended large model requires substantial disk/RAM/VRAM; the UI shows its size and hardware guidance. Engine/model installation and model weights are not part of the Win64 ZIP.

PhotoCoach starts its own Ollama instance as needed at `127.0.0.1:11435`, with a separate model directory and cloud access disabled for that instance. It does not manage unrelated Ollama processes. Readiness requires vision capability and a generated-image structured-output probe tied to model digest and engine version. This probe does not certify editing taste. Analysis remains manual.

Local inference does not use an API key or the cloud dollar budget. It has a separate session request limit. Closing the model window cancels its task; installed model data remains available for the next session.

## Settings and credentials

Nonsecret connection settings live under `%LOCALAPPDATA%/PhotoCoach`, outside the repository and photo projects. A cloud key can be session-only, explicitly saved to Windows Credential Manager, or resolved from a named environment variable. There is no fallback between key sources. JSON settings, model metadata and projects do not contain key values. Startup restores configuration without reading the credential vault or contacting providers.

Deleting/changing a connection does not erase photo projects or local model downloads. Session budgets are application guards, not a provider billing guarantee. Restarting resets local counters, not provider charges or quota.

Developer mock coaching is available only with `--dev` or `PHOTOCOACH_DEV=1`. It is a rules demonstration and cannot interpret arbitrary questions.
