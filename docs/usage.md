# Usage

## Edit a photo

Open a Sony Bayer `.ARW` file. Use the Edit panel for basic tone, the RGB curve, Color Mixer, crop/straighten, and local adjustments. Edits change parameters and previews; the source RAW remains untouched.

During a drag, **Adjusting** displays a reduced sample. **Refining** keeps the image visible while a formal render runs. Wait for **Full detail** before judging fine detail. Use Fit for composition and 100% for pixel inspection. Space + drag pans the image.

The histogram samples the whole edited photo. Red and blue clipping overlays describe the displayed SDR limits, not sensor clipping or RAW highlight recoverability. Small regions can be missed in reduced previews. Overlays are excluded from exports and coach images.

## Local adjustments

Expand **Local adjustments** and add a linear or radial mask. Up to four masks can affect exposure, highlights, and shadows. Drag the on-image handles to change the geometry, and use **Done** to hide them. Each adjustment is undoable. Mask selection and visibility do not alter the image.

## Save and export

Save a `.photocoach` project to preserve the source reference, editing values, and operation history. Project files contain the source path and hash; they do not embed the RAW. Treat a project file as local working data when it contains a private path.

Export JPEG for an 8-bit image or TIFF for 16-bit RGB. Both use sRGB encoding and embed an sRGB ICC profile. Source EXIF/GPS and editing overlays are omitted. Export renders a captured current state at full resolution instead of reusing a temporary preview.

Monitor ICC conversion and HDR display are not implemented. Color-critical work needs separate evaluation on the intended display and output workflow.

## Configure the coach

Select **Set up AI coach**, then choose a protocol, endpoint, model, and key source. Supported protocols are OpenAI Responses, Chat Completions compatible APIs, and native Gemini. A compatible HTTP protocol alone does not prove that a model supports image input or the required response schema.

Model refresh loads metadata and sends the selected key to the configured destination; it does not upload a photo or generate a coaching answer. Unknown model IDs can be entered manually. No model is selected, tested, or retried automatically.

Write a goal and a specific question, such as "How can I lift these midtones without losing contrast?" Select **Ask coach**, inspect the review, then send. The coach returns an answer, suggested steps, a stopping condition, and supporting explanation. Suggestions do not modify editing values.

Control-navigation links recognize the English control names used in the interface. Provider text is rendered as escaped text. A response remains attached to its original question and editing revision when later changes make it a reference.

## Data sent to a provider

A manual analysis sends reduced current/before images, selected editing parameters and history context, and the goal/question you enter. The review shows the content and destination. You may omit the before image.

Requests exclude the original RAW, original filesystem path, source EXIF/GPS, and display overlays. Text entered in the goal or question is transmitted as part of the request. Provider retention and billing rules apply to that provider.

Startup restores nonsecret configuration without contacting a provider. There are no automatic cloud analyses. Developer mode (`--dev` or `PHOTOCOACH_DEV=1`) exposes a local mock and optional automatic local mock feedback.

## Keys and limits

Connection settings are stored under `%LOCALAPPDATA%\PhotoCoach`. Remembering a connection saves nonsecret metadata. **Remember API key on this PC** separately stores a credential in Windows Credential Manager for the current Windows account. Keys are not written to settings, model caches, or projects.

A key can instead remain in memory for the session or come from a named environment variable. There is no silent fallback between key sources. Startup does not read the credential vault. Credential access is not a defense against another malicious process running as the same user.

Session attempt limits count failed and cancelled requests too. Optional USD limits depend on the rates and reserve values you configure; they cannot guarantee the provider's final bill. Restarting the application resets the local session ledger, not provider charges or quotas. A ChatGPT subscription does not pay API charges.

Cancelling or editing invalidates pending feedback locally. It cannot undo an upload or charge that has already reached the provider. There are no automatic retries.
