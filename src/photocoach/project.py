"""Versioned local projects with audit replay and source-content verification."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import re
from uuid import uuid4

from . import __version__
from .imaging import dependency_versions
from .safe_io import atomic_write, sha256_file
from .state import EDIT_PIPELINE_VERSION, EditState, EditValues, HistoryEntry, StateStore
from .gradients import LinearGradient

MAX_PROJECT_BYTES = 64 * 1024 * 1024


class SourceMissing(FileNotFoundError):
    """The UI can offer to locate the same source by its stored hash."""


def snapshot_project(store: StateStore) -> dict:
    """Called on the UI thread after finishing the current gesture; no I/O."""
    store.end_drag()
    state = store.current
    if state is None:
        raise ValueError("Open a photo before saving a project.")
    return {"format": "photocoach-project", "schema_version": 7,
            "pipeline_version": EDIT_PIPELINE_VERSION, "project_id": store.project_id,
            "created_utc": store.created_utc, "saved_utc": store._now(),
            "source": {"path": str(state.source_path), "sha256": state.photo_id},
            "state": {"revision": state.revision, "parameters": state.values.to_dict(),
                      "compare_baseline": state.compare_baseline},
            "history": {"events": store.project_events(),
                        "undo": [entry.to_dict() for entry in store.undo_stack],
                        "redo": [entry.to_dict() for entry in store.redo_stack]},
            "provenance": {"application_version": __version__, "dependencies": dependency_versions()},
            "coach": {"mode": "mock", "automatic": False}}


def _require(condition: bool, message="Project state and operation history disagree.") -> None:
    if not condition:
        raise ValueError(message)


def upgrade_project(data: dict) -> dict:
    """Normalize only known v1–v6 contracts, without writing the source project.

    Old history is enriched with neutral defaults for newer controls. The rest is still validated
    by full audit replay. New-schema files never get missing parameters filled in.
    """
    _require(isinstance(data, dict), "Invalid project document.")
    version = data.get("schema_version")
    if version not in (1, 2, 3, 4, 5, 6) or type(version) is not int:
        return data
    _require(data.get("format") == "photocoach-project" and
             data.get("pipeline_version") == f"phase1-sdr-srgb-v{version}", "Unsupported legacy rendering pipeline.")
    data = copy.deepcopy(data)
    additions = {"local_gradients": []} if version<=5 else {}
    if version <= 4:
        additions.update(whites=0.0,blacks=0.0,vibrance=0.0)
    if version <= 3:
        additions["color_mixer"] = EditValues().color_mixer.to_dict()
    if version <= 2:
        additions["composition"] = EditValues().composition.to_dict()
    if version == 1:
        additions.update(highlights=0.0, shadows=0.0)
    legacy_keys = set(EditValues().to_dict()) - set(additions)
    def parameters(value):
        _require(isinstance(value, dict) and set(value) == legacy_keys, "Invalid legacy editing parameters.")
        if version==6:
            _require(isinstance(value['local_gradients'],list),'Invalid legacy gradient collection.')
            for gradient in value['local_gradients']:
                LinearGradient.from_dict(gradient)
        return dict(value, **copy.deepcopy(additions))
    def entry(item):
        item["before"] = parameters(item["before"])
        item["after"] = parameters(item["after"])
    try:
        data["state"]["parameters"] = parameters(data["state"]["parameters"])
        for event in data["history"]["events"]:
            event["edit_before"] = parameters(event["edit_before"])
            event["edit_after"] = parameters(event["edit_after"])
            if event["parameter"] == "all":
                event["before"], event["after"] = parameters(event["before"]), parameters(event["after"])
            if event["history_entry"] is not None:
                entry(event["history_entry"])
        for item in data["history"]["undo"] + data["history"]["redo"]:
            entry(item)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("Malformed legacy project.") from error
    data["schema_version"] = 7
    data["pipeline_version"] = EDIT_PIPELINE_VERSION
    return data


def validate_project(data: dict) -> tuple[list[HistoryEntry], list[HistoryEntry]]:
    """Reconstruct undo/redo from audit events instead of trusting stored stacks."""
    data = upgrade_project(data)
    _require(data.get("format") == "photocoach-project" and type(data.get("schema_version")) is int
             and data["schema_version"] == 7, "Unsupported project format or schema version. Newer projects require a newer PhotoCoach.")
    _require(data.get("pipeline_version") == EDIT_PIPELINE_VERSION, "This project uses an unsupported rendering pipeline.")
    try:
        photo_id = data["source"]["sha256"]
        _require(isinstance(photo_id, str) and re.fullmatch("[0-9a-f]{64}", photo_id) is not None, "Invalid source SHA-256.")
        _require(isinstance(data["source"]["path"], str) and bool(data["source"]["path"]), "Invalid source reference.")
        _require(isinstance(data["project_id"], str) and bool(data["project_id"]), "Invalid project identity.")
        _require(isinstance(data["created_utc"], str), "Invalid project creation time.")
        target = EditValues.from_dict(data["state"]["parameters"])
        events = data["history"]["events"]
        _require(isinstance(events, list) and 1 <= len(events) <= 200000, "Invalid operation history length.")
        values, revision, comparison = EditValues(), 0, False
        undo: list[HistoryEntry] = []
        redo: list[HistoryEntry] = []
        gesture = None
        gesture_before = None
        ids, entry_ids = set(), set()
        previous_sequence = events[0]["sequence"] - 1
        for index, event in enumerate(events):
            _require(type(event["sequence"]) is int and event["sequence"] == previous_sequence + 1)
            previous_sequence = event["sequence"]
            _require(isinstance(event["event_id"], str) and event["event_id"] not in ids)
            ids.add(event["event_id"])
            _require(event["photo_id"] == photo_id and isinstance(event["timestamp"], str))
            _require(type(event["state_version"]) is int and type(event["state_version_before"]) is int)
            _require(event["state_version_before"] == revision)
            before = EditValues.from_dict(event["edit_before"])
            after = EditValues.from_dict(event["edit_after"])
            _require(before == values)
            kind = event["type"]
            entry_before = None
            if index == 0:
                _require(kind == "photo_switch" and event["after"] == photo_id and after == EditValues())
            elif kind in ("parameter", "reset", "cancel", "undo", "redo"):
                _require(after != before)
                revision += 1
                if kind in ("parameter", "cancel"):
                    parameter = event["parameter"]
                    _require(parameter in EditValues.__dataclass_fields__)
                    _require(event["before"] == StateStore._parameter_value(before, parameter))
                    _require(event["after"] == StateStore._parameter_value(after, parameter))
                    _require(replace(before, **{parameter: getattr(after, parameter)}) == after)
                else:
                    _require(event["parameter"] == "all" and event["before"] == before.to_dict() and event["after"] == after.to_dict())
                if kind == "reset":
                    _require(after == EditValues(goal=before.goal) and gesture is None)
                    entry_before = before
                elif kind == "parameter":
                    if gesture is None:
                        _require(event["transaction_id"] is None and event["phase"] == "end")
                        entry_before = before
                    else:
                        _require(event["transaction_id"] == gesture and event["phase"] == "update")
                elif kind == "cancel":
                    _require(gesture is not None and after == gesture_before)
                elif kind == "undo":
                    _require(gesture is None and bool(undo))
                    entry = undo.pop()
                    _require(event["undo_of"] == entry.id and before == entry.after and after == entry.before)
                    redo.append(entry)
                elif kind == "redo":
                    _require(gesture is None and bool(redo))
                    entry = redo.pop()
                    _require(event["redo_of"] == entry.id and before == entry.before and after == entry.after)
                    undo.append(entry)
            elif kind == "compare":
                _require(gesture is None and before == after and type(event["after"]) is bool)
                _require(event["before"] == comparison and event["after"] != comparison)
                comparison = event["after"]
                revision += 1
            elif kind == "gesture":
                _require(before == after)
                if event["phase"] == "begin":
                    _require(gesture is None and isinstance(event["transaction_id"], str))
                    gesture, gesture_before = event["transaction_id"], before
                else:
                    _require(gesture is not None and event["transaction_id"] == gesture)
                    _require(event["phase"] in ("end", "cancel"))
                    if event["phase"] == "cancel":
                        _require(after == gesture_before)
                    elif after != gesture_before:
                        entry_before = gesture_before
                    gesture, gesture_before = None, None
            elif kind in ("preview_ready", "project_open"):
                _require(before == after)
                if kind == "preview_ready":
                    _require(event["after"] == revision)
            else:
                raise ValueError("Unknown operation in project history.")
            _require(event["state_version"] == revision and event["requested_preview_version"] == revision)
            shown = event["displayed_preview_version"]
            _require(shown is None or (type(shown) is int and 0 <= shown <= revision))
            if entry_before is not None:
                entry_data = event["history_entry"]
                _require(isinstance(entry_data, dict) and isinstance(entry_data["id"], str) and entry_data["id"] not in entry_ids)
                entry_ids.add(entry_data["id"])
                entry = HistoryEntry(entry_data["id"], entry_data["label"],
                                     EditValues.from_dict(entry_data["before"]), EditValues.from_dict(entry_data["after"]))
                _require(entry.before == entry_before and entry.after == after)
                undo.append(entry)
                redo.clear()
            else:
                _require(event["history_entry"] is None)
            values = after
        _require(gesture is None, "Project contains an unfinished editing gesture.")
        _require(type(data["state"]["revision"]) is int and data["state"]["revision"] == revision and target == values)
        _require(type(data["state"]["compare_baseline"]) is bool and data["state"]["compare_baseline"] == comparison)
        _require(data["history"]["undo"] == [entry.to_dict() for entry in undo])
        _require(data["history"]["redo"] == [entry.to_dict() for entry in redo])
        return undo, redo
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("Malformed or inconsistent project data.") from error


def write_project(path: Path, data: dict, *, overwrite: bool = False) -> Path:
    """Run in a worker. Overwrite is permitted only after explicit UI consent."""
    if path.suffix.lower() not in (".photocoach", ".json"):
        raise ValueError("Projects use .photocoach or .json files.")
    data = upgrade_project(copy.deepcopy(data))
    source = Path(data["source"]["path"]).resolve(strict=True)
    _require(sha256_file(source) == data["source"]["sha256"], "The original photo has changed; project save stopped.")
    data["source"]["byte_length"] = source.stat().st_size
    validate_project(data)
    content = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    if len(content.encode("utf-8")) > MAX_PROJECT_BYTES:
        raise ValueError("Project history exceeds the 64 MiB format limit.")
    return atomic_write(path, lambda temporary: temporary.write_text(content, encoding="utf-8"),
                        forbidden_paths=(source,), overwrite=overwrite)


def read_project(path: Path, *, source_override: Path | None = None) -> StateStore:
    """Read, replay, and authenticate a project before it can replace UI state."""
    if path.stat().st_size > MAX_PROJECT_BYTES:
        raise ValueError("Project exceeds the 64 MiB format limit.")
    def invalid_constant(value):
        raise ValueError(f"Invalid JSON number: {value}")
    data = json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid_constant)
    legacy = isinstance(data, dict) and data.get("schema_version") in (1, 2, 3, 4, 5, 6)
    data = upgrade_project(data)
    undo, redo = validate_project(data)
    source = source_override or Path(data["source"]["path"])
    if not source.is_absolute():
        source = path.parent / source
    if not source.is_file():
        raise SourceMissing("The project's ARW could not be found. Locate the same original photo.")
    source = source.resolve(strict=True)
    _require(source.suffix.lower() == ".arw", "The project must reference a Sony ARW.")
    _require(sha256_file(source) == data["source"]["sha256"], "The selected ARW does not match this project's source hash.")
    _require(data["source"].get("byte_length") == source.stat().st_size, "The source file size does not match the project.")
    store = StateStore()
    if legacy:
        store.load_warnings.append("Legacy project restored; existing edits are preserved. Saving uses format v7, which older PhotoCoach versions cannot open.")
    recorded = data.get("provenance", {}).get("dependencies", {})
    current_versions = dependency_versions()
    changed = [name for name in ("rawpy", "libraw", "numpy") if recorded.get(name) != current_versions.get(name)]
    if changed:
        store.load_warnings.append("Rendering dependencies differ: " + ", ".join(changed) +
                                   ". Parameters were restored; pixel-identical rendering across versions is not guaranteed.")
    store.project_id = data["project_id"]
    store.created_utc = data["created_utc"]
    store.current = EditState(data["source"]["sha256"], source, str(uuid4()),
                              revision=data["state"]["revision"], compare_baseline=data["state"]["compare_baseline"])
    store.current = store.current.with_values(EditValues.from_dict(data["state"]["parameters"]))
    store.events = copy.deepcopy(data["history"]["events"])
    store.undo_stack, store.redo_stack = undo, redo
    store._saved_revision = store.current.revision
    store._record("project_open", None, None, None)
    return store
