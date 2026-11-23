"""Validated edit state, reversible transactions, and append-only audit."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import copy
import math
from pathlib import Path
from typing import Any
from uuid import uuid4

from .imaging import RawOptions, BASE_RENDERINGS
from .composition import Composition
from .lens import LensCorrections
from .detail import DetailControls
from .color_mixer import ColorMixer
from .color_grading import ColorGrading
from .gradients import LinearGradient, validate_gradients, gradient_from_dict
from .spots import SpotRemoval, validate_spots
from .snapshots import EditSnapshot, MAX_SNAPSHOTS, validate_snapshot_name, validate_snapshots

EDIT_PIPELINE_VERSION = "phase1-sdr-srgb-v14"
IDENTITY_CURVE = ((0.0, 0.0), (1.0, 1.0))


def finite_number(value: Any, low: float, high: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number.")
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite and between {low:g} and {high:g}.")
    return value


def validate_curve(points) -> tuple[tuple[float, float], ...]:
    if not isinstance(points, (list, tuple)) or not 2 <= len(points) <= 16:
        raise ValueError("A curve requires 2–16 control points.")
    result = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Curve points must contain x and y.")
        result.append((finite_number(point[0], 0, 1, "Curve x"), finite_number(point[1], 0, 1, "Curve y")))
    if result[0][0] != 0 or result[-1][0] != 1 or any(a[0] >= b[0] for a, b in zip(result, result[1:])):
        raise ValueError("Curve x values must increase strictly, from 0 to 1.")
    return tuple(result)


@dataclass(frozen=True)
class EditValues:
    exposure_ev: float = 0.0
    raw_options: RawOptions = RawOptions()
    contrast: float = 0.0
    saturation: float = 1.0
    rgb_curve: tuple[tuple[float, float], ...] = IDENTITY_CURVE
    goal: str = ""
    highlights: float = 0.0
    shadows: float = 0.0
    composition: Composition = Composition()
    color_mixer: ColorMixer = ColorMixer()
    whites: float = 0.0
    blacks: float = 0.0
    vibrance: float = 0.0
    local_gradients: tuple[LinearGradient, ...] = ()
    lens_corrections: LensCorrections = LensCorrections()
    detail: DetailControls = DetailControls()
    base_rendering: str = "legacy_v1"
    color_grading: ColorGrading = ColorGrading()
    spots: tuple[SpotRemoval, ...] = ()

    def __post_init__(self):
        if self.base_rendering not in BASE_RENDERINGS:
            raise ValueError("Unknown base rendering version.")
        validate_gradients(self.local_gradients)
        validate_spots(self.spots)
        if not isinstance(self.lens_corrections, LensCorrections):
            raise ValueError("Invalid manual lens correction parameters.")
        if not isinstance(self.detail, DetailControls):
            raise ValueError("Invalid detail parameters.")
        if not isinstance(self.color_grading, ColorGrading):
            raise ValueError("Invalid color grading parameters.")
        object.__setattr__(self, "exposure_ev", finite_number(self.exposure_ev, -5, 5, "Exposure"))
        object.__setattr__(self, "contrast", finite_number(self.contrast, -1, 1, "Contrast"))
        object.__setattr__(self, "saturation", finite_number(self.saturation, 0, 2, "Saturation"))
        object.__setattr__(self, "highlights", finite_number(self.highlights, -1, 1, "Highlights"))
        object.__setattr__(self, "shadows", finite_number(self.shadows, -1, 1, "Shadows"))
        for name in ("whites", "blacks", "vibrance"):
            object.__setattr__(self, name, finite_number(getattr(self, name), -1, 1, name.title()))
        object.__setattr__(self, "rgb_curve", validate_curve(self.rgb_curve))
        if not isinstance(self.color_mixer, ColorMixer):
            raise ValueError("Invalid Color Mixer parameters.")
        if not isinstance(self.composition, Composition):
            raise ValueError("Invalid crop/straighten parameters.")
        if not isinstance(self.raw_options, RawOptions):
            raise ValueError("Invalid white balance options.")
        if not isinstance(self.goal, str) or len(self.goal) > 2000:
            raise ValueError("The editing goal must contain at most 2000 characters.")

    def fields(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def to_dict(self) -> dict:
        return {"base_rendering": self.base_rendering, "exposure_ev": self.exposure_ev, "white_balance": {
            "mode": self.raw_options.wb_mode, "relative_rgb_gain": list(self.raw_options.wb_gains)},
            "contrast": self.contrast, "saturation": self.saturation,
            "whites": self.whites, "blacks": self.blacks, "vibrance": self.vibrance,
            "highlights": self.highlights, "shadows": self.shadows,
            "composition": self.composition.to_dict(),
            "lens_corrections": self.lens_corrections.to_dict(),
            "color_mixer": self.color_mixer.to_dict(),
            "color_grading": self.color_grading.to_dict(),
            "local_gradients": [g.to_dict() for g in self.local_gradients],
            "spots": [s.to_dict() for s in self.spots],
            "rgb_curve": [list(point) for point in self.rgb_curve], "goal": self.goal,
            **({"detail": self.detail.to_dict()} if self.detail != DetailControls() else {})}

    @classmethod
    def from_dict(cls, value: dict) -> EditValues:
        base_keys = set(cls().to_dict())
        if not isinstance(value, dict) or not (base_keys <= set(value) <= (base_keys | {"detail"})):
            raise ValueError("Unknown or missing editing parameters.")
        wb = value["white_balance"]
        if not isinstance(wb, dict) or set(wb) != {"mode", "relative_rgb_gain"}:
            raise ValueError("Invalid white balance parameters.")
        gains = wb["relative_rgb_gain"]
        if not isinstance(gains, list) or len(gains) != 3:
            raise ValueError("White balance requires three relative RGB gains.")
        gains = tuple(finite_number(v, .25, 4, "WB gain") for v in gains)
        if not isinstance(value["local_gradients"],list):
            raise ValueError("Invalid gradient collection.")
        if not isinstance(value["spots"], list):
            raise ValueError("Invalid spot removal collection.")
        return cls(value["exposure_ev"], RawOptions(wb["mode"], gains), value["contrast"],
                   value["saturation"], validate_curve(value["rgb_curve"]), value["goal"],
                   value["highlights"], value["shadows"], Composition.from_dict(value["composition"]),
                   ColorMixer.from_dict(value["color_mixer"]), value["whites"], value["blacks"], value["vibrance"],
                   tuple(gradient_from_dict(g) for g in value["local_gradients"]),
                   LensCorrections.from_dict(value["lens_corrections"]),
                   DetailControls.from_dict(value["detail"]) if "detail" in value else DetailControls(),
                   value["base_rendering"],
                   ColorGrading.from_dict(value["color_grading"]),
                   tuple(SpotRemoval.from_dict(s) for s in value["spots"]))


@dataclass(frozen=True)
class EditState:
    photo_id: str
    source_path: Path
    session_id: str
    revision: int = 0
    exposure_ev: float = 0.0
    raw_options: RawOptions = RawOptions()
    compare_baseline: bool = False
    contrast: float = 0.0
    saturation: float = 1.0
    rgb_curve: tuple[tuple[float, float], ...] = IDENTITY_CURVE
    goal: str = ""
    highlights: float = 0.0
    shadows: float = 0.0
    composition: Composition = Composition()
    color_mixer: ColorMixer = ColorMixer()
    whites: float = 0.0
    blacks: float = 0.0
    vibrance: float = 0.0
    local_gradients: tuple[LinearGradient, ...] = ()
    lens_corrections: LensCorrections = LensCorrections()
    detail: DetailControls = DetailControls()
    base_rendering: str = "legacy_v1"
    color_grading: ColorGrading = ColorGrading()
    spots: tuple[SpotRemoval, ...] = ()
    compare_snapshot: EditSnapshot | None = None

    @property
    def compare_snapshot_id(self) -> str | None:
        return None if self.compare_snapshot is None else self.compare_snapshot.id

    @property
    def values(self) -> EditValues:
        return EditValues(self.exposure_ev, self.raw_options, self.contrast, self.saturation, self.rgb_curve,
                          self.goal, self.highlights, self.shadows, self.composition, self.color_mixer,
                          self.whites, self.blacks, self.vibrance, self.local_gradients, self.lens_corrections, self.detail, self.base_rendering, self.color_grading, self.spots)

    def with_values(self, values: EditValues) -> EditState:
        return replace(self, **values.fields())


@dataclass(frozen=True)
class HistoryEntry:
    id: str
    label: str
    before: EditValues
    after: EditValues

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "before": self.before.to_dict(), "after": self.after.to_dict()}


class StateStore:
    """The only application boundary allowed to apply edits or history actions."""
    def __init__(self) -> None:
        self.current: EditState | None = None
        self.events: list[dict[str, Any]] = []
        self.preview_revision: int | None = None
        self.undo_stack: list[HistoryEntry] = []
        self.redo_stack: list[HistoryEntry] = []
        self.snapshots: list[EditSnapshot] = []
        self.project_id = str(uuid4())
        self.created_utc = self._now()
        self._transaction: str | None = None
        self._transaction_before: EditValues | None = None
        self._transaction_parameter = "exposure_ev"
        self._event_start = 0
        self._saved_revision = 0
        self._saved_snapshots: tuple[EditSnapshot, ...] = ()
        self.load_warnings: list[str] = []

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @property
    def dirty(self) -> bool:
        return self.current is not None and (
            self.current.revision != self._saved_revision
            or tuple(self.snapshots) != self._saved_snapshots
        )

    @property
    def gesture_active(self) -> bool:
        return self._transaction is not None

    def mark_saved(self, revision: int, snapshots: tuple[EditSnapshot, ...] | None = None) -> None:
        self._saved_revision = revision
        self._saved_snapshots = tuple(self.snapshots) if snapshots is None else tuple(snapshots)

    def _record(self, kind, parameter, before, after, phase="end", *, previous_revision=None,
                edit_before=None, entry=None, undo_of=None, redo_of=None) -> dict:
        state = self.current
        assert state is not None
        event = {
            "event_id": str(uuid4()), "sequence": self.events[-1]["sequence"] + 1 if self.events else 1,
            "photo_id": state.photo_id, "session_id": state.session_id,
            "state_version": state.revision,
            "state_version_before": state.revision if previous_revision is None else previous_revision,
            "timestamp": self._now(), "type": kind, "parameter": parameter,
            "before": copy.deepcopy(before), "after": copy.deepcopy(after), "phase": phase,
            "transaction_id": self._transaction, "undo_of": undo_of, "redo_of": redo_of,
            "requested_preview_version": state.revision, "displayed_preview_version": self.preview_revision,
            "edit_before": (edit_before or state.values).to_dict(), "edit_after": state.values.to_dict(),
            "history_entry": None if entry is None else entry.to_dict(),
        }
        self.events.append(event)
        return event

    def activate(self, path: Path, photo_id: str, *, base_rendering: str = "legacy_v1") -> EditState:
        self.end_drag()
        previous = self.current.photo_id if self.current else None
        if base_rendering not in BASE_RENDERINGS:
            raise ValueError("Unknown base rendering version.")
        self.current = EditState(photo_id, Path(path), str(uuid4()), base_rendering=base_rendering)
        self.preview_revision = None
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.snapshots.clear()
        self._event_start = len(self.events)
        self._saved_revision = 0
        self._saved_snapshots = ()
        self.project_id = str(uuid4())
        self.created_utc = self._now()
        self._record("photo_switch", "photo_id", previous, photo_id)
        return self.current

    def begin_drag(self, parameter: str = "exposure_ev") -> None:
        if self.current is not None:
            self.end_drag()
            if parameter not in ("exposure_ev", "contrast", "saturation", "rgb_curve", "highlights", "shadows", "color_mixer", "color_grading", "whites", "blacks", "vibrance", "local_gradients", "spots", "lens_corrections", "detail", "raw_options"):
                raise ValueError("This parameter cannot be dragged.")
            self._transaction = str(uuid4())
            self._transaction_before = self.current.values
            self._transaction_parameter = parameter
            value = self._parameter_value(self.current.values, parameter)
            self._record("gesture", parameter, value, value, "begin")

    def _commit(self, before: EditValues, label: str) -> HistoryEntry | None:
        assert self.current is not None
        if before == self.current.values:
            return None
        entry = HistoryEntry(self._transaction or str(uuid4()), label, before, self.current.values)
        self.undo_stack.append(entry)
        self.redo_stack.clear()
        return entry

    def end_drag(self, *, cancel: bool = False) -> None:
        if self.current is None or self._transaction is None:
            return
        before = self._transaction_before
        assert before is not None
        if cancel and self.current.values != before:
            self._apply_values(before, "cancel", self._transaction_parameter)
        entry = None if cancel else self._commit(before, self._transaction_parameter)
        self._record("gesture", self._transaction_parameter,
                     self._parameter_value(before, self._transaction_parameter),
                     self._parameter_value(self.current.values, self._transaction_parameter),
                     "cancel" if cancel else "end", entry=entry)
        self._transaction = None
        self._transaction_before = None

    @staticmethod
    def _parameter_value(values: EditValues, parameter: str):
        if parameter == "detail":
            return values.detail.to_dict()
        key = "white_balance" if parameter == "raw_options" else parameter
        return values.to_dict().get(key)

    def _apply_values(self, values: EditValues, kind: str, parameter: str, *, history=True,
                      undo_of=None, redo_of=None, label: str | None = None) -> None:
        if self.current is None:
            raise RuntimeError("Open a photo first.")
        old = self.current
        if old.values == values:
            return
        self.current = replace(old.with_values(values), revision=old.revision + 1)
        entry = (
            self._commit(old.values, parameter if label is None else label)
            if history and self._transaction is None
            else None
        )
        all_values = parameter == "all"
        self._record(kind, parameter,
                     old.values.to_dict() if all_values else self._parameter_value(old.values, parameter),
                     values.to_dict() if all_values else self._parameter_value(values, parameter),
                     "update" if self._transaction else "end", previous_revision=old.revision,
                     edit_before=old.values, entry=entry, undo_of=undo_of, redo_of=redo_of)

    def set_parameter(self, parameter: str, value) -> None:
        if self.current is None:
            raise RuntimeError("Open a photo first.")
        if parameter not in EditValues.__dataclass_fields__:
            raise ValueError("Unknown editing parameter.")
        values = replace(self.current.values, **{parameter: value})
        if self._transaction and parameter != self._transaction_parameter:
            self.end_drag()
        self._apply_values(values, "parameter", parameter)

    def apply_changes(self, changes, primary_parameter: str | None = None) -> bool:
        """Apply one or more (parameter, value) updates atomically as a single Undo HistoryEntry."""
        if self.current is None:
            raise RuntimeError("Open a photo first.")
        self.end_drag()
        effective: list[tuple[str, object]] = []
        temp_values = self.current.values
        for param, val in changes:
            if param not in EditValues.__dataclass_fields__:
                raise ValueError(f"Unknown editing parameter: {param}")
            if getattr(temp_values, param) != val:
                effective.append((param, val))
                temp_values = replace(temp_values, **{param: val})
        if not effective:
            return False
        if len(effective) == 1:
            self.set_parameter(effective[0][0], effective[0][1])
            return True
        tx_param = primary_parameter if primary_parameter in EditValues.__dataclass_fields__ else effective[-1][0]
        self.begin_drag(tx_param)
        try:
            for param, val in effective:
                values = replace(self.current.values, **{param: val})
                self._apply_values(values, "parameter", param)
        finally:
            self.end_drag()
        return True

    def set_exposure(self, value):
        self.set_parameter("exposure_ev", value)

    def set_whitebalance(self, options):
        self.set_parameter("raw_options", options)

    def set_compare(self, enabled: bool) -> None:
        self.end_drag()
        if self.current is None:
            raise RuntimeError("Open a photo first.")
        enabled = bool(enabled)
        if enabled and self.current.compare_snapshot is not None:
            self.set_compare_snapshot(None)
        old = self.current
        if old.compare_baseline != enabled:
            self.current = replace(old, compare_baseline=enabled, revision=old.revision + 1)
            self._record("compare", "compare_baseline", old.compare_baseline, enabled, previous_revision=old.revision)

    def get_snapshot(self, snapshot_id: str) -> EditSnapshot:
        for snapshot in self.snapshots:
            if snapshot.id == snapshot_id:
                return snapshot
        raise ValueError("Snapshot not found.")

    def save_snapshot(self, name: str) -> EditSnapshot:
        if self.current is None:
            raise RuntimeError("Open a photo first.")
        self.end_drag()
        cleaned = validate_snapshot_name(name)
        if len(self.snapshots) >= MAX_SNAPSHOTS:
            raise ValueError(f"At most {MAX_SNAPSHOTS} snapshots are supported per project.")
        if any(existing.name.casefold() == cleaned.casefold() for existing in self.snapshots):
            raise ValueError(f"Snapshot '{cleaned}' already exists. Choose a new name or delete the old one.")
        snapshot = EditSnapshot(str(uuid4()), cleaned, self._now(), self.current.values)
        self.snapshots.append(snapshot)
        validate_snapshots(self.snapshots)
        self._record("snapshot_save", "snapshots", None, {"id": snapshot.id, "name": snapshot.name})
        return snapshot

    def delete_snapshot(self, snapshot_id: str) -> EditSnapshot:
        if self.current is None:
            raise RuntimeError("Open a photo first.")
        self.end_drag()
        index = next((idx for idx, item in enumerate(self.snapshots) if item.id == snapshot_id), None)
        if index is None:
            raise ValueError("Snapshot not found.")
        if self.current.compare_snapshot_id == snapshot_id:
            self.set_compare_snapshot(None)
        removed = self.snapshots.pop(index)
        self._record("snapshot_delete", "snapshots", {"id": removed.id, "name": removed.name}, None)
        return removed

    def restore_snapshot(self, snapshot_id: str) -> bool:
        if self.current is None:
            raise RuntimeError("Open a photo first.")
        self.end_drag()
        snapshot = self.get_snapshot(snapshot_id)
        if self.current.compare_baseline:
            self.set_compare(False)
        if self.current.compare_snapshot is not None:
            self.set_compare_snapshot(None)
        if self.current.values == snapshot.values:
            return False
        self._apply_values(
            snapshot.values,
            "snapshot_restore",
            "all",
            label=f"Restore snapshot: {snapshot.name}",
        )
        return True

    def set_compare_snapshot(self, snapshot_id: str | None) -> None:
        self.end_drag()
        if self.current is None:
            raise RuntimeError("Open a photo first.")
        target = None if snapshot_id is None else self.get_snapshot(snapshot_id)
        if target is not None and self.current.compare_baseline:
            self.set_compare(False)
        old = self.current
        target_id = None if target is None else target.id
        if old.compare_snapshot_id != target_id:
            self.current = replace(old, compare_snapshot=target, revision=old.revision + 1)
            self._record(
                "compare_snapshot",
                "compare_snapshot_id",
                old.compare_snapshot_id,
                target_id,
                previous_revision=old.revision,
            )

    def reset(self) -> None:
        self.end_drag()
        if self.current:
            self._apply_values(EditValues(goal=self.current.goal, base_rendering=self.current.base_rendering), "reset", "all")

    def undo(self) -> None:
        self.end_drag()
        if self.undo_stack:
            entry = self.undo_stack.pop()
            self._apply_values(entry.before, "undo", "all", history=False, undo_of=entry.id)
            self.redo_stack.append(entry)

    def redo(self) -> None:
        self.end_drag()
        if self.redo_stack:
            entry = self.redo_stack.pop()
            self._apply_values(entry.after, "redo", "all", history=False, redo_of=entry.id)
            self.undo_stack.append(entry)

    def accepts(self, snapshot: EditState) -> bool:
        return self.current is not None and (snapshot.session_id, snapshot.photo_id, snapshot.revision) == (
            self.current.session_id, self.current.photo_id, self.current.revision)

    def mark_preview(self, snapshot: EditState) -> bool:
        if not self.accepts(snapshot):
            return False
        self.preview_revision = snapshot.revision
        self._record("preview_ready", None, None, snapshot.revision)
        return True

    def project_events(self) -> list[dict]:
        return copy.deepcopy(self.events[self._event_start:])

    def before_last_edit(self) -> EditState:
        assert self.current is not None
        for event in reversed(self.events[self._event_start:]):
            entry = event.get("history_entry")
            if entry or event["type"] in ("undo", "redo"):
                values = EditValues.from_dict(entry["before"] if entry else event["edit_before"])
                version = event["state_version_before"]
                if event["type"] == "gesture":
                    start = next(e for e in self.events[self._event_start:]
                                 if e["transaction_id"] == event["transaction_id"] and e["phase"] == "begin")
                    version = start["state_version"]
                return replace(self.current.with_values(values), revision=version, compare_baseline=False, compare_snapshot=None)
        return replace(self.current, compare_baseline=False, compare_snapshot=None)
