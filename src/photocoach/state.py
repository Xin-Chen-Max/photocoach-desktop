"""Validated edit state, reversible transactions, and append-only audit."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import copy
import math
from pathlib import Path
from typing import Any
from uuid import uuid4

from .imaging import RawOptions
from .composition import Composition
from .color_mixer import ColorMixer
from .gradients import LinearGradient, validate_gradients, gradient_from_dict

EDIT_PIPELINE_VERSION = "phase1-sdr-srgb-v7"
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

    def __post_init__(self):
        validate_gradients(self.local_gradients)
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
        return {"exposure_ev": self.exposure_ev, "white_balance": {
            "mode": self.raw_options.wb_mode, "relative_rgb_gain": list(self.raw_options.wb_gains)},
            "contrast": self.contrast, "saturation": self.saturation,
            "whites": self.whites, "blacks": self.blacks, "vibrance": self.vibrance,
            "highlights": self.highlights, "shadows": self.shadows,
            "composition": self.composition.to_dict(),
            "color_mixer": self.color_mixer.to_dict(),
            "local_gradients": [g.to_dict() for g in self.local_gradients],
            "rgb_curve": [list(point) for point in self.rgb_curve], "goal": self.goal}

    @classmethod
    def from_dict(cls, value: dict) -> EditValues:
        if not isinstance(value, dict) or set(value) != set(cls().to_dict()):
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
        return cls(value["exposure_ev"], RawOptions(wb["mode"], gains), value["contrast"],
                   value["saturation"], validate_curve(value["rgb_curve"]), value["goal"],
                   value["highlights"], value["shadows"], Composition.from_dict(value["composition"]),
                   ColorMixer.from_dict(value["color_mixer"]), value["whites"], value["blacks"], value["vibrance"],
                   tuple(gradient_from_dict(g) for g in value["local_gradients"]))


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

    @property
    def values(self) -> EditValues:
        return EditValues(self.exposure_ev, self.raw_options, self.contrast, self.saturation, self.rgb_curve,
                          self.goal, self.highlights, self.shadows, self.composition, self.color_mixer,
                          self.whites, self.blacks, self.vibrance, self.local_gradients)

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
        self.project_id = str(uuid4())
        self.created_utc = self._now()
        self._transaction: str | None = None
        self._transaction_before: EditValues | None = None
        self._transaction_parameter = "exposure_ev"
        self._event_start = 0
        self._saved_revision = 0
        self.load_warnings: list[str] = []

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @property
    def dirty(self) -> bool:
        return self.current is not None and self.current.revision != self._saved_revision

    @property
    def gesture_active(self) -> bool:
        return self._transaction is not None

    def mark_saved(self, revision: int) -> None:
        self._saved_revision = revision

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

    def activate(self, path: Path, photo_id: str) -> EditState:
        self.end_drag()
        previous = self.current.photo_id if self.current else None
        self.current = EditState(photo_id, Path(path), str(uuid4()))
        self.preview_revision = None
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._event_start = len(self.events)
        self._saved_revision = 0
        self.project_id = str(uuid4())
        self.created_utc = self._now()
        self._record("photo_switch", "photo_id", previous, photo_id)
        return self.current

    def begin_drag(self, parameter: str = "exposure_ev") -> None:
        if self.current is not None:
            self.end_drag()
            if parameter not in ("exposure_ev", "contrast", "saturation", "rgb_curve", "highlights", "shadows", "color_mixer", "whites", "blacks", "vibrance", "local_gradients"):
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
        key = "white_balance" if parameter == "raw_options" else parameter
        return values.to_dict().get(key)

    def _apply_values(self, values: EditValues, kind: str, parameter: str, *, history=True,
                      undo_of=None, redo_of=None) -> None:
        if self.current is None:
            raise RuntimeError("Open a photo first.")
        old = self.current
        if old.values == values:
            return
        self.current = replace(old.with_values(values), revision=old.revision + 1)
        entry = self._commit(old.values, parameter) if history and self._transaction is None else None
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

    def set_exposure(self, value):
        self.set_parameter("exposure_ev", value)

    def set_whitebalance(self, options):
        self.set_parameter("raw_options", options)

    def set_compare(self, enabled: bool) -> None:
        self.end_drag()
        if self.current is None:
            raise RuntimeError("Open a photo first.")
        enabled = bool(enabled)
        old = self.current
        if old.compare_baseline != enabled:
            self.current = replace(old, compare_baseline=enabled, revision=old.revision + 1)
            self._record("compare", "compare_baseline", old.compare_baseline, enabled, previous_revision=old.revision)

    def reset(self) -> None:
        self.end_drag()
        if self.current:
            self._apply_values(EditValues(goal=self.current.goal), "reset", "all")

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
                return replace(self.current.with_values(values), revision=version, compare_baseline=False)
        return replace(self.current, compare_baseline=False)
