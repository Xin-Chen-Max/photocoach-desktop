"""Immutable named edit snapshots with strict validation and project serialization."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import EditValues

MAX_SNAPSHOTS = 24
MAX_SNAPSHOT_NAME_LENGTH = 80


def _has_control_chars(text: str) -> bool:
    return any(ord(ch) < 32 or ord(ch) == 127 for ch in text)


def validate_snapshot_name(name: str) -> str:
    """Validate and trim a human-readable snapshot name (1..80 characters)."""
    if not isinstance(name, str):
        raise ValueError("Snapshot name must be text.")
    cleaned = name.strip()
    if not cleaned or len(cleaned) > MAX_SNAPSHOT_NAME_LENGTH:
        raise ValueError(f"Snapshot name must be 1–{MAX_SNAPSHOT_NAME_LENGTH} characters.")
    if _has_control_chars(cleaned):
        raise ValueError("Snapshot name cannot contain control characters.")
    return cleaned


def _validate_snapshot_id(snapshot_id: str) -> str:
    if not isinstance(snapshot_id, str):
        raise ValueError("Snapshot ID must be text.")
    cleaned = snapshot_id.strip()
    if not cleaned or cleaned != snapshot_id or len(cleaned) > 128 or _has_control_chars(cleaned):
        raise ValueError("Invalid snapshot ID.")
    return cleaned


def _validate_utc(timestamp: str) -> str:
    if not isinstance(timestamp, str) or not timestamp or len(timestamp) > 64 or _has_control_chars(timestamp):
        raise ValueError("Invalid snapshot timestamp.")
    try:
        datetime.fromisoformat(timestamp)
    except ValueError as error:
        raise ValueError("Invalid snapshot timestamp.") from error
    return timestamp


@dataclass(frozen=True)
class EditSnapshot:
    """Immutable named checkpoint of complete EditValues for a single photo."""

    id: str
    name: str
    created_utc: str
    values: EditValues

    def __post_init__(self) -> None:
        from .state import EditValues

        object.__setattr__(self, "id", _validate_snapshot_id(self.id))
        object.__setattr__(self, "name", validate_snapshot_name(self.name))
        object.__setattr__(self, "created_utc", _validate_utc(self.created_utc))
        if not isinstance(self.values, EditValues):
            raise ValueError("Snapshot values must be an EditValues instance.")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "created_utc": self.created_utc,
            "values": self.values.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> EditSnapshot:
        from .state import EditValues

        if type(data) is not dict or set(data) != {"id", "name", "created_utc", "values"}:
            raise ValueError("Invalid snapshot record.")
        return cls(
            data["id"],
            data["name"],
            data["created_utc"],
            EditValues.from_dict(data["values"]),
        )


def validate_snapshots(snapshots) -> tuple[EditSnapshot, ...]:
    """Validate snapshot collection bounds, unique IDs, and case-insensitive unique names."""
    if not isinstance(snapshots, (list, tuple)):
        raise ValueError("Invalid snapshot collection.")
    if len(snapshots) > MAX_SNAPSHOTS:
        raise ValueError(f"At most {MAX_SNAPSHOTS} snapshots are supported per project.")
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    result: list[EditSnapshot] = []
    for item in snapshots:
        if not isinstance(item, EditSnapshot):
            raise ValueError("Snapshot collection contains an invalid item.")
        if item.id in seen_ids:
            raise ValueError("Duplicate snapshot ID.")
        name_key = item.name.casefold()
        if name_key in seen_names:
            raise ValueError(f"Snapshot '{item.name}' already exists.")
        seen_ids.add(item.id)
        seen_names.add(name_key)
        result.append(item)
    return tuple(result)
