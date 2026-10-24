"""Multiple saved AI connections (settings schema 2) with safe migration from schema 1.

The file holds no key or credential target. Each connection resolves only its own
key source (see ``settings.resolve_key_source``): a Gemini key is never offered to
another provider. Upgrading never loses the old settings: the first write over a
valid schema-1 file first stores ``settings.json.bak`` atomically, a valid backup is
never overwritten, and an unreadable settings file is recovered in memory from a valid backup.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
import re
import tempfile
import uuid
from urllib.parse import urlsplit

from .api_coach import ApiConfig
from .settings import (MAX_SETTINGS_BYTES, ApiSettings, SettingsError, SettingsStore, _BUDGET_FIELDS,
                       _CONFIG_FIELDS, _config_values, _reject_duplicate_fields, _reject_json_constant,
                       _require, _validate_budget, _validate_config, _validate_source, credential_target,
                       parse_schema1, resolve_key_source)

SCHEMA_VERSION = 2
MAX_CONNECTIONS = 16
MAX_FAVORITES = 50
MAX_RECENT = 8
PROVIDERS = ("gemini", "openai", "custom")
_ID = re.compile(r"[A-Za-z0-9_-]{1,32}")
_CONNECTION_FIELDS = frozenset(("id", "provider", "label", "config", "key_source", "key_environment",
                                "favorites", "recent"))
_DOCUMENT_FIELDS = frozenset(("schema_version", "connections", "active", "budget"))


def infer_provider(config: ApiConfig) -> str:
    if config.protocol == "gemini":
        return "gemini"
    host = (urlsplit(config.endpoint).hostname or "").lower()
    return "openai" if host == "api.openai.com" else "custom"


def _model_ids(values, limit):
    _require(type(values) is tuple and len(values) <= limit)
    for value in values:
        _require(type(value) is str and 0 < len(value) <= 200 and not any(c.isspace() or ord(c) < 32 for c in value))
    _require(len(set(values)) == len(values))
    return values


@dataclass(frozen=True)
class Connection:
    id: str
    provider: str
    label: str
    config: ApiConfig
    key_source: str = "saved"
    key_environment: str = ""
    favorites: tuple = ()
    recent: tuple = ()

    def __post_init__(self):
        _require(type(self.id) is str and _ID.fullmatch(self.id) is not None)
        _require(self.provider in PROVIDERS)
        _require(type(self.label) is str and 0 < len(self.label) <= 60 and self.label.isprintable())
        _require(isinstance(self.config, ApiConfig))
        _validate_source(self.key_source, self.key_environment)
        _model_ids(self.favorites, MAX_FAVORITES)
        _model_ids(self.recent, MAX_RECENT)

    @property
    def credential_target(self) -> str:
        return credential_target(self.config)

    def resolve_key(self, entered_key, credentials) -> str:
        """Resolve this connection's own key source only; never another connection's."""
        return resolve_key_source(self.config, self.key_source, entered_key, self.key_environment, credentials)


@dataclass(frozen=True)
class ConnectionBook:
    connections: tuple = ()
    active: str | None = None
    budget_settings: dict = field(default_factory=dict)

    def __post_init__(self):
        _require(type(self.connections) is tuple and len(self.connections) <= MAX_CONNECTIONS)
        _require(all(isinstance(c, Connection) for c in self.connections))
        ids = [c.id for c in self.connections]
        _require(all(isinstance(c, Connection) for c in self.connections) and len(set(ids)) == len(ids))
        _require(self.active is None or self.active in ids)
        _validate_budget(self.budget_settings)

    def get(self, connection_id):
        for connection in self.connections:
            if connection.id == connection_id:
                return connection
        return None

    @property
    def active_connection(self):
        return None if self.active is None else self.get(self.active)

    def connection_for_provider(self, provider: str) -> Connection | None:
        for c in self.connections:
            if c.provider == provider:
                return c
        return None

    def default_for_provider(self, provider: str) -> Connection:
        existing = self.connection_for_provider(provider)
        if existing is not None:
            return existing
        from .api_coach import GEMINI_ENDPOINT
        defaults = {
            "gemini": (GEMINI_ENDPOINT, "gemini-2.5-flash", "gemini", "Google Gemini"),
            "openai": ("https://api.openai.com/v1/responses", "gpt-4o", "responses", "OpenAI"),
            "custom": ("https://api.openai.com/v1/chat/completions", "gpt-4o", "chat_completions", "Custom service"),
        }
        endpoint, model, protocol, label = defaults.get(provider, defaults["custom"])
        config = ApiConfig(endpoint=endpoint, model=model, protocol=protocol)
        return Connection(new_connection_id(), provider, label, config)

    def with_connection(self, connection: Connection, *, activate=True) -> "ConnectionBook":
        others = tuple(c for c in self.connections if c.id != connection.id)
        position = next((i for i, c in enumerate(self.connections) if c.id == connection.id), len(others))
        merged = others[:position] + (connection,) + others[position:]
        return replace(self, connections=merged, active=connection.id if activate or self.active is None else self.active)

    def without(self, connection_id) -> "ConnectionBook":
        remaining = tuple(c for c in self.connections if c.id != connection_id)
        active = self.active if self.active != connection_id else (remaining[0].id if remaining else None)
        return replace(self, connections=remaining, active=active)

    def with_active(self, connection_id) -> "ConnectionBook":
        _require(self.get(connection_id) is not None)
        return replace(self, active=connection_id)

    def with_model(self, connection_id, model) -> "ConnectionBook":
        """Select a model for one connection and remember it as most recent."""
        connection = self.get(connection_id)
        _require(connection is not None)
        config = _validate_config({**{name: getattr(connection.config, name) for name in _CONFIG_FIELDS}, "model": model})
        recent = (model,) + tuple(m for m in connection.recent if m != model)
        return self.with_connection(replace(connection, config=config, recent=recent[:MAX_RECENT]), activate=False)

    def toggle_favorite(self, connection_id, model) -> "ConnectionBook":
        connection = self.get(connection_id)
        _require(connection is not None)
        favorites = (tuple(m for m in connection.favorites if m != model) if model in connection.favorites
                     else connection.favorites + (model,))
        return self.with_connection(replace(connection, favorites=favorites), activate=False)


def new_connection_id() -> str:
    return "c" + uuid.uuid4().hex[:12]


def connection_from_settings(settings: ApiSettings, *, connection_id=None, label=None) -> Connection:
    config, _ = _config_values(settings.config)
    provider = infer_provider(config)
    names = {"gemini": "Google Gemini", "openai": "OpenAI", "custom": "Custom service"}
    return Connection(connection_id or new_connection_id(), provider, label or names[provider], config,
                      settings.key_source, settings.key_environment)


def settings_for(connection: Connection, budget_settings: dict, *, api_key="", remember=True) -> ApiSettings:
    """Adapter to the existing single-connection runtime types."""
    return ApiSettings(connection.config, dict(budget_settings), api_key=api_key,
                       key_environment=connection.key_environment, key_source=connection.key_source,
                       remember_settings=remember)


def _connection_document(connection: Connection):
    _, config = _config_values(connection.config)
    return {"id": connection.id, "provider": connection.provider, "label": connection.label, "config": config,
            "key_source": connection.key_source, "key_environment": connection.key_environment,
            "favorites": list(connection.favorites), "recent": list(connection.recent)}


def _document(book: ConnectionBook):
    _require(isinstance(book, ConnectionBook))
    budget = _validate_budget(book.budget_settings)
    return {"schema_version": SCHEMA_VERSION, "connections": [_connection_document(c) for c in book.connections],
            "active": book.active, "budget": {k: (str(v) if v is not None and k != "call_limit" else v)
                                              for k, v in budget.items()}}


def _parse_connection(data) -> Connection:
    _require(type(data) is dict and data.keys() == _CONNECTION_FIELDS)
    _require(type(data["favorites"]) is list and type(data["recent"]) is list)
    return Connection(data["id"], data["provider"], data["label"], _validate_config(data["config"]),
                      data["key_source"], data["key_environment"],
                      tuple(data["favorites"]), tuple(data["recent"]))


def _parse(raw: bytes):
    """Return a ConnectionBook from schema 1 or 2 bytes, or raise SettingsError."""
    try:
        _require(len(raw) <= MAX_SETTINGS_BYTES)
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_fields,
                          parse_constant=_reject_json_constant)
        _require(type(data) is dict and type(data.get("schema_version")) is int)
        if data["schema_version"] == 1:
            settings = _read_schema1(raw)
            connection = connection_from_settings(settings, connection_id="migrated")
            return ConnectionBook((connection,), connection.id, settings.budget_settings)
        _require(data["schema_version"] == SCHEMA_VERSION and data.keys() == _DOCUMENT_FIELDS)
        _require(type(data["connections"]) is list)
        connections = tuple(_parse_connection(item) for item in data["connections"])
        return ConnectionBook(connections, data["active"], _validate_budget(data["budget"], document=True))
    except (UnicodeError, ValueError, TypeError, KeyError, RecursionError):
        raise SettingsError() from None


def _read_schema1(raw: bytes) -> ApiSettings:
    return parse_schema1(raw)


def _is_schema1(raw: bytes) -> bool:
    try:
        return json.loads(raw.decode("utf-8")).get("schema_version") == 1
    except (UnicodeError, ValueError, AttributeError, RecursionError):
        return False


class ConnectionStore:
    """Atomic schema-2 store beside ``settings.json``; see module docstring for upgrade safety."""

    def __init__(self, path: Path | None = None):
        self.path = SettingsStore(path).path
        self.backup_path = self.path.with_name(self.path.name + ".bak")
        self.recovered_from_backup = False

    def _read(self, path: Path):
        with path.open("rb") as stream:
            return stream.read(MAX_SETTINGS_BYTES + 1)

    def _valid_backup(self) -> bool:
        try:
            _parse(self._read(self.backup_path))
            return True
        except (OSError, ValueError, SettingsError):
            return False

    def _atomic_write(self, destination: Path, raw: bytes) -> None:
        temporary = None
        descriptor = None
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(prefix=".photocoach-settings-", suffix=".tmp", dir=destination.parent)
            temporary = Path(name)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = None
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            temporary = None
        except (OSError, ValueError):
            raise SettingsError("save") from None
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def load(self) -> ConnectionBook | None:
        """Read schema 1 or 2; recovery is in memory and never overwrites evidence."""
        self.recovered_from_backup = False
        try:
            raw = self._read(self.path)
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            raw = None
        if raw is not None:
            try:
                return _parse(raw)
            except SettingsError:
                pass
        # The settings file is unreadable or invalid: restore the last valid backup, if any.
        if self._valid_backup():
            try:
                backup_raw = self._read(self.backup_path)
                restored = _parse(backup_raw)
                self.recovered_from_backup = True
                return restored
            except (OSError, SettingsError):
                pass
        raise SettingsError("read" if raw is None else "invalid")

    def save(self, book: ConnectionBook) -> None:
        # Validate and serialize first: nothing is touched when the new document is invalid.
        document = _document(book)
        try:
            raw = (json.dumps(document, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
        except (TypeError, ValueError, UnicodeError):
            raise SettingsError() from None
        _require(len(raw) <= MAX_SETTINGS_BYTES)
        # First upgrade over a valid schema-1 file: keep it as the backup, unless a valid backup exists.
        try:
            existing = self._read(self.path)
        except FileNotFoundError:
            existing = None
        except (OSError, ValueError):
            raise SettingsError("read") from None
        if existing is not None and _is_schema1(existing) and not self._valid_backup():
            try:
                _parse(existing)
            except SettingsError:
                pass  # Preserve invalid schema-1 bytes through the damaged-file path below.
            else:
                self._atomic_write(self.backup_path, existing)  # failure aborts before the main file changes
        if existing is not None:
            if len(existing) > MAX_SETTINGS_BYTES:
                raise SettingsError('save')  # never replace an oversized file with a truncated backup
            try:
                _parse(existing)
            except SettingsError:
                # Explicit save preserves the damaged bytes before replacing settings.
                damaged = self.path.with_name(self.path.name + '.damaged-' + uuid.uuid4().hex[:12])
                self._atomic_write(damaged, existing)
        self._atomic_write(self.path, raw)

    def clear(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except (OSError, ValueError):
            raise SettingsError("clear") from None
