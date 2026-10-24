"""Allowlisted local connection settings; credentials are stored separately.

Reading settings never looks up a credential, changes the filesystem, enables
cloud mode, or contacts a provider. The caller controls when to save or resolve
the explicitly selected credential source.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urlsplit, urlunsplit

from .api_coach import ApiConfig, ApiError, GeminiModelsConfig, ProviderModelsConfig
from .budget import SessionBudget


MAX_SETTINGS_BYTES = 64 * 1024
_SCHEMA_VERSION = 1
_CONFIG_FIELDS = frozenset(("endpoint", "model", "protocol", "timeout_seconds", "max_output_tokens"))
_BUDGET_FIELDS = frozenset(("call_limit", "budget_usd", "reserve_usd",
                            "input_usd_per_million", "output_usd_per_million"))
_DOCUMENT_FIELDS = frozenset(("schema_version", "config", "budget", "key_source", "key_environment"))
_KEY_SOURCES = frozenset(("session", "environment", "saved"))


@dataclass(frozen=True)
class ApiSettings:
    config: ApiConfig
    budget_settings: dict
    api_key: str = field(default="", repr=False)
    key_environment: str = "PHOTOCOACH_API_KEY"
    key_source: str = "session"
    remember_settings: bool = False


class SettingsError(RuntimeError):
    """Fixed messages only: filesystem paths and settings values stay private."""

    def __init__(self, code="invalid"):
        messages = {
            "invalid": "Saved connection settings are invalid or use an unsupported version. Configure the connection again.",
            "location": "The local settings folder is unavailable. Connection settings can still be used for this session.",
            "read": "Connection settings could not be read. Configure the connection for this session.",
            "save": "Connection settings could not be saved. Your existing saved settings were not intentionally cleared.",
            "clear": "Saved connection settings could not be removed.",
        }
        self.code = code if code in messages else "invalid"
        super().__init__(messages[self.code])


def _require(condition):
    if not condition:
        raise SettingsError()


def _validate_source(source, environment):
    _require(isinstance(source, str) and source in _KEY_SOURCES)
    _require(isinstance(environment, str) and len(environment) <= 128)
    _require((environment == "" and source != "environment") or
             re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", environment) is not None)


def _config_values(config):
    _require(isinstance(config, ApiConfig))
    values = {name: getattr(config, name) for name in _CONFIG_FIELDS}
    return _validate_config(values), values


def _validate_config(values):
    _require(type(values) is dict and values.keys() == _CONFIG_FIELDS)
    for name in ("endpoint", "model", "protocol"):
        _require(type(values[name]) is str)
    _require(len(values["endpoint"]) <= 2048)
    try:
        return ApiConfig(**values)
    except (ApiError, ValueError, TypeError):
        raise SettingsError() from None


def _validate_budget(values, *, document=False):
    _require(type(values) is dict and not values.keys() - _BUDGET_FIELDS)
    if document:
        _require(values.keys() == _BUDGET_FIELDS)
    for name, value in values.items():
        if name == "call_limit":
            _require(type(value) is int)
        elif value is not None:
            # JSON money fields are strings, preserving Decimal precision.
            if document:
                _require(type(value) is str)
            else:
                _require(isinstance(value, (str, int, float, Decimal)) and not isinstance(value, bool))
            _require(len(str(value)) <= 128)
    try:
        budget = SessionBudget(**values)
    except (ValueError, TypeError, ArithmeticError):
        raise SettingsError() from None
    return {name: getattr(budget, name) for name in _BUDGET_FIELDS}


def _document(settings):
    _require(isinstance(settings, ApiSettings))
    _require(type(settings.api_key) is str and type(settings.remember_settings) is bool)
    _validate_source(settings.key_source, settings.key_environment)
    _, config = _config_values(settings.config)
    values = _validate_budget(settings.budget_settings)
    budget = {name: str(value) if isinstance(value, Decimal) else value for name, value in values.items()}
    return {"schema_version": _SCHEMA_VERSION, "config": config, "budget": budget,
            "key_source": settings.key_source, "key_environment": settings.key_environment}


def _reject_duplicate_fields(pairs):
    result = {}
    for name, value in pairs:
        _require(name not in result)
        result[name] = value
    return result


def _reject_json_constant(_value):
    raise SettingsError()


def parse_schema1(raw: bytes) -> ApiSettings:
    try:
        _require(len(raw) <= MAX_SETTINGS_BYTES)
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_fields,
                          parse_constant=_reject_json_constant)
        _require(type(data) is dict and data.keys() == _DOCUMENT_FIELDS)
        _require(type(data["schema_version"]) is int and data["schema_version"] == _SCHEMA_VERSION)
        config = _validate_config(data["config"])
        budget = _validate_budget(data["budget"], document=True)
        _validate_source(data["key_source"], data["key_environment"])
        return ApiSettings(config, budget, key_environment=data["key_environment"],
                           key_source=data["key_source"], remember_settings=True)
    except (UnicodeError, ValueError, TypeError, KeyError, RecursionError):
        raise SettingsError() from None


class SettingsStore:
    """A small atomic JSON file; the file contains no key or credential target."""

    def __init__(self, path: Path | None = None):
        if path is None:
            local = os.environ.get("LOCALAPPDATA", "")
            if not local or not Path(local).is_absolute():
                raise SettingsError("location")
            path = Path(local) / "PhotoCoach" / "settings.json"
        self.path = Path(path)

    def load(self) -> ApiSettings | None:
        try:
            with self.path.open("rb") as stream:
                raw = stream.read(MAX_SETTINGS_BYTES + 1)
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            raise SettingsError("read") from None
        try:
            return parse_schema1(raw)
        except SettingsError:
            try:
                from .connections import ConnectionStore, settings_for
                book = ConnectionStore(self.path).load()
                if book is not None and book.active_connection is not None:
                    return settings_for(book.active_connection, book.budget_settings)
            except Exception:
                pass
            raise

    def save(self, settings: ApiSettings) -> None:
        # Validate before creating a directory or touching an existing file.
        document = _document(settings)
        try:
            raw = (json.dumps(document, ensure_ascii=False, allow_nan=False,
                              sort_keys=True, indent=2) + "\n").encode("utf-8")
        except (TypeError, ValueError, UnicodeError):
            raise SettingsError() from None
        _require(len(raw) <= MAX_SETTINGS_BYTES)
        temporary = None
        descriptor = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(prefix=".photocoach-settings-", suffix=".tmp", dir=self.path.parent)
            temporary = Path(name)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = None  # The stream now owns and closes this handle.
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
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

    def clear(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except (OSError, ValueError):
            raise SettingsError("clear") from None


def credential_target(config: ApiConfig | GeminiModelsConfig) -> str:
    """Bind a credential to its interface and endpoint, independently of model."""
    if isinstance(config, ProviderModelsConfig):
        config = ApiConfig(config.endpoint, 'metadata-key-scope', config.protocol)
    elif isinstance(config, GeminiModelsConfig):
        config = GeminiModelsConfig(config.endpoint, config.timeout_seconds)
    else:
        config, _ = _config_values(config)
    url = urlsplit(config.endpoint)
    hostname = url.hostname.lower()
    if ":" in hostname:
        hostname = "[" + hostname + "]"
    netloc = hostname if url.port in (None, 443) else f"{hostname}:{url.port}"
    normalized = urlunsplit((url.scheme.lower(), netloc, url.path.rstrip("/"), "", ""))
    identity = json.dumps([config.protocol, normalized], ensure_ascii=False, separators=(",", ":"))
    return "PhotoCoach/api/" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def resolve_api_key(settings: ApiSettings, credentials) -> str:
    """Resolve exactly one source; never fall back to a different saved key."""
    _require(isinstance(settings, ApiSettings) and type(settings.api_key) is str)
    return resolve_key_source(settings.config, settings.key_source, settings.api_key,
                              settings.key_environment, credentials)


def resolve_key_source(config: ApiConfig | GeminiModelsConfig, source: str, entered_key: str,
                       environment: str, credentials) -> str:
    """Shared explicit key resolution for generation settings or model-free metadata."""
    _require(type(entered_key) is str)
    _validate_source(source, environment)
    if source == "session":
        return entered_key
    if source == "environment":
        return os.environ.get(environment, "")
    # A typed replacement can be reviewed/tested before saving it in the vault.
    result = entered_key or credentials.read(credential_target(config))
    _require(result is None or type(result) is str)
    return result or ""
