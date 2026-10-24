"""Bounded, destination-scoped model metadata cache. No credential or photo data."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
import re
import tempfile
from .settings import SettingsStore, SettingsError, _reject_duplicate_fields, _reject_json_constant

CATALOG_SCHEMA_VERSION = 2
MAX_CATALOG_BYTES = 4 * 1024 * 1024
CHECKED_UTC = '2026-10-22'
# Exact IDs only. A newly introduced suffix/alias is deliberately unverified.
SOURCES = {
    'gemini': 'https://ai.google.dev/gemini-api/docs/models/',
    'openai': 'https://developers.openai.com/api/docs/models/',
}
KNOWN_VISION_MODELS = {
    'gemini': {'gemini-3.5-flash-lite'},
    'openai': {'gpt-6-astra', 'gpt-6.1-sol', 'gpt-6-luna', 'gpt-4.1', 'gpt-4o'},
}
DEFAULT_CURATED = {p: sorted(ids) for p, ids in KNOWN_VISION_MODELS.items()}
DEFAULT_CURATED['custom'] = []
EXPLICIT_NOT_SUITABLE = {'o1-mini', 'o1-preview', 'o3-mini', 'gpt-3.5-turbo', 'gpt-4'}
NON_VISION_PREFIXES = ('dall-e', 'tts', 'whisper', 'text-embedding', 'text-search',
                      'text-moderation', 'babbage', 'davinci', 'curie', 'ada', 'gpt-image', 'imagen', 'veo')
NON_VISION_SUBSTRINGS = ('-tts', '-image-generation', '-audio', '-realtime', '-transcribe',
                        '-transcription', 'embedding', 'moderation', '-live', 'robotics')

class ModelCapability(str, Enum):
    SUPPORTED = 'supported'
    NEEDS_VERIFICATION = 'manual'
    UNSUPPORTED = 'unsupported'


def classify_model(provider: str, model_id: str) -> str:
    mid = model_id.lower().strip()
    if provider not in ('gemini', 'openai'):
        return 'manual'
    if (mid in EXPLICIT_NOT_SUITABLE or any(mid == p or mid.startswith(p+'-') for p in NON_VISION_PREFIXES)
            or any(part in mid for part in NON_VISION_SUBSTRINGS)):
        return 'unsupported'
    return 'supported' if mid in KNOWN_VISION_MODELS.get(provider, ()) else 'manual'


def capability_evidence(provider, model_id):
    if classify_model(provider, model_id) == 'supported':
        url = SOURCES[provider] + (model_id if provider == 'gemini' else model_id)
        if model_id.startswith('gpt-6'):
            url = SOURCES[provider]
        return url, CHECKED_UTC
    return None, None


def _valid_id(value):
    return type(value) is str and 0 < len(value) <= 200 and all(c.isprintable() and not c.isspace() for c in value)


def _valid_time(value):
    if value is None:
        return True
    try:
        return type(value) is str and len(value) <= 40 and datetime.fromisoformat(value).utcoffset() is not None
    except ValueError:
        return False


@dataclass(frozen=True)
class CatalogModel:
    id: str
    display_name: str
    provider: str
    capability: str
    verified_utc: str | None = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(data['id'], data.get('display_name', data['id']), data.get('provider', 'custom'),
                   classify_model(data.get('provider', 'custom'), data['id']))


class CatalogError(RuntimeError):
    def __init__(self):
        super().__init__('Model cache could not be saved. The previous cache is preserved; loaded models remain available for this session.')


class ModelCatalogStore:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else SettingsStore().path.with_name('model_catalog.json')
        self.notice = ''
        self._data = None

    def _read_data(self):
        if self._data is not None:
            return self._data
        self._data = {'schema_version': 2, 'scopes': {}}
        try:
            with self.path.open('rb') as stream:
                raw = stream.read(MAX_CATALOG_BYTES + 1)
            if len(raw) > MAX_CATALOG_BYTES:
                raise ValueError
            data = json.loads(raw.decode('utf-8'), object_pairs_hook=_reject_duplicate_fields,
                              parse_constant=_reject_json_constant)
            if type(data) is not dict or set(data) != {'schema_version', 'scopes'} or type(data['schema_version']) is not int or data['schema_version'] != 2:
                raise ValueError
            scopes = data['scopes']
            if type(scopes) is not dict or len(scopes) > 32:
                raise ValueError
            for scope, entry in scopes.items():
                if not re.fullmatch(r'(?:gemini|openai|custom|PhotoCoach/api/[a-f0-9]{64})', scope):
                    raise ValueError
                if type(entry) is not dict or set(entry) != {'provider', 'updated_utc', 'models', 'verified'}:
                    raise ValueError
                if entry['provider'] not in SOURCES.keys() | {'custom'} or not _valid_time(entry['updated_utc']):
                    raise ValueError
                rows, verified = entry['models'], entry['verified']
                if type(rows) is not list or len(rows) > 2000 or type(verified) is not dict or len(verified) > 2000:
                    raise ValueError
                for row in rows:
                    if (type(row) is not dict or set(row) != {'id', 'display_name'} or not _valid_id(row['id'])
                            or type(row['display_name']) is not str or not 0 < len(row['display_name']) <= 128
                            or not row['display_name'].isprintable()):
                        raise ValueError
                if any(not _valid_id(mid) or not _valid_time(stamp) or stamp is None for mid, stamp in verified.items()):
                    raise ValueError
            self._data = data
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError, KeyError, UnicodeError, RecursionError, SettingsError):
            self.notice = 'Saved model cache is unavailable or from an older version. Refresh models manually; no request was sent.'
        return self._data

    def _entry(self, provider, scope):
        entry = self._read_data()['scopes'].get(scope or provider)
        return entry if entry and entry['provider'] == provider else None

    def get_models(self, provider, scope=None):
        entry = self._entry(provider, scope)
        rows = entry['models'] if entry is not None else [{'id': mid, 'display_name': mid} for mid in DEFAULT_CURATED.get(provider, ())]
        verified = entry['verified'] if entry else {}
        return [CatalogModel(row['id'], row['display_name'], provider, classify_model(provider, row['id']), verified.get(row['id'])) for row in rows]

    def get_last_updated(self, provider, scope=None):
        entry = self._entry(provider, scope)
        return entry['updated_utc'] if entry else None

    def _atomic_write(self, data):
        temporary = None
        try:
            raw = (json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2)+'\n').encode('utf-8')
            if len(raw) > MAX_CATALOG_BYTES or len(data['scopes']) > 32:
                raise ValueError
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(prefix='.photocoach-catalog-', suffix='.tmp', dir=self.path.parent)
            temporary = Path(name)
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except (OSError, ValueError, TypeError):
            raise CatalogError() from None
        finally:
            if temporary:
                try: temporary.unlink(missing_ok=True)
                except OSError: pass

    def update_provider_models(self, provider, raw_models, scope=None):
        if (provider not in ('gemini', 'openai', 'custom') or len(raw_models) > 2000 or
                re.fullmatch(r'(?:gemini|openai|custom|PhotoCoach/api/[a-f0-9]{64})', scope or provider) is None):
            raise CatalogError()
        entry = self._entry(provider, scope)
        rows = {}
        for item in raw_models:
            mid = item if isinstance(item, str) else item.get('id')
            display = mid if isinstance(item, str) else item.get('display_name', mid)
            if not _valid_id(mid) or not isinstance(display, str) or not display.isprintable() or not 0 < len(display) <= 128:
                raise CatalogError()
            rows[mid] = {'id': mid, 'display_name': display}
        data = self._read_data()
        data['scopes'][scope or provider] = {'provider': provider, 'updated_utc': datetime.now(timezone.utc).isoformat(),
                                           'models': list(rows.values()), 'verified': dict(entry['verified']) if entry else {}}
        # Retain fresh data in memory if disk persistence fails; surface the fixed notice.
        self._atomic_write(data)
        return self.get_models(provider, scope)

    def mark_model_verified(self, model_id, *, provider, scope):
        if not _valid_id(model_id):
            raise CatalogError()
        entry = self._entry(provider, scope)
        if entry is None:
            self.update_provider_models(provider, [model_id], scope)
            entry = self._entry(provider, scope)
        entry['verified'][model_id] = datetime.now(timezone.utc).isoformat()
        self._atomic_write(self._read_data())
