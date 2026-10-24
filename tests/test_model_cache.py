"""Tests for model catalog, capability classification, and caching."""
from pathlib import Path
import pytest

from photocoach.model_catalog import (
    ModelCapability,
    CatalogModel,
    classify_model,
    ModelCatalogStore,
    KNOWN_VISION_MODELS,
)


def test_classify_supported_vision_models():
    for model in ("gemini-3.5-flash-lite",):
        assert classify_model("gemini", model) == ModelCapability.SUPPORTED.value
    for model in ("gpt-4o", "gpt-4.1", "gpt-6-astra", "gpt-6.1-sol", "gpt-6-luna"):
        assert classify_model("openai", model) == ModelCapability.SUPPORTED.value


def test_classify_unsupported_non_vision_models():
    unsupported = [
        "dall-e-3",
        "tts-1",
        "tts-1-hd",
        "whisper-1",
        "text-embedding-3-small",
        "text-embedding-ada-002",
        "text-moderation-latest",
        "babbage-002",
        "davinci-002",
        "gemini-2.0-flash-preview-tts",
        "gemini-2.0-flash-preview-image-generation",
    ]
    for mid in unsupported:
        provider = "gemini" if "gemini" in mid else "openai"
        assert classify_model(provider, mid) == ModelCapability.UNSUPPORTED.value, f"Expected {mid} to be unsupported"


def test_classify_manual_unverified_models():
    assert classify_model("gemini", "gemini-3.0-experimental") == ModelCapability.NEEDS_VERIFICATION.value
    assert classify_model("openai", "gpt-5-preview") == ModelCapability.NEEDS_VERIFICATION.value
    assert classify_model("custom", "my-finetuned-llama") == ModelCapability.NEEDS_VERIFICATION.value


def test_catalog_store_default_and_persistence(tmp_path):
    catalog_path = tmp_path / "model_catalog.json"
    store = ModelCatalogStore(catalog_path)

    # Defaults load when file does not exist
    gemini_models = store.get_models("gemini")
    assert len(gemini_models) >= 1
    assert any(m.id == "gemini-3.5-flash-lite" and m.capability == "supported" for m in gemini_models)
    assert store.get_last_updated("gemini") is None

    # Update models from API fetch
    fetched = ["gemini-2.5-flash", "gemini-custom-future", "gemini-test-tts"]
    updated = store.update_provider_models("gemini", fetched)
    assert len(updated) == 3
    assert store.get_last_updated("gemini") is not None
    assert catalog_path.exists()

    # Reload from disk
    new_store = ModelCatalogStore(catalog_path)
    reloaded = new_store.get_models("gemini")
    assert [m.id for m in reloaded] == fetched
    tts_model = next(m for m in reloaded if m.id == "gemini-test-tts")
    assert tts_model.capability == "unsupported"

    # Mark verified
    new_store.mark_model_verified("gemini-custom-future", provider="gemini", scope="gemini")
    verified_models = new_store.get_models("gemini")
    custom_model = next(m for m in verified_models if m.id == "gemini-custom-future")
    assert custom_model.verified_utc is not None


@pytest.mark.parametrize('mid', ['o1-mini', 'o1-preview', 'o3-mini'])
def test_text_only_models_are_not_marked_vision(mid):
    assert classify_model('openai', mid) == 'unsupported'


def test_new_suffixes_and_custom_services_are_unverified():
    assert classify_model('openai', 'gpt-4o-made-up') == 'manual'
    assert classify_model('custom', 'gpt-4o') == 'manual'


def test_cache_is_scoped_and_legacy_verified_claims_are_not_trusted(tmp_path):
    import json
    path = tmp_path/'catalog.json'
    path.write_text(json.dumps({'schema_version': 1, 'verified_models': {'gpt-4o': '2026-01-01'}}))
    store = ModelCatalogStore(path)
    assert not any(m.verified_utc for m in store.get_models('openai'))
    a, b = 'PhotoCoach/api/'+'a'*64, 'PhotoCoach/api/'+'b'*64
    store.update_provider_models('custom', ['same-name', 'only-a'], a)
    store.update_provider_models('custom', ['same-name', 'only-b'], b)
    store.mark_model_verified('same-name', provider='custom', scope=a)
    fresh = ModelCatalogStore(path)
    assert [m.id for m in fresh.get_models('custom', a)] == ['same-name', 'only-a']
    assert fresh.get_models('custom', a)[0].verified_utc
    assert not fresh.get_models('custom', b)[0].verified_utc


@pytest.mark.parametrize('raw', ['{', '{"schema_version":2,"scopes":[]}',
    '{"schema_version":2,"schema_version":2,"scopes":{}}'])
def test_invalid_cache_falls_back_without_overwriting(tmp_path, raw):
    path=tmp_path/'catalog.json';path.write_text(raw)
    store=ModelCatalogStore(path)
    assert store.get_models('openai')
    assert store.notice and path.read_text()==raw


def test_cache_failure_preserves_disk_and_cleans_temporary_files(tmp_path, monkeypatch):
    import photocoach.model_catalog as module
    path=tmp_path/'catalog.json';store=ModelCatalogStore(path)
    store.update_provider_models('gemini',['gemini-first'])
    previous=path.read_bytes()
    def fail(*args):raise OSError('private filesystem detail')
    monkeypatch.setattr(module.os,'replace',fail)
    with pytest.raises(module.CatalogError,match='previous cache is preserved'):
        store.update_provider_models('gemini',['gemini-next'])
    assert path.read_bytes()==previous
    assert not list(tmp_path.glob('.photocoach-catalog-*'))


def test_capability_has_official_source_and_date():
    from photocoach.model_catalog import capability_evidence
    source, checked=capability_evidence('openai','gpt-6-luna')
    assert source.startswith('https://developers.openai.com/') and checked=='2026-10-22'
    assert capability_evidence('custom','gpt-6-luna')==(None,None)
