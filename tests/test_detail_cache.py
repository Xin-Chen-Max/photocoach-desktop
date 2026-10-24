"""Pixel equivalence, cache lifetime and combined-memory regression tests."""
from dataclasses import replace
import hashlib
import weakref

import numpy as np
import pytest

from photocoach.adjustments import render_photo
from photocoach.composition import Composition, output_size
from photocoach.detail_cache import DetailGeometryCache, GEOMETRY_CACHE_BYTES
from photocoach.imaging import DecodedPhoto, RawOptions, quantize_srgb
from photocoach.preview import DetailRequest, render_region
from photocoach.state import EditValues, EditState


def cached_region(cache, image, composition, rect, *, identity="sample", budget=GEOMETRY_CACHE_BYTES):
    return cache.region(image, identity, composition, rect, max_bytes=budget, reserve=lambda n: None)


@pytest.mark.parametrize("angle", [-15, -3.25, 4.75, 15])
@pytest.mark.parametrize("step", [1, 2, 3])
def test_cached_native_geometry_keeps_export_pixels_and_box_phase(angle, step):
    image = np.random.default_rng(17).random((391, 507, 3), dtype=np.float32)
    image.setflags(write=False)
    composition = Composition((.08, .12, .92, .94), angle)
    values = EditValues(exposure_ev=.35, highlights=-.75, shadows=.65, contrast=.2,
                        saturation=1.12, rgb_curve=((0, 0), (.45, .52), (1, 1)),
                        composition=composition)
    size = output_size((507, 391), composition)
    cache = DetailGeometryCache()
    full = quantize_srgb(render_photo(image, values))
    for rect in [(37, 21, 204, 170), (30, 16, 197, 165),
                 (size[0]-173, size[1]-143, size[0], size[1])]:
        request = DetailRequest(1, size, rect, step)
        region = cached_region(cache, image, composition, rect)
        cached = render_region(image, values, request, composed_region=region)
        np.testing.assert_array_equal(cached, render_region(image, values, request))
        if step == 1:
            l, t, r, b = rect
            np.testing.assert_array_equal(cached, full[t:b, l:r])
    assert cache.hits >= 1


def test_patch_owns_memory_reuses_nearby_pan_and_invalidates_dependencies():
    cache = DetailGeometryCache()
    image = np.ones((900, 1200, 3), np.float32)
    source = weakref.ref(image)
    composition = Composition(angle=4)
    rect = (300, 300, 501, 449)
    first = cached_region(cache, image, composition, rect)
    assert not first.flags.writeable and cache._pixels.flags.owndata
    assert not np.shares_memory(first, image)
    nearby = cached_region(cache, image, composition, (307, 305, 508, 454))
    assert np.shares_memory(first, nearby) and cache.builds == 1 and cache.hits == 1
    cached_region(cache, image, composition, (800, 300, 1001, 449))
    assert cache.builds == 2
    cached_region(cache, image, composition, rect, identity="other-WB")
    assert cache.builds == 3
    cached_region(cache, image, Composition(angle=5), rect, identity="other-WB")
    assert cache.builds == 4
    replacement = image.copy()
    cached_region(cache, replacement, Composition(angle=5), rect, identity="other-WB")
    assert cache.builds == 5  # Same logical identity, new decode allocation.
    del image
    assert source() is None  # Returned patch views do not pin the source either.
    cache.clear()
    assert cache.retained_bytes == 0


def test_patch_shrinks_padding_or_falls_back_without_changing_supported_images():
    cache = DetailGeometryCache()
    image = np.ones((500, 700, 3), np.float32)
    composition = Composition(angle=3)
    rect = (200, 200, 300, 300)
    reserve_calls = []
    region = cache.region(image, "sample", composition, rect,
                          max_bytes=120_000, reserve=reserve_calls.append)
    assert region.shape == (100, 100, 3)
    assert reserve_calls == [120_000] and cache.retained_bytes == 120_000
    assert cached_region(cache, image, composition, rect, budget=119_999) is None
    assert cache.retained_bytes == 0
    assert cached_region(cache, image, Composition((.1, .1, .9, .9)), rect) is None
    assert cached_region(cache, image.astype(np.float64), composition, rect) is None


def test_optional_cache_allocation_failure_uses_uncached_renderer(monkeypatch):
    import photocoach.detail_cache as module
    image = np.ones((500, 700, 3), np.float32)
    cache = DetailGeometryCache()
    def allocation_failed(*args, **kwargs):
        raise MemoryError("simulated allocation failure")
    monkeypatch.setattr(module.np, "empty", allocation_failed)
    assert cached_region(cache, image, Composition(angle=4), (10, 10, 50, 50)) is None
    assert cache.retained_bytes == 0


def test_engine_budget_reuse_wb_source_and_clear(monkeypatch, tmp_path):
    import photocoach.app as desktop
    import photocoach.imaging as imaging

    budget = 4_000_000
    monkeypatch.setattr(desktop, "LINEAR_CACHE_BYTES", budget)
    monkeypatch.setattr(desktop, "SMALL_RESERVE_BYTES", 256*192*12)
    engine = desktop.PreviewEngine()
    path = tmp_path / "synthetic.ARW"
    path.write_bytes(b"synthetic source")
    def decode(source, options, *, half_size):
        assert engine._geometry.retained_bytes == 0  # Before any RAW allocation.
        size = (256, 192) if half_size else (512, 384)
        image = np.full((size[1], size[0], 3), .15*options.wb_gains[0], np.float32)
        image.setflags(write=False)
        metadata = {"sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "visible_size": [512, 384], "output_size": list(size),
                    "half_size": half_size, "orientation_flip": 0}
        return DecodedPhoto(image, metadata, options, half_size, source)
    monkeypatch.setattr(imaging, "decode_raw", decode)
    photo, _ = engine.load(path)
    state = EditState(photo.metadata["sha256"], path, "session", composition=Composition(angle=3.25))
    del photo  # No external decoded references in a real serial worker.
    size = output_size((512, 384), state.composition)
    request = DetailRequest(1, size, (100, 100, 250, 220), source_size=(512, 384))
    engine.detail(state, request)
    assert engine.retained_bytes <= budget
    assert engine._geometry.retained_bytes > 0
    assert len(engine.cache._entries) == 1  # LRU half decode made room for patch.
    decodes = engine.cache.decode_count
    for contrast in [.1, .2, -.1]:
        engine.detail(replace(state, contrast=contrast), request)
        assert engine.retained_bytes <= budget
    assert engine.cache.decode_count == decodes and engine._geometry.hits == 3
    changed_wb = replace(state, raw_options=RawOptions("custom", (1.1, 1, 1)))
    engine.detail(changed_wb, request)
    assert engine.cache.decode_count == decodes+1 and engine.retained_bytes <= budget
    engine.detail(replace(changed_wb, compare_baseline=True), request)
    assert engine.cache.decode_count == decodes+2 and engine.retained_bytes <= budget
    engine.render(state)
    assert engine._geometry.retained_bytes == 0 and engine.retained_bytes <= budget
    engine.detail(state, request)
    path.write_bytes(b"different synthetic source bytes")
    with pytest.raises(RuntimeError, match="source file has changed"):
        engine.detail(state, request)
    assert engine._geometry.retained_bytes == 0 and engine.retained_bytes <= budget
    engine.clear()
    assert engine.retained_bytes == 0
