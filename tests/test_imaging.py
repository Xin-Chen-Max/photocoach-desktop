"""Pipeline contracts tested without depending on a private RAW fixture."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from PIL import Image, ImageCms
import pytest
import tifffile

from photocoach import imaging
from photocoach.imaging import (
    DecodedCache,
    RawOptions,
    effective_whitebalance,
    export_image,
    quantize_srgb,
    render_srgb,
)


def inverse_srgb(encoded: np.ndarray) -> np.ndarray:
    """Independent standard transfer used to check linear-light exposure."""
    value = encoded.astype(np.float64)
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


@pytest.mark.parametrize("ev", [-5.0, -1.25, 0.0, 1.0, 3.5, 5.0])
def test_exposure_scales_linear_light_without_mutating_input(ev: float) -> None:
    linear = np.linspace(0.00001, 0.025, 129 * 3 * 3, dtype=np.float32).reshape(129, 3, 3)
    original = linear.copy()
    linear.setflags(write=False)
    rendered = render_srgb(linear, ev)
    np.testing.assert_allclose(inverse_srgb(rendered), original * 2.0**ev, rtol=2e-6, atol=1e-8)
    np.testing.assert_array_equal(linear, original)
    assert rendered.dtype == np.float32
    assert not np.shares_memory(rendered, linear)
    assert not linear.flags.writeable


def test_srgb_breakpoint_and_sdr_clipping_are_deterministic() -> None:
    linear = np.array([[[-0.2, 0.0, 0.0031308], [0.003131, 0.18, 2.0]]], dtype=np.float32)
    expected = np.array([[[0.0, 0.0, 12.92 * 0.0031308],
                          [1.055 * 0.003131 ** (1 / 2.4) - 0.055,
                           1.055 * 0.18 ** (1 / 2.4) - 0.055, 1.0]]])
    first = render_srgb(linear)
    np.testing.assert_allclose(first, expected, atol=1e-7, rtol=0)
    np.testing.assert_array_equal(first, render_srgb(linear))
    assert first.min() == 0
    assert first.max() == pytest.approx(1.0, abs=1e-7)


@pytest.mark.parametrize("ev", [-5.001, 5.001, np.inf, -np.inf, np.nan])
def test_render_rejects_invalid_exposure(ev: float) -> None:
    with pytest.raises(ValueError):
        render_srgb(np.ones((1, 1, 3), dtype=np.float32), ev)


@pytest.mark.parametrize("image", [
    np.zeros((1, 1, 3), dtype=np.uint16),
    np.zeros((2, 2), dtype=np.float32),
    np.zeros((1, 1, 4), dtype=np.float32),
    np.zeros((0, 1, 3), dtype=np.float32),
    np.array([[[np.nan, 0.0, 0.0]]], dtype=np.float32),
    np.array([[[0.0, np.inf, 0.0]]], dtype=np.float32),
])
def test_render_and_quantization_reject_invalid_images(image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        render_srgb(image)
    with pytest.raises(ValueError):
        quantize_srgb(image)


@pytest.mark.parametrize("bits,dtype,expected", [
    (8, np.uint8, [0, 0, 128, 255, 255, 64]),
    (16, np.uint16, [0, 0, 32768, 65535, 65535, 16384]),
])
def test_quantization_occurs_once_with_rounding_and_clipping(bits, dtype, expected) -> None:
    values = np.array([[[-0.1, 0.0, 0.5], [1.0, 1.1, 0.25]]], dtype=np.float32)
    saved = values.copy()
    actual = quantize_srgb(values, bits=bits)
    assert actual.dtype == dtype
    np.testing.assert_array_equal(actual.ravel(), expected)
    np.testing.assert_array_equal(values, saved)


@pytest.mark.parametrize("bits", [0, 10, 12, 32])
def test_quantization_rejects_unsupported_depth(bits: int) -> None:
    with pytest.raises(ValueError):
        quantize_srgb(np.zeros((1, 1, 3), dtype=np.float32), bits=bits)


def test_camera_and_custom_whitebalance_have_explicit_reproducible_meanings() -> None:
    camera = [512, 256, 384, 256]
    assert effective_whitebalance(camera, RawOptions()) == (2, 1, 1.5, 1)
    # Camera mode ignores stored custom gains, avoiding hidden tint adjustments.
    assert effective_whitebalance(camera, RawOptions("camera", (2, 3, 4))) == (2, 1, 1.5, 1)
    custom = RawOptions("custom", (1.5, 0.5, 2.0))
    assert effective_whitebalance(camera, custom) == (6, 1, 6, 1)
    # A uniform gain is normalized away; it must not double as exposure.
    assert effective_whitebalance(camera, RawOptions("custom", (2, 2, 2))) == (2, 1, 1.5, 1)


def test_second_green_fallback_is_only_allowed_for_three_color_sensor() -> None:
    assert effective_whitebalance([2, 1, 3, 0], RawOptions(), num_colors=3) == (2, 1, 3, 1)
    with pytest.raises(ValueError, match="second green"):
        effective_whitebalance([2, 1, 3, 0], RawOptions(), num_colors=4)


@pytest.mark.parametrize("coefficients", [[], [2, 1, 3], [0, 1, 3, 1],
                                                   [2, np.nan, 3, 1], [2, 1, -3, 1],
                                                   [2, 1, 3, np.inf], [2, 1, 3, -1]])
@pytest.mark.parametrize("mode", ["camera", "custom"])
def test_missing_invalid_camera_whitebalance_never_silently_uses_auto(coefficients, mode) -> None:
    with pytest.raises(ValueError):
        effective_whitebalance(coefficients, RawOptions(mode))


@pytest.mark.parametrize("description,num_colors", [("GRBG", 3), ("RGBG", 1), ("CMYG", 4)])
def test_unsupported_sensor_layout_is_rejected(description, num_colors) -> None:
    with pytest.raises(ValueError):
        effective_whitebalance([2, 1, 3, 1], RawOptions(), color_desc=description, num_colors=num_colors)


@pytest.mark.parametrize("kwargs", [
    {"wb_mode": "auto"}, {"wb_gains": (1, 1)}, {"wb_gains": (1, 1, np.nan)},
    {"wb_gains": (0.249, 1, 1)}, {"wb_gains": (1, 4.001, 1)},
])
def test_raw_options_reject_ambiguous_or_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError):
        RawOptions(**kwargs)


@pytest.fixture
def fake_raw_decoder(monkeypatch, tmp_path):
    """A fake oriented LibRaw result checks our contract, not camera support."""
    source = tmp_path / "fixture.ARW"
    original_bytes = b"synthetic read-only RAW test fixture"
    source.write_bytes(original_bytes)
    pixels = np.arange(4 * 3 * 3, dtype=np.uint16).reshape(4, 3, 3) * 1500
    calls = []
    reads = []

    class FakeRaw:
        sizes = SimpleNamespace(raw_width=8, raw_height=6, width=8, height=6, flip=6)
        camera_whitebalance = [512.0, 256.0, 384.0, 0.0]
        color_desc = b"RGBG"
        num_colors = 3
        black_level_per_channel = [512, 512, 512, 512]
        white_level = 16383

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def postprocess(self, **kwargs):
            calls.append(kwargs)
            return raw_result[0]

    raw_result = [pixels]

    def imread(stream):
        reads.append((stream.mode, stream.readable(), stream.writable()))
        assert stream.read() == original_bytes
        return FakeRaw()

    monkeypatch.setitem(sys.modules, "rawpy", SimpleNamespace(
        imread=imread,
        DemosaicAlgorithm=SimpleNamespace(AHD="AHD"),
        ColorSpace=SimpleNamespace(sRGB="sRGB"),
        HighlightMode=SimpleNamespace(Clip="clip"),
        FBDDNoiseReductionMode=SimpleNamespace(Off="off"),
    ))
    monkeypatch.setattr(imaging, "dependency_versions", lambda: {"rawpy": "fake"})
    monkeypatch.setattr(imaging, "_shooting_metadata", lambda path: {"model": "fixture"})
    return SimpleNamespace(source=source, original=original_bytes, pixels=pixels,
                           calls=calls, reads=reads, raw_result=raw_result)


def test_decode_has_explicit_raw_controls_readonly_source_and_one_orientation(fake_raw_decoder) -> None:
    fake = fake_raw_decoder
    decoded = imaging.decode_raw(fake.source)
    expected_hash = hashlib.sha256(fake.original).hexdigest()
    assert fake.source.read_bytes() == fake.original
    assert decoded.metadata["sha256"] == expected_hash
    assert decoded.metadata["photo_id"] == expected_hash
    assert fake.reads == [("rb", True, False)]
    assert decoded.metadata["orientation_flip"] == 6
    assert decoded.metadata["output_size"] == [3, 4]
    assert decoded.linear_rgb.shape == (4, 3, 3)
    np.testing.assert_allclose(decoded.linear_rgb, fake.pixels / 65535.0, rtol=1e-7)
    assert decoded.linear_rgb.dtype == np.float32
    assert not decoded.linear_rgb.flags.writeable
    with pytest.raises(ValueError):
        decoded.linear_rgb[0, 0, 0] = 1
    assert fake.calls == [{
        "demosaic_algorithm": "AHD", "half_size": True, "four_color_rgb": False,
        "use_camera_wb": False, "use_auto_wb": False, "user_wb": [2.0, 1.0, 1.5, 1.0],
        "output_color": "sRGB", "output_bps": 16, "gamma": (1.0, 1.0),
        "no_auto_bright": True, "no_auto_scale": False, "adjust_maximum_thr": 0.0,
        "bright": 1.0, "highlight_mode": "clip", "exp_shift": 1.0,
        "exp_preserve_highlights": 0.0, "median_filter_passes": 0,
        "fbdd_noise_reduction": "off",
    }]
    assert "user_flip" not in fake.calls[0]
    assert not any("path" in key.lower() or "gps" in key.lower() for key in decoded.metadata)


@pytest.mark.parametrize("invalid_pixels", [
    np.zeros((2, 3, 3), dtype=np.uint8),
    np.zeros((2, 3, 3), dtype=np.float32),
    np.zeros((2, 3), dtype=np.uint16),
    np.zeros((2, 3, 4), dtype=np.uint16),
])
def test_decode_rejects_non_16bit_rgb_decoder_output(fake_raw_decoder, invalid_pixels) -> None:
    fake_raw_decoder.raw_result[0] = invalid_pixels
    with pytest.raises(ValueError, match="16-bit RGB"):
        imaging.decode_raw(fake_raw_decoder.source)


def test_decode_rejects_other_extensions_before_decoder_call(fake_raw_decoder, tmp_path) -> None:
    source = tmp_path / "preview.jpg"
    source.write_bytes(b"not RAW")
    with pytest.raises(ValueError, match="Sony"):
        imaging.decode_raw(source)
    assert not fake_raw_decoder.calls


def test_cache_reuses_decode_for_exposure_and_invalidates_wb_size_source_and_pipeline(fake_raw_decoder, monkeypatch) -> None:
    source = fake_raw_decoder.source
    cache = DecodedCache(max_entries=6)
    original = cache.get(source)
    baseline = render_srgb(original.linear_rgb, 0)
    changed = render_srgb(original.linear_rgb, 1)
    assert cache.get(source) is original
    assert len(fake_raw_decoder.calls) == 1
    assert not np.array_equal(baseline, changed)
    custom = cache.get(source, RawOptions("custom", (1.1, 1, 1)))
    assert custom is not original
    assert fake_raw_decoder.calls[-1]["user_wb"] == [2.2, 1.0, 1.5, 1.0]
    full = cache.get(source, half_size=False)
    assert full is not original and not fake_raw_decoder.calls[-1]["half_size"]
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert cache.get(source) is not original
    monkeypatch.setattr(imaging, "PIPELINE_VERSION", "future-test-pipeline")
    cache.get(source)
    assert len(fake_raw_decoder.calls) == 5
    cache.clear()
    cache.get(source)
    assert len(fake_raw_decoder.calls) == 6


def test_cache_has_bounded_eviction_and_validates_capacity(fake_raw_decoder) -> None:
    with pytest.raises(ValueError):
        DecodedCache(0)
    cache = DecodedCache(1)
    original = cache.get(fake_raw_decoder.source)
    cache.get(fake_raw_decoder.source, half_size=False)
    assert cache.get(fake_raw_decoder.source) is not original
    assert len(fake_raw_decoder.calls) == 3


def test_cache_bounds_bytes_and_one_full_frame(fake_raw_decoder):
    source = fake_raw_decoder.source
    size = fake_raw_decoder.pixels.size * 4
    cache = DecodedCache(10, max_bytes=size * 2)
    cache.get(source)
    cache.get(source, half_size=False, expected_bytes=size)
    assert cache.retained_bytes == size * 2
    cache.get(source, RawOptions("custom", (1.1,1,1)), half_size=False, expected_bytes=size)
    assert sum(not photo.half_size for photo in cache._entries.values()) == 1
    assert cache.retained_bytes <= cache.max_bytes
    assert cache.high_water_bytes <= cache.max_bytes
    calls = cache.decode_count
    with pytest.raises(MemoryError, match="budget"):
        cache.get(source, RawOptions("custom", (1.2,1,1)), half_size=False, expected_bytes=size * 3)
    assert cache.decode_count == calls
    cache.clear()
    assert cache.retained_bytes == 0


def test_cache_trim_preserves_recent_full_frame_and_before_decode_only_on_miss(fake_raw_decoder):
    from photocoach.imaging import DecodedCache
    cache = DecodedCache(3)
    callbacks = []
    source = fake_raw_decoder.source
    half = cache.get(source, before_decode=lambda: callbacks.append("half"))
    full = cache.get(source, half_size=False, before_decode=lambda: callbacks.append("full"))
    assert cache.get(source, half_size=False, before_decode=lambda: callbacks.append("hit")) is full
    assert callbacks == ["half", "full"]
    cache.trim_to(full.linear_rgb.nbytes)
    assert cache.retained_bytes == full.linear_rgb.nbytes
    assert cache.full_identity == (full.metadata["sha256"], full.options)
    assert cache.get(source) is not half
    with pytest.raises(ValueError):
        cache.trim_to(-1)
    cache.trim_to(0)
    assert cache.retained_bytes == 0 and cache.full_identity is None


def test_16bit_tiff_roundtrip_retains_precision_icc_and_omits_source_metadata(tmp_path) -> None:
    source = tmp_path / "original.ARW"
    source.write_bytes(b"private source file including hypothetical GPS")
    pixels = np.linspace(0, 1, 300, dtype=np.float32).reshape(10, 10, 3)
    target = export_image(tmp_path / "result.tiff", pixels, source_path=source)
    with tifffile.TiffFile(target) as tiff:
        page = tiff.pages[0]
        actual = page.asarray()
        assert actual.dtype == np.uint16
        assert page.samplesperpixel == 3
        np.testing.assert_array_equal(actual, quantize_srgb(pixels, bits=16))
        assert page.tags[34675].value == imaging.srgb_icc_profile()
        assert 34853 not in page.tags  # GPS IFD
        assert 34665 not in page.tags  # source EXIF IFD
        assert 270 not in page.tags  # no private JSON/path in ImageDescription
        assert 274 not in page.tags  # already oriented; no rotation on reopen
    assert source.read_bytes() == b"private source file including hypothetical GPS"
    # Demonstrate that output has genuinely more than 256 representable steps.
    assert np.unique(actual).size > 256


def test_jpeg_has_valid_srgb_profile_and_no_source_exif(tmp_path) -> None:
    target = export_image(tmp_path / "result.jpg", np.full((4, 5, 3), 0.5, dtype=np.float32))
    with Image.open(target) as image:
        assert image.mode == "RGB" and image.size == (5, 4)
        assert image.info["icc_profile"] == imaging.srgb_icc_profile()
        assert not image.getexif()
        assert np.abs(np.asarray(image).astype(int) - 128).max() <= 1
    import io
    profile = ImageCms.ImageCmsProfile(io.BytesIO(imaging.srgb_icc_profile()))
    assert "sRGB" in ImageCms.getProfileDescription(profile)


@pytest.mark.parametrize("suffix,quality", [(".png", 95), (".jpg", 0), (".jpg", 101)])
def test_export_rejects_unsupported_target_without_creating_file(tmp_path, suffix, quality) -> None:
    path = tmp_path / ("bad" + suffix)
    with pytest.raises(ValueError):
        export_image(path, np.zeros((1, 1, 3), dtype=np.float32), jpeg_quality=quality)
    assert not path.exists()
    assert not list(tmp_path.glob(".photocoach-*"))
