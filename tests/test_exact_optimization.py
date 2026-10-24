"""Exact-pixel optimization contract against frozen 0.7.0 reference implementations."""
import hashlib
import numpy as np
import pytest

from photocoach.adjustments import render_edit, render_photo
from photocoach.color_mixer import ColorMixer, apply_color_mixer, CENTERS
from photocoach.composition import Composition, output_size, sample_composition
from photocoach.imaging import quantize_srgb
from photocoach.preview import DetailRequest, render_region
from photocoach.scratch import Scratch, ScratchPool
from photocoach.state import EditValues

from reference_pipeline import reference_apply_color_mixer, reference_sample_composition


def assert_bit_identical_float32(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.dtype == np.float32 and expected.dtype == np.float32
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def boundary_rgb_fixture() -> np.ndarray:
    """Include exact 0/1, grays, channel ties, near-endpoints with tiny chroma, and band centers."""
    rng = np.random.default_rng(1071)
    random_part = rng.random((45, 73, 3), dtype=np.float32)
    special = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5],
        [1e-6, 1e-6, 1e-6],
        # Channel ties: R==G, G==B, R==B, R==G==B
        [0.6, 0.6, 0.2],
        [0.2, 0.7, 0.7],
        [0.8, 0.1, 0.8],
        # Near-white and near-black with nonzero chroma (endpoint denominator guard)
        [1.0, 1.0 - 1e-7, 1.0 - 2e-7],
        [1e-7, 2e-7, 0.0],
        [0.9999999, 0.9999998, 0.9999990],
        # Hue wrap around 0 / 1 and band centers
        [0.9, 0.1, 0.10001],
        [0.9, 0.10001, 0.1],
        [0.8, 0.5, 0.2],
        [0.8, 0.8, 0.2],
        [0.2, 0.8, 0.2],
        [0.2, 0.8, 0.8],
        [0.2, 0.2, 0.8],
        [0.5, 0.2, 0.8],
        [0.8, 0.2, 0.8],
    ], dtype=np.float32).reshape(1, -1, 3)
    tiled = np.resize(special, (4, 73, 3))
    return np.concatenate([random_part, tiled], axis=0)


MIXER_CASES = [
    ColorMixer(),
    ColorMixer().with_value(0, 0, 30).with_value(0, 1, 100).with_value(0, 2, -100),
    ColorMixer().with_value(2, 2, 15).with_value(5, 1, -10),
    ColorMixer().with_value(1, 0, -30).with_value(4, 1, 75).with_value(7, 2, 60),
    # Single-axis active cases (hue only, saturation only, lightness only)
    ColorMixer().with_value(3, 0, 22.5).with_value(6, 0, -18.0),
    ColorMixer().with_value(0, 1, -100).with_value(2, 1, 100),
    ColorMixer().with_value(1, 2, -85).with_value(5, 2, 95),
    # All 8 bands active at extreme limits
    ColorMixer((
        (30.0, 100.0, -100.0),
        (-30.0, -100.0, 100.0),
        (15.0, 50.0, -50.0),
        (-15.0, -50.0, 50.0),
        (29.5, -99.5, 99.5),
        (-29.5, 99.5, -99.5),
        (5.0, -12.0, 34.0),
        (-5.0, 12.0, -34.0),
    )),
]


@pytest.mark.parametrize("mixer", MIXER_CASES)
@pytest.mark.parametrize("with_scratch", [False, True])
def test_color_mixer_bit_identical_to_reference(mixer, with_scratch):
    source = boundary_rgb_fixture()
    expected = source.copy()
    actual = source.copy()
    reference_apply_color_mixer(expected, mixer)
    scratch = Scratch() if with_scratch else None
    apply_color_mixer(actual, mixer, scratch=scratch)
    assert_bit_identical_float32(actual, expected)
    np.testing.assert_array_equal(quantize_srgb(actual, bits=8), quantize_srgb(expected, bits=8))
    np.testing.assert_array_equal(quantize_srgb(actual, bits=16), quantize_srgb(expected, bits=16))


def test_color_mixer_scratch_reuse_across_varying_shapes():
    scratch = Scratch()
    mixer = MIXER_CASES[-1]
    rng = np.random.default_rng(99)
    for shape in ((1, 1, 3), (128, 333, 3), (7, 19, 3), (128, 512, 3), (3, 5, 3)):
        block = rng.random(shape, dtype=np.float32)
        expected = block.copy()
        actual = block.copy()
        reference_apply_color_mixer(expected, mixer)
        apply_color_mixer(actual, mixer, scratch=scratch)
        assert_bit_identical_float32(actual, expected)


@pytest.mark.parametrize("composition,rect,target_size", [
    (Composition(), None, None),
    (Composition(crop=(0.08, 0.12, 0.92, 0.94), angle=0.0), (10, 5, 190, 130), None),
    (Composition(angle=3.25), None, None),
    (Composition(angle=-15.0), (0, 0, 180, 95), None),
    (Composition(angle=0.01), (13, 17, 211, 141), None),
    (Composition(crop=(0.08, 0.12, 0.92, 0.94), angle=3.25), (5, 7, 160, 110), None),
    (Composition(crop=(0.1, 0.15, 0.85, 0.9), angle=-7.5), None, (97, 63)),
    (Composition(angle=15.0), (0, 0, 50, 33), (25, 17)),
])
@pytest.mark.parametrize("with_scratch", [False, True])
def test_sample_composition_bit_identical_to_reference(composition, rect, target_size, with_scratch):
    rng = np.random.default_rng(2026)
    linear = rng.random((181, 263, 3), dtype=np.float32)
    linear[0, 0] = [0.0, 0.0, 0.0]
    linear[-1, -1] = [1.0, 1.0, 1.0]
    linear.setflags(write=False)
    digest = hashlib.sha256(linear.tobytes()).hexdigest()
    expected = reference_sample_composition(
        linear, composition, source_size=(526, 362), rect=rect, target_size=target_size
    )
    scratch = Scratch() if with_scratch else None
    actual = sample_composition(
        linear, composition, source_size=(526, 362), rect=rect, target_size=target_size, scratch=scratch
    )
    assert_bit_identical_float32(actual, expected)
    assert hashlib.sha256(linear.tobytes()).hexdigest() == digest


def test_sample_composition_scratch_reuse_across_strips():
    rng = np.random.default_rng(555)
    linear = rng.random((240, 360, 3), dtype=np.float32)
    linear.setflags(write=False)
    composition = Composition(crop=(0.05, 0.08, 0.95, 0.92), angle=4.75)
    ow, oh = output_size((360, 240), composition)
    scratch = Scratch()
    for top in range(0, oh, 37):
        bottom = min(oh, top + 37)
        rect = (0, top, ow, bottom)
        expected = reference_sample_composition(linear, composition, rect=rect)
        actual = sample_composition(linear, composition, rect=rect, scratch=scratch)
        assert_bit_identical_float32(actual, expected)


def test_native_preview_and_export_agreement_with_parallel_strips():
    rng = np.random.default_rng(808)
    linear = rng.random((320, 480, 3), dtype=np.float32)
    linear.setflags(write=False)
    values = EditValues(
        exposure_ev=0.35, highlights=-0.3, shadows=0.4, whites=0.15, blacks=-0.12,
        contrast=0.18, saturation=1.12, vibrance=0.22,
        rgb_curve=((0.0, 0.0), (0.3, 0.25), (0.7, 0.76), (1.0, 1.0)),
        color_mixer=MIXER_CASES[-1],
        composition=Composition(crop=(0.06, 0.09, 0.94, 0.91), angle=3.25),
    )
    size = output_size((480, 320), values.composition)
    request = DetailRequest(0, size, (0, 0, *size), step=1)
    serial = render_region(linear, values, request, workers=1)
    parallel = render_region(linear, values, request, workers=4)
    exported = quantize_srgb(render_photo(linear, values), bits=8)
    np.testing.assert_array_equal(serial, parallel)
    np.testing.assert_array_equal(parallel, exported)


@pytest.mark.parametrize("with_scratch", [False, True])
def test_render_srgb_and_render_edit_bit_identical_to_reference(with_scratch):
    from reference_pipeline import reference_render_srgb, reference_render_edit
    from photocoach.imaging import render_srgb
    source = boundary_rgb_fixture()
    source.setflags(write=False)
    scratch = Scratch() if with_scratch else None
    assert_bit_identical_float32(
        render_srgb(source, 0.35, highlights=-0.4, shadows=0.5, scratch=scratch),
        reference_render_srgb(source, 0.35, highlights=-0.4, shadows=0.5),
    )
    values = EditValues(
        exposure_ev=-0.25, highlights=-0.35, shadows=0.45,
        whites=0.22, blacks=-0.18, contrast=0.15, saturation=1.15, vibrance=0.28,
        rgb_curve=((0.0, 0.0), (0.25, 0.2), (0.75, 0.82), (1.0, 1.0)),
        color_mixer=MIXER_CASES[-1],
    )
    actual = render_edit(source, values, scratch=scratch)
    expected = reference_render_edit(source, values)
    assert_bit_identical_float32(actual, expected)
    np.testing.assert_array_equal(quantize_srgb(actual, bits=8), quantize_srgb(expected, bits=8))
    np.testing.assert_array_equal(quantize_srgb(actual, bits=16), quantize_srgb(expected, bits=16))
