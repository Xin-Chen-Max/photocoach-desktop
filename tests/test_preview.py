"""Float preview reduction preserves linear-light data, including sub-byte steps."""

from __future__ import annotations

import numpy as np
import pytest

from photocoach.preview import reduce_linear


def test_preview_averages_linear_samples_without_eight_bit_intermediate() -> None:
    image = np.array([
        [[0.0, 0.0001, 0.25], [1.0, 0.0003, 0.75]],
        [[0.0, 0.0001, 0.25], [1.0, 0.0003, 0.75]],
    ], dtype=np.float32)
    original = image.copy()
    image.setflags(write=False)
    small = reduce_linear(image, max_edge=1)
    assert small.dtype == np.float32 and small.shape == (1, 1, 3)
    np.testing.assert_allclose(small, [[[0.5, 0.0002, 0.5]]], rtol=1e-6)
    np.testing.assert_array_equal(image, original)
    assert not small.flags.writeable
    # A linear 50% average becomes ~0.735 sRGB later; the preview reducer
    # must retain 0.5 here rather than averaging already encoded pixels.
    assert float(small[0, 0, 0]) == 0.5


def test_preview_keeps_hdr_linear_values_and_aspect_ratio() -> None:
    image = np.full((4, 8, 3), [-0.05, 0.123456, 2.5], dtype=np.float32)
    small = reduce_linear(image, max_edge=4)
    assert small.shape == (2, 4, 3)
    np.testing.assert_allclose(small, np.broadcast_to([-0.05, 0.123456, 2.5], small.shape), rtol=1e-6)


def test_preview_does_not_upscale_or_copy_already_small_immutable_data() -> None:
    image = np.zeros((3, 4, 3), dtype=np.float32)
    image.setflags(write=False)
    assert reduce_linear(image, max_edge=10) is image


@pytest.mark.parametrize("image", [
    np.zeros((2, 2, 3), dtype=np.uint8),
    np.zeros((2, 2, 4), dtype=np.float32),
    np.zeros((0, 2, 3), dtype=np.float32),
    np.zeros((2, 2), dtype=np.float32),
    np.full((2, 2, 3), np.nan, dtype=np.float32),
])
def test_preview_rejects_invalid_images_before_noop_or_resize(image) -> None:
    with pytest.raises(ValueError):
        reduce_linear(image, max_edge=10)


@pytest.mark.parametrize("max_edge", [0, -1, 2.5, float("nan"), True])
def test_preview_rejects_invalid_dimensions(max_edge) -> None:
    with pytest.raises(ValueError):
        reduce_linear(np.ones((2, 2, 3), dtype=np.float32), max_edge=max_edge)
