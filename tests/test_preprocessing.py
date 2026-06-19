"""Unit tests for data.preprocessing primitives."""

import numpy as np
import pytest

from data.preprocessing import (
    apply_row_masking,
    normalize_images,
    to_image,
    to_sequence,
)


def test_normalize_images_scales_uint8_to_unit_float():
    x = np.full((1, 32, 32, 3), 255, dtype=np.uint8)
    x[0, 0, 0, 0] = 0
    x[0, 0, 0, 1] = 127
    out = normalize_images(x)
    assert out.dtype == np.float32
    assert out.min() == 0.0
    assert np.isclose(out.max(), 1.0)
    assert np.isclose(out[0, 0, 0, 1], 127.0 / 255.0)


def test_to_image_preserves_shape_and_returns_float32():
    x = np.zeros((4, 32, 32, 3), dtype=np.float32)
    out = to_image(x)
    assert out.shape == (4, 32, 32, 3)
    assert out.dtype == np.float32


def test_to_image_rejects_wrong_shape():
    x = np.zeros((4, 16, 16, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        to_image(x)


def test_to_sequence_reshapes_to_T32_F96():
    x = np.arange(32 * 32 * 3, dtype=np.float32).reshape(1, 32, 32, 3)
    out = to_sequence(x)
    assert out.shape == (1, 32, 96)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out[0, 0], x[0, 0].reshape(96))


def test_apply_row_masking_zero_prob_is_identity():
    seq = np.ones((2, 32, 96), dtype=np.float32) * 0.5
    out = apply_row_masking(seq, drop_prob=0.0)
    np.testing.assert_array_equal(out, seq)


def test_apply_row_masking_full_prob_masks_everything():
    seq = np.ones((2, 32, 96), dtype=np.float32) * 0.5
    out = apply_row_masking(seq, drop_prob=1.0, mask_value=-1.0, seed=0)
    assert np.all(out == -1.0)


def test_apply_row_masking_is_deterministic_with_seed():
    seq = np.ones((4, 32, 96), dtype=np.float32) * 0.5
    a = apply_row_masking(seq, drop_prob=0.3, mask_value=-1.0, seed=123)
    b = apply_row_masking(seq, drop_prob=0.3, mask_value=-1.0, seed=123)
    np.testing.assert_array_equal(a, b)


def test_apply_row_masking_rejects_wrong_shape():
    seq = np.zeros((2, 16, 96), dtype=np.float32)
    with pytest.raises(ValueError):
        apply_row_masking(seq, drop_prob=0.5)
