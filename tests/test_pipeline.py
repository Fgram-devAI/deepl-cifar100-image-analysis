"""Tests for data.loaders.make_pipeline view contract."""

import numpy as np
import pytest
import tensorflow as tf

from data.loaders import make_pipeline


@pytest.fixture
def small_arrays():
    images = (np.arange(8 * 32 * 32 * 3, dtype=np.uint8) % 256).reshape(
        8, 32, 32, 3
    )
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    return images, labels


def test_image_view_batch_shape(small_arrays):
    images, labels = small_arrays
    ds = make_pipeline(images, labels, view="image", batch_size=4)
    x, y = next(iter(ds))
    assert tuple(x.shape) == (4, 32, 32, 3)
    assert tuple(y.shape) == (4,)
    assert x.dtype == tf.float32
    assert y.dtype == tf.int64


def test_image_view_pixels_normalized_to_unit_range(small_arrays):
    images, labels = small_arrays
    ds = make_pipeline(images, labels, view="image", batch_size=4)
    x, _ = next(iter(ds))
    assert float(tf.reduce_min(x).numpy()) >= 0.0
    assert float(tf.reduce_max(x).numpy()) <= 1.0


def test_sequence_view_batch_shape(small_arrays):
    images, labels = small_arrays
    ds = make_pipeline(images, labels, view="sequence", batch_size=2)
    x, y = next(iter(ds))
    assert tuple(x.shape) == (2, 32, 96)
    assert tuple(y.shape) == (2,)
    assert x.dtype == tf.float32


def test_shuffle_with_seed_is_deterministic(small_arrays):
    images, labels = small_arrays
    a = list(
        make_pipeline(
            images, labels, view="image", batch_size=8, shuffle=True, seed=7
        )
    )
    b = list(
        make_pipeline(
            images, labels, view="image", batch_size=8, shuffle=True, seed=7
        )
    )
    np.testing.assert_array_equal(a[0][1].numpy(), b[0][1].numpy())


def test_unknown_view_raises(small_arrays):
    images, labels = small_arrays
    with pytest.raises(ValueError):
        make_pipeline(images, labels, view="other", batch_size=2)  # type: ignore[arg-type]
