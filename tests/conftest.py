"""Shared pytest fixtures: seeded RNGs and a synthetic CIFAR-100 stand-in."""

from typing import Tuple

import numpy as np
import pytest
import tensorflow as tf


@pytest.fixture(autouse=True)
def _seed_everything() -> None:
    """Reset NumPy and TensorFlow seeds before every test for determinism."""
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)


@pytest.fixture
def synthetic_cifar100() -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Return a tiny shape-correct stand-in for CIFAR-100.

    Shape contract matches what `tf.keras.datasets.cifar100.load_data` returns,
    but with 200 train / 50 test samples and ids drawn from the full
    [0, 100) fine and [0, 20) coarse ranges.
    """
    rng = np.random.default_rng(0)
    x_train = rng.integers(0, 256, size=(200, 32, 32, 3), dtype=np.uint8)
    y_fine_train = rng.integers(0, 100, size=(200, 1), dtype=np.int64)
    y_coarse_train = rng.integers(0, 20, size=(200, 1), dtype=np.int64)
    x_test = rng.integers(0, 256, size=(50, 32, 32, 3), dtype=np.uint8)
    y_fine_test = rng.integers(0, 100, size=(50, 1), dtype=np.int64)
    y_coarse_test = rng.integers(0, 20, size=(50, 1), dtype=np.int64)
    return (
        (x_train, y_fine_train, y_coarse_train),
        (x_test, y_fine_test, y_coarse_test),
    )
