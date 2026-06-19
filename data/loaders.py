"""CIFAR-100 dataset loading. Downloads happen only inside `load_cifar100`."""

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np
import tensorflow as tf


Split = Literal["train", "test"]


@dataclass(frozen=True)
class Cifar100Split:
    """A CIFAR-100 split with both label levels.

    images: (N, 32, 32, 3) uint8
    fine_labels: (N,) int64 in [0, 100)
    coarse_labels: (N,) int64 in [0, 20)
    """

    images: np.ndarray
    fine_labels: np.ndarray
    coarse_labels: np.ndarray


def load_cifar100(
    split: Split = "train",
    *,
    _loader: Optional[Callable] = None,
) -> Cifar100Split:
    """Load a CIFAR-100 split with both fine and coarse labels.

    Args:
        split: ``"train"`` or ``"test"``.
        _loader: Test-only injection point with the same signature as
            ``tf.keras.datasets.cifar100.load_data(label_mode=...)``.

    The Keras loader caches to ``~/.keras/datasets/`` and only downloads on
    first call. Nothing in this module triggers a download at import time.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test'; got {split!r}")

    loader = _loader if _loader is not None else tf.keras.datasets.cifar100.load_data
    (x_train_f, y_train_f), (x_test_f, y_test_f) = loader(label_mode="fine")
    (_, y_train_c), (_, y_test_c) = loader(label_mode="coarse")

    if split == "train":
        images, y_fine, y_coarse = x_train_f, y_train_f, y_train_c
    else:
        images, y_fine, y_coarse = x_test_f, y_test_f, y_test_c

    return Cifar100Split(
        images=np.asarray(images, dtype=np.uint8),
        fine_labels=np.asarray(y_fine, dtype=np.int64).reshape(-1),
        coarse_labels=np.asarray(y_coarse, dtype=np.int64).reshape(-1),
    )
