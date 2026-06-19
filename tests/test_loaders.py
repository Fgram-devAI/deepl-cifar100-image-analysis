"""Tests for data.loaders.load_cifar100 using an injected synthetic loader."""

import numpy as np
import pytest

from data.loaders import Cifar100Split, load_cifar100


def _make_fake_loader(synthetic_cifar100, label_mode_holder):
    train, test = synthetic_cifar100

    def fake(label_mode: str = "fine"):
        label_mode_holder.append(label_mode)
        x_train, y_fine_train, y_coarse_train = train
        x_test, y_fine_test, y_coarse_test = test
        y_train = y_fine_train if label_mode == "fine" else y_coarse_train
        y_test = y_fine_test if label_mode == "fine" else y_coarse_test
        return (x_train, y_train), (x_test, y_test)

    return fake


def test_load_cifar100_train_split_shapes(synthetic_cifar100):
    seen: list[str] = []
    loader = _make_fake_loader(synthetic_cifar100, seen)
    out = load_cifar100("train", _loader=loader)
    assert isinstance(out, Cifar100Split)
    assert out.images.shape == (200, 32, 32, 3)
    assert out.images.dtype == np.uint8
    assert out.fine_labels.shape == (200,)
    assert out.coarse_labels.shape == (200,)
    assert "fine" in seen and "coarse" in seen


def test_load_cifar100_test_split_shapes(synthetic_cifar100):
    loader = _make_fake_loader(synthetic_cifar100, [])
    out = load_cifar100("test", _loader=loader)
    assert out.images.shape == (50, 32, 32, 3)
    assert out.fine_labels.shape == (50,)
    assert out.coarse_labels.shape == (50,)


def test_load_cifar100_rejects_unknown_split(synthetic_cifar100):
    loader = _make_fake_loader(synthetic_cifar100, [])
    with pytest.raises(ValueError):
        load_cifar100("val", _loader=loader)  # type: ignore[arg-type]


def test_labels_are_1d_int64(synthetic_cifar100):
    loader = _make_fake_loader(synthetic_cifar100, [])
    out = load_cifar100("train", _loader=loader)
    assert out.fine_labels.ndim == 1
    assert out.coarse_labels.ndim == 1
    assert out.fine_labels.dtype == np.int64
    assert out.coarse_labels.dtype == np.int64


@pytest.mark.slow
def test_load_cifar100_real_download_train_shapes():
    """Opt-in test that hits the Keras CIFAR-100 download. Run with `-m slow`."""
    out = load_cifar100("train")
    assert out.images.shape == (50000, 32, 32, 3)
    assert out.fine_labels.shape == (50000,)
    assert out.coarse_labels.shape == (50000,)
    assert int(out.fine_labels.max()) == 99
    assert int(out.coarse_labels.max()) == 19
