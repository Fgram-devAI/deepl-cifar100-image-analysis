"""Tests for CIFAR-100 label-name metadata and lookups."""

import pytest

from data.labels import (
    CIFAR100_COARSE_LABEL_NAMES,
    CIFAR100_FINE_LABEL_NAMES,
    get_cifar100_label_names,
    get_label_ids,
)


def test_fine_label_names_length_and_uniqueness():
    assert len(CIFAR100_FINE_LABEL_NAMES) == 100
    assert len(set(CIFAR100_FINE_LABEL_NAMES)) == 100


def test_coarse_label_names_length_and_uniqueness():
    assert len(CIFAR100_COARSE_LABEL_NAMES) == 20
    assert len(set(CIFAR100_COARSE_LABEL_NAMES)) == 20


def test_known_fine_names_present():
    # `cattle` is the canonical Keras name; `cow` is exposed via the synonym map.
    assert "cattle" in CIFAR100_FINE_LABEL_NAMES
    assert "apple" in CIFAR100_FINE_LABEL_NAMES
    assert "orchid" in CIFAR100_FINE_LABEL_NAMES
    assert get_label_ids("fine", ["cow"]) == get_label_ids("fine", ["cattle"])


def test_known_coarse_names_present():
    for name in ("aquatic_mammals", "vehicles_1", "large_carnivores"):
        assert name in CIFAR100_COARSE_LABEL_NAMES


def test_get_cifar100_label_names_dispatches_by_level():
    assert get_cifar100_label_names("fine") == CIFAR100_FINE_LABEL_NAMES
    assert get_cifar100_label_names("coarse") == CIFAR100_COARSE_LABEL_NAMES


def test_get_cifar100_label_names_rejects_unknown_level():
    with pytest.raises(ValueError):
        get_cifar100_label_names("super")  # type: ignore[arg-type]


def test_get_label_ids_returns_correct_indices():
    cow_id = CIFAR100_FINE_LABEL_NAMES.index("cattle")
    assert get_label_ids("fine", ["cow"]) == [cow_id]


def test_get_label_ids_rejects_unknown_name():
    with pytest.raises(KeyError):
        get_label_ids("fine", ["not_a_real_class"])
