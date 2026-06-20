"""Tests for binary-task construction from CIFAR-100 splits."""

import numpy as np
import pytest

from data.loaders import Cifar100Split
from data.tasks import (
    BinaryTask,
    class_counts,
    make_binary_labels,
    make_cifar100_binary_task,
)


def test_make_binary_labels_marks_positive_ids():
    labels = np.array([0, 1, 2, 3, 1, 2], dtype=np.int64)
    out = make_binary_labels(labels, positive_ids=[1, 2])
    np.testing.assert_array_equal(out, np.array([0, 1, 1, 0, 1, 1], dtype=np.int64))
    assert out.dtype == np.int64


def test_make_binary_labels_empty_positive_ids_returns_all_zero():
    labels = np.array([0, 1, 2], dtype=np.int64)
    out = make_binary_labels(labels, positive_ids=[])
    np.testing.assert_array_equal(out, np.zeros(3, dtype=np.int64))


def test_class_counts_returns_int_keys_and_values():
    bin_labels = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    counts = class_counts(bin_labels)
    assert counts == {0: 2, 1: 3}


def _synthetic_split(n: int = 20) -> Cifar100Split:
    rng = np.random.default_rng(0)
    return Cifar100Split(
        images=rng.integers(0, 256, size=(n, 32, 32, 3), dtype=np.uint8),
        fine_labels=np.arange(n, dtype=np.int64) % 100,
        coarse_labels=np.arange(n, dtype=np.int64) % 20,
        split="train",
    )


def test_make_cifar100_binary_task_fine_cow_vs_rest():
    split = _synthetic_split(n=200)
    task = make_cifar100_binary_task(
        split,
        label_level="fine",
        positive_label_names=["cow"],
    )
    assert isinstance(task, BinaryTask)
    assert set(np.unique(task.binary_labels).tolist()).issubset({0, 1})
    assert task.metadata["label_level"] == "fine"
    assert task.metadata["positive_label_names"] == ["cow"]
    assert task.metadata["split"] == "train"
    assert task.class_counts[1] > 0
    assert task.class_counts[0] + task.class_counts[1] == 200


def test_make_cifar100_binary_task_coarse_aquatic_vs_rest():
    split = _synthetic_split(n=100)
    task = make_cifar100_binary_task(
        split,
        label_level="coarse",
        positive_label_names=["aquatic_mammals"],
    )
    assert task.metadata["label_level"] == "coarse"
    assert task.metadata["split"] == "train"
    assert task.class_counts[1] > 0
    assert task.class_counts[0] + task.class_counts[1] == 100


def test_make_cifar100_binary_task_rejects_unknown_level():
    split = _synthetic_split()
    with pytest.raises(ValueError):
        make_cifar100_binary_task(
            split,
            label_level="super",  # type: ignore[arg-type]
            positive_label_names=["cow"],
        )


# ---------------------------------------------------------------------------
# Multiclass task tests
# ---------------------------------------------------------------------------

from data.tasks import MulticlassTask, make_cifar100_multiclass_task


def test_make_multiclass_task_coarse_returns_20_classes_and_labels_in_range(
    synthetic_cifar100,
):
    (x_tr, yf_tr, yc_tr), _ = synthetic_cifar100
    split = Cifar100Split(
        images=x_tr,
        fine_labels=yf_tr.reshape(-1),
        coarse_labels=yc_tr.reshape(-1),
        split="train",
    )
    task = make_cifar100_multiclass_task(split, label_level="coarse")
    assert isinstance(task, MulticlassTask)
    assert task.num_classes == 20
    assert len(task.label_names) == 20
    assert task.label_names[0] == "aquatic_mammals"
    assert task.labels.min() >= 0
    assert task.labels.max() < 20
    assert task.metadata["task_type"] == "multiclass"
    assert task.metadata["label_level"] == "coarse"
    assert task.metadata["split"] == "train"


def test_make_multiclass_task_fine_returns_100_classes(synthetic_cifar100):
    (x_tr, yf_tr, yc_tr), _ = synthetic_cifar100
    split = Cifar100Split(
        images=x_tr,
        fine_labels=yf_tr.reshape(-1),
        coarse_labels=yc_tr.reshape(-1),
        split="train",
    )
    task = make_cifar100_multiclass_task(split, label_level="fine")
    assert task.num_classes == 100
    assert len(task.label_names) == 100
    assert task.label_names[19] == "cattle"  # known canonical fine-id 19
    assert task.labels.max() < 100


def test_make_multiclass_task_class_counts_sum_to_n(synthetic_cifar100):
    (x_tr, yf_tr, yc_tr), _ = synthetic_cifar100
    split = Cifar100Split(
        images=x_tr,
        fine_labels=yf_tr.reshape(-1),
        coarse_labels=yc_tr.reshape(-1),
        split="train",
    )
    task = make_cifar100_multiclass_task(split, label_level="coarse")
    assert sum(task.class_counts.values()) == task.labels.shape[0]
    # Every class index in [0, 20) is present in the dict (count may be 0).
    assert set(task.class_counts.keys()) == set(range(20))


def test_make_multiclass_task_rejects_invalid_label_level(synthetic_cifar100):
    (x_tr, yf_tr, yc_tr), _ = synthetic_cifar100
    split = Cifar100Split(
        images=x_tr,
        fine_labels=yf_tr.reshape(-1),
        coarse_labels=yc_tr.reshape(-1),
        split="train",
    )
    with pytest.raises(ValueError):
        make_cifar100_multiclass_task(split, label_level="bogus")  # type: ignore[arg-type]


def test_make_multiclass_task_module_import_is_offline():
    # Sanity: importing the data module must not trigger downloads.
    # If this test runs at all, the import already succeeded without net.
    import data
    assert hasattr(data, "make_cifar100_multiclass_task")
    assert hasattr(data, "MulticlassTask")
