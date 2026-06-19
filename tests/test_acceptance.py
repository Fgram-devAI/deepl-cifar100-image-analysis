"""End-to-end acceptance test for the CIFAR-100 data pipeline.

Walks through the spec's acceptance criteria with a synthetic CIFAR-100
stand-in (no network). Verifies:
  - image batch shape (B, 32, 32, 3)
  - sequence batch shape (B, 32, 96)
  - binary labels are 0/1
  - class counts are visible for the selected task
"""

import numpy as np

from data import (
    BinaryTask,
    Cifar100Split,
    load_cifar100,
    make_cifar100_binary_task,
    make_pipeline,
)


def _fake_loader_factory(synthetic_cifar100):
    train, test = synthetic_cifar100

    def fake(label_mode: str = "fine"):
        x_tr, yf_tr, yc_tr = train
        x_te, yf_te, yc_te = test
        y_tr = yf_tr if label_mode == "fine" else yc_tr
        y_te = yf_te if label_mode == "fine" else yc_te
        return (x_tr, y_tr), (x_te, y_te)

    return fake


def test_acceptance_fine_class_image_view(synthetic_cifar100):
    loader = _fake_loader_factory(synthetic_cifar100)
    split = load_cifar100("train", source="keras", _loader=loader)
    assert isinstance(split, Cifar100Split)

    task = make_cifar100_binary_task(
        split, label_level="fine", positive_label_names=["cow"]
    )
    assert isinstance(task, BinaryTask)
    assert set(np.unique(task.binary_labels).tolist()).issubset({0, 1})
    assert task.metadata["split"] == "train"
    assert task.class_counts.keys() == {0, 1}
    assert task.class_counts[0] + task.class_counts[1] == task.binary_labels.shape[0]

    ds = make_pipeline(
        task.images, task.binary_labels, view="image", batch_size=8
    )
    x, y = next(iter(ds))
    assert tuple(x.shape) == (8, 32, 32, 3)
    assert tuple(y.shape) == (8,)


def test_acceptance_coarse_superclass_sequence_view(synthetic_cifar100):
    loader = _fake_loader_factory(synthetic_cifar100)
    split = load_cifar100("train", source="keras", _loader=loader)

    task = make_cifar100_binary_task(
        split,
        label_level="coarse",
        positive_label_names=["aquatic_mammals"],
    )
    assert set(np.unique(task.binary_labels).tolist()).issubset({0, 1})
    assert task.metadata["split"] == "train"
    assert task.class_counts[0] + task.class_counts[1] == task.binary_labels.shape[0]

    ds = make_pipeline(
        task.images, task.binary_labels, view="sequence", batch_size=4
    )
    x, y = next(iter(ds))
    assert tuple(x.shape) == (4, 32, 96)
    assert tuple(y.shape) == (4,)
