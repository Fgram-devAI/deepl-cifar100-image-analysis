"""Binary-task construction over CIFAR-100 splits.

A binary task labels a chosen positive class (or set of classes) as 1 and
everything else as 0, at either the fine or coarse label level.
"""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from data.labels import LabelLevel, get_cifar100_label_names, get_label_ids
from data.loaders import Cifar100Split, Split


@dataclass(frozen=True)
class BinaryTask:
    """Materialized binary task plus the metadata needed to reproduce it."""

    images: np.ndarray  # (N, 32, 32, 3) uint8
    binary_labels: np.ndarray  # (N,) int64 in {0, 1}
    class_counts: dict[int, int]
    metadata: dict = field(default_factory=dict)


def make_binary_labels(
    labels: np.ndarray, positive_ids: Sequence[int]
) -> np.ndarray:
    """Return a 0/1 int64 array where positives are members of `positive_ids`."""
    positive_set = {int(i) for i in positive_ids}
    if not positive_set:
        return np.zeros_like(labels, dtype=np.int64)
    mask = np.isin(labels, list(positive_set))
    return mask.astype(np.int64)


def class_counts(binary_labels: np.ndarray) -> dict[int, int]:
    """Return {0: n_neg, 1: n_pos}; missing classes appear with count 0."""
    values, counts = np.unique(binary_labels, return_counts=True)
    out = {0: 0, 1: 0}
    for v, c in zip(values.tolist(), counts.tolist()):
        out[int(v)] = int(c)
    return out


def make_cifar100_binary_task(
    split_data: Cifar100Split,
    *,
    label_level: LabelLevel,
    positive_label_names: Sequence[str],
    split: Split | str | None = None,
    seed: int = 42,
) -> BinaryTask:
    """Build a binary task from a loaded CIFAR-100 split.

    The metadata block records the positive definition, negative definition,
    split, class counts, and seed so downstream code can log a reproducible
    task spec.
    """
    if label_level == "fine":
        source_labels = split_data.fine_labels
    elif label_level == "coarse":
        source_labels = split_data.coarse_labels
    else:
        raise ValueError(
            f"label_level must be 'fine' or 'coarse'; got {label_level!r}"
        )

    positive_ids = get_label_ids(label_level, positive_label_names)
    binary = make_binary_labels(source_labels, positive_ids=positive_ids)
    counts = class_counts(binary)

    metadata = {
        "label_level": label_level,
        "positive_label_names": list(positive_label_names),
        "positive_ids": positive_ids,
        "negative_definition": "all other classes",
        "split": split or split_data.split or "unknown",
        "n_total": int(binary.shape[0]),
        "class_counts": counts,
        "seed": seed,
    }
    return BinaryTask(
        images=split_data.images,
        binary_labels=binary,
        class_counts=counts,
        metadata=metadata,
    )


@dataclass(frozen=True)
class MulticlassTask:
    """Materialized multiclass task plus the metadata needed to reproduce it."""

    images: np.ndarray              # (N, 32, 32, 3) uint8
    labels: np.ndarray              # (N,) int64 in [0, num_classes)
    num_classes: int
    label_level: LabelLevel
    label_names: tuple[str, ...]
    class_counts: dict[int, int]
    metadata: dict = field(default_factory=dict)


def make_cifar100_multiclass_task(
    split_data: Cifar100Split,
    *,
    label_level: LabelLevel = "coarse",
    split: Split | str | None = None,
    seed: int = 42,
) -> MulticlassTask:
    """Build a multiclass task directly from a CIFAR-100 split.

    For label_level="coarse", labels span [0, 20); for "fine", [0, 100).
    Metadata records task_type, label_level, num_classes, label_names,
    split, and seed.
    """
    if label_level == "fine":
        source_labels = split_data.fine_labels
    elif label_level == "coarse":
        source_labels = split_data.coarse_labels
    else:
        raise ValueError(
            f"label_level must be 'fine' or 'coarse'; got {label_level!r}"
        )

    label_names = get_cifar100_label_names(label_level)
    num_classes = len(label_names)
    labels = np.asarray(source_labels, dtype=np.int64).reshape(-1)

    # Count every class index in [0, num_classes); zeros for absent classes.
    counts: dict[int, int] = {i: 0 for i in range(num_classes)}
    values, raw_counts = np.unique(labels, return_counts=True)
    for v, c in zip(values.tolist(), raw_counts.tolist()):
        counts[int(v)] = int(c)

    metadata = {
        "task_type": "multiclass",
        "label_level": label_level,
        "num_classes": int(num_classes),
        "label_names": list(label_names),
        "split": split or split_data.split or "unknown",
        "n_total": int(labels.shape[0]),
        "class_counts": counts,
        "seed": seed,
    }
    return MulticlassTask(
        images=split_data.images,
        labels=labels,
        num_classes=int(num_classes),
        label_level=label_level,
        label_names=tuple(label_names),
        class_counts=counts,
        metadata=metadata,
    )
