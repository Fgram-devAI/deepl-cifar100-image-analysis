"""Data module: CIFAR-100 loaders, label metadata, binary tasks, pipelines."""

from data.labels import (
    CIFAR100_COARSE_LABEL_NAMES,
    CIFAR100_FINE_LABEL_NAMES,
    get_cifar100_label_names,
    get_label_ids,
)
from data.loaders import Cifar100Split, load_cifar100, make_pipeline
from data.preprocessing import (
    apply_row_masking,
    normalize_images,
    to_image,
    to_sequence,
)
from data.tasks import (
    BinaryTask,
    class_counts,
    make_binary_labels,
    make_cifar100_binary_task,
)

__all__ = [
    "BinaryTask",
    "CIFAR100_COARSE_LABEL_NAMES",
    "CIFAR100_FINE_LABEL_NAMES",
    "Cifar100Split",
    "apply_row_masking",
    "class_counts",
    "get_cifar100_label_names",
    "get_label_ids",
    "load_cifar100",
    "make_binary_labels",
    "make_cifar100_binary_task",
    "make_pipeline",
    "normalize_images",
    "to_image",
    "to_sequence",
]
