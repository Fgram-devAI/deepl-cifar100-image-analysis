"""CIFAR-100 dataset loading. Downloads happen only inside `load_cifar100`."""

import argparse
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

import numpy as np
import tensorflow as tf


Split = Literal["train", "test"]
Source = Literal["huggingface", "keras"]


@dataclass(frozen=True)
class Cifar100Split:
    """A CIFAR-100 split with both label levels.

    images: (N, 32, 32, 3) uint8
    fine_labels: (N,) int64 in [0, 100)
    coarse_labels: (N,) int64 in [0, 20)
    split: train/test split name, if known
    """

    images: np.ndarray
    fine_labels: np.ndarray
    coarse_labels: np.ndarray
    split: Optional[Split] = None


def load_cifar100(
    split: Split = "train",
    *,
    source: Source = "huggingface",
    _loader: Optional[Callable] = None,
) -> Cifar100Split:
    """Load a CIFAR-100 split with both fine and coarse labels.

    Args:
        split: ``"train"`` or ``"test"``.
        source: Dataset backend. ``"huggingface"`` uses
            ``uoft-cs/cifar100``; ``"keras"`` uses TensorFlow/Keras.
        _loader: Test-only injection point. For Hugging Face, it receives
            the split name and returns row records with ``img``,
            ``fine_label``, and ``coarse_label``. For Keras, it has the same
            signature as ``tf.keras.datasets.cifar100.load_data``.

    Hugging Face caches through the ``datasets`` cache. Keras caches to
    ``~/.keras/datasets/``. Nothing in this module triggers a download at
    import time.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test'; got {split!r}")
    if source == "huggingface":
        return _load_cifar100_huggingface(split, _loader=_loader)
    if source == "keras":
        return _load_cifar100_keras(split, _loader=_loader)
    raise ValueError(f"source must be 'huggingface' or 'keras'; got {source!r}")


def _load_cifar100_keras(
    split: Split,
    *,
    _loader: Optional[Callable] = None,
) -> Cifar100Split:
    """Load CIFAR-100 from the TensorFlow/Keras built-in dataset backend."""
    loader = _loader if _loader is not None else tf.keras.datasets.cifar100.load_data
    # Keras exposes fine and coarse labels through separate label modes, so the
    # fallback backend must call the loader twice. The archive cache avoids a
    # second download, but the arrays are decoded twice.
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
        split=split,
    )


def _load_cifar100_huggingface(
    split: Split,
    *,
    _loader: Optional[Callable] = None,
) -> Cifar100Split:
    """Load CIFAR-100 from Hugging Face dataset ``uoft-cs/cifar100``."""
    if _loader is None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "Hugging Face loading requires the `datasets` package. "
                "Install it with `pip install -r requirements-hf.txt`."
            ) from exc

        rows = load_dataset("uoft-cs/cifar100", split=split)
    else:
        rows = _loader(split)

    images: list[np.ndarray] = []
    fine_labels: list[int] = []
    coarse_labels: list[int] = []
    for row in rows:
        images.append(_image_to_array(row["img"]))
        fine_labels.append(int(row["fine_label"]))
        coarse_labels.append(int(row["coarse_label"]))

    return Cifar100Split(
        images=np.asarray(images, dtype=np.uint8),
        fine_labels=np.asarray(fine_labels, dtype=np.int64).reshape(-1),
        coarse_labels=np.asarray(coarse_labels, dtype=np.int64).reshape(-1),
        split=split,
    )


def _image_to_array(image: Any) -> np.ndarray:
    """Convert a Hugging Face image object to a 32x32 RGB uint8 array."""
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    array = np.asarray(image, dtype=np.uint8)
    if array.shape != (32, 32, 3):
        raise ValueError(f"expected image shape (32, 32, 3); got {array.shape}")
    return array


View = Literal["image", "sequence"]


def make_pipeline(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    view: View,
    batch_size: int = 32,
    shuffle: bool = False,
    cache: bool = False,
    prefetch: bool = True,
    shuffle_buffer: int = 1024,
    seed: Optional[int] = None,
) -> tf.data.Dataset:
    """Build a ``tf.data.Dataset`` yielding (images, labels) in the chosen view.

    Image view yields ``(B, 32, 32, 3)`` float32 in [0, 1]; sequence view
    yields ``(B, 32, 96)`` float32 in [0, 1]. Shuffle is deterministic when
    ``seed`` is given. Cache is in-memory and only useful for splits that
    comfortably fit.
    """
    if view not in ("image", "sequence"):
        raise ValueError(f"view must be 'image' or 'sequence'; got {view!r}")
    if images.ndim != 4 or images.shape[1:] != (32, 32, 3):
        raise ValueError(
            f"images must have shape (N, 32, 32, 3); got {images.shape}"
        )
    if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
        raise ValueError(
            f"labels must be 1-D with len == N; got {labels.shape}"
        )

    images_f32 = images.astype(np.float32, copy=False) / 255.0
    if view == "sequence":
        images_f32 = images_f32.reshape(images_f32.shape[0], 32, 96)

    labels_i64 = labels.astype(np.int64, copy=False)

    ds = tf.data.Dataset.from_tensor_slices((images_f32, labels_i64))
    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(
            buffer_size=shuffle_buffer,
            seed=seed,
            reshuffle_each_iteration=False,
        )
    ds = ds.batch(batch_size, drop_remainder=False)
    if prefetch:
        ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _print_split_summary(split: Split, source: Source) -> None:
    """Load a split and print a compact shape/label summary."""
    data = load_cifar100(split, source=source)
    print(
        f"{source}/{split}: images={data.images.shape} "
        f"fine={data.fine_labels.shape} coarse={data.coarse_labels.shape} "
        f"fine_range=({data.fine_labels.min()}, {data.fine_labels.max()}) "
        f"coarse_range=({data.coarse_labels.min()}, {data.coarse_labels.max()})"
    )


def main() -> None:
    """CLI entrypoint for fetching/caching CIFAR-100 and checking split shapes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("train", "test", "both"),
        default="both",
        help="CIFAR-100 split to fetch/cache and summarize.",
    )
    parser.add_argument(
        "--source",
        choices=("huggingface", "keras"),
        default="huggingface",
        help="Dataset backend to fetch/cache from.",
    )
    args = parser.parse_args()

    splits: tuple[Split, ...] = (
        ("train", "test") if args.split == "both" else (args.split,)
    )
    for split in splits:
        _print_split_summary(split, args.source)


if __name__ == "__main__":
    main()
