"""Dataset loaders for CIFAKE and OOD Midjourney sets."""

from typing import Optional, Tuple, Literal
import tensorflow as tf


def load_cifake(
    subset_size: Optional[int] = None,
    seed: int = 42,
) -> tf.data.Dataset:
    """
    Load CIFAKE dataset with optional balanced subsampling.

    Args:
        subset_size: If set, return a balanced subset of this size. If None, return full set.
        seed: Random seed for reproducible subsampling.

    Returns:
        `tf.data.Dataset` yielding (image, label) tuples where:
        - image: (32, 32, 3) uint8 or float32 (depends on preprocessing).
        - label: 0 (FAKE) or 1 (REAL) for binary; or 0-9 for CIFAR-10 class if multiclass.

    Notes:
        Full CIFAKE is ~120k images. Ablation runs use balanced subset (15-25k).
        Seed is set for determinism.
    """
    # TODO: Implement loader for CIFAKE from data/raw/cifake/.
    # Load REAL/ and FAKE/ subdirectories as class 1 and 0.
    # If subset_size is set, balance and subsample.
    # Return tf.data.Dataset.
    raise NotImplementedError("load_cifake not yet implemented")


def load_mj_ood() -> tf.data.Dataset:
    """
    Load Midjourney v6 OOD test set.

    Returns:
        `tf.data.Dataset` yielding (image, label) tuples (32, 32, 3) and labels.

    Notes:
        This set is **eval-only**. Never train, validate, or tune on it.
        Used to measure train-to-OOD generalization gap for binary models.
    """
    # TODO: Implement loader for Midjourney OOD set from data/raw/midjourney_ood/.
    # Assert/comment that this is eval-only.
    raise NotImplementedError("load_mj_ood not yet implemented")


def make_pipeline(
    dataset: tf.data.Dataset,
    batch_size: int = 32,
    view: Literal["sequence", "image"] = "sequence",
    apply_masking: bool = False,
    drop_prob: float = 0.1,
    shuffle: bool = True,
    prefetch: bool = True,
) -> tf.data.Dataset:
    """
    Build a tf.data pipeline from a raw dataset.

    Args:
        dataset: Raw dataset yielding (image, label).
        batch_size: Batch size.
        view: "sequence" for row-as-timestep (N, 32, 96), "image" for (N, 32, 32, 3).
        apply_masking: If True, apply row masking (only for "sequence" view).
        drop_prob: Row masking probability.
        shuffle: Whether to shuffle the dataset.
        prefetch: Whether to prefetch for performance.

    Returns:
        Batched, optionally shuffled and prefetched `tf.data.Dataset`.

    Notes:
        - Sequential models consume "sequence" view.
        - CNN/transfer models consume "image" view.
        - Masking is only meaningful for sequential models.
    """
    # TODO: Implement pipeline construction.
    # Apply view transformation (to_sequence or to_image).
    # Optionally apply masking.
    # Batch, shuffle (if requested), and prefetch.
    raise NotImplementedError("make_pipeline not yet implemented")
