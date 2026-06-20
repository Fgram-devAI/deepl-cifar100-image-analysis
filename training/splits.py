"""Deterministic train/validation splits for binary and multiclass tasks."""

import numpy as np


def simple_random_train_val_split(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split images/labels into train and validation by random shuffle.

    Unlike :func:`stratified_train_val_split`, no per-class stratification is
    applied. This is appropriate for multiclass tasks where stratifying across
    up to 100 classes is not needed.

    Args:
        images: Array of shape ``(N, H, W, C)``.
        labels: Integer label array of shape ``(N,)``.
        val_fraction: Fraction of samples to put in validation. Must be in
            ``(0, 1)``.
        seed: RNG seed for determinism.

    Returns:
        ``(x_train, y_train, x_val, y_val)`` — disjoint, covering all N rows.

    Raises:
        ValueError: If ``val_fraction`` is not in ``(0, 1)`` or arrays differ
            in length.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(
            f"val_fraction must be in (0, 1); got {val_fraction}"
        )
    labels = np.asarray(labels).reshape(-1)
    if images.shape[0] != labels.shape[0]:
        raise ValueError(
            f"images and labels length mismatch: "
            f"{images.shape[0]} vs {labels.shape[0]}"
        )

    n = images.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_val = max(1, int(round(n * val_fraction)))
    n_val = min(n_val, n - 1)  # ensure at least one train sample

    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    return (
        images[train_idx],
        labels[train_idx],
        images[val_idx],
        labels[val_idx],
    )


def stratified_train_val_split(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split images/labels into train and validation, stratified by label.

    Stratification preserves the positive/negative ratio in both splits, so
    binary metrics on the validation set remain meaningful even when the
    positive class is rare.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(
            f"val_fraction must be in (0, 1); got {val_fraction}"
        )
    labels = np.asarray(labels).reshape(-1)
    if images.shape[0] != labels.shape[0]:
        raise ValueError(
            f"images and labels length mismatch: "
            f"{images.shape[0]} vs {labels.shape[0]}"
        )

    rng = np.random.default_rng(seed)
    train_idx_parts: list[np.ndarray] = []
    val_idx_parts: list[np.ndarray] = []
    for cls in (0, 1):
        cls_idx = np.flatnonzero(labels == cls)
        rng.shuffle(cls_idx)
        n_val = int(round(cls_idx.size * val_fraction))
        if cls_idx.size > 1:
            n_val = max(1, min(n_val, cls_idx.size - 1))
        val_idx_parts.append(cls_idx[:n_val])
        train_idx_parts.append(cls_idx[n_val:])

    train_idx = np.concatenate(train_idx_parts)
    val_idx = np.concatenate(val_idx_parts)
    # Reshuffle so positives/negatives are interleaved.
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return (
        images[train_idx],
        labels[train_idx],
        images[val_idx],
        labels[val_idx],
    )
