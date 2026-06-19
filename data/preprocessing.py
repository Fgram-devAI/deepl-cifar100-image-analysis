"""Image preprocessing: normalization, row-as-timestep conversion, row masking."""

from typing import Optional

import numpy as np


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Scale uint8 images in [0, 255] to float32 in [0, 1]."""
    return (images.astype(np.float32)) / 255.0


def to_image(images: np.ndarray) -> np.ndarray:
    """Return a (N, 32, 32, 3) float32 view for CNN/transfer branches."""
    if images.ndim != 4 or images.shape[1:] != (32, 32, 3):
        raise ValueError(
            f"to_image expects shape (N, 32, 32, 3); got {images.shape}"
        )
    return images.astype(np.float32, copy=False)


def to_sequence(images: np.ndarray) -> np.ndarray:
    """Reshape (N, 32, 32, 3) images to (N, 32, 96) row-as-timestep sequences.

    Each row becomes one timestep; the 32 pixels x 3 channels per row become
    96 features.
    """
    if images.ndim != 4 or images.shape[1:] != (32, 32, 3):
        raise ValueError(
            f"to_sequence expects shape (N, 32, 32, 3); got {images.shape}"
        )
    n = images.shape[0]
    return images.astype(np.float32, copy=False).reshape(n, 32, 96)


def apply_row_masking(
    seq: np.ndarray,
    drop_prob: float = 0.0,
    mask_value: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Randomly replace whole rows in (batch, T=32, 96) sequences with mask_value.

    Sentinel-collision gotcha (CLAUDE.md §10): with mask_value=0.0 and pixels
    normalized to [0, 1], legitimately black rows look identical to masked rows.
    If a downstream Masking layer relies on the sentinel, prefer an out-of-range
    value (e.g. -1.0) and pass the same value to Masking(mask_value=...).
    """
    if seq.ndim != 3 or seq.shape[1:] != (32, 96):
        raise ValueError(
            f"apply_row_masking expects shape (batch, 32, 96); got {seq.shape}"
        )
    if not 0.0 <= drop_prob <= 1.0:
        raise ValueError(f"drop_prob must be in [0, 1]; got {drop_prob}")
    if drop_prob == 0.0:
        return seq.astype(np.float32, copy=False)
    rng = np.random.default_rng(seed)
    mask = rng.random(size=seq.shape[:2]) < drop_prob
    out = seq.astype(np.float32, copy=True)
    out[mask] = mask_value
    return out
