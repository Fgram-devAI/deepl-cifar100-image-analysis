"""Image preprocessing: row-as-timestep conversion and masking."""

from typing import Tuple
import numpy as np
import tensorflow as tf


def to_sequence(images: np.ndarray) -> np.ndarray:
    """
    Convert images from (N, 32, 32, 3) to row-as-timestep sequence (N, 32, 96).

    Each row of the image becomes a timestep; the 32 pixels × 3 channels = 96 features.

    Args:
        images: Array of shape (N, 32, 32, 3) with values in [0, 1].

    Returns:
        Sequence array of shape (N, 32, 96).

    Notes:
        This contract is fixed per CLAUDE.md. Do not change T or feature dimension.
    """
    # TODO: Implement row-as-timestep reshaping.
    # Expected: (N, 32, 32, 3) → (N, 32, 32*3) → (N, 32, 96)
    raise NotImplementedError("to_sequence not yet implemented")


def to_image(images: np.ndarray) -> np.ndarray:
    """
    Return images in (N, 32, 32, 3) format for the CNN branch.

    May apply normalization or identity passthrough.

    Args:
        images: Array of shape (N, 32, 32, 3) with values in [0, 1].

    Returns:
        Array of shape (N, 32, 32, 3).
    """
    # TODO: Implement normalization or passthrough for CNN input.
    raise NotImplementedError("to_image not yet implemented")


def apply_row_masking(
    seq: np.ndarray, drop_prob: float = 0.0, mask_value: float = 0.0
) -> np.ndarray:
    """
    Randomly mask (replace) whole rows in a sequence to simulate missing timesteps.

    Args:
        seq: Sequence array of shape (batch, T=32, features=96).
        drop_prob: Probability of masking each row. If 0.0, no masking applied.
        mask_value: Value to fill masked rows with.

    Returns:
        Masked sequence array of shape (batch, 32, 96).

    Notes:
        **Sentinel collision gotcha (from CLAUDE.md):**
        If mask_value=0.0 and normalized pixels can legitimately be 0.0, this will
        incorrectly skip valid rows. Consider:
          1. Masking rows to 0.0 *before* normalizing to [0, 1].
          2. Using an out-of-range sentinel (e.g., -1.0) with matching Masking(mask_value).
        Document your choice in the implementation.
    """
    # TODO: Implement row masking. Generate a mask for each row in each batch,
    # and replace selected rows with mask_value.
    raise NotImplementedError("apply_row_masking not yet implemented")
