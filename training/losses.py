"""Loss functions for binary and multiclass tasks."""

from typing import Literal
import tensorflow as tf
from tensorflow.keras.losses import Loss


def get_loss(head: str = "binary") -> Loss:
    """
    Get the appropriate loss function for the output head.

    Args:
        head: "binary" or "multiclass".

    Returns:
        A `keras.Loss` instance.

    Notes:
        - Binary: sigmoid + BinaryCrossentropy
        - Multiclass (10-class): softmax + SparseCategoricalCrossentropy or CategoricalCrossentropy
    """
    # TODO: Implement loss selection.
    # Binary: BinaryCrossentropy (from_logits=False if already sigmoid-applied in model)
    # Multiclass: SparseCategoricalCrossentropy or CategoricalCrossentropy
    # Return the selected loss.
    raise NotImplementedError("get_loss not yet implemented")
