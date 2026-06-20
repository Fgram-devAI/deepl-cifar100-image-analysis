"""Loss functions for CIFAR-100 binary and multiclass tasks."""

import tensorflow as tf

BinaryCrossentropy = tf.keras.losses.BinaryCrossentropy
Loss = tf.keras.losses.Loss
SparseCategoricalCrossentropy = tf.keras.losses.SparseCategoricalCrossentropy


def get_loss(head: str = "binary") -> Loss:
    """Return the Keras loss for the chosen output head.

    Args:
        head: Either ``"binary"`` (sigmoid + BinaryCrossentropy) or
              ``"multiclass"`` (softmax + SparseCategoricalCrossentropy).

    Returns:
        A compiled Keras Loss instance configured with ``from_logits=False``.

    Raises:
        ValueError: If ``head`` is not ``"binary"`` or ``"multiclass"``.
    """
    if head == "binary":
        return BinaryCrossentropy(from_logits=False)
    if head == "multiclass":
        return SparseCategoricalCrossentropy(from_logits=False)
    raise ValueError(
        f"head must be 'binary' or 'multiclass'; got {head!r}"
    )
