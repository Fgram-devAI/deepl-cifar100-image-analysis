"""Transfer learning: MobileNetV3Small with frozen backbone and new head."""

from typing import Literal
import tensorflow as tf

keras = tf.keras
layers = tf.keras.layers


def build_transfer(
    head: str = "binary",
    num_classes: int = 2,
    freeze_backbone: bool = True,
    input_size: int = 96,
) -> keras.Model:
    """
    Build a transfer-learning model using MobileNetV3Small as backbone.

    Args:
        head: Output head type ("binary" or "multiclass").
        num_classes: Number of output classes (2 for binary, 10 for multiclass).
        freeze_backbone: If True, freeze the backbone weights.
        input_size: Input image size after resizing (default 96×96, upscaled from 32).

    Returns:
        Compiled `keras.Model`.

    Notes:
        Input shape: (batch, 32, 32, 3) — the image view.
        Architecture:
          1. Resize 32×32 → 96×96 (MobileNetV3 prefers larger inputs)
          2. MobileNetV3Small pretrained backbone
          3. Optionally freeze backbone
          4. New classification head
        MobileNetV3 is pretrained on ImageNet; performance on 32×32 is lower than on larger
        images, but this is expected. This is the transfer-learning upper benchmark.
    """
    # TODO: Implement transfer model.
    # 1. Input: (batch, 32, 32, 3)
    # 2. Resize to (input_size, input_size, 3)
    # 3. Load MobileNetV3Small with ImageNet weights, include_top=False
    # 4. Optionally freeze backbone
    # 5. Global average pooling
    # 6. Dense layers + classification head (sigmoid or softmax)
    raise NotImplementedError("build_transfer not yet implemented")
