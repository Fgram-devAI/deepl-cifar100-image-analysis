"""Baseline CNN for CIFAR-100 binary tasks (image view)."""

import tensorflow as tf

from models.augmentation import build_augmentation

keras = tf.keras
layers = tf.keras.layers


def build_baseline_cnn(
    input_shape: tuple[int, int, int] = (32, 32, 3),
    dropout: float = 0.3,
    num_classes: int = 1,
    augmentation: dict | None = None,
) -> keras.Model:
    """Build a compact 2-block CNN for image classification.

    Two conv blocks (32 then 64 filters, 3x3, ReLU), max-pool + dropout after
    each block, then a 128-unit dense head. The output head adapts to the
    task: ``num_classes=1`` yields a single sigmoid unit (binary); any
    ``num_classes > 1`` yields a softmax head with that many units
    (multiclass). The model is returned uncompiled; ``training.train``
    compiles it with the configured loss and optimizer.

    Args:
        input_shape: Spatial dimensions of the input image (H, W, C).
        dropout: Dropout rate applied after each conv block and the dense layer.
        num_classes: Number of output classes. Must be >= 1.
        augmentation: Optional augmentation config dict (from YAML
            ``augmentation:`` block). ``None`` or ``enabled: false`` disables
            augmentation entirely. Keras Random* layers are no-ops at
            inference time.

    Raises:
        ValueError: If ``num_classes`` is less than 1.
    """
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1; got {num_classes}")

    inputs = keras.Input(shape=input_shape, name="image")

    aug_layer = build_augmentation(augmentation)
    x = aug_layer(inputs) if aug_layer is not None else inputs

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D(pool_size=2)(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D(pool_size=2)(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)

    if num_classes == 1:
        outputs = layers.Dense(1, activation="sigmoid", name="prob")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax", name="prob")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="baseline_cnn")
