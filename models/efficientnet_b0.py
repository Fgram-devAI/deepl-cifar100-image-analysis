"""EfficientNetB0 transfer-learning model for CIFAR-100 multiclass tasks."""

import tensorflow as tf

from models.augmentation import build_augmentation

keras = tf.keras
layers = tf.keras.layers


def build_efficientnet_b0(
    input_shape: tuple[int, int, int] = (32, 32, 3),
    num_classes: int = 20,
    dropout: float = 0.3,
    freeze_backbone: bool = True,
    input_size: int = 96,
    augmentation: dict | None = None,
) -> keras.Model:
    """Build an EfficientNetB0 transfer-learning classifier for CIFAR-100.

    Resizes 32x32 -> (input_size, input_size), rescales back to [0, 255]
    (EfficientNetB0 has its own internal Rescaling/Normalization layers
    that expect raw pixel values, unlike most keras.applications models),
    runs the ImageNet-pretrained backbone (optionally frozen), then a
    dropout + softmax head sized to num_classes. Returned uncompiled.
    """
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1; got {num_classes}")

    inputs = keras.Input(shape=input_shape, name="image")

    aug_layer = build_augmentation(augmentation)
    x = aug_layer(inputs) if aug_layer is not None else inputs

    x = layers.Resizing(input_size, input_size, interpolation="bilinear")(x)
    x = layers.Rescaling(255.0)(x)

    backbone = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(input_size, input_size, 3),
        pooling="avg",
    )
    backbone.trainable = not freeze_backbone

    x = backbone(x, training=not freeze_backbone)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="prob")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="efficientnet_b0")
