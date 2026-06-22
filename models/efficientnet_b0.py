"""EfficientNetB0 transfer-learning model builder for CIFAR-100."""

import tensorflow as tf

from models.augmentation import build_augmentation

keras = tf.keras
layers = tf.keras.layers


def build_efficientnet_b0(
    input_shape: tuple[int, int, int] = (32, 32, 3),
    num_classes: int = 20,
    dropout: float = 0.3,
    trainable_backbone: bool = False,
    resize_to: int = 96,
    weights: str | None = "imagenet",
    augmentation: dict | None = None,
) -> keras.Model:
    """Build an EfficientNetB0 transfer-learning classifier.

    The upstream image pipeline yields CIFAR-100 images as float32 values in
    ``[0, 1]``. Keras EfficientNet includes its own preprocessing layers and
    expects raw pixel-like values, so the graph resizes and rescales back to
    ``[0, 255]`` before the backbone.
    """
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1; got {num_classes}")
    if resize_to < 32:
        raise ValueError(f"resize_to must be >= 32; got {resize_to}")

    inputs = keras.Input(shape=input_shape, name="image")

    aug_layer = build_augmentation(augmentation)
    x = aug_layer(inputs) if aug_layer is not None else inputs

    x = layers.Resizing(
        resize_to,
        resize_to,
        interpolation="bilinear",
        name="resize",
    )(x)
    x = layers.Rescaling(255.0, name="rescale_to_0_255")(x)

    backbone = keras.applications.EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=(resize_to, resize_to, 3),
        pooling="avg",
    )
    backbone.trainable = trainable_backbone

    # Keep BatchNorm statistics stable during frozen and fine-tune passes.
    x = backbone(x, training=False)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    if num_classes == 1:
        outputs = layers.Dense(1, activation="sigmoid", name="prob")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax", name="prob")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="efficientnet_b0_transfer")
