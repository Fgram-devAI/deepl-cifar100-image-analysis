"""EfficientNetB3 transfer-learning model for CIFAR-100 multiclass tasks."""

import tensorflow as tf

from models.augmentation import build_augmentation

keras = tf.keras
layers = tf.keras.layers

# EfficientNetB3 backbone has 384 layers total (vs B0's 237, B7's 813),
# grouped into 7 MBConv blocks plus stem/top. Layer-index where each block
# starts (measured at input_size=160; stable across input sizes since it's
# topology, not shape, that determines block boundaries):
#   stem -> 3   block1 -> 7   block2 -> 29   block3 -> 73   block4 -> 117
#   block5 -> 191   block6 -> 264   block7 -> 353   top -> 381   (384 total)
# unfreeze_from="block6" leaves the first ~69% frozen and trains the last
# ~120 layers (block6, block7, top) — same sweet-spot ratio as B0/B7's
# block6 configs (~30% of the network trainable).


def build_efficientnet_b3(
    input_shape: tuple[int, int, int] = (32, 32, 3),
    num_classes: int = 100,
    dropout: float = 0.4,
    freeze_backbone: bool = True,
    unfreeze_from: str | None = None,   # None | "all" | "block5" | "block6" | "block7"
    freeze_bn: bool = True,             # keep BatchNorm stats frozen (recommended)
    input_size: int = 160,              # B3's native res is 300; 160 is a practical CIFAR compromise
    augmentation: dict | None = None,
) -> keras.Model:
    """EfficientNetB3 classifier for CIFAR-100.

    Same structure/semantics as build_efficientnet_b0 / build_efficientnet_b7.
    """
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1; got {num_classes}")

    inputs = keras.Input(shape=input_shape, name="image")

    aug_layer = build_augmentation(augmentation)
    x = aug_layer(inputs) if aug_layer is not None else inputs

    x = layers.Resizing(input_size, input_size, interpolation="bilinear")(x)
    x = layers.Rescaling(255.0)(x)

    backbone = keras.applications.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(input_size, input_size, 3),
        pooling="avg",
    )

    if freeze_backbone:
        backbone.trainable = False
    else:
        backbone.trainable = True
        if unfreeze_from not in (None, "all"):
            trainable = False
            for layer in backbone.layers:
                if layer.name.startswith(unfreeze_from):
                    trainable = True
                layer.trainable = trainable
        if freeze_bn:
            for layer in backbone.layers:
                if isinstance(layer, layers.BatchNormalization):
                    layer.trainable = False

    backbone_is_fully_frozen = freeze_backbone
    x = backbone(x, training=not backbone_is_fully_frozen)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="prob")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="efficientnet_b3")
