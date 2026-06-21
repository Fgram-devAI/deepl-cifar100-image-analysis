
"""EfficientNetB0 transfer-learning model for CIFAR-100 multiclass tasks."""

import tensorflow as tf

from models.augmentation import build_augmentation

keras = tf.keras
layers = tf.keras.layers

# EfficientNetB0 backbone has 237 layers total, grouped into 7 MBConv blocks
# (block1..block7) plus a stem and a top. Layer-index where each block starts
# (measured at input_size=128; stable across input sizes since it's topology,
# not shape, that determines block boundaries):
#   stem -> 3   block1 -> 7   block2 -> 17   block3 -> 46   block4 -> 75
#   block5 -> 119   block6 -> 162   block7 -> 221   top -> 234   (237 total)
# So unfreeze_from="block6" leaves the first ~68% of the network frozen and
# trains only the last ~75 layers (block6, block7, top) — usually the best
# accuracy/stability tradeoff for a single-stage fine-tune.


def build_efficientnet_b0(
    input_shape: tuple[int, int, int] = (32, 32, 3),
    num_classes: int = 100,
    dropout: float = 0.4,
    freeze_backbone: bool = True,
    unfreeze_from: str | None = None,   # None | "all" | "block5" | "block6" | "block7" | "top"
    freeze_bn: bool = True,             # keep BatchNorm stats frozen (recommended)
    input_size: int = 128,
    augmentation: dict | None = None,
) -> keras.Model:
    """EfficientNetB0 classifier for CIFAR-100.

    32x32 -> resize(input_size) -> rescale to [0,255] (EfficientNet has its own
    internal normalization) -> ImageNet backbone -> dropout + softmax head.
    Returned uncompiled.

    Backbone freezing modes:
      - freeze_backbone=True: everything frozen (fastest, weakest).
      - freeze_backbone=False, unfreeze_from=None or "all": everything
        trainable (strongest ceiling, but riskiest single-stage — needs a
        very low learning_rate, e.g. 1e-5).
      - freeze_backbone=False, unfreeze_from="block6" (etc.): only layers
        from that block onward are trainable; everything before it stays
        frozen. This is the recommended middle ground — adapts the
        high-level semantic features without disturbing low-level filters,
        and tolerates a higher learning_rate (~1e-4).

    freeze_bn=True additionally keeps every BatchNormalization layer frozen
    (non-trainable + inference-mode stats) even when its block is unfrozen.
    This avoids corrupting running mean/variance on small batches — almost
    always what you want when fine-tuning a pretrained backbone.
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

    if freeze_backbone:
        backbone.trainable = False
    else:
        backbone.trainable = True
        if unfreeze_from not in (None, "all"):
            # Freeze everything before `unfreeze_from`; train from there on.
            trainable = False
            for layer in backbone.layers:
                if layer.name.startswith(unfreeze_from):
                    trainable = True
                layer.trainable = trainable
        if freeze_bn:
            for layer in backbone.layers:
                if isinstance(layer, layers.BatchNormalization):
                    layer.trainable = False

    # Run in inference mode (training=False) whenever the backbone is fully
    # frozen, so BN uses its pretrained running stats rather than batch stats.
    backbone_is_fully_frozen = freeze_backbone
    x = backbone(x, training=not backbone_is_fully_frozen)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="prob")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="efficientnet_b0")
