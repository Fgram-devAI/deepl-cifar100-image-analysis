"""ResNet-family transfer-learning model builder for CIFAR-100.

Wraps `tf.keras.applications.resnet_v2` and `tf.keras.applications.densenet`
backbones with a small classification head sized for CIFAR-100 binary or
multiclass tasks. Preprocessing (Rescaling + Resizing + family-specific
`preprocess_input`) lives inside the model graph so saved weights and
prediction reloads use the same transform.
"""

from typing import Callable

import tensorflow as tf

keras = tf.keras
layers = tf.keras.layers


_BACKBONE_REGISTRY: dict[str, tuple[Callable[..., keras.Model], Callable]] = {
    "resnet50v2": (
        tf.keras.applications.ResNet50V2,
        tf.keras.applications.resnet_v2.preprocess_input,
    ),
    "resnet101v2": (
        tf.keras.applications.ResNet101V2,
        tf.keras.applications.resnet_v2.preprocess_input,
    ),
    "resnet152v2": (
        tf.keras.applications.ResNet152V2,
        tf.keras.applications.resnet_v2.preprocess_input,
    ),
    "densenet121": (
        tf.keras.applications.DenseNet121,
        tf.keras.applications.densenet.preprocess_input,
    ),
}


SUPPORTED_BACKBONES: tuple[str, ...] = tuple(_BACKBONE_REGISTRY.keys())


def build_resnet_family_model(
    backbone_name: str,
    num_classes: int,
    input_shape: tuple[int, int, int] = (32, 32, 3),
    resize_to: int = 224,
    dropout: float = 0.2,
    trainable_backbone: bool = False,
    fine_tune_at: int | None = None,
    weights: str | None = "imagenet",
) -> keras.Model:
    """Build a ResNet/DenseNet transfer-learning model for CIFAR-100.

    Graph: Input(32,32,3) → Rescaling(255.0) → Resizing → preprocess_input
    → backbone(include_top=False) → GlobalAveragePooling2D → Dropout →
    Dense head (sigmoid for binary, softmax for multiclass).

    Args:
        backbone_name: One of `SUPPORTED_BACKBONES`.
        num_classes: 1 for binary, >=2 for multiclass.
        input_shape: Image-view input shape from `make_pipeline`.
        resize_to: Internal backbone input edge size (square).
        dropout: Dropout rate applied between GAP and the head.
        trainable_backbone: If False, freeze the entire backbone.
        fine_tune_at: When `trainable_backbone=True`, freeze layers below
            this index. BatchNorm layers above the index are also kept
            frozen to keep moving statistics stable.
        weights: `"imagenet"` or `None` (testing/offline).

    Raises:
        ValueError: If `backbone_name` is unsupported, `num_classes < 1`, or
            `fine_tune_at` is out of range.

    Note:
        The graph includes a ``Lambda(preprocess_input)`` layer, so use
        ``model.save_weights(...)`` / ``model.load_weights(...)`` for
        persistence. Full-model save via ``model.save(...)`` or
        ``tf.keras.models.load_model(...)`` will fail to round-trip the Lambda.
    """
    if backbone_name not in _BACKBONE_REGISTRY:
        raise ValueError(
            f"backbone_name must be one of {SUPPORTED_BACKBONES}; "
            f"got {backbone_name!r}"
        )
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1; got {num_classes}")

    constructor, preprocess_input = _BACKBONE_REGISTRY[backbone_name]

    backbone = constructor(
        include_top=False,
        weights=weights,
        input_shape=(resize_to, resize_to, 3),
    )

    if fine_tune_at is not None:
        if fine_tune_at < 0:
            raise ValueError(
                f"fine_tune_at must be non-negative; got {fine_tune_at}"
            )
        if fine_tune_at >= len(backbone.layers):
            raise ValueError(
                f"fine_tune_at must be less than the backbone layer count "
                f"{len(backbone.layers)}; got {fine_tune_at}"
            )

    if not trainable_backbone:
        backbone.trainable = False
    else:
        backbone.trainable = True
        if fine_tune_at is not None:
            for layer in backbone.layers[:fine_tune_at]:
                layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

    inputs = keras.Input(shape=input_shape, name="image")
    # Upstream pipeline yields [0, 1]; multiply by 255 so preprocess_input receives [0, 255].
    x = layers.Rescaling(255.0, name="rescale_to_0_255")(inputs)
    x = layers.Resizing(resize_to, resize_to, name="resize")(x)
    x = layers.Lambda(preprocess_input, name="preprocess_input")(x)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)

    if num_classes == 1:
        outputs = layers.Dense(1, activation="sigmoid", name="prob")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax", name="prob")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=f"{backbone_name}_transfer")
