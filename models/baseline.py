"""From-scratch CNN baselines for CIFAR-100 image-view tasks."""

import tensorflow as tf

from models.augmentation import build_augmentation

keras = tf.keras
layers = tf.keras.layers
regularizers = tf.keras.regularizers


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


def _classification_head(
    x: tf.Tensor,
    *,
    num_classes: int,
    name: str = "prob",
) -> tf.Tensor:
    if num_classes == 1:
        return layers.Dense(1, activation="sigmoid", name=name)(x)
    return layers.Dense(num_classes, activation="softmax", name=name)(x)


def _conv_bn_relu(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(1e-4),
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)


def build_strong_cnn(
    input_shape: tuple[int, int, int] = (32, 32, 3),
    dropout: float = 0.35,
    num_classes: int = 1,
    augmentation: dict | None = None,
) -> keras.Model:
    """Build a stronger from-scratch CNN control model.

    This model is intentionally still a CIFAR-sized from-scratch baseline, not
    a transfer-learning backbone. It adds the ingredients missing from the
    compact baseline CNN: repeated conv blocks, BatchNorm, MaxPooling, and a
    GlobalAveragePooling head. Use it as a control when comparing row-sequence
    RNN/LSTM/Bi-LSTM models against convolutional inductive bias.
    """
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1; got {num_classes}")

    inputs = keras.Input(shape=input_shape, name="image")

    aug_layer = build_augmentation(augmentation)
    x = aug_layer(inputs) if aug_layer is not None else inputs

    x = _conv_bn_relu(x, 32, "block1_conv1")
    x = _conv_bn_relu(x, 32, "block1_conv2")
    x = layers.MaxPool2D(pool_size=2, name="block1_pool")(x)
    x = layers.SpatialDropout2D(dropout * 0.5, name="block1_dropout")(x)

    x = _conv_bn_relu(x, 64, "block2_conv1")
    x = _conv_bn_relu(x, 64, "block2_conv2")
    x = layers.MaxPool2D(pool_size=2, name="block2_pool")(x)
    x = layers.SpatialDropout2D(dropout * 0.75, name="block2_dropout")(x)

    x = _conv_bn_relu(x, 128, "block3_conv1")
    x = _conv_bn_relu(x, 128, "block3_conv2")
    x = layers.MaxPool2D(pool_size=2, name="block3_pool")(x)
    x = layers.SpatialDropout2D(dropout, name="block3_dropout")(x)

    x = _conv_bn_relu(x, 256, "block4_conv1")
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(
        256,
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(1e-4),
        name="head_dense",
    )(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Activation("relu", name="head_relu")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)

    outputs = _classification_head(x, num_classes=num_classes)
    return keras.Model(inputs=inputs, outputs=outputs, name="strong_cnn")
