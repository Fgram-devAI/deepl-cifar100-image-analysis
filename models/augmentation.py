"""Optional light augmentation layer factory for image-view CNN models."""

import tensorflow as tf

keras = tf.keras
layers = tf.keras.layers


def build_augmentation(config: dict | None) -> keras.layers.Layer | None:
    """Build a Keras augmentation Sequential from a config dict, or return None.

    Returns ``None`` when ``config`` is ``None`` or ``config["enabled"]`` is
    falsy, so callers can skip wiring entirely when augmentation is off.

    Only the five light ops are supported: horizontal_flip, translation, zoom,
    rotation, contrast. A zero or False value for any field omits that layer.
    Keras Random* layers are no-ops at inference time (training=False).

    Args:
        config: Augmentation config dict (from YAML ``augmentation:`` block),
            or ``None`` to disable.

    Returns:
        A named ``tf.keras.Sequential`` of augmentation layers, or ``None``.
    """
    if config is None or not config.get("enabled", False):
        return None

    aug_layers = []
    if config.get("horizontal_flip", False):
        aug_layers.append(layers.RandomFlip("horizontal"))
    if config.get("translation", 0.0) > 0:
        t = float(config["translation"])
        aug_layers.append(layers.RandomTranslation(t, t))
    if config.get("zoom", 0.0) > 0:
        aug_layers.append(layers.RandomZoom(float(config["zoom"])))
    if config.get("rotation", 0.0) > 0:
        aug_layers.append(layers.RandomRotation(float(config["rotation"])))
    if config.get("contrast", 0.0) > 0:
        aug_layers.append(layers.RandomContrast(float(config["contrast"])))

    return keras.Sequential(aug_layers, name="augmentation")
