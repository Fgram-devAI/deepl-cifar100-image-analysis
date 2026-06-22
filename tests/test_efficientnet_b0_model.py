"""Tests for the EfficientNetB0 transfer-learning model builder."""

import pytest
import tensorflow as tf

from models.efficientnet_b0 import build_efficientnet_b0


def _backbone(model: tf.keras.Model) -> tf.keras.Model:
    return [layer for layer in model.layers if isinstance(layer, tf.keras.Model)][0]


def test_efficientnet_b0_multiclass_shape():
    model = build_efficientnet_b0(num_classes=20, resize_to=64, weights=None)

    assert model.name == "efficientnet_b0_transfer"
    assert model.input_shape == (None, 32, 32, 3)
    assert model.output_shape == (None, 20)


def test_efficientnet_b0_binary_head_shape():
    model = build_efficientnet_b0(num_classes=1, resize_to=64, weights=None)

    assert model.output_shape == (None, 1)
    assert model.layers[-1].activation == tf.keras.activations.sigmoid


def test_efficientnet_b0_rejects_invalid_num_classes():
    with pytest.raises(ValueError, match="num_classes"):
        build_efficientnet_b0(num_classes=0, resize_to=64, weights=None)


def test_efficientnet_b0_rejects_too_small_resize():
    with pytest.raises(ValueError, match="resize_to"):
        build_efficientnet_b0(num_classes=20, resize_to=31, weights=None)


def test_efficientnet_b0_freezes_backbone_by_default():
    model = build_efficientnet_b0(num_classes=20, resize_to=64, weights=None)

    assert not _backbone(model).trainable


def test_efficientnet_b0_can_make_backbone_trainable():
    model = build_efficientnet_b0(
        num_classes=20,
        resize_to=64,
        trainable_backbone=True,
        weights=None,
    )

    assert _backbone(model).trainable


def test_efficientnet_b0_includes_augmentation_when_enabled():
    model = build_efficientnet_b0(
        num_classes=20,
        resize_to=64,
        weights=None,
        augmentation={"enabled": True, "horizontal_flip": True},
    )

    assert "augmentation" in [layer.name for layer in model.layers]
