"""Unit tests for the training.train module."""

import pytest

from training.train import _build_model


def test_build_model_wires_enabled_augmentation_config():
    model = _build_model(
        {
            "architecture": "baseline_cnn",
            "dropout": 0.3,
            "augmentation": {
                "enabled": True,
                "horizontal_flip": True,
                "translation": 0.05,
            },
        },
        num_classes=1,
    )

    layer_names = [layer.name for layer in model.layers]
    assert "augmentation" in layer_names


def test_build_model_routes_resnet_family_to_new_builder():
    model = _build_model(
        {
            "architecture": "resnet_family",
            "backbone_name": "resnet50v2",
            "resize_to": 64,
            "dropout": 0.2,
            "trainable_backbone": False,
            "weights": None,
        },
        num_classes=1,
    )

    assert model.name == "resnet50v2_transfer"
    assert model.input_shape == (None, 32, 32, 3)
    assert model.output_shape == (None, 1)


def test_build_model_resnet_family_multiclass_head_shape():
    model = _build_model(
        {
            "architecture": "resnet_family",
            "backbone_name": "resnet50v2",
            "resize_to": 64,
            "weights": None,
        },
        num_classes=20,
    )
    assert model.output_shape == (None, 20)


def test_build_model_rejects_unknown_architecture():
    with pytest.raises(ValueError, match="architecture"):
        _build_model({"architecture": "vit"}, num_classes=1)


def test_build_model_resnet_family_requires_backbone_name():
    with pytest.raises(KeyError, match="backbone_name"):
        _build_model(
            {"architecture": "resnet_family", "weights": None},
            num_classes=1,
        )


def test_build_model_resnet_family_propagates_fine_tune_at():
    model = _build_model(
        {
            "architecture": "resnet_family",
            "backbone_name": "resnet50v2",
            "resize_to": 64,
            "trainable_backbone": True,
            "fine_tune_at": 50,
            "weights": None,
        },
        num_classes=1,
    )
    import tensorflow as tf
    backbone = [l for l in model.layers if isinstance(l, tf.keras.Model)][0]
    assert all(not l.trainable for l in backbone.layers[:50])
