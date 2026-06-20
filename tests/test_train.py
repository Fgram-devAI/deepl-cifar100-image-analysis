"""Unit tests for the training.train module."""

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
