"""Tests for the baseline CNN model builder."""

import numpy as np
import tensorflow as tf

from models.baseline import build_baseline_cnn, build_strong_cnn


def test_baseline_cnn_input_and_output_shapes():
    model = build_baseline_cnn()
    assert model.input_shape == (None, 32, 32, 3)
    assert model.output_shape == (None, 1)


def test_baseline_cnn_is_named_baseline_cnn():
    model = build_baseline_cnn()
    assert model.name == "baseline_cnn"


def test_baseline_cnn_forward_pass_returns_probabilities():
    model = build_baseline_cnn()
    x = np.zeros((2, 32, 32, 3), dtype=np.float32)
    y = model(x, training=False).numpy()
    assert y.shape == (2, 1)
    assert float(y.min()) >= 0.0
    assert float(y.max()) <= 1.0


def test_baseline_cnn_returns_uncompiled_model():
    model = build_baseline_cnn()
    assert model.optimizer is None


def test_baseline_cnn_accepts_custom_dropout():
    model = build_baseline_cnn(dropout=0.5)
    dropout_rates = [
        layer.rate for layer in model.layers
        if isinstance(layer, tf.keras.layers.Dropout)
    ]
    assert dropout_rates and all(r == 0.5 for r in dropout_rates)


def test_baseline_cnn_coarse_multiclass_head_has_20_softmax_units():
    model = build_baseline_cnn(num_classes=20)
    assert model.output_shape == (None, 20)
    last = model.layers[-1]
    assert last.activation.__name__ == "softmax"
    assert last.units == 20


def test_baseline_cnn_fine_multiclass_head_has_100_softmax_units():
    model = build_baseline_cnn(num_classes=100)
    assert model.output_shape == (None, 100)
    last = model.layers[-1]
    assert last.activation.__name__ == "softmax"
    assert last.units == 100


def test_baseline_cnn_binary_head_still_uses_sigmoid():
    model = build_baseline_cnn(num_classes=1)
    last = model.layers[-1]
    assert last.activation.__name__ == "sigmoid"
    assert last.units == 1
    assert model.output_shape == (None, 1)


def test_baseline_cnn_rejects_num_classes_below_one():
    import pytest
    with pytest.raises(ValueError):
        build_baseline_cnn(num_classes=0)
    with pytest.raises(ValueError):
        build_baseline_cnn(num_classes=-3)


def test_baseline_cnn_no_augmentation_layer_by_default():
    model = build_baseline_cnn()
    layer_names = [l.name for l in model.layers]
    assert "augmentation" not in layer_names


def test_baseline_cnn_with_augmentation_includes_augmentation_layer():
    aug_cfg = {
        "enabled": True,
        "horizontal_flip": True,
        "translation": 0.05,
        "zoom": 0.05,
        "rotation": 0.0,
        "contrast": 0.0,
    }
    model = build_baseline_cnn(augmentation=aug_cfg)
    layer_names = [l.name for l in model.layers]
    assert "augmentation" in layer_names


def test_baseline_cnn_augmentation_preserves_binary_output_shape():
    aug_cfg = {"enabled": True, "horizontal_flip": True, "translation": 0.05}
    model = build_baseline_cnn(augmentation=aug_cfg)
    assert model.output_shape == (None, 1)


def test_baseline_cnn_augmentation_preserves_multiclass_output_shape():
    aug_cfg = {"enabled": True, "horizontal_flip": True, "translation": 0.05}
    model = build_baseline_cnn(num_classes=20, augmentation=aug_cfg)
    assert model.output_shape == (None, 20)


def test_baseline_cnn_disabled_augmentation_config_has_no_augmentation_layer():
    model = build_baseline_cnn(augmentation={"enabled": False})
    layer_names = [l.name for l in model.layers]
    assert "augmentation" not in layer_names


def test_strong_cnn_coarse_multiclass_head_has_20_softmax_units():
    model = build_strong_cnn(num_classes=20)
    assert model.name == "strong_cnn"
    assert model.input_shape == (None, 32, 32, 3)
    assert model.output_shape == (None, 20)
    last = model.layers[-1]
    assert last.activation.__name__ == "softmax"
    assert last.units == 20


def test_strong_cnn_includes_batch_norm_and_pooling():
    model = build_strong_cnn(num_classes=20)
    layer_types = {type(layer).__name__ for layer in model.layers}
    assert "BatchNormalization" in layer_types
    assert "MaxPooling2D" in layer_types
    assert "GlobalAveragePooling2D" in layer_types
