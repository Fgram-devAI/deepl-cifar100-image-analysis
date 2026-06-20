"""Tests for the augmentation factory."""

import tensorflow as tf

from models.augmentation import build_augmentation


def test_build_augmentation_none_returns_none():
    assert build_augmentation(None) is None


def test_build_augmentation_disabled_returns_none():
    assert build_augmentation({"enabled": False}) is None


def test_build_augmentation_missing_enabled_returns_none():
    assert build_augmentation({}) is None


def test_build_augmentation_enabled_returns_keras_layer():
    config = {
        "enabled": True,
        "horizontal_flip": True,
        "translation": 0.05,
        "zoom": 0.05,
        "rotation": 0.0,
        "contrast": 0.0,
    }
    result = build_augmentation(config)
    assert result is not None
    assert isinstance(result, tf.keras.layers.Layer)


def test_build_augmentation_result_is_named_augmentation():
    result = build_augmentation({"enabled": True, "horizontal_flip": True})
    assert result.name == "augmentation"


def test_build_augmentation_horizontal_flip_adds_random_flip():
    result = build_augmentation({"enabled": True, "horizontal_flip": True})
    types = [type(l).__name__ for l in result.layers]
    assert "RandomFlip" in types


def test_build_augmentation_no_flip_flag_omits_random_flip():
    result = build_augmentation({"enabled": True, "translation": 0.05})
    types = [type(l).__name__ for l in result.layers]
    assert "RandomFlip" not in types


def test_build_augmentation_nonzero_translation_adds_random_translation():
    result = build_augmentation({"enabled": True, "translation": 0.05})
    types = [type(l).__name__ for l in result.layers]
    assert "RandomTranslation" in types


def test_build_augmentation_zero_translation_omits_random_translation():
    result = build_augmentation({"enabled": True, "translation": 0.0})
    types = [type(l).__name__ for l in result.layers]
    assert "RandomTranslation" not in types


def test_build_augmentation_nonzero_zoom_adds_random_zoom():
    result = build_augmentation({"enabled": True, "zoom": 0.05})
    types = [type(l).__name__ for l in result.layers]
    assert "RandomZoom" in types


def test_build_augmentation_zero_zoom_omits_random_zoom():
    result = build_augmentation({"enabled": True, "zoom": 0.0})
    types = [type(l).__name__ for l in result.layers]
    assert "RandomZoom" not in types


def test_build_augmentation_nonzero_rotation_adds_random_rotation():
    result = build_augmentation({"enabled": True, "rotation": 0.1})
    types = [type(l).__name__ for l in result.layers]
    assert "RandomRotation" in types


def test_build_augmentation_zero_rotation_omits_random_rotation():
    result = build_augmentation({"enabled": True, "rotation": 0.0})
    types = [type(l).__name__ for l in result.layers]
    assert "RandomRotation" not in types


def test_build_augmentation_nonzero_contrast_adds_random_contrast():
    result = build_augmentation({"enabled": True, "contrast": 0.1})
    types = [type(l).__name__ for l in result.layers]
    assert "RandomContrast" in types


def test_build_augmentation_zero_contrast_omits_random_contrast():
    result = build_augmentation({"enabled": True, "contrast": 0.0})
    types = [type(l).__name__ for l in result.layers]
    assert "RandomContrast" not in types


def test_build_augmentation_is_stable_when_not_training():
    result = build_augmentation({"enabled": True, "contrast": 0.2})
    x = tf.ones((2, 32, 32, 3), dtype=tf.float32) * 0.5

    y1 = result(x, training=False)
    y2 = result(x, training=False)

    tf.debugging.assert_near(y1, x)
    tf.debugging.assert_near(y2, x)
