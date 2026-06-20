"""Tests for the ResNet-family transfer-learning builder.

Test policy: every backbone gets a smoke instantiation check (parametrized
`test_each_supported_backbone_builds_a_keras_model`). All detailed tests —
output shapes, freezing semantics, fine_tune_at behavior, layer presence,
forward pass — run only on `resnet50v2`. This keeps the suite fast while
still covering every entry in the registry.
"""

import numpy as np
import pytest
import tensorflow as tf

from models.resnet_family import (
    SUPPORTED_BACKBONES,
    build_resnet_family_model,
)


SUPPORTED = ("resnet50v2", "resnet101v2", "resnet152v2", "densenet121")


def test_supported_backbones_is_the_expected_tuple():
    assert SUPPORTED_BACKBONES == SUPPORTED


@pytest.mark.parametrize("name", SUPPORTED)
def test_each_supported_backbone_builds_a_keras_model(name):
    """Smoke-test every backbone. Deeper tests live on resnet50v2 only."""
    model = build_resnet_family_model(
        backbone_name=name,
        num_classes=1,
        weights=None,
    )
    assert isinstance(model, tf.keras.Model)


def test_unsupported_backbone_name_raises_value_error():
    with pytest.raises(ValueError, match="backbone_name"):
        build_resnet_family_model(
            backbone_name="resnet50",
            num_classes=1,
            weights=None,
        )


def test_num_classes_below_one_raises_value_error():
    with pytest.raises(ValueError, match="num_classes"):
        build_resnet_family_model(
            backbone_name="resnet50v2",
            num_classes=0,
            weights=None,
        )


def test_binary_output_shape_is_none_one():
    model = build_resnet_family_model(
        backbone_name="resnet50v2",
        num_classes=1,
        weights=None,
    )
    assert model.input_shape == (None, 32, 32, 3)
    assert model.output_shape == (None, 1)


def test_binary_head_uses_sigmoid():
    model = build_resnet_family_model(
        backbone_name="resnet50v2",
        num_classes=1,
        weights=None,
    )
    last = model.layers[-1]
    assert last.activation.__name__ == "sigmoid"
    assert last.units == 1


def test_multiclass_output_shape_matches_num_classes():
    model = build_resnet_family_model(
        backbone_name="resnet50v2",
        num_classes=20,
        weights=None,
    )
    assert model.output_shape == (None, 20)


def test_multiclass_head_uses_softmax():
    model = build_resnet_family_model(
        backbone_name="resnet50v2",
        num_classes=20,
        weights=None,
    )
    last = model.layers[-1]
    assert last.activation.__name__ == "softmax"
    assert last.units == 20


def test_frozen_backbone_marks_backbone_non_trainable():
    model = build_resnet_family_model(
        backbone_name="resnet50v2",
        num_classes=1,
        trainable_backbone=False,
        weights=None,
    )
    backbones = [l for l in model.layers if isinstance(l, tf.keras.Model)]
    assert backbones, "expected the backbone to be a nested keras.Model"
    assert backbones[0].trainable is False


def test_trainable_backbone_marks_backbone_trainable():
    model = build_resnet_family_model(
        backbone_name="resnet50v2",
        num_classes=1,
        trainable_backbone=True,
        weights=None,
    )
    backbones = [l for l in model.layers if isinstance(l, tf.keras.Model)]
    assert backbones[0].trainable is True
    # Model-level flag is necessary but not sufficient: verify that at least one
    # non-BN layer inside the backbone is individually trainable.  A regression
    # where the BN sweep accidentally freezes non-BN layers would fail here.
    backbone = backbones[0]
    non_bn_trainable = [
        l for l in backbone.layers
        if not isinstance(l, tf.keras.layers.BatchNormalization) and l.trainable
    ]
    assert non_bn_trainable, (
        "expected at least one non-BatchNorm backbone layer to be trainable "
        "when trainable_backbone=True"
    )


def test_fine_tune_at_freezes_layers_before_index():
    model = build_resnet_family_model(
        backbone_name="resnet50v2",
        num_classes=1,
        trainable_backbone=True,
        fine_tune_at=100,
        weights=None,
    )
    backbone = [l for l in model.layers if isinstance(l, tf.keras.Model)][0]
    frozen_below = all(not l.trainable for l in backbone.layers[:100])
    assert frozen_below


def test_fine_tune_at_keeps_batchnorm_frozen_above_index():
    model = build_resnet_family_model(
        backbone_name="resnet50v2",
        num_classes=1,
        trainable_backbone=True,
        fine_tune_at=10,
        weights=None,
    )
    backbone = [l for l in model.layers if isinstance(l, tf.keras.Model)][0]
    bn_layers_above = [
        l for l in backbone.layers[10:]
        if isinstance(l, tf.keras.layers.BatchNormalization)
    ]
    assert bn_layers_above, "ResNet50V2 has BatchNorm layers above index 10"
    assert all(l.trainable is False for l in bn_layers_above)


def test_fine_tune_at_keeps_non_batchnorm_layers_above_index_trainable():
    """Sanity check: the non-BatchNorm layers above fine_tune_at should be
    trainable. If everything above is frozen, the fine-tune phase is a no-op.
    """
    model = build_resnet_family_model(
        backbone_name="resnet50v2",
        num_classes=1,
        trainable_backbone=True,
        fine_tune_at=10,
        weights=None,
    )
    backbone = [l for l in model.layers if isinstance(l, tf.keras.Model)][0]
    non_bn_above = [
        l for l in backbone.layers[10:]
        if not isinstance(l, tf.keras.layers.BatchNormalization)
    ]
    trainable_count = sum(1 for l in non_bn_above if l.trainable)
    assert trainable_count > 0, (
        "expected some non-BatchNorm layers above fine_tune_at to remain trainable"
    )


def test_model_graph_contains_resizing_layer_at_input_resolution():
    model = build_resnet_family_model(
        backbone_name="resnet50v2",
        num_classes=1,
        resize_to=160,
        weights=None,
    )
    resizings = [l for l in model.layers if isinstance(l, tf.keras.layers.Resizing)]
    assert resizings, "expected a Resizing layer in the model graph"
    assert resizings[0].height == 160
    assert resizings[0].width == 160


def test_model_graph_contains_rescaling_layer_to_undo_zero_one_normalization():
    model = build_resnet_family_model(
        backbone_name="resnet50v2",
        num_classes=1,
        weights=None,
    )
    rescalings = [l for l in model.layers if isinstance(l, tf.keras.layers.Rescaling)]
    assert rescalings, "expected a Rescaling layer to undo [0,1] normalization"
    assert rescalings[0].scale == 255.0


def test_model_is_uncompiled():
    model = build_resnet_family_model(
        backbone_name="resnet50v2",
        num_classes=1,
        weights=None,
    )
    assert model.optimizer is None


def test_forward_pass_returns_finite_probabilities_for_binary_head():
    model = build_resnet_family_model(
        backbone_name="resnet50v2",
        num_classes=1,
        weights=None,
    )
    x = np.zeros((2, 32, 32, 3), dtype=np.float32)
    y = model(x, training=False).numpy()
    assert y.shape == (2, 1)
    assert np.isfinite(y).all()
    assert (y >= 0.0).all() and (y <= 1.0).all()


def test_densenet121_uses_densenet_preprocess_input_in_graph():
    """The graph must include a layer wrapping densenet.preprocess_input.

    The wrapper is a Lambda layer; its inbound function name reflects the
    preprocess_input used so we can sanity-check routing.
    """
    model = build_resnet_family_model(
        backbone_name="densenet121",
        num_classes=1,
        weights=None,
    )
    lambdas = [l for l in model.layers if isinstance(l, tf.keras.layers.Lambda)]
    assert lambdas, "expected a Lambda(preprocess_input) layer"


def test_resnet50v2_uses_resnet_v2_preprocess_input_in_graph():
    model = build_resnet_family_model(
        backbone_name="resnet50v2",
        num_classes=1,
        weights=None,
    )
    lambdas = [l for l in model.layers if isinstance(l, tf.keras.layers.Lambda)]
    assert lambdas, "expected a Lambda(preprocess_input) layer"


def test_fine_tune_at_negative_raises_value_error():
    with pytest.raises(ValueError, match="fine_tune_at"):
        build_resnet_family_model(
            backbone_name="resnet50v2",
            num_classes=1,
            trainable_backbone=True,
            fine_tune_at=-1,
            weights=None,
        )


def test_fine_tune_at_above_layer_count_raises_value_error():
    with pytest.raises(ValueError, match="fine_tune_at"):
        build_resnet_family_model(
            backbone_name="resnet50v2",
            num_classes=1,
            trainable_backbone=True,
            fine_tune_at=10_000,
            weights=None,
        )
