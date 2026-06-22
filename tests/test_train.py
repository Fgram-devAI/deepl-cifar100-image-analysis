"""Unit tests for the training.train module."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import tensorflow as tf

from data import Cifar100Split
from training.train import _build_model, load_config, train_from_config


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


def test_build_model_routes_strong_cnn_to_new_builder():
    model = _build_model(
        {
            "architecture": "strong_cnn",
            "dropout": 0.35,
        },
        num_classes=20,
    )

    assert model.name == "strong_cnn"
    assert model.input_shape == (None, 32, 32, 3)
    assert model.output_shape == (None, 20)


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


def test_build_model_routes_efficientnet_b0_to_new_builder():
    model = _build_model(
        {
            "architecture": "efficientnet_b0",
            "resize_to": 64,
            "dropout": 0.2,
            "trainable_backbone": False,
            "weights": None,
        },
        num_classes=20,
    )

    assert model.name == "efficientnet_b0_transfer"
    assert model.input_shape == (None, 32, 32, 3)
    assert model.output_shape == (None, 20)


def test_build_model_efficientnet_b0_supports_legacy_freeze_key():
    model = _build_model(
        {
            "architecture": "efficientnet_b0",
            "input_size": 64,
            "freeze_backbone": True,
            "weights": None,
        },
        num_classes=20,
    )
    backbone = [layer for layer in model.layers if isinstance(layer, tf.keras.Model)][0]
    assert not backbone.trainable


def test_build_model_routes_efficientnet_b3_to_new_builder():
    model = _build_model(
        {
            "architecture": "efficientnet_b3",
            "resize_to": 64,
            "dropout": 0.4,
            "trainable_backbone": False,
            "weights": None,
        },
        num_classes=100,
    )

    assert model.name == "efficientnet_b3_transfer"
    assert model.input_shape == (None, 32, 32, 3)
    assert model.output_shape == (None, 100)


def test_build_model_efficientnet_b3_supports_partial_unfreeze_config():
    model = _build_model(
        {
            "architecture": "efficientnet_b3",
            "input_size": 64,
            "freeze_backbone": False,
            "unfreeze_from": "block6",
            "freeze_bn": True,
            "weights": None,
        },
        num_classes=100,
    )

    backbone = [layer for layer in model.layers if isinstance(layer, tf.keras.Model)][0]
    block5_layers = [layer for layer in backbone.layers if layer.name.startswith("block5")]
    block6_layers = [layer for layer in backbone.layers if layer.name.startswith("block6")]
    assert block5_layers
    assert block6_layers
    assert all(not layer.trainable for layer in block5_layers)
    assert any(layer.trainable for layer in block6_layers)


def test_build_model_routes_bilstm_to_sequence_model():
    model = _build_model(
        {
            "architecture": "bilstm",
            "hidden_units": 8,
            "dropout": 0.2,
        },
        num_classes=20,
    )

    assert model.name == "bilstm_sequence"
    assert model.input_shape == (None, 32, 96)
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
    backbone = [l for l in model.layers if isinstance(l, tf.keras.Model)][0]
    assert all(not l.trainable for l in backbone.layers[:50])


# ---------------------------------------------------------------------------
# Task 3: _maybe_load_initial_weights + initial_weights config key
# ---------------------------------------------------------------------------


def _tiny_split(seed: int, n: int = 32, positives: int = 6) -> Cifar100Split:
    """Synthetic CIFAR-100 split with a controlled positive count.

    Stamps `positives` rows with fine label 0 (apple). Stratified val/test
    splits need enough positives in each partition for stable metrics, so
    keep `positives >= 4` when `validation.stratify=True`.
    """
    if positives > n:
        raise ValueError("positives must be <= n")
    rng = np.random.default_rng(seed)
    images = rng.integers(0, 256, size=(n, 32, 32, 3), dtype=np.uint8)
    fine = rng.integers(1, 100, size=(n,), dtype=np.int64)  # avoid id 0 first
    coarse = rng.integers(0, 20, size=(n,), dtype=np.int64)
    fine[:positives] = 0  # apple
    return Cifar100Split(
        images=images, fine_labels=fine, coarse_labels=coarse, split="train"
    )


def test_maybe_load_initial_weights_no_op_when_key_absent():
    from training.train import _maybe_load_initial_weights
    mock_model = MagicMock()
    _maybe_load_initial_weights(mock_model, {})
    mock_model.load_weights.assert_not_called()


def test_maybe_load_initial_weights_no_op_when_key_is_none():
    from training.train import _maybe_load_initial_weights
    mock_model = MagicMock()
    _maybe_load_initial_weights(mock_model, {"initial_weights": None})
    mock_model.load_weights.assert_not_called()


def test_maybe_load_initial_weights_calls_load_weights_with_path(tmp_path: Path):
    from training.train import _maybe_load_initial_weights
    weights_path = tmp_path / "fake.h5"
    weights_path.write_text("placeholder")
    mock_model = MagicMock()
    _maybe_load_initial_weights(
        mock_model, {"initial_weights": str(weights_path)}
    )
    mock_model.load_weights.assert_called_once_with(str(weights_path))


def test_maybe_load_initial_weights_raises_when_path_missing(tmp_path: Path):
    from training.train import _maybe_load_initial_weights
    mock_model = MagicMock()
    with pytest.raises(FileNotFoundError, match="initial_weights"):
        _maybe_load_initial_weights(
            mock_model, {"initial_weights": str(tmp_path / "missing.h5")}
        )
    mock_model.load_weights.assert_not_called()


def test_train_from_config_spies_load_weights_with_initial_weights(
    tmp_path: Path, monkeypatch
):
    """End-to-end proof: train_from_config calls model.load_weights(...) with
    exactly the configured initial_weights path. Spying on the keras Model
    method captures the call without depending on a real .h5 file.
    """
    captured: list[str] = []

    def spy(self, filepath, *args, **kwargs):
        captured.append(str(filepath))
        return None  # no-op: we are only verifying the call

    monkeypatch.setattr(tf.keras.Model, "load_weights", spy)

    fake_weights = tmp_path / "spy.h5"
    fake_weights.write_text("placeholder")

    config = {
        "architecture": "baseline_cnn",
        "run_name": "spy_run",
        "seed": 7,
        "task": {
            "type": "binary",
            "label_level": "fine",
            "positive_label_names": ["apple"],
        },
        "validation": {"fraction": 0.25, "stratify": True},
        "class_imbalance": {"strategy": "none"},
        "batch_size": 8,
        "shuffle_buffer": 8,
        "dropout": 0.3,
        "epochs": 1,
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "early_stopping": {"monitor": "val_loss", "patience": 1},
        "output_dir": str(tmp_path / "out"),
        "save_weights": False,
        "initial_weights": str(fake_weights),
    }

    train_from_config(
        config,
        train_split=_tiny_split(seed=1),
        test_split=_tiny_split(seed=2),
    )

    assert captured == [str(fake_weights)], (
        f"expected one load_weights call with {fake_weights}; got {captured}"
    )


def test_initial_weights_load_changes_starting_parameters(tmp_path: Path):
    config_a = {
        "architecture": "baseline_cnn",
        "run_name": "frozen_run",
        "seed": 7,
        "task": {
            "type": "binary",
            "label_level": "fine",
            "positive_label_names": ["apple"],
        },
        "validation": {"fraction": 0.25, "stratify": True},
        "class_imbalance": {"strategy": "none"},
        "batch_size": 8,
        "shuffle_buffer": 8,
        "dropout": 0.3,
        "epochs": 1,
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "early_stopping": {"monitor": "val_loss", "patience": 1},
        "output_dir": str(tmp_path / "out"),
        "save_weights": True,
    }
    train_from_config(
        config_a,
        train_split=_tiny_split(seed=1),
        test_split=_tiny_split(seed=2),
    )

    weights_path = tmp_path / "out" / "frozen_run" / "weights.h5"
    assert weights_path.exists(), "Task T3 test requires save_weights=True to produce weights.h5"

    config_b = dict(config_a)
    config_b["run_name"] = "fine_tune_run"
    config_b["initial_weights"] = str(weights_path)
    config_b["learning_rate"] = 1e-6
    config_b["epochs"] = 1

    # Capture the post-load weights of the head Dense layer by replaying
    # the load against a fresh model and comparing to a randomly-init one.
    from training.train import _build_model
    fresh = _build_model(config_b, num_classes=1)
    loaded = _build_model(config_b, num_classes=1)
    loaded.load_weights(str(weights_path))

    fresh_dense = [l for l in fresh.layers if l.name == "prob"][0]
    loaded_dense = [l for l in loaded.layers if l.name == "prob"][0]
    fresh_w = fresh_dense.get_weights()[0]
    loaded_w = loaded_dense.get_weights()[0]
    assert not np.allclose(fresh_w, loaded_w), (
        "loaded weights should differ from a freshly-initialized model"
    )

    # And confirm train_from_config with initial_weights runs end-to-end.
    train_from_config(
        config_b,
        train_split=_tiny_split(seed=1),
        test_split=_tiny_split(seed=2),
    )
    assert (tmp_path / "out" / "fine_tune_run" / "metrics.json").exists()


def test_initial_weights_missing_path_raises_file_not_found(tmp_path: Path):
    config = {
        "architecture": "baseline_cnn",
        "run_name": "bad_run",
        "seed": 7,
        "task": {
            "type": "binary",
            "label_level": "fine",
            "positive_label_names": ["apple"],
        },
        "validation": {"fraction": 0.25, "stratify": True},
        "class_imbalance": {"strategy": "none"},
        "batch_size": 8,
        "shuffle_buffer": 8,
        "dropout": 0.3,
        "epochs": 1,
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "early_stopping": {"monitor": "val_loss", "patience": 1},
        "output_dir": str(tmp_path / "out"),
        "save_weights": False,
        "initial_weights": str(tmp_path / "does_not_exist.h5"),
    }
    with pytest.raises(FileNotFoundError):
        train_from_config(
            config,
            train_split=_tiny_split(seed=1),
            test_split=_tiny_split(seed=2),
        )


# ---------------------------------------------------------------------------
# Task 4: ResNet-family recommended-experiment YAML configs + routing test
# ---------------------------------------------------------------------------

_RESNET_FAMILY_CONFIGS = [
    "configs/transfer/resnet_family/binary/coarse/resnet50v2_food_containers.yaml",
    "configs/transfer/resnet_family/binary/fine/resnet50v2_skyscraper.yaml",
    "configs/transfer/resnet_family/binary/fine/resnet50v2_orange.yaml",
    "configs/transfer/resnet_family/multiclass/resnet50v2_coarse.yaml",
    "configs/transfer/resnet_family/binary/fine/resnet50v2_skyscraper_finetune.yaml",
    "configs/transfer/resnet_family/binary/fine/densenet121_skyscraper.yaml",
]


@pytest.mark.parametrize("config_path", _RESNET_FAMILY_CONFIGS)
def test_resnet_family_config_parses_and_routes_to_builder(config_path):
    config = load_config(config_path)
    assert config["architecture"] == "resnet_family"
    assert config["backbone_name"] in {
        "resnet50v2", "resnet101v2", "resnet152v2", "densenet121",
    }

    # Override weights to None so the test does not hit the network.
    config["weights"] = None

    num_classes = 1 if config["task"]["type"] == "binary" else 20
    model = _build_model(config, num_classes=num_classes)
    assert model.name.endswith("_transfer")


@pytest.mark.parametrize(
    ("config_path", "label_level", "num_classes"),
    [
        (
            "configs/transfer/efficientnet/multiclass/efficientnet_b0_coarse.yaml",
            "coarse",
            20,
        ),
        (
            "configs/transfer/efficientnet/multiclass/efficientnet_b0_fine.yaml",
            "fine",
            100,
        ),
    ],
)
def test_efficientnet_b0_config_parses_and_routes_to_builder(
    config_path, label_level, num_classes
):
    config = load_config(config_path)
    assert config["architecture"] == "efficientnet_b0"
    assert config["task"] == {"type": "multiclass", "label_level": label_level}

    # Override weights to None so the test does not hit the network.
    config["weights"] = None

    model = _build_model(config, num_classes=num_classes)
    assert model.name == "efficientnet_b0_transfer"
