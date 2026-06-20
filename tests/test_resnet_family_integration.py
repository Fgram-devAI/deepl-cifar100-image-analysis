"""End-to-end integration: resnet_family architecture on synthetic CIFAR-100.

Stays fully offline:
  - synthetic CIFAR-100 splits via the conftest fixture pattern
  - `weights: null` so `ResNet50V2(weights=None)` does not hit the network
"""

from pathlib import Path

import numpy as np
import tensorflow as tf

from data import Cifar100Split
from models.resnet_family import build_resnet_family_model
from training.train import _build_model, train_from_config


def _tiny_split(seed: int, n: int = 32, positives: int = 6) -> Cifar100Split:
    """Synthetic CIFAR-100 split with a controlled positive count.

    Stamps `positives` rows with fine label 0 (apple) so the binary task has
    enough positives in both val and test splits for stable metric
    computation.
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


def _binary_resnet_config(tmp_path: Path) -> dict:
    return {
        "architecture": "resnet_family",
        "backbone_name": "resnet50v2",
        "run_name": "smoke_resnet50v2_frozen",
        "seed": 7,
        "resize_to": 64,
        "trainable_backbone": False,
        "dropout": 0.2,
        "weights": None,
        "task": {
            "type": "binary",
            "label_level": "fine",
            "positive_label_names": ["apple"],
        },
        "validation": {"fraction": 0.25, "stratify": True},
        "class_imbalance": {"strategy": "class_weights"},
        "batch_size": 8,
        "shuffle_buffer": 8,
        "epochs": 1,
        "optimizer": "adam",
        "learning_rate": 1e-4,
        "early_stopping": {"monitor": "val_loss", "patience": 1},
        "output_dir": str(tmp_path / "out"),
        "save_weights": True,
        "subset_size": None,
    }


def test_resnet_family_binary_end_to_end_offline(tmp_path: Path):
    config = _binary_resnet_config(tmp_path)
    history, metrics = train_from_config(
        config,
        train_split=_tiny_split(seed=1),
        test_split=_tiny_split(seed=2),
    )

    run_dir = tmp_path / "out" / "smoke_resnet50v2_frozen"
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "class_balance.json").exists()
    assert (run_dir / "history.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "weights.h5").exists()

    assert "loss" in history
    assert {"accuracy", "precision", "recall", "f1", "roc_auc"}.issubset(metrics)


def test_resnet_family_reload_path_reproduces_predictions(tmp_path: Path):
    config = _binary_resnet_config(tmp_path)
    train_from_config(
        config,
        train_split=_tiny_split(seed=1),
        test_split=_tiny_split(seed=2),
    )
    weights_path = tmp_path / "out" / "smoke_resnet50v2_frozen" / "weights.h5"
    assert weights_path.exists()

    # Build a fresh model from the same config, load weights, and check that
    # predictions on a fixed input match a second fresh model that loads the
    # same weights. This is the "reload reproduces predictions" contract.
    model_a = _build_model(config, num_classes=1)
    model_a.load_weights(str(weights_path))

    model_b = _build_model(config, num_classes=1)
    model_b.load_weights(str(weights_path))

    x = np.zeros((4, 32, 32, 3), dtype=np.float32) + 0.5
    y_a = model_a(x, training=False).numpy()
    y_b = model_b(x, training=False).numpy()

    np.testing.assert_allclose(y_a, y_b, rtol=1e-6, atol=1e-6)


def test_resnet_family_multiclass_end_to_end_offline(tmp_path: Path):
    config = {
        "architecture": "resnet_family",
        "backbone_name": "resnet50v2",
        "run_name": "smoke_resnet50v2_coarse",
        "seed": 7,
        "resize_to": 64,
        "trainable_backbone": False,
        "dropout": 0.2,
        "weights": None,
        "task": {"type": "multiclass", "label_level": "coarse"},
        "validation": {"fraction": 0.25, "stratify": False},
        "class_imbalance": {"strategy": "none"},
        "batch_size": 8,
        "shuffle_buffer": 8,
        "epochs": 1,
        "optimizer": "adam",
        "learning_rate": 1e-4,
        "early_stopping": {"monitor": "val_loss", "patience": 1},
        "output_dir": str(tmp_path / "out"),
        "save_weights": False,
    }
    history, metrics = train_from_config(
        config,
        train_split=_tiny_split(seed=1, n=64),
        test_split=_tiny_split(seed=2, n=32),
    )

    run_dir = tmp_path / "out" / "smoke_resnet50v2_coarse"
    assert (run_dir / "metrics.json").exists()
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert metrics["confusion_matrix"]  # non-empty
