"""Tests for training helpers: losses, callbacks, splits, class weights, train loop."""

import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from training.callbacks import JSONLogger, get_callbacks
from training.losses import get_loss


# --- losses ---------------------------------------------------------------

def test_get_loss_binary_returns_binary_crossentropy():
    loss = get_loss("binary")
    assert isinstance(loss, tf.keras.losses.BinaryCrossentropy)
    assert loss.get_config()["from_logits"] is False


def test_get_loss_rejects_unknown_head():
    with pytest.raises(ValueError):
        get_loss("multiclass")


# --- callbacks ------------------------------------------------------------

def test_json_logger_writes_history_after_each_epoch(tmp_path: Path):
    path = tmp_path / "history.json"
    cb = JSONLogger(path)
    cb.on_epoch_end(0, {"loss": 0.5, "val_loss": 0.6})
    cb.on_epoch_end(1, {"loss": 0.4, "val_loss": 0.55})
    written = json.loads(path.read_text())
    assert written == {"loss": [0.5, 0.4], "val_loss": [0.6, 0.55]}


def test_json_logger_creates_parent_directory(tmp_path: Path):
    path = tmp_path / "nested" / "history.json"
    cb = JSONLogger(path)
    cb.on_epoch_end(0, {"loss": 0.3})
    assert path.exists()


def test_get_callbacks_includes_early_stopping_csv_and_json(tmp_path: Path):
    cbs = get_callbacks(tmp_path, patience=3, monitor="val_loss")
    types = {type(c).__name__ for c in cbs}
    assert "EarlyStopping" in types
    assert "CSVLogger" in types
    assert "JSONLogger" in types
    es = next(c for c in cbs if type(c).__name__ == "EarlyStopping")
    assert es.patience == 3
    assert es.monitor == "val_loss"
    assert es.restore_best_weights is True


# --- splits ---------------------------------------------------------------

from training.class_weights import compute_balanced_class_weights  # noqa: E402
from training.splits import stratified_train_val_split  # noqa: E402


def _toy_dataset(n_pos: int = 30, n_neg: int = 70):
    rng = np.random.default_rng(0)
    images = rng.integers(
        0, 256, size=(n_pos + n_neg, 32, 32, 3), dtype=np.uint8
    )
    labels = np.concatenate(
        [np.ones(n_pos, dtype=np.int64), np.zeros(n_neg, dtype=np.int64)]
    )
    return images, labels


def test_stratified_split_preserves_class_ratio():
    images, labels = _toy_dataset(n_pos=30, n_neg=70)
    x_tr, y_tr, x_val, y_val = stratified_train_val_split(
        images, labels, val_fraction=0.2, seed=42
    )
    assert x_tr.shape[0] + x_val.shape[0] == 100
    assert x_val.shape[0] == 20
    assert int(y_val.sum()) == 6   # 30% of 20 is 6 positives
    assert int(y_tr.sum()) == 24


def test_stratified_split_is_deterministic():
    images, labels = _toy_dataset()
    a = stratified_train_val_split(images, labels, val_fraction=0.2, seed=7)
    b = stratified_train_val_split(images, labels, val_fraction=0.2, seed=7)
    np.testing.assert_array_equal(a[1], b[1])
    np.testing.assert_array_equal(a[3], b[3])


def test_stratified_split_rejects_invalid_fraction():
    images, labels = _toy_dataset()
    with pytest.raises(ValueError):
        stratified_train_val_split(images, labels, val_fraction=0.0, seed=0)
    with pytest.raises(ValueError):
        stratified_train_val_split(images, labels, val_fraction=1.0, seed=0)


# --- class weights --------------------------------------------------------

def test_compute_balanced_class_weights_inverse_to_frequency():
    labels = np.array([0] * 90 + [1] * 10, dtype=np.int64)
    weights = compute_balanced_class_weights(labels)
    assert set(weights.keys()) == {0, 1}
    # Positive class is rarer, so its weight should be larger.
    assert weights[1] > weights[0]
    # sklearn 'balanced' formula: n_samples / (n_classes * count)
    assert weights[0] == pytest.approx(100 / (2 * 90))
    assert weights[1] == pytest.approx(100 / (2 * 10))


def test_compute_balanced_class_weights_single_class_returns_unit_weights():
    labels = np.zeros(50, dtype=np.int64)
    weights = compute_balanced_class_weights(labels)
    assert weights == {0: 1.0, 1: 1.0}


# --- training entrypoint --------------------------------------------------

import yaml  # noqa: E402

from data.loaders import Cifar100Split  # noqa: E402
from training.train import load_config, train_from_config  # noqa: E402


def _synthetic_split(
    images: np.ndarray, fine: np.ndarray, coarse: np.ndarray, name: str
) -> Cifar100Split:
    return Cifar100Split(
        images=images,
        fine_labels=fine.reshape(-1),
        coarse_labels=coarse.reshape(-1),
        split=name,
    )


def test_load_config_reads_yaml(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("run_name: demo\nepochs: 1\n")
    cfg = load_config(cfg_path)
    assert cfg["run_name"] == "demo"
    assert cfg["epochs"] == 1


def test_train_from_config_smoke_runs_and_writes_artifacts(
    synthetic_cifar100, tmp_path: Path
):
    (x_tr, yf_tr, yc_tr), (x_te, yf_te, yc_te) = synthetic_cifar100
    train_split = _synthetic_split(x_tr, yf_tr, yc_tr, "train")
    test_split = _synthetic_split(x_te, yf_te, yc_te, "test")

    # Ensure both classes appear in train+val by picking a coarse label that
    # is plausibly present in the synthetic stand-in.
    config = {
        "architecture": "baseline_cnn",
        "run_name": "smoke",
        "seed": 42,
        "task": {
            "label_level": "coarse",
            "positive_label_names": ["aquatic_mammals"],
        },
        "validation": {"fraction": 0.2, "stratify": True},
        "class_imbalance": {"strategy": "class_weights"},
        "batch_size": 16,
        "shuffle_buffer": 64,
        "dropout": 0.3,
        "epochs": 1,
        "learning_rate": 0.001,
        "early_stopping": {"monitor": "val_loss", "patience": 1},
        "output_dir": str(tmp_path),
        "save_weights": False,
        "subset_size": None,
    }

    history, metrics = train_from_config(
        config, train_split=train_split, test_split=test_split
    )

    run_dir = tmp_path / "smoke"
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "class_balance.json").exists()
    assert (run_dir / "history.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert not (run_dir / "weights.h5").exists()

    # history has at least loss recorded.
    assert "loss" in history
    # metrics for the test split include all required binary keys plus
    # class counts and confusion matrix.
    saved_metrics = json.loads((run_dir / "metrics.json").read_text())
    for key in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "confusion_matrix",
        "class_counts",
    ):
        assert key in saved_metrics

    # Class balance JSON records both train and val splits.
    balance = json.loads((run_dir / "class_balance.json").read_text())
    assert set(balance.keys()) == {"train", "val", "test"}
    for split_name in ("train", "val", "test"):
        assert set(balance[split_name].keys()) == {"0", "1"}

    # Config snapshot round-trips.
    snap = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert snap["run_name"] == "smoke"


def test_train_from_config_writes_weights_when_requested(
    synthetic_cifar100, tmp_path: Path
):
    (x_tr, yf_tr, yc_tr), (x_te, yf_te, yc_te) = synthetic_cifar100
    train_split = _synthetic_split(x_tr, yf_tr, yc_tr, "train")
    test_split = _synthetic_split(x_te, yf_te, yc_te, "test")
    config = {
        "architecture": "baseline_cnn",
        "run_name": "smoke_weights",
        "seed": 42,
        "task": {
            "label_level": "coarse",
            "positive_label_names": ["aquatic_mammals"],
        },
        "validation": {"fraction": 0.2, "stratify": True},
        "class_imbalance": {"strategy": "none"},
        "batch_size": 16,
        "shuffle_buffer": 64,
        "dropout": 0.3,
        "epochs": 1,
        "learning_rate": 0.001,
        "early_stopping": {"monitor": "val_loss", "patience": 1},
        "output_dir": str(tmp_path),
        "save_weights": True,
        "subset_size": None,
    }
    train_from_config(
        config, train_split=train_split, test_split=test_split
    )
    assert (tmp_path / "smoke_weights" / "weights.h5").exists()
