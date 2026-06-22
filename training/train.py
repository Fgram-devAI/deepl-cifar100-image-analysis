"""Config-driven training entrypoint for the baseline CNN."""

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np
import tensorflow as tf
import yaml

from data import (
    Cifar100Split,
    load_cifar100,
    make_cifar100_binary_task,
    make_pipeline,
)
from data.tasks import make_cifar100_multiclass_task
from evaluation.metrics import (
    compute_confusion_matrix,
    compute_metrics,
    compute_multiclass_metrics,
    find_best_threshold,
)
from models.baseline import build_baseline_cnn, build_strong_cnn
from models.efficientnet_b0 import build_efficientnet_b0
from models.efficientnet_b3_fine import build_efficientnet_b3
from models.resnet_family import build_resnet_family_model
from models.sequential import build_bilstm, build_lstm, build_rnn
from training.callbacks import get_callbacks
from training.class_weights import compute_balanced_class_weights
from training.losses import get_loss
from training.optimizers import build_optimizer
from training.splits import simple_random_train_val_split, stratified_train_val_split


def load_config(config_path) -> Dict[str, Any]:
    """Load a YAML config from disk."""
    with Path(config_path).open() as f:
        return yaml.safe_load(f)


def _build_model(config: Dict[str, Any], *, num_classes: int = 1) -> tf.keras.Model:
    architecture = config.get("architecture", "baseline_cnn")
    if architecture in {"rnn", "lstm", "bilstm"}:
        builder = {
            "rnn": build_rnn,
            "lstm": build_lstm,
            "bilstm": build_bilstm,
        }[architecture]
        return builder(
            activation=str(config.get("activation", "tanh")),
            use_masking=bool(config.get("use_masking", False)),
            hidden_units=int(config.get("hidden_units", 64)),
            dropout=float(config.get("dropout", 0.2)),
            num_classes=num_classes,
        )
    if architecture == "baseline_cnn":
        return build_baseline_cnn(
            dropout=float(config.get("dropout", 0.3)),
            num_classes=num_classes,
            augmentation=config.get("augmentation"),
        )
    if architecture == "strong_cnn":
        return build_strong_cnn(
            dropout=float(config.get("dropout", 0.35)),
            num_classes=num_classes,
            augmentation=config.get("augmentation"),
        )
    if architecture == "efficientnet_b0":
        trainable_backbone = bool(config.get("trainable_backbone", False))
        if "freeze_backbone" in config:
            trainable_backbone = not bool(config["freeze_backbone"])
        return build_efficientnet_b0(
            num_classes=num_classes,
            resize_to=int(config.get("resize_to", config.get("input_size", 96))),
            dropout=float(config.get("dropout", 0.3)),
            trainable_backbone=trainable_backbone,
            weights=config.get("weights", "imagenet"),
            augmentation=config.get("augmentation"),
        )
    if architecture == "efficientnet_b3":
        trainable_backbone = bool(config.get("trainable_backbone", False))
        if "freeze_backbone" in config:
            trainable_backbone = not bool(config["freeze_backbone"])
        return build_efficientnet_b3(
            num_classes=num_classes,
            resize_to=int(config.get("resize_to", config.get("input_size", 160))),
            dropout=float(config.get("dropout", 0.4)),
            trainable_backbone=trainable_backbone,
            weights=config.get("weights", "imagenet"),
            augmentation=config.get("augmentation"),
            unfreeze_from=config.get("unfreeze_from"),
            freeze_bn=bool(config.get("freeze_bn", True)),
        )
    if architecture == "resnet_family":
        if "backbone_name" not in config:
            raise KeyError(
                "resnet_family architecture requires a 'backbone_name' key "
                "in the config (e.g. 'resnet50v2')."
            )
        return build_resnet_family_model(
            backbone_name=str(config["backbone_name"]),
            num_classes=num_classes,
            resize_to=int(config.get("resize_to", 224)),
            dropout=float(config.get("dropout", 0.2)),
            trainable_backbone=bool(config.get("trainable_backbone", False)),
            fine_tune_at=config.get("fine_tune_at"),
            weights=config.get("weights", "imagenet"),
        )
    raise ValueError(
        f"Unsupported architecture {architecture!r}. "
        "Supported: 'baseline_cnn', 'strong_cnn', 'rnn', 'lstm', 'bilstm', "
        "'efficientnet_b0', 'efficientnet_b3', 'resnet_family'."
    )


def _binary_task(
    split: Cifar100Split, config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    task_cfg = config["task"]
    task = make_cifar100_binary_task(
        split,
        label_level=task_cfg["label_level"],
        positive_label_names=task_cfg["positive_label_names"],
        seed=int(config.get("seed", 42)),
    )
    return task.images, task.binary_labels, task.class_counts


def _stratified_subset(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    subset_size: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Optionally take a deterministic stratified subset after binary labeling.

    This avoids fine-class smoke runs with zero positives, which can happen
    when a raw CIFAR-100 head slice is used before task labels are built.
    """
    if subset_size is None or subset_size >= labels.shape[0]:
        return images, labels
    if subset_size < 2:
        raise ValueError("subset_size must be at least 2 when set")

    rng = np.random.default_rng(seed)
    selected_parts: list[np.ndarray] = []
    for cls in (0, 1):
        cls_idx = np.flatnonzero(labels == cls)
        if cls_idx.size == 0:
            continue
        n_cls = max(1, int(round(subset_size * cls_idx.size / labels.shape[0])))
        n_cls = min(n_cls, cls_idx.size)
        selected_parts.append(rng.choice(cls_idx, size=n_cls, replace=False))

    selected = np.concatenate(selected_parts)
    if selected.shape[0] > subset_size:
        selected = rng.choice(selected, size=subset_size, replace=False)
    elif selected.shape[0] < subset_size:
        remaining = np.setdiff1d(np.arange(labels.shape[0]), selected, assume_unique=False)
        extra = rng.choice(
            remaining,
            size=min(subset_size - selected.shape[0], remaining.shape[0]),
            replace=False,
        )
        selected = np.concatenate([selected, extra])
    rng.shuffle(selected)
    return images[selected], labels[selected]


def _simple_subset(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    subset_size: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Optionally take a deterministic head-slice for multiclass data.

    A uniformly-shuffled slice is safe for multiclass tasks where all classes
    are plentiful; no per-class stratification is needed.
    """
    if subset_size is None or subset_size >= labels.shape[0]:
        return images, labels
    if subset_size < 2:
        raise ValueError("subset_size must be at least 2 when set")

    rng = np.random.default_rng(seed)
    idx = np.arange(labels.shape[0])
    rng.shuffle(idx)
    selected = idx[:subset_size]
    return images[selected], labels[selected]


def _class_counts_binary(labels: np.ndarray) -> Dict[str, int]:
    return {
        "0": int((labels == 0).sum()),
        "1": int((labels == 1).sum()),
    }


def _class_counts_multiclass(labels: np.ndarray, num_classes: int) -> Dict[str, int]:
    return {str(c): int((labels == c).sum()) for c in range(num_classes)}


def _make_image_pipelines(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    *,
    batch_size: int,
    shuffle_buffer: int,
    seed: int,
    view: str = "image",
) -> tuple:
    """Build train, val, and test tf.data pipelines."""
    train_ds = make_pipeline(
        x_tr, y_tr,
        view=view,
        batch_size=batch_size,
        shuffle=True,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
    )
    val_ds = make_pipeline(
        x_val, y_val,
        view=view,
        batch_size=batch_size,
        shuffle=False,
    )
    test_ds = make_pipeline(
        test_images, test_labels,
        view=view,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_ds, val_ds, test_ds


def _data_view(config: Dict[str, Any]) -> str:
    """Return the tf.data view requested by the config."""
    view = str(config.get("view", "image"))
    if view not in ("image", "sequence"):
        raise ValueError(f"view must be 'image' or 'sequence'; got {view!r}")
    return view


def _iter_all_layers(model: tf.keras.Model):
    """Yield ``model`` and every nested sublayer (including those inside
    nested Functional sub-models)."""
    yield model
    for layer in getattr(model, "layers", []):
        yield from _iter_all_layers(layer)


@contextmanager
def _frozen_trainability(model: tf.keras.Model) -> Iterator[None]:
    """Context manager: stash per-layer ``trainable``, force everything frozen,
    restore on exit.

    Keras's legacy ``.h5`` weight format pairs file values to model variables
    by index using ``layer.trainable_weights + layer.non_trainable_weights``
    order. With a nested Functional backbone, toggling ``backbone.trainable``
    between save and load reshuffles that list (e.g. BatchNorm ``gamma``/
    ``beta`` move out of the non-trainable bucket, ``moving_mean``/
    ``moving_variance`` stay in it), so position ``i`` in the file no longer
    targets the same variable. Forcing every layer frozen during the fallback
    load collapses everything into the same layer-by-layer ordering and makes
    the frozen -> fine-tune chain safe without changing how new weight files
    are saved.
    """
    states = [(layer, layer.trainable) for layer in _iter_all_layers(model)]
    try:
        for layer, _ in states:
            layer.trainable = False
        yield
    finally:
        for layer, original in states:
            layer.trainable = original


def _maybe_load_initial_weights(
    model: tf.keras.Model, config: Dict[str, Any]
) -> None:
    """Load weights from ``config['initial_weights']`` if the key is set.

    Used to chain a fine-tune run on top of a previous (e.g. frozen)
    run by reusing its saved ``weights.h5``.

    Raises:
        FileNotFoundError: If the path is set but does not exist.
    """
    path = config.get("initial_weights")
    if path is None:
        return
    weights_path = Path(path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"initial_weights path does not exist: {weights_path}"
        )
    try:
        model.load_weights(str(weights_path))
    except ValueError as exc:
        if "axes don't match array" not in str(exc):
            raise
        with _frozen_trainability(model):
            model.load_weights(str(weights_path))


def _run_binary(
    config: Dict[str, Any],
    train_split: Cifar100Split,
    test_split: Cifar100Split,
    run_dir: Path,
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute the binary classification training branch."""
    train_images, train_labels, _ = _binary_task(train_split, config)
    test_images, test_labels, test_counts = _binary_task(test_split, config)

    train_images, train_labels = _stratified_subset(
        train_images,
        train_labels,
        subset_size=config.get("subset_size"),
        seed=seed,
    )
    test_images, test_labels = _stratified_subset(
        test_images,
        test_labels,
        subset_size=config.get("subset_size"),
        seed=seed + 1,
    )
    test_counts = {
        0: int((test_labels == 0).sum()),
        1: int((test_labels == 1).sum()),
    }

    val_cfg = config["validation"]
    x_tr, y_tr, x_val, y_val = stratified_train_val_split(
        train_images,
        train_labels,
        val_fraction=float(val_cfg["fraction"]),
        seed=seed,
    )

    class_balance = {
        "train": _class_counts_binary(y_tr),
        "val": _class_counts_binary(y_val),
        "test": _class_counts_binary(test_labels),
    }
    (run_dir / "class_balance.json").write_text(json.dumps(class_balance, indent=2))

    batch_size = int(config.get("batch_size", 64))
    shuffle_buffer = int(config.get("shuffle_buffer", 4096))
    train_ds, val_ds, test_ds = _make_image_pipelines(
        x_tr, y_tr, x_val, y_val, test_images, test_labels,
        batch_size=batch_size,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
        view=_data_view(config),
    )

    strategy = config.get("class_imbalance", {}).get("strategy", "none")
    class_weight = (
        compute_balanced_class_weights(y_tr)
        if strategy == "class_weights"
        else None
    )

    model = _build_model(config, num_classes=1)
    model.compile(
        optimizer=build_optimizer(config),
        loss=get_loss("binary"),
        metrics=["accuracy"],
    )
    _maybe_load_initial_weights(model, config)

    es_cfg = config.get("early_stopping", {})
    callbacks = get_callbacks(
        run_dir,
        patience=int(es_cfg.get("patience", 5)),
        monitor=str(es_cfg.get("monitor", "val_loss")),
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(config.get("epochs", 1)),
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    val_prob = model.predict(val_ds, verbose=1).reshape(-1)
    best_threshold, val_threshold_metrics = find_best_threshold(
        y_val,
        val_prob,
        metric="f1",
    )

    y_prob = model.predict(test_ds, verbose=1).reshape(-1)
    metrics = compute_metrics(test_labels, y_prob, threshold=best_threshold)
    cm = compute_confusion_matrix(test_labels, y_prob, threshold=best_threshold)

    metrics["threshold"] = best_threshold
    metrics["validation_threshold_metrics"] = val_threshold_metrics
    metrics["confusion_matrix"] = cm.tolist()
    metrics["class_counts"] = {str(k): int(v) for k, v in test_counts.items()}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    history_dict = dict(history.history)
    (run_dir / "history.json").write_text(json.dumps(history_dict, indent=2))

    if config.get("save_weights", False):
        model.save_weights(str(run_dir / "weights.h5"))

    return history_dict, metrics


def _run_multiclass(
    config: Dict[str, Any],
    train_split: Cifar100Split,
    test_split: Cifar100Split,
    run_dir: Path,
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute the multiclass classification training branch."""
    task_cfg = config["task"]
    label_level = task_cfg["label_level"]

    train_task = make_cifar100_multiclass_task(
        train_split,
        label_level=label_level,
        seed=seed,
    )
    test_task = make_cifar100_multiclass_task(
        test_split,
        label_level=label_level,
        seed=seed,
    )

    num_classes = train_task.num_classes
    train_images, train_labels = train_task.images, train_task.labels
    test_images, test_labels = test_task.images, test_task.labels

    # Optional subset (simple random — all classes are plentiful)
    train_images, train_labels = _simple_subset(
        train_images,
        train_labels,
        subset_size=config.get("subset_size"),
        seed=seed,
    )
    test_images, test_labels = _simple_subset(
        test_images,
        test_labels,
        subset_size=config.get("subset_size"),
        seed=seed + 1,
    )

    val_cfg = config["validation"]
    x_tr, y_tr, x_val, y_val = simple_random_train_val_split(
        train_images,
        train_labels,
        val_fraction=float(val_cfg["fraction"]),
        seed=seed,
    )

    class_balance = {
        "train": _class_counts_multiclass(y_tr, num_classes),
        "val": _class_counts_multiclass(y_val, num_classes),
        "test": _class_counts_multiclass(test_labels, num_classes),
    }
    (run_dir / "class_balance.json").write_text(json.dumps(class_balance, indent=2))

    batch_size = int(config.get("batch_size", 64))
    shuffle_buffer = int(config.get("shuffle_buffer", 4096))
    train_ds, val_ds, test_ds = _make_image_pipelines(
        x_tr, y_tr, x_val, y_val, test_images, test_labels,
        batch_size=batch_size,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
        view=_data_view(config),
    )

    model = _build_model(config, num_classes=num_classes)
    model.compile(
        optimizer=build_optimizer(config),
        loss=get_loss("multiclass"),
        metrics=["accuracy"],
    )
    _maybe_load_initial_weights(model, config)

    es_cfg = config.get("early_stopping", {})
    callbacks = get_callbacks(
        run_dir,
        patience=int(es_cfg.get("patience", 5)),
        monitor=str(es_cfg.get("monitor", "val_loss")),
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(config.get("epochs", 1)),
        callbacks=callbacks,
        verbose=1,
    )

    y_prob = model.predict(test_ds, verbose=1).reshape(-1, num_classes)
    metrics = compute_multiclass_metrics(test_labels, y_prob)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    history_dict = dict(history.history)
    (run_dir / "history.json").write_text(json.dumps(history_dict, indent=2))

    if config.get("save_weights", False):
        model.save_weights(str(run_dir / "weights.h5"))

    return history_dict, metrics


def train_from_config(
    config: Dict[str, Any],
    *,
    train_split: Optional[Cifar100Split] = None,
    test_split: Optional[Cifar100Split] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run the full baseline training pipeline.

    When ``train_split`` / ``test_split`` aren't injected, they're loaded via
    :func:`data.load_cifar100`, which downloads CIFAR-100 from Hugging Face on
    first use. Tests inject synthetic splits to keep the suite offline.

    The task type is read from ``config["task"]["type"]``. It defaults to
    ``"binary"`` when the key is absent (backward-compatible). ``"multiclass"``
    activates the multiclass branch.

    Returns:
        ``(history_dict, test_metrics_dict)``.

    Raises:
        ValueError: If ``task.type`` is not ``"binary"`` or ``"multiclass"``.
    """
    seed = int(config.get("seed", 42))
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    task_cfg = config["task"]
    task_type = task_cfg.get("type", "binary")
    if task_type not in ("binary", "multiclass"):
        raise ValueError(
            f"task.type must be 'binary' or 'multiclass'; got {task_type!r}"
        )

    run_dir = Path(config["output_dir"]) / config["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    if train_split is None:
        train_split = load_cifar100("train")
    if test_split is None:
        test_split = load_cifar100("test")

    # Write config snapshot early so it's always present even on error.
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    if task_type == "binary":
        history_dict, metrics = _run_binary(
            config, train_split, test_split, run_dir, seed
        )
    else:
        history_dict, metrics = _run_multiclass(
            config, train_split, test_split, run_dir, seed
        )

    return history_dict, metrics


def train(config_path: str) -> None:
    """CLI shim: load YAML and run `train_from_config` with default splits."""
    config = load_config(config_path)
    train_from_config(config)


def main() -> None:
    """Command-line entrypoint: python -m training.train --config <path>."""
    parser = argparse.ArgumentParser(
        description="Train the baseline CNN from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help=(
            "Path to YAML config file "
            "(e.g. configs/binary/coarse/baseline_cnn_flowers.yaml)."
        ),
    )
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
