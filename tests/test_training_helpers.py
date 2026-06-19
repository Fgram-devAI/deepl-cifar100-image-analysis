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
