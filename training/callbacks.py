"""Callbacks for training: early stopping, CSV history, JSON history."""

import json
from pathlib import Path
from typing import Union

import tensorflow as tf

Callback = tf.keras.callbacks.Callback
CSVLogger = tf.keras.callbacks.CSVLogger
EarlyStopping = tf.keras.callbacks.EarlyStopping


class JSONLogger(Callback):
    """Append per-epoch metrics to an in-memory dict and rewrite a JSON file.

    Useful for downstream summaries that prefer JSON over CSV (e.g. result
    aggregation across runs). The full history is rewritten every epoch so a
    partial run still leaves a readable file.
    """

    def __init__(self, filepath: Union[str, Path], **kwargs):
        super().__init__(**kwargs)
        self.filepath = Path(filepath)
        self.history: dict[str, list[float]] = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            self.history.setdefault(k, []).append(float(v))
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with self.filepath.open("w") as f:
            json.dump(self.history, f, indent=2)


def get_callbacks(
    output_dir: Union[str, Path],
    *,
    patience: int = 5,
    monitor: str = "val_loss",
) -> list[Callback]:
    """Build the standard training callback list.

    EarlyStopping restores best weights so the final saved model corresponds
    to the best monitored epoch, not the last one.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
        ),
        CSVLogger(str(output_dir / "history.csv")),
        JSONLogger(output_dir / "history.json"),
    ]
