"""Callbacks for training: early stopping, metrics logging, etc."""

import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, CSVLogger
import json
from pathlib import Path


class JSONLogger(Callback):
    """
    Log metrics to a JSON file after each epoch.

    Useful for tracking metrics in a format that's easy to parse post-training.
    """

    def __init__(self, filepath: str, **kwargs):
        """
        Args:
            filepath: Path to save JSON file.
        """
        super().__init__(**kwargs)
        self.filepath = Path(filepath)
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        """Record epoch metrics to history."""
        # TODO: Implement JSON logging.
        # Append epoch metrics to self.history and write to self.filepath.
        raise NotImplementedError("JSONLogger.on_epoch_end not yet implemented")


def get_callbacks(
    output_dir: str,
    patience: int = 5,
    monitor: str = "val_loss",
) -> list:
    """
    Get a list of callbacks for training.

    Args:
        output_dir: Directory to save callback outputs.
        patience: Early stopping patience.
        monitor: Metric to monitor for early stopping.

    Returns:
        List of `keras.callbacks.Callback` instances.
    """
    # TODO: Implement callback construction.
    # - EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True)
    # - CSVLogger (optional)
    # - JSONLogger
    # Return list of callbacks.
    raise NotImplementedError("get_callbacks not yet implemented")
