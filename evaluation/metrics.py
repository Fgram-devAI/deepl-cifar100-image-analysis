"""Metrics computation: accuracy, precision, recall, F1, ROC-AUC."""

from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str = "binary",
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels or probabilities.
        task: "binary" or "multiclass".

    Returns:
        Dictionary of metrics: accuracy, precision, recall, f1, etc.
        For binary task, also includes roc_auc.
    """
    # TODO: Implement metric computation using sklearn.
    # For binary: accuracy, precision, recall, f1, roc_auc (if y_pred are probabilities)
    # For multiclass: accuracy, macro precision/recall/f1
    raise NotImplementedError("compute_metrics not yet implemented")


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Confusion matrix of shape (num_classes, num_classes).
    """
    # TODO: Use sklearn.metrics.confusion_matrix.
    raise NotImplementedError("compute_confusion_matrix not yet implemented")
