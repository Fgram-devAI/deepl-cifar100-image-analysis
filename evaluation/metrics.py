"""Metrics computation for CIFAR-100 binary tasks.

Binary tasks here always use {0, 1} integer labels and sigmoid probabilities
in [0, 1]; thresholding (default 0.5) produces hard predictions.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _flatten_prob(y_prob: np.ndarray) -> np.ndarray:
    arr = np.asarray(y_prob)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.reshape(-1)
    elif arr.ndim != 1:
        raise ValueError(
            f"y_prob must be 1-D or (N, 1); got shape {arr.shape}"
        )
    return arr.astype(np.float64, copy=False)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute binary classification metrics from probabilities.

    Returns accuracy, precision, recall, f1, and roc_auc. roc_auc is NaN when
    `y_true` contains a single class (sklearn raises in that case, but the
    training loop logs the rest of the metrics anyway).
    """
    y_true_arr = np.asarray(y_true).astype(np.int64).reshape(-1)
    prob = _flatten_prob(y_prob)
    y_pred = (prob >= threshold).astype(np.int64)

    try:
        roc_auc = float(roc_auc_score(y_true_arr, prob))
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = float(average_precision_score(y_true_arr, prob))
    except ValueError:
        pr_auc = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "precision": float(
            precision_score(y_true_arr, y_pred, zero_division=0)
        ),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    """Return a 2x2 confusion matrix (rows true, cols pred) over labels [0, 1]."""
    y_true_arr = np.asarray(y_true).astype(np.int64).reshape(-1)
    prob = _flatten_prob(y_prob)
    y_pred = (prob >= threshold).astype(np.int64)
    return confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).astype(np.int64)


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
    metric: str = "f1",
) -> tuple[float, dict[str, float]]:
    """Find the threshold with best validation metric."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    best_threshold = 0.5
    best_metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    best_score = best_metrics[metric]

    for threshold in thresholds:
        metrics = compute_metrics(y_true, y_prob, threshold=float(threshold))
        score = metrics[metric]
        if score > best_score:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_score = score

    return best_threshold, best_metrics
