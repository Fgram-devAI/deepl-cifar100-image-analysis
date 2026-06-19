"""Metrics computation for CIFAR-100 binary and multiclass tasks.

Binary tasks here always use {0, 1} integer labels and sigmoid probabilities
in [0, 1]; thresholding (default 0.5) produces hard predictions.

Multiclass tasks use integer labels in [0, C) and a (N, C) probability matrix
where each row sums to 1 (softmax output).
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


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    top_k: tuple[int, ...] = (3, 5),
) -> dict:
    """Multiclass metrics from class-probability matrix.

    y_true: (N,) int64 in [0, C).
    y_prob: (N, C) float in [0, 1], rows sum to 1.
    top_k: iterable of k values. Each top_k_accuracy entry is included only
    when k <= C; entries with k > C are omitted (NOT NaN).

    Returns: accuracy, top_<k>_accuracy for valid k, macro_/weighted_
    precision/recall/f1 (all with zero_division=0), confusion_matrix (list
    of lists for JSON), class_counts ({"0": n, "1": n, ...} for JSON).
    """
    y_true_arr = np.asarray(y_true).astype(np.int64).reshape(-1)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)

    C = y_prob_arr.shape[1]
    y_pred = np.argmax(y_prob_arr, axis=1)

    metrics: dict = {}

    # Accuracy
    metrics["accuracy"] = float(accuracy_score(y_true_arr, y_pred))

    # Top-k accuracies (only when k <= C)
    for k in top_k:
        if k <= C:
            top_k_indices = np.argsort(-y_prob_arr, axis=1)[:, :k]
            correct = (top_k_indices == y_true_arr[:, None]).any(axis=1)
            metrics[f"top_{k}_accuracy"] = float(correct.mean())

    # Macro averages
    metrics["macro_precision"] = float(
        precision_score(y_true_arr, y_pred, average="macro", zero_division=0)
    )
    metrics["macro_recall"] = float(
        recall_score(y_true_arr, y_pred, average="macro", zero_division=0)
    )
    metrics["macro_f1"] = float(
        f1_score(y_true_arr, y_pred, average="macro", zero_division=0)
    )

    # Weighted averages
    metrics["weighted_precision"] = float(
        precision_score(y_true_arr, y_pred, average="weighted", zero_division=0)
    )
    metrics["weighted_recall"] = float(
        recall_score(y_true_arr, y_pred, average="weighted", zero_division=0)
    )
    metrics["weighted_f1"] = float(
        f1_score(y_true_arr, y_pred, average="weighted", zero_division=0)
    )

    # Confusion matrix as nested list (JSON-serializable)
    cm = confusion_matrix(y_true_arr, y_pred, labels=np.arange(C))
    metrics["confusion_matrix"] = cm.astype(int).tolist()

    # Class counts as string-keyed dict (JSON-serializable)
    metrics["class_counts"] = {
        str(c): int((y_true_arr == c).sum()) for c in range(C)
    }

    return metrics


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
