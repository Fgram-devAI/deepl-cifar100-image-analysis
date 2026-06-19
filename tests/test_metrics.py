"""Tests for evaluation.metrics."""

import json

import numpy as np
import pytest

from evaluation.metrics import (
    compute_confusion_matrix,
    compute_metrics,
    compute_multiclass_metrics,
    find_best_threshold,
)


def test_compute_metrics_returns_all_required_keys():
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_prob = np.array([0.1, 0.4, 0.6, 0.9], dtype=np.float32)
    metrics = compute_metrics(y_true, y_prob)
    assert set(metrics.keys()) == {
        "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"
    }
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0


def test_compute_metrics_handles_two_dimensional_probabilities():
    y_true = np.array([0, 1], dtype=np.int64)
    y_prob = np.array([[0.2], [0.8]], dtype=np.float32)
    metrics = compute_metrics(y_true, y_prob)
    assert metrics["accuracy"] == 1.0


def test_compute_metrics_applies_threshold():
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_prob = np.array([0.1, 0.6, 0.4, 0.9], dtype=np.float32)
    default = compute_metrics(y_true, y_prob)
    high = compute_metrics(y_true, y_prob, threshold=0.7)
    assert default["accuracy"] == 0.5
    assert high["accuracy"] == 0.75


def test_compute_metrics_roc_auc_falls_back_when_single_class():
    y_true = np.array([1, 1, 1, 1], dtype=np.int64)
    y_prob = np.array([0.9, 0.8, 0.7, 0.95], dtype=np.float32)
    metrics = compute_metrics(y_true, y_prob)
    assert np.isnan(metrics["roc_auc"])
    assert metrics["accuracy"] == 1.0


def test_compute_confusion_matrix_is_fixed_size_2x2():
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_prob = np.array([0.2, 0.7, 0.6, 0.9], dtype=np.float32)
    cm = compute_confusion_matrix(y_true, y_prob)
    assert cm.shape == (2, 2)
    assert cm.dtype == np.int64
    assert cm.sum() == 4


def test_compute_confusion_matrix_keeps_2x2_when_one_class_missing():
    y_true = np.array([1, 1, 1, 1], dtype=np.int64)
    y_prob = np.array([0.9, 0.8, 0.6, 0.7], dtype=np.float32)
    cm = compute_confusion_matrix(y_true, y_prob)
    assert cm.shape == (2, 2)


def test_find_best_threshold_maximizes_requested_metric():
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_prob = np.array([0.1, 0.2, 0.3, 0.9], dtype=np.float32)
    threshold, metrics = find_best_threshold(
        y_true,
        y_prob,
        thresholds=np.array([0.5, 0.25]),
        metric="f1",
    )
    assert threshold == 0.25
    assert metrics["f1"] == 1.0


# ---------------------------------------------------------------------------
# Multiclass metrics tests
# ---------------------------------------------------------------------------

def test_compute_multiclass_metrics_returns_all_keys_for_3_class():
    """3-class input: top_3_accuracy present, top_5_accuracy omitted (k > C)."""
    rng = np.random.default_rng(42)
    C = 3
    N = 30
    y_true = rng.integers(0, C, size=N).astype(np.int64)
    # Build valid prob matrix (rows sum to 1)
    raw = rng.random((N, C)).astype(np.float32)
    y_prob = raw / raw.sum(axis=1, keepdims=True)

    metrics = compute_multiclass_metrics(y_true, y_prob)

    expected_keys = {
        "accuracy",
        "top_3_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "confusion_matrix",
        "class_counts",
    }
    assert expected_keys.issubset(metrics.keys())
    assert "top_5_accuracy" not in metrics


def test_compute_multiclass_metrics_5_class_includes_top_3_and_top_5():
    """5-class input: both top_3_accuracy and top_5_accuracy present."""
    rng = np.random.default_rng(0)
    C = 5
    N = 50
    y_true = rng.integers(0, C, size=N).astype(np.int64)
    raw = rng.random((N, C)).astype(np.float32)
    y_prob = raw / raw.sum(axis=1, keepdims=True)

    metrics = compute_multiclass_metrics(y_true, y_prob)

    assert "top_3_accuracy" in metrics
    assert "top_5_accuracy" in metrics


def test_compute_multiclass_metrics_top_k_correct_on_perfect_predictions():
    """Perfect one-hot predictions yield accuracy, top_3, top_5 all == 1.0."""
    C = 5
    N = 20
    y_true = np.tile(np.arange(C, dtype=np.int64), N // C)
    # One-hot prob matrix
    y_prob = np.zeros((N, C), dtype=np.float32)
    y_prob[np.arange(N), y_true] = 1.0

    metrics = compute_multiclass_metrics(y_true, y_prob)

    assert metrics["accuracy"] == 1.0
    assert metrics["top_3_accuracy"] == 1.0
    assert metrics["top_5_accuracy"] == 1.0


def test_compute_multiclass_metrics_top_k_below_perfect_when_true_class_not_in_top_k():
    """True class outside top-3 for one sample but inside top-5; top_3 < 1.0, top_5 == 1.0."""
    C = 6
    N = 6
    # Each sample's true class matches index
    y_true = np.arange(N, dtype=np.int64)

    # Build prob matrix where for sample 0 the true class (0) is ranked 4th
    y_prob = np.zeros((N, C), dtype=np.float32)
    # Samples 1-5: true class gets highest probability
    for i in range(1, N):
        y_prob[i, i] = 0.9
        rest = [j for j in range(C) if j != i]
        for j, col in enumerate(rest):
            y_prob[i, col] = 0.02
    # Sample 0: true class 0 ranks 4th (positions 1,2,3 get higher prob)
    y_prob[0, 0] = 0.05   # rank 4
    y_prob[0, 1] = 0.40   # rank 1
    y_prob[0, 2] = 0.30   # rank 2
    y_prob[0, 3] = 0.20   # rank 3
    y_prob[0, 4] = 0.03   # rank 5
    y_prob[0, 5] = 0.02   # rank 6
    # Normalise to ensure rows sum to 1
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

    metrics = compute_multiclass_metrics(y_true, y_prob)

    assert metrics["top_3_accuracy"] < 1.0
    assert metrics["top_5_accuracy"] == 1.0


def test_compute_multiclass_metrics_confusion_matrix_is_json_serializable():
    """Entire metrics dict (including confusion_matrix) must serialize via json.dumps."""
    rng = np.random.default_rng(7)
    C = 4
    N = 40
    y_true = rng.integers(0, C, size=N).astype(np.int64)
    raw = rng.random((N, C)).astype(np.float32)
    y_prob = raw / raw.sum(axis=1, keepdims=True)

    metrics = compute_multiclass_metrics(y_true, y_prob)

    # Must not raise
    serialized = json.dumps(metrics)
    assert isinstance(serialized, str)


def test_compute_multiclass_metrics_class_counts_match_y_true():
    """Sum of class_counts values equals N."""
    rng = np.random.default_rng(99)
    C = 5
    N = 100
    y_true = rng.integers(0, C, size=N).astype(np.int64)
    raw = rng.random((N, C)).astype(np.float32)
    y_prob = raw / raw.sum(axis=1, keepdims=True)

    metrics = compute_multiclass_metrics(y_true, y_prob)

    assert sum(metrics["class_counts"].values()) == N


def test_compute_multiclass_metrics_handles_missing_classes_in_y_true():
    """y_true has only classes {0, 2} but C=3; confusion_matrix is 3x3, no NaN in macro_f1."""
    C = 3
    N = 20
    # Only classes 0 and 2 appear in y_true
    y_true = np.array([0, 2] * (N // 2), dtype=np.int64)
    raw = np.random.default_rng(11).random((N, C)).astype(np.float32)
    y_prob = raw / raw.sum(axis=1, keepdims=True)

    metrics = compute_multiclass_metrics(y_true, y_prob)

    cm = metrics["confusion_matrix"]
    assert len(cm) == C
    assert all(len(row) == C for row in cm)
    assert not np.isnan(metrics["macro_f1"])
