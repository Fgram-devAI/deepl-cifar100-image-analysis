"""Tests for evaluation.metrics."""

import numpy as np
import pytest

from evaluation.metrics import (
    compute_confusion_matrix,
    compute_metrics,
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
