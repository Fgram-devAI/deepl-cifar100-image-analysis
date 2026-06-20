"""Tests for evaluation.summarize_results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from evaluation.summarize_results import COLUMNS, summarize_runs, write_csv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BINARY_METRICS = {
    "accuracy": 0.9113,
    "precision": 0.243,
    "recall": 0.366,
    "f1": 0.292,
    "roc_auc": 0.789,
    "pr_auc": 0.216,
    "threshold": 0.34,
    "confusion_matrix": [[8930, 570], [317, 183]],
    "class_counts": {"0": 9500, "1": 500},
}

MULTICLASS_METRICS = {
    "accuracy": 0.72,
    "top_3_accuracy": 0.91,
    "top_5_accuracy": 0.95,
    "macro_f1": 0.68,
    "weighted_f1": 0.71,
    "confusion_matrix": [[10, 1], [2, 9]],
    "class_counts": {"0": 11, "1": 11},
}

BINARY_CONFIG = {
    "run_name": "run_a",
    "task": {
        "type": "binary",
        "label_level": "coarse",
        "positive_label_names": ["aquatic_mammals"],
    },
}

MULTICLASS_CONFIG = {
    "run_name": "run_mc",
    "task": {
        "type": "multiclass",
        "label_level": "fine",
    },
}


def _make_run(
    results_dir: Path,
    run_name: str,
    metrics: dict | None = None,
    config: dict | None = None,
) -> Path:
    """Create a run directory with optional metrics.json and config.yaml."""
    run_dir = results_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if metrics is not None:
        (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    if config is not None:
        (run_dir / "config.yaml").write_text(yaml.dump(config), encoding="utf-8")
    return run_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_summarize_binary_run_produces_expected_row(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _make_run(results_dir, "run_a", metrics=BINARY_METRICS, config=BINARY_CONFIG)

    rows = summarize_runs(results_dir)

    assert len(rows) == 1
    row = rows[0]
    assert row["task_name"] == ""
    assert row["run_name"] == "run_a"
    assert row["task_type"] == "binary"
    assert row["label_level"] == "coarse"
    assert row["target_labels"] == "aquatic_mammals"
    assert row["accuracy"] == str(BINARY_METRICS["accuracy"])
    assert row["precision"] == str(BINARY_METRICS["precision"])
    assert row["f1"] == str(BINARY_METRICS["f1"])
    assert row["threshold"] == str(BINARY_METRICS["threshold"])
    # Multiclass-only fields must be empty for binary runs
    assert row["top_3_accuracy"] == ""
    assert row["top_5_accuracy"] == ""
    assert row["macro_f1"] == ""
    assert row["weighted_f1"] == ""


def test_summarize_multiclass_run_produces_expected_row(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _make_run(results_dir, "run_mc", metrics=MULTICLASS_METRICS, config=MULTICLASS_CONFIG)

    rows = summarize_runs(results_dir)

    assert len(rows) == 1
    row = rows[0]
    assert row["task_name"] == ""
    assert row["task_type"] == "multiclass"
    assert row["label_level"] == "fine"
    assert row["target_labels"] == ""
    assert row["top_3_accuracy"] == str(MULTICLASS_METRICS["top_3_accuracy"])
    assert row["macro_f1"] == str(MULTICLASS_METRICS["macro_f1"])
    assert row["weighted_f1"] == str(MULTICLASS_METRICS["weighted_f1"])
    # Binary-only fields should be empty
    assert row["threshold"] == ""
    assert row["roc_auc"] == ""
    assert row["pr_auc"] == ""


def test_summarize_skips_directories_without_metrics_json(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    # Empty directory — no metrics.json
    (results_dir / "run_empty").mkdir(parents=True)
    # Directory with config only — no metrics.json
    _make_run(results_dir, "run_no_metrics", config=BINARY_CONFIG)
    # Populated directory
    _make_run(results_dir, "run_ok", metrics=BINARY_METRICS, config=BINARY_CONFIG)

    rows = summarize_runs(results_dir)

    assert len(rows) == 1
    assert rows[0]["run_name"] == "run_ok"


def test_summarize_handles_missing_config_yaml(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _make_run(results_dir, "run_no_cfg", metrics=BINARY_METRICS)  # no config

    rows = summarize_runs(results_dir)

    assert len(rows) == 1
    row = rows[0]
    assert row["run_name"] == "run_no_cfg"
    assert row["task_type"] == "unknown"
    assert row["label_level"] == ""
    assert row["target_labels"] == ""
    # Metrics still populated from metrics.json
    assert row["accuracy"] == str(BINARY_METRICS["accuracy"])
    assert row["f1"] == str(BINARY_METRICS["f1"])


def test_write_csv_writes_header_for_empty_rows(tmp_path: Path) -> None:
    output_path = tmp_path / "out.csv"
    write_csv([], output_path)

    assert output_path.exists()
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1  # header only
    assert lines[0] == ",".join(COLUMNS)


def test_write_csv_round_trips(tmp_path: Path) -> None:
    row = {col: "" for col in COLUMNS}
    row["run_name"] = "run_a"
    row["task_type"] = "binary"
    row["accuracy"] = "0.91"
    row["f1"] = "0.29"

    output_path = tmp_path / "out.csv"
    write_csv([row], output_path)

    with output_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        read_rows = list(reader)

    assert len(read_rows) == 1
    read_row = read_rows[0]
    assert read_row["run_name"] == "run_a"
    assert read_row["task_type"] == "binary"
    assert read_row["accuracy"] == "0.91"
    assert read_row["f1"] == "0.29"
    # Unpopulated fields round-trip as empty strings
    assert read_row["top_3_accuracy"] == ""


def test_summarize_handles_multiple_target_labels(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    config = {
        "run_name": "run_multi_label",
        "task": {
            "type": "binary",
            "label_level": "coarse",
            "positive_label_names": ["aquatic_mammals", "fish"],
        },
    }
    _make_run(results_dir, "run_multi_label", metrics=BINARY_METRICS, config=config)

    rows = summarize_runs(results_dir)

    assert len(rows) == 1
    assert rows[0]["target_labels"] == "aquatic_mammals;fish"


def test_summarize_infers_legacy_binary_config_without_task_type(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    config = {
        "run_name": "legacy_binary",
        "task": {
            "label_level": "coarse",
            "positive_label_names": ["food_containers"],
        },
    }
    _make_run(results_dir, "legacy_binary", metrics=BINARY_METRICS, config=config)

    rows = summarize_runs(results_dir)

    assert len(rows) == 1
    assert rows[0]["task_type"] == "binary"
    assert rows[0]["target_labels"] == "food_containers"


def test_summarize_supports_grouped_task_run_layout(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    run_dir = results_dir / "food_containers" / "baseline_cnn_lr_3e-5"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(BINARY_METRICS),
        encoding="utf-8",
    )
    (run_dir / "config.yaml").write_text(
        yaml.dump(BINARY_CONFIG),
        encoding="utf-8",
    )

    rows = summarize_runs(results_dir)

    assert len(rows) == 1
    assert rows[0]["task_name"] == "food_containers"
    assert rows[0]["run_name"] == "baseline_cnn_lr_3e-5"
    assert rows[0]["task_type"] == "binary"


def test_summarize_supports_nested_binary_layout(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    run_dir = results_dir / "binary" / "coarse" / "flowers" / "run_a"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(BINARY_METRICS),
        encoding="utf-8",
    )
    (run_dir / "config.yaml").write_text(
        yaml.dump(BINARY_CONFIG),
        encoding="utf-8",
    )

    rows = summarize_runs(results_dir)

    assert len(rows) == 1
    assert rows[0]["task_name"] == "flowers"
    assert rows[0]["run_name"] == "run_a"


def test_summarize_supports_nested_multiclass_layout(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    run_dir = results_dir / "multiclass" / "coarse" / "run_mc"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(MULTICLASS_METRICS),
        encoding="utf-8",
    )
    (run_dir / "config.yaml").write_text(
        yaml.dump(MULTICLASS_CONFIG),
        encoding="utf-8",
    )

    rows = summarize_runs(results_dir)

    assert len(rows) == 1
    assert rows[0]["task_name"] == "coarse_multiclass"
    assert rows[0]["run_name"] == "run_mc"
