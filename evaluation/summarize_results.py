"""Summarize experiment results from multiple training runs into a CSV.

Usage::

    python -m evaluation.summarize_results --results-dir results --output results/summary.csv

Each training run directory is expected to contain:
- ``metrics.json``: required — the run is skipped if absent or malformed.
- ``config.yaml``: optional — used to populate ``task_type``, ``label_level``,
  and ``target_labels``.  If missing, those columns are left as empty strings.

Both flat and grouped result layouts are supported:

- ``results/<run_name>/metrics.json``
- ``results/<task_name>/<run_name>/metrics.json``
- ``results/binary/<label_level>/<task_name>/<run_name>/metrics.json``
- ``results/multiclass/<label_level>/<run_name>/metrics.json``
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml

# Fixed column order used in every CSV row.
COLUMNS = [
    "task_name",
    "run_name",
    "task_type",
    "label_level",
    "target_labels",
    # Binary metrics
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "threshold",
    # Multiclass metrics
    "top_3_accuracy",
    "top_5_accuracy",
    "macro_f1",
    "weighted_f1",
]


def _load_json(path: Path) -> dict[str, Any] | None:
    """Return parsed JSON from *path*, or ``None`` on any error."""
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: could not read {path}: {exc}", file=sys.stderr)
        return None


def _load_yaml(path: Path) -> dict[str, Any] | None:
    """Return parsed YAML from *path*, or ``None`` on any error (silently)."""
    try:
        with path.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return None


def _str(value: Any) -> str:
    """Convert *value* to string, returning empty string for ``None``."""
    if value is None:
        return ""
    return str(value)


def _iter_run_dirs(results_dir: Path) -> list[Path]:
    """Return directories below *results_dir* that contain ``metrics.json``."""
    return sorted(
        {
            metrics_path.parent
            for metrics_path in results_dir.rglob("metrics.json")
            if metrics_path.is_file()
        }
    )


def _task_name_for_run(results_dir: Path, run_dir: Path) -> str:
    """Return the grouped task folder name for *run_dir*, or empty if flat."""
    relative = run_dir.relative_to(results_dir)
    if len(relative.parts) >= 4 and relative.parts[0] == "binary":
        return relative.parts[2]
    if len(relative.parts) >= 3 and relative.parts[0] == "multiclass":
        return f"{relative.parts[1]}_multiclass"
    if len(relative.parts) >= 2:
        return relative.parts[0]
    return ""


def summarize_runs(results_dir: Path) -> list[dict[str, str]]:
    """Scan *results_dir* recursively for training runs and collect metrics.

    Each run directory is expected to contain ``metrics.json`` and, optionally,
    ``config.yaml``.  Both flat ``results/<run_name>`` and grouped
    ``results/<task_name>/<run_name>`` layouts are supported.

    Parameters
    ----------
    results_dir:
        Root directory whose immediate children are individual run directories.

    Returns
    -------
    list[dict[str, str]]
        A list of flat row dicts.  Every dict has exactly the keys listed in
        :data:`COLUMNS`.  Missing metric keys become empty strings; the list
        never contains ``None`` values.
    """
    rows: list[dict[str, str]] = []

    if not results_dir.is_dir():
        print(
            f"WARNING: results directory does not exist: {results_dir}",
            file=sys.stderr,
        )
        return rows

    for run_dir in _iter_run_dirs(results_dir):
        metrics_path = run_dir / "metrics.json"
        metrics = _load_json(metrics_path)
        if metrics is None:
            # Warning already printed by _load_json.
            continue

        # Start with a blank row so every column is always present.
        row: dict[str, str] = {col: "" for col in COLUMNS}
        row["task_name"] = _task_name_for_run(results_dir, run_dir)
        row["run_name"] = run_dir.name

        # --- Config (optional) -------------------------------------------
        config_path = run_dir / "config.yaml"
        config = _load_yaml(config_path)
        if config is not None:
            task_section = config.get("task") or {}
            task_type = task_section.get("type", "")
            row["label_level"] = _str(task_section.get("label_level", ""))
            positive_labels = task_section.get("positive_label_names")
            if positive_labels:
                row["target_labels"] = ";".join(str(lbl) for lbl in positive_labels)
            if task_type:
                row["task_type"] = _str(task_type)
            elif positive_labels:
                row["task_type"] = "binary"
            else:
                row["task_type"] = "unknown"
        else:
            row["task_type"] = "unknown"

        # --- Binary metrics -----------------------------------------------
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "threshold"):
            row[key] = _str(metrics.get(key))

        # --- Multiclass metrics -------------------------------------------
        for key in ("top_3_accuracy", "top_5_accuracy", "macro_f1", "weighted_f1"):
            row[key] = _str(metrics.get(key))

        rows.append(row)

    return rows


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    """Write *rows* to *output_path* as a CSV with a fixed column order.

    The header is always written, even when *rows* is empty.

    Parameters
    ----------
    rows:
        Flat dicts as returned by :func:`summarize_runs`.
    output_path:
        Destination file.  Parent directories are created if they do not exist.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Summarize experiment results into a CSV file.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        type=Path,
        help="Directory containing per-run sub-directories (default: results).",
    )
    parser.add_argument(
        "--output",
        default="results/summary.csv",
        type=Path,
        help="Output CSV path (default: results/summary.csv).",
    )
    args = parser.parse_args()

    rows = summarize_runs(args.results_dir)
    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} row(s) to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
