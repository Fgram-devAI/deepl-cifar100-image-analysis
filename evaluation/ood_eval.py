"""OOD (out-of-distribution) evaluation on Midjourney v6 test set."""

import argparse
import json
import numpy as np
import tensorflow as tf
from pathlib import Path

# TODO: Import as they are implemented.
# from data import load_mj_ood, make_pipeline
# from evaluation.metrics import compute_metrics


def ood_eval(weights_path: str, output_dir: str = "results"):
    """
    Evaluate a trained binary model on the Midjourney OOD test set.

    Args:
        weights_path: Path to saved model weights (.h5 file).
        output_dir: Directory to save OOD evaluation results.

    Notes:
        - Midjourney set is used **only for evaluation**, never for training/tuning.
        - Reports train-vs-OOD accuracy gap (generalization measure).
        - Results are saved to output_dir/ood_eval.json.
    """
    # TODO: Implement OOD evaluation.
    # 1. Load trained model (from weights_path).
    # 2. Load Midjourney OOD dataset (load_mj_ood).
    # 3. Evaluate on OOD set (model.evaluate or manual prediction).
    # 4. Compute metrics (compute_metrics).
    # 5. Save results (metrics, optional confusion matrix) to JSON.
    raise NotImplementedError("ood_eval not yet implemented")


def main():
    """Command-line entrypoint: python -m evaluation.ood_eval --weights results/<run>/weights.h5"""
    parser = argparse.ArgumentParser(description="Evaluate a model on OOD Midjourney set.")
    parser.add_argument("--weights", type=str, required=True, help="Path to saved model weights.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for OOD results.",
    )
    args = parser.parse_args()

    ood_eval(args.weights, args.output_dir)


if __name__ == "__main__":
    main()
