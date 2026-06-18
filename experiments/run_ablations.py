"""Ablation sweep: iterate the experiment matrix across architectures, activations, heads, and masking."""

import itertools
from pathlib import Path
from typing import List, Dict, Any

# TODO: Import as implemented.
# from training.train import train


def run_ablations():
    """
    Run full ablation sweep on the subsampled dataset.

    Iterates the experiment matrix and dispatches training jobs.

    Axes:
        - Architecture: RNN, LSTM, BiLSTM, ViT, Transfer
        - Activation (RNN only): tanh, relu
        - Head: binary, multiclass
        - Masking: off, on

    Notes:
        - All ablations run on the subset (15-25k images).
        - Winners are promoted to full-data runs separately.
        - OOD eval is performed after ablations, on the best binary models.
    """
    # TODO: Implement ablation sweep.
    # 1. Define axes (architectures, activations, heads, masking).
    # 2. Generate all combinations using itertools.product.
    # 3. For each combination, create a config (or load from template).
    # 4. Call training.train(config) and track results.
    # 5. Log summary of all runs (metrics, best run, etc.).
    raise NotImplementedError("run_ablations not yet implemented")


if __name__ == "__main__":
    run_ablations()
