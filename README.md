# Image-as-Sequence: Sequential vs Attention Architectures for Vision

A deep-learning evaluation framework benchmarking how fundamentally different neural architectures (RNNs, LSTMs, Transformers, and transfer learning) process computer-vision data when images are fed row-by-row as a time series.

## Quick Start

### Python Environment

Use Python 3.11 for local development. The project targets TensorFlow/Keras and Colab T4, so a
dedicated virtual environment keeps the dependency set predictable.

```bash
python3.11 -m venv venv
source venv/bin/activate
python --version
pip install --upgrade pip
```

The version check should report Python 3.11.x. If your shell still shows a Conda `(base)` prompt,
either deactivate Conda first or make sure `which python` points inside `./venv/`.

### Dependencies

For the core project environment:

```bash
pip install -r requirements.txt
```

For Apple Silicon Macs, use the macOS overlay to include TensorFlow Metal acceleration:

```bash
pip install -r requirements-macos.txt
```

On Colab, TensorFlow is preinstalled. Prefer the notebook/runtime TensorFlow version and avoid
reinstalling it unless the environment explicitly requires it.

Quick TensorFlow check:

```bash
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices())"
```

### Commands

```bash
# Single training run with a given config
python -m training.train --config configs/lstm.yaml

# Full ablation sweep on the subsampled training set
python -m experiments.run_ablations

# OOD evaluation on trained binary model
python -m evaluation.ood_eval --weights results/<run>/weights.h5
```

## Git Setup

This scaffold is ready to initialize as a new repository. Keep generated data, virtual
environments, and run outputs out of git; `.gitignore` already covers `data/raw/`, `results/*`
except `results/.gitkeep`, caches, notebooks, and local env files.

```bash
git init
git add README.md CLAUDE.md CODEX_INIT_PROMPT.md requirements.txt requirements-macos.txt .gitignore
git add configs data models training evaluation experiments results/.gitkeep
git commit -m "Initial project scaffold"
```

Then create an empty remote repository on your Git host and connect it:

```bash
git remote add origin <repo-url>
git branch -M main
git push -u origin main
```

## Project Structure

- **`data/`** — dataset loaders, preprocessing (row-as-timestep conversion), and masking.
- **`models/`** — sequential (RNN/LSTM/BiLSTM), from-scratch ViT, and transfer-learning (MobileNetV3) architectures.
- **`training/`** — config-driven training loop, loss functions, and callbacks.
- **`evaluation/`** — metrics computation and OOD evaluation.
- **`experiments/`** — ablation sweep matrix and runner.
- **`configs/`** — YAML configurations for each model variant.
- **`results/`** — output directory for run artifacts (weights, metrics, config snapshots).

## Design Constraints

- **T = 32:** Images are 32×32, processed one row per timestep → sequence shape (T=32, features=96).
- **Hardware:** Single Colab T4 (16 GB) — no OOM, use `tf.data` streaming.
- **Data:** CIFAKE (120k, real vs. fake) + OOD eval on Midjourney v6 subset.
- **Dual task:** Binary (real-vs-fake, sigmoid+BCE) and 10-class (CIFAR-10 objects, softmax+CE).
- **Masking:** Keras `Masking` layer simulates missing rows; observe temporal robustness.

For locked design decisions and constraints, see `CLAUDE.md`.
