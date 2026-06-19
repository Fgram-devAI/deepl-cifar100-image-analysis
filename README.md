# CIFAR-100 Deep Learning Benchmark

Deep-learning MSc project for NCSR Demokritos and the University of Piraeus. The project
benchmarks sequential, attention-based, and transfer-learning architectures on CIFAR-100 using
binary classification tasks derived from fine classes and superclasses.

## Project Direction

The main dataset is CIFAR-100. The project studies binary questions such as "cow vs. not cow" or
superclass-level tasks such as "aquatic mammals vs. all other classes". The goal is to compare how
different model families handle the same 32x32 visual data under controlled binary classification
settings.

The project has two parallel deliverables:

- **Local source-code implementation:** reusable Python modules for data loading, preprocessing,
  model building, training, evaluation, and ablation runs.
- **Standalone Colab notebooks:** notebook workflows that run independently in Colab and do not
  import source-code modules from this repository. They should be reproducible, instructional, and
  aligned with the same CIFAR-100 tasks.

## Roadmap

1. Define the CIFAR-100 binary task plan: one or more fine-class targets and one or more superclass
   targets.
2. Implement local data loading and preprocessing, including image view `(32, 32, 3)` and
   row-as-timestep sequence view `(32, 96)`.
3. Build baseline sequential models: RNN, LSTM, and Bi-LSTM.
4. Build attention and transfer-learning models: small ViT, MobileNetV3, EfficientNet, and ResNet.
5. Add training guardrails: config-driven runs, deterministic seeds, callbacks, metrics, and saved
   result artifacts.
6. Run ablations for data augmentation, frozen-backbone transfer learning, partial fine-tuning,
   and full fine-tuning where appropriate.
7. Create standalone Colab notebooks that mirror the main experiment stages without importing the
   local package code.
8. Compare local-code and notebook results in a final evaluation summary.

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

For Hugging Face dataset fetching:

```bash
pip install -r requirements-hf.txt
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

## Local Code Commands

```bash
# Fetch/cache CIFAR-100 through Hugging Face
venv/bin/python data/loaders.py --split both

# Optional fallback: fetch/cache CIFAR-100 through Keras
venv/bin/python data/loaders.py --source keras --split both

# Data pipeline checks
venv/bin/pytest -q tests/test_preprocessing.py tests/test_labels.py tests/test_loaders.py tests/test_tasks.py tests/test_pipeline.py tests/test_acceptance.py

# Single training run with a given config
python -m training.train --config configs/lstm.yaml

# Full ablation sweep on the configured subset/task
python -m experiments.run_ablations

# Evaluation of a trained model
python -m evaluation.ood_eval --weights results/<run>/weights.h5
```

## Notebook Workflow

Notebooks live under `notebooks/` and are intended for Colab. They should be self-contained:
dataset loading, preprocessing, model definitions, training loops, and plots should exist inside
the notebook rather than importing from `data/`, `models/`, or `training/`.

Suggested notebook sequence:

- `01_cifar100_data_exploration.ipynb`
- `02_binary_baseline_cnn.ipynb`
- `03_data_augmentation.ipynb`
- `04_transfer_learning_feature_extraction.ipynb`
- `05_fine_tuning_efficientnet_resnet.ipynb`
- `06_sequence_models.ipynb`
- `07_attention_vit_comparison.ipynb`

## Project Structure

- **`data/`** - CIFAR-100 loaders, binary task construction, preprocessing, row-as-timestep
  conversion, and masking.
- **`models/`** - sequential models, small ViT, MobileNetV3, EfficientNet, and ResNet model
  builders for the local implementation.
- **`training/`** - config-driven training loop, loss functions, callbacks, and result logging.
- **`evaluation/`** - metrics computation and evaluation utilities.
- **`experiments/`** - ablation sweep matrix and runners.
- **`notebooks/`** - standalone Colab notebooks that do not depend on local source modules.
- **`configs/`** - YAML configurations for local-code runs.
- **`results/`** - output directory for run artifacts.

## Design Constraints

- **Dataset:** CIFAR-100, using fine labels and coarse superclasses to define binary tasks.
- **Primary task style:** binary classification such as target class vs. rest or superclass vs.
  rest.
- **Input views:** sequential models consume row-wise sequences of shape `(32, 96)`; image models
  consume standard image tensors of shape `(32, 32, 3)`.
- **Hardware:** single Colab T4 target for notebooks and lightweight local runs.
- **Guardrails:** deterministic seeds, small batches, `tf.data` pipelines, and clear separation
  between training and evaluation.
- **Notebooks:** runnable independently in Colab without importing local project modules.
