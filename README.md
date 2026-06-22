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

For the core project environment, including tests, notebooks, and Hugging Face dataset loading:

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

## Local Code Commands

```bash
# Fetch/cache CIFAR-100 through Hugging Face
venv/bin/python data/loaders.py --split both

# Optional fallback: fetch/cache CIFAR-100 through Keras
venv/bin/python data/loaders.py --source keras --split both

# Data pipeline checks
venv/bin/pytest -q tests/test_preprocessing.py tests/test_labels.py tests/test_loaders.py tests/test_tasks.py tests/test_pipeline.py tests/test_acceptance.py

# Baseline CNN binary task (fine `cow` vs. rest, default config)
venv/bin/python -m training.train --config configs/binary/fine/baseline_cnn_cow.yaml

# Baseline CNN binary coarse task (aquatic_mammals vs. rest)
venv/bin/python -m training.train --config configs/binary/coarse/baseline_cnn_aquatic_mammals.yaml

# Baseline CNN binary coarse task (flowers vs. rest)
venv/bin/python -m training.train --config configs/binary/coarse/baseline_cnn_flowers.yaml

# Baseline CNN coarse multiclass (20 superclasses)
venv/bin/python -m training.train --config configs/multiclass/baseline_cnn_coarse.yaml

# Baseline CNN fine multiclass (100 classes; longer run)
venv/bin/python -m training.train --config configs/multiclass/baseline_cnn_fine.yaml

# EfficientNetB0 coarse multiclass transfer learning (20 superclasses)
venv/bin/python -m training.train --config configs/transfer/efficientnet/multiclass/efficientnet_b0_coarse.yaml

# EfficientNetB0 fine multiclass transfer learning (100 classes; longer run)
venv/bin/python -m training.train --config configs/transfer/efficientnet/multiclass/efficientnet_b0_fine.yaml

# Summarize the results/ directory into a CSV
venv/bin/python -m evaluation.summarize_results --results-dir results --output results/summary.csv

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

- [01 CIFAR-100 data exploration](notebooks/01_cifar100_data_exploration.ipynb)
- [02 Baseline CNN training](notebooks/02_baseline_cnn_training.ipynb)
  [![Open 02 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fgram-devAI/deepl-cifar100-image-analysis/blob/main/notebooks/02_baseline_cnn_training.ipynb)
- [03 ResNet-family transfer learning](notebooks/03_resnet_family_transfer_learning.ipynb)
  [![Open 03 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fgram-devAI/deepl-cifar100-image-analysis/blob/main/notebooks/03_resnet_family_transfer_learning.ipynb)
- [04 EfficientNetB0 coarse transfer learning](notebooks/04_efficientnet_b0_transfer_learning.ipynb)
  [![Open 04 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fgram-devAI/deepl-cifar100-image-analysis/blob/main/notebooks/04_efficientnet_b0_transfer_learning.ipynb)
- [05 EfficientNetB0 fine transfer learning](notebooks/05_efficientnet_b0_fine_transfer_learning.ipynb)
  [![Open 05 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fgram-devAI/deepl-cifar100-image-analysis/blob/main/notebooks/05_efficientnet_b0_fine_transfer_learning.ipynb)
- [06 MobileNetV3 coarse frozen/unfrozen transfer learning](notebooks/06_mobilenetv3_coarse_frozen_unfrozen.ipynb)
  [![Open 06 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fgram-devAI/deepl-cifar100-image-analysis/blob/main/notebooks/06_mobilenetv3_coarse_frozen_unfrozen.ipynb)
- [07 MobileNetV3 fine frozen/unfrozen transfer learning](notebooks/07_mobilenetv3_fine_frozen_unfrozen.ipynb)
  [![Open 07 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fgram-devAI/deepl-cifar100-image-analysis/blob/main/notebooks/07_mobilenetv3_fine_frozen_unfrozen.ipynb)
- [08 EfficientNetB3 fine transfer learning](notebooks/08_efficientnet_b3_fine_transfer_learning.ipynb)
  [![Open 08 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fgram-devAI/deepl-cifar100-image-analysis/blob/main/notebooks/08_efficientnet_b3_fine_transfer_learning.ipynb)
- Future notebooks: data augmentation, sequence models, and
  attention/ViT comparison.

Imported EfficientNet run summaries:

| Run | Task | Accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| EfficientNetB0 coarse frozen | 20 coarse classes | 0.7396 | 0.7385 |
| EfficientNetB0 fine frozen | 100 fine classes | 0.6863 | 0.6831 |
| EfficientNetB0 fine fine-tuned | 100 fine classes | 0.7850 | 0.7840 |
| EfficientNetB0 fine unfreeze block 6 | 100 fine classes | 0.7671 | 0.7661 |
| EfficientNetB3 fine unfreeze block 6 | 100 fine classes | 0.7819 | 0.7806 |
| EfficientNetB3 fine fine-tuned | 100 fine classes | 0.8321 | 0.8319 |

Imported MobileNetV3Small run summaries:

| Run | Task | Top-1 | Top-5 | Macro F1 | Cohen κ |
| --- | --- | ---: | ---: | ---: | ---: |
| MobileNetV3Small coarse frozen | 20 coarse classes | 0.7939 | 0.9693 | 0.7935 | 0.7831 |
| MobileNetV3Small coarse unfrozen | 20 coarse classes | 0.8408 | 0.9832 | 0.8409 | 0.8324 |
| MobileNetV3Small fine frozen | 100 fine classes | 0.6929 | 0.9213 | 0.6902 | 0.6898 |
| MobileNetV3Small fine unfrozen | 100 fine classes | 0.7461 | 0.9482 | 0.7445 | 0.7435 |

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
