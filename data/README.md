# Data Module

## CIFAR-100 Dataset

The main dataset is CIFAR-100: 32x32 RGB images with 100 fine-grained object classes grouped into
20 coarse superclasses.

The project uses CIFAR-100 to build binary classification tasks, for example:

- fine class vs. rest: `cow` vs. `not cow`;
- fine class vs. rest: `apple` vs. `not apple`;
- superclass vs. rest: `aquatic mammals` vs. all other classes.

## Loading

Prefer explicit loader functions that return `tf.data.Dataset` objects. The local code loads
CIFAR-100 from the Hugging Face `uoft-cs/cifar100` dataset by default, with
`tf.keras.datasets.cifar100` available as a fallback source.

Install `requirements-hf.txt` before using the default Hugging Face backend.

No dataset download should happen at module import time.

## Preprocessing

- Images are normalized to floating-point tensors.
- Sequential models use row-as-timestep conversion from `(32, 32, 3)` to `(32, 96)`.
- Image models preserve `(32, 32, 3)` and may resize inside the model/pipeline for transfer
  learning.
- Binary task construction must record the positive label definition, negative label definition,
  split, class counts, and seed.

## Notebooks

Colab notebooks should include their own CIFAR-100 loading and preprocessing logic. Do not import
the local `data/` module from notebooks.
