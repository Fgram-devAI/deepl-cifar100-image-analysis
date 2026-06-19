# Spec: CIFAR-100 Data Pipeline

Branch: `feat/cifar100-data-pipeline`

## Objective

Replace the old dataset direction with a CIFAR-100 data pipeline that supports binary
classification tasks from both fine labels and coarse superclasses.

## Scope

This branch should implement the local-code data foundation only. Model training, transfer
learning, notebook authoring, and full experiment orchestration should happen in later branches.

## Required Capabilities

- Load CIFAR-100 train/test splits with both fine and coarse labels.
- Expose label-name mappings for:
  - 100 fine classes;
  - 20 coarse superclasses.
- Build binary labels for:
  - fine class vs. rest;
  - coarse superclass vs. rest.
- Report or expose class counts for each binary task.
- Produce two model-ready views:
  - image view: `(batch, 32, 32, 3)`;
  - sequence view: `(batch, 32, 96)`.
- Build `tf.data.Dataset` pipelines with batching, optional shuffle, cache, and prefetch.
- Keep downloads/loading inside explicit functions. Do not download data at import time.

## Proposed Public API

The exact names can change if the codebase suggests better local conventions, but the branch
should end with equivalent functionality:

```python
load_cifar100(split: str = "train") -> ...
get_cifar100_label_names(label_level: str) -> list[str]
make_binary_labels(labels, positive_ids) -> ...
make_cifar100_binary_task(...)
make_pipeline(..., view: str = "image" | "sequence") -> tf.data.Dataset
```

## Acceptance Criteria

- `data/preprocessing.py` implements image normalization, row-as-timestep conversion, and optional
  row masking.
- `data/loaders.py` no longer references the previous dataset direction.
- The pipeline can build at least one fine-class task, for example `cow` vs. rest.
- The pipeline can build at least one superclass task, for example `aquatic mammals` vs. rest.
- A smoke check verifies:
  - image batch shape is `(batch, 32, 32, 3)`;
  - sequence batch shape is `(batch, 32, 96)`;
  - binary labels are 0/1;
  - class counts are visible for the selected task.

## Out of Scope

- Training model architectures.
- Implementing notebooks.
- Transfer-learning backbones.
- Full experiment sweeps.
- Report writing.
