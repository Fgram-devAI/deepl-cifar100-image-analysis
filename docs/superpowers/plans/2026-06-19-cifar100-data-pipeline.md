# CIFAR-100 Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the prior CIFAKE/Midjourney data stubs with a CIFAR-100 data foundation that produces image- and sequence-view `tf.data.Dataset` pipelines for fine-class-vs-rest and superclass-vs-rest binary tasks.

**Architecture:** Pure-function data layer split across four files in `data/`: `labels.py` (label-name metadata + lookups), `loaders.py` (CIFAR-100 load + `make_pipeline`), `preprocessing.py` (normalize, row-as-timestep, row-mask), `tasks.py` (binary-task construction). Loading is dependency-injectable so tests don't hit the network. No download at import time. View contract: image `(B, 32, 32, 3)` vs sequence `(B, 32, 96)`.

**Tech Stack:** Python 3.11, TensorFlow/Keras 2.13.x, NumPy 1.24.x, pytest (added as dev dep).

## Global Constraints

- TensorFlow/Keras only; no PyTorch, no extra ML frameworks.
- Python 3.11; `tensorflow>=2.13,<2.14` and `numpy>=1.24,<2.0` per `requirements.txt`.
- No dataset download at module import — only inside explicit functions.
- Sequence view fixed at `T=32`, features `=96`. Image view fixed at `(32, 32, 3)`.
- Binary labels are integers in `{0, 1}`.
- Determinism: every public function that randomizes accepts a `seed`; defaults are explicit.
- Notebooks under `notebooks/` MUST NOT import from `data/`. This plan does not touch `notebooks/`.
- Type hints + short docstring on every public function (CLAUDE.md §12).
- Mask sentinel: `apply_row_masking` defaults to `mask_value=0.0` but documents the sentinel-collision gotcha and accepts an explicit override.
- All tests must run without network access. The real CIFAR-100 download is exercised only via an opt-in `@pytest.mark.slow` test.

---

### Task 1: Test Scaffolding and Dev Dependencies

**Files:**
- Create: `requirements-dev.txt`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_smoke.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a pytest setup where seeded fixtures (`seed`, `synthetic_cifar100`) are available to every later task's tests.

- [ ] **Step 1: Add dev requirements file**

Create `requirements-dev.txt`:

```text
-r requirements.txt

pytest>=7.4,<8.0
```

- [ ] **Step 2: Add empty test package marker**

Create `tests/__init__.py`:

```python
```

(Empty file — pytest discovers tests without it, but this prevents `tests/` from being treated as an import-search root by editors.)

- [ ] **Step 3: Add `conftest.py` with seeded fixtures**

Create `tests/conftest.py`:

```python
"""Shared pytest fixtures: seeded RNGs and a synthetic CIFAR-100 stand-in."""

from typing import Tuple

import numpy as np
import pytest
import tensorflow as tf


@pytest.fixture(autouse=True)
def _seed_everything() -> None:
    """Reset NumPy and TensorFlow seeds before every test for determinism."""
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)


@pytest.fixture
def synthetic_cifar100() -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Return a tiny shape-correct stand-in for CIFAR-100.

    Shape contract matches what `tf.keras.datasets.cifar100.load_data` returns,
    but with 200 train / 50 test samples and ids drawn from the full
    [0, 100) fine and [0, 20) coarse ranges.

    Returns:
        ((x_train, y_fine_train, y_coarse_train),
         (x_test, y_fine_test, y_coarse_test))
    """
    rng = np.random.default_rng(0)
    x_train = rng.integers(0, 256, size=(200, 32, 32, 3), dtype=np.uint8)
    y_fine_train = rng.integers(0, 100, size=(200, 1), dtype=np.int64)
    y_coarse_train = rng.integers(0, 20, size=(200, 1), dtype=np.int64)
    x_test = rng.integers(0, 256, size=(50, 32, 32, 3), dtype=np.uint8)
    y_fine_test = rng.integers(0, 100, size=(50, 1), dtype=np.int64)
    y_coarse_test = rng.integers(0, 20, size=(50, 1), dtype=np.int64)
    return (
        (x_train, y_fine_train, y_coarse_train),
        (x_test, y_fine_test, y_coarse_test),
    )
```

- [ ] **Step 4: Add a smoke test to prove the scaffold works**

Create `tests/test_smoke.py`:

```python
def test_pytest_runs():
    assert True
```

- [ ] **Step 5: Install dev deps and run the smoke test**

Run:
```bash
pip install -r requirements-dev.txt
pytest tests/test_smoke.py -v
```
Expected: `1 passed`.

- [ ] **Step 6: Commit**

```bash
git add requirements-dev.txt tests/__init__.py tests/conftest.py tests/test_smoke.py
git commit -m "test: scaffold pytest setup with seeded fixtures"
```

---

### Task 2: Preprocessing Primitives

**Files:**
- Modify: `data/preprocessing.py` (replace all three stub bodies)
- Create: `tests/test_preprocessing.py`

**Interfaces:**
- Consumes: NumPy, TensorFlow.
- Produces:
  - `normalize_images(images: np.ndarray) -> np.ndarray` — uint8 `[0,255]` → float32 `[0,1]`.
  - `to_image(images: np.ndarray) -> np.ndarray` — float32 passthrough enforcing shape `(N, 32, 32, 3)`.
  - `to_sequence(images: np.ndarray) -> np.ndarray` — reshape `(N, 32, 32, 3)` → `(N, 32, 96)` float32.
  - `apply_row_masking(seq: np.ndarray, drop_prob: float = 0.0, mask_value: float = 0.0, seed: int | None = None) -> np.ndarray` — replace whole rows with `mask_value` at probability `drop_prob`.

- [ ] **Step 1: Write failing tests for `normalize_images`, `to_image`, `to_sequence`**

Create `tests/test_preprocessing.py`:

```python
"""Unit tests for data.preprocessing primitives."""

import numpy as np
import pytest

from data.preprocessing import (
    apply_row_masking,
    normalize_images,
    to_image,
    to_sequence,
)


def test_normalize_images_scales_uint8_to_unit_float():
    x = np.array([[0, 127, 255]], dtype=np.uint8).reshape(1, 1, 3, 1)
    x = np.broadcast_to(x, (1, 32, 32, 3)).copy()
    out = normalize_images(x)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.isclose(out.max(), 1.0)


def test_to_image_preserves_shape_and_returns_float32():
    x = np.zeros((4, 32, 32, 3), dtype=np.float32)
    out = to_image(x)
    assert out.shape == (4, 32, 32, 3)
    assert out.dtype == np.float32


def test_to_image_rejects_wrong_shape():
    x = np.zeros((4, 16, 16, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        to_image(x)


def test_to_sequence_reshapes_to_T32_F96():
    x = np.arange(32 * 32 * 3, dtype=np.float32).reshape(1, 32, 32, 3)
    out = to_sequence(x)
    assert out.shape == (1, 32, 96)
    assert out.dtype == np.float32
    # Row 0 of the image must equal timestep 0, flattened across (col, channel).
    np.testing.assert_array_equal(out[0, 0], x[0, 0].reshape(96))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_preprocessing.py -v`
Expected: all four tests FAIL with `NotImplementedError` (current stubs).

- [ ] **Step 3: Implement `normalize_images`, `to_image`, `to_sequence`**

Rewrite `data/preprocessing.py`:

```python
"""Image preprocessing: normalization, row-as-timestep conversion, row masking."""

from typing import Optional

import numpy as np


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Scale uint8 images in [0, 255] to float32 in [0, 1]."""
    return (images.astype(np.float32)) / 255.0


def to_image(images: np.ndarray) -> np.ndarray:
    """Return a (N, 32, 32, 3) float32 view for CNN/transfer branches."""
    if images.ndim != 4 or images.shape[1:] != (32, 32, 3):
        raise ValueError(
            f"to_image expects shape (N, 32, 32, 3); got {images.shape}"
        )
    return images.astype(np.float32, copy=False)


def to_sequence(images: np.ndarray) -> np.ndarray:
    """
    Reshape (N, 32, 32, 3) images to (N, 32, 96) row-as-timestep sequences.

    Each row becomes one timestep; the 32 pixels x 3 channels per row become 96 features.
    """
    if images.ndim != 4 or images.shape[1:] != (32, 32, 3):
        raise ValueError(
            f"to_sequence expects shape (N, 32, 32, 3); got {images.shape}"
        )
    n = images.shape[0]
    return images.astype(np.float32, copy=False).reshape(n, 32, 96)


def apply_row_masking(
    seq: np.ndarray,
    drop_prob: float = 0.0,
    mask_value: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Randomly replace whole rows in a (batch, T=32, 96) sequence with `mask_value`.

    Sentinel-collision gotcha (CLAUDE.md §10): with `mask_value=0.0` and
    pixels normalized to [0, 1], legitimately black rows look identical to masked
    rows. If a downstream `Masking` layer relies on the sentinel, prefer an
    out-of-range value (e.g. -1.0) and pass the same value to `Masking(mask_value=...)`.
    """
    if seq.ndim != 3 or seq.shape[1:] != (32, 96):
        raise ValueError(
            f"apply_row_masking expects shape (batch, 32, 96); got {seq.shape}"
        )
    if not 0.0 <= drop_prob <= 1.0:
        raise ValueError(f"drop_prob must be in [0, 1]; got {drop_prob}")
    if drop_prob == 0.0:
        return seq.astype(np.float32, copy=False)
    rng = np.random.default_rng(seed)
    mask = rng.random(size=seq.shape[:2]) < drop_prob  # (batch, 32)
    out = seq.astype(np.float32, copy=True)
    out[mask] = mask_value
    return out
```

- [ ] **Step 4: Run preprocessing tests; expect the three non-masking ones to pass**

Run: `pytest tests/test_preprocessing.py -v`
Expected: `test_normalize_images_scales_uint8_to_unit_float`, `test_to_image_preserves_shape_and_returns_float32`, `test_to_image_rejects_wrong_shape`, `test_to_sequence_reshapes_to_T32_F96` all PASS.

- [ ] **Step 5: Add failing tests for `apply_row_masking`**

Append to `tests/test_preprocessing.py`:

```python
def test_apply_row_masking_zero_prob_is_identity():
    seq = np.ones((2, 32, 96), dtype=np.float32) * 0.5
    out = apply_row_masking(seq, drop_prob=0.0)
    np.testing.assert_array_equal(out, seq)


def test_apply_row_masking_full_prob_masks_everything():
    seq = np.ones((2, 32, 96), dtype=np.float32) * 0.5
    out = apply_row_masking(seq, drop_prob=1.0, mask_value=-1.0, seed=0)
    assert np.all(out == -1.0)


def test_apply_row_masking_is_deterministic_with_seed():
    seq = np.ones((4, 32, 96), dtype=np.float32) * 0.5
    a = apply_row_masking(seq, drop_prob=0.3, mask_value=-1.0, seed=123)
    b = apply_row_masking(seq, drop_prob=0.3, mask_value=-1.0, seed=123)
    np.testing.assert_array_equal(a, b)


def test_apply_row_masking_rejects_wrong_shape():
    seq = np.zeros((2, 16, 96), dtype=np.float32)
    with pytest.raises(ValueError):
        apply_row_masking(seq, drop_prob=0.5)
```

- [ ] **Step 6: Run; expect all preprocessing tests to pass**

Run: `pytest tests/test_preprocessing.py -v`
Expected: all 8 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add data/preprocessing.py tests/test_preprocessing.py
git commit -m "feat(data): implement preprocessing primitives for CIFAR-100"
```

---

### Task 3: CIFAR-100 Label Metadata

**Files:**
- Create: `data/labels.py`
- Create: `tests/test_labels.py`

**Interfaces:**
- Consumes: nothing (constants are inline).
- Produces:
  - `CIFAR100_FINE_LABEL_NAMES: tuple[str, ...]` — 100 names, indexed by fine id.
  - `CIFAR100_COARSE_LABEL_NAMES: tuple[str, ...]` — 20 names, indexed by coarse id.
  - `get_cifar100_label_names(label_level: Literal["fine", "coarse"]) -> tuple[str, ...]`.
  - `get_label_ids(label_level: Literal["fine", "coarse"], names: Sequence[str]) -> list[int]` — raises `KeyError` if a name is unknown.

- [ ] **Step 1: Write failing tests**

Create `tests/test_labels.py`:

```python
"""Tests for CIFAR-100 label-name metadata and lookups."""

import pytest

from data.labels import (
    CIFAR100_COARSE_LABEL_NAMES,
    CIFAR100_FINE_LABEL_NAMES,
    get_cifar100_label_names,
    get_label_ids,
)


def test_fine_label_names_length_and_uniqueness():
    assert len(CIFAR100_FINE_LABEL_NAMES) == 100
    assert len(set(CIFAR100_FINE_LABEL_NAMES)) == 100


def test_coarse_label_names_length_and_uniqueness():
    assert len(CIFAR100_COARSE_LABEL_NAMES) == 20
    assert len(set(CIFAR100_COARSE_LABEL_NAMES)) == 20


def test_known_fine_names_present():
    for name in ("cow", "apple", "orchid"):
        assert name in CIFAR100_FINE_LABEL_NAMES


def test_known_coarse_names_present():
    for name in ("aquatic_mammals", "vehicles_1", "large_carnivores"):
        assert name in CIFAR100_COARSE_LABEL_NAMES


def test_get_cifar100_label_names_dispatches_by_level():
    assert get_cifar100_label_names("fine") == CIFAR100_FINE_LABEL_NAMES
    assert get_cifar100_label_names("coarse") == CIFAR100_COARSE_LABEL_NAMES


def test_get_cifar100_label_names_rejects_unknown_level():
    with pytest.raises(ValueError):
        get_cifar100_label_names("super")  # type: ignore[arg-type]


def test_get_label_ids_returns_correct_indices():
    cow_id = CIFAR100_FINE_LABEL_NAMES.index("cow")
    assert get_label_ids("fine", ["cow"]) == [cow_id]


def test_get_label_ids_rejects_unknown_name():
    with pytest.raises(KeyError):
        get_label_ids("fine", ["not_a_real_class"])
```

- [ ] **Step 2: Run; expect ImportError / failures**

Run: `pytest tests/test_labels.py -v`
Expected: collection error — `data.labels` does not exist.

- [ ] **Step 3: Implement `data/labels.py`**

Create `data/labels.py`:

```python
"""CIFAR-100 label-name metadata (fine + coarse) and lookup helpers.

Names follow the canonical CIFAR-100 ordering used by
`tf.keras.datasets.cifar100.load_data`. Underscores join multi-word coarse
superclass names so each label is a single token (e.g. ``aquatic_mammals``).
"""

from typing import Literal, Sequence


CIFAR100_FINE_LABEL_NAMES: tuple[str, ...] = (
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose",
    "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
    "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
    "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman",
    "worm",
)

# Synonym used in CIFAR-100 docs: "cow" is the fine label id 19.
# Keras returns the name "cattle" for that id; we expose both as valid lookups.
_FINE_SYNONYMS: dict[str, str] = {"cow": "cattle"}

CIFAR100_COARSE_LABEL_NAMES: tuple[str, ...] = (
    "aquatic_mammals", "fish", "flowers", "food_containers",
    "fruit_and_vegetables", "household_electrical_devices",
    "household_furniture", "insects", "large_carnivores",
    "large_man-made_outdoor_things", "large_natural_outdoor_scenes",
    "large_omnivores_and_herbivores", "medium_mammals",
    "non-insect_invertebrates", "people", "reptiles", "small_mammals",
    "trees", "vehicles_1", "vehicles_2",
)

LabelLevel = Literal["fine", "coarse"]


def get_cifar100_label_names(label_level: LabelLevel) -> tuple[str, ...]:
    """Return the ordered tuple of label names for the requested level."""
    if label_level == "fine":
        return CIFAR100_FINE_LABEL_NAMES
    if label_level == "coarse":
        return CIFAR100_COARSE_LABEL_NAMES
    raise ValueError(
        f"label_level must be 'fine' or 'coarse'; got {label_level!r}"
    )


def get_label_ids(label_level: LabelLevel, names: Sequence[str]) -> list[int]:
    """Resolve label names to their integer ids; raises KeyError if unknown."""
    table = get_cifar100_label_names(label_level)
    index = {name: i for i, name in enumerate(table)}
    if label_level == "fine":
        for synonym, canonical in _FINE_SYNONYMS.items():
            index.setdefault(synonym, index[canonical])
    out: list[int] = []
    for name in names:
        if name not in index:
            raise KeyError(f"Unknown {label_level} label: {name!r}")
        out.append(index[name])
    return out
```

Note on the `cow` synonym: CIFAR-100's fine id 19 is officially named `cattle` in the Keras dataset. The spec's example task is `cow` vs. rest, so the synonym map lets callers pass either name. The test for `"cow"` membership uses `in CIFAR100_FINE_LABEL_NAMES`, which would fail under this design — fix that test now.

- [ ] **Step 4: Fix the `test_known_fine_names_present` test to use the synonym path**

Edit `tests/test_labels.py`:

```python
def test_known_fine_names_present():
    # `cattle` is the canonical Keras name; `cow` is exposed via the synonym map.
    assert "cattle" in CIFAR100_FINE_LABEL_NAMES
    assert "apple" in CIFAR100_FINE_LABEL_NAMES
    assert "orchid" in CIFAR100_FINE_LABEL_NAMES
    assert get_label_ids("fine", ["cow"]) == get_label_ids("fine", ["cattle"])
```

- [ ] **Step 5: Run; expect all label tests to pass**

Run: `pytest tests/test_labels.py -v`
Expected: all 8 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add data/labels.py tests/test_labels.py
git commit -m "feat(data): add CIFAR-100 fine/coarse label metadata and lookups"
```

---

### Task 4: CIFAR-100 Loader (Dependency-Injected)

**Files:**
- Modify: `data/loaders.py` (delete CIFAKE/Midjourney functions; keep file, add CIFAR-100 loader)
- Create: `tests/test_loaders.py`

**Interfaces:**
- Consumes: a Keras-shaped loader callable (default `tf.keras.datasets.cifar100.load_data`).
- Produces:
  - `Cifar100Split` dataclass: `images: np.ndarray (N, 32, 32, 3) uint8`, `fine_labels: np.ndarray (N,) int64`, `coarse_labels: np.ndarray (N,) int64`.
  - `load_cifar100(split: Literal["train", "test"] = "train", *, _loader: Callable | None = None) -> Cifar100Split`.

- [ ] **Step 1: Write failing loader tests using the synthetic fixture**

Create `tests/test_loaders.py`:

```python
"""Tests for data.loaders.load_cifar100 using an injected synthetic loader."""

import numpy as np
import pytest

from data.loaders import Cifar100Split, load_cifar100


def _make_fake_loader(synthetic_cifar100, label_mode_holder):
    (train_fine, test_fine) = synthetic_cifar100[0], synthetic_cifar100[1]

    def fake(label_mode: str = "fine"):
        label_mode_holder.append(label_mode)
        x_train, y_fine_train, y_coarse_train = train_fine
        x_test, y_fine_test, y_coarse_test = test_fine
        y_train = y_fine_train if label_mode == "fine" else y_coarse_train
        y_test = y_fine_test if label_mode == "fine" else y_coarse_test
        return (x_train, y_train), (x_test, y_test)

    return fake


def test_load_cifar100_train_split_shapes(synthetic_cifar100):
    seen: list[str] = []
    loader = _make_fake_loader(synthetic_cifar100, seen)
    out = load_cifar100("train", _loader=loader)
    assert isinstance(out, Cifar100Split)
    assert out.images.shape == (200, 32, 32, 3)
    assert out.images.dtype == np.uint8
    assert out.fine_labels.shape == (200,)
    assert out.coarse_labels.shape == (200,)
    assert "fine" in seen and "coarse" in seen


def test_load_cifar100_test_split_shapes(synthetic_cifar100):
    loader = _make_fake_loader(synthetic_cifar100, [])
    out = load_cifar100("test", _loader=loader)
    assert out.images.shape == (50, 32, 32, 3)
    assert out.fine_labels.shape == (50,)
    assert out.coarse_labels.shape == (50,)


def test_load_cifar100_rejects_unknown_split(synthetic_cifar100):
    loader = _make_fake_loader(synthetic_cifar100, [])
    with pytest.raises(ValueError):
        load_cifar100("val", _loader=loader)  # type: ignore[arg-type]


def test_labels_are_1d_int64(synthetic_cifar100):
    loader = _make_fake_loader(synthetic_cifar100, [])
    out = load_cifar100("train", _loader=loader)
    assert out.fine_labels.ndim == 1
    assert out.coarse_labels.ndim == 1
    assert out.fine_labels.dtype == np.int64
    assert out.coarse_labels.dtype == np.int64
```

- [ ] **Step 2: Run; expect import error**

Run: `pytest tests/test_loaders.py -v`
Expected: collection error — `Cifar100Split` and the new `load_cifar100` don't exist yet (the old `data/loaders.py` exports `load_cifake`).

- [ ] **Step 3: Replace `data/loaders.py` with CIFAR-100 loader + dataclass**

Rewrite `data/loaders.py` (we'll add `make_pipeline` in Task 6):

```python
"""CIFAR-100 dataset loading. Downloads happen only inside `load_cifar100`."""

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np
import tensorflow as tf


Split = Literal["train", "test"]


@dataclass(frozen=True)
class Cifar100Split:
    """A CIFAR-100 split with both label levels.

    images: (N, 32, 32, 3) uint8
    fine_labels: (N,) int64 in [0, 100)
    coarse_labels: (N,) int64 in [0, 20)
    """

    images: np.ndarray
    fine_labels: np.ndarray
    coarse_labels: np.ndarray


def load_cifar100(
    split: Split = "train",
    *,
    _loader: Optional[Callable] = None,
) -> Cifar100Split:
    """Load a CIFAR-100 split with both fine and coarse labels.

    Args:
        split: ``"train"`` or ``"test"``.
        _loader: Test-only injection point with the same signature as
            ``tf.keras.datasets.cifar100.load_data(label_mode=...)``.

    The Keras loader caches to ``~/.keras/datasets/`` and only downloads on
    first call. Nothing in this module triggers a download at import time.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test'; got {split!r}")

    loader = _loader if _loader is not None else tf.keras.datasets.cifar100.load_data
    (x_train_f, y_train_f), (x_test_f, y_test_f) = loader(label_mode="fine")
    (_, y_train_c), (_, y_test_c) = loader(label_mode="coarse")

    if split == "train":
        images, y_fine, y_coarse = x_train_f, y_train_f, y_train_c
    else:
        images, y_fine, y_coarse = x_test_f, y_test_f, y_test_c

    return Cifar100Split(
        images=np.asarray(images, dtype=np.uint8),
        fine_labels=np.asarray(y_fine, dtype=np.int64).reshape(-1),
        coarse_labels=np.asarray(y_coarse, dtype=np.int64).reshape(-1),
    )
```

- [ ] **Step 4: Run loader tests; expect all to pass**

Run: `pytest tests/test_loaders.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Add opt-in real-download test (skipped by default)**

Append to `tests/test_loaders.py`:

```python
@pytest.mark.slow
def test_load_cifar100_real_download_train_shapes():
    """Opt-in test that hits the Keras CIFAR-100 download. Run with `-m slow`."""
    out = load_cifar100("train")
    assert out.images.shape == (50000, 32, 32, 3)
    assert out.fine_labels.shape == (50000,)
    assert out.coarse_labels.shape == (50000,)
    assert int(out.fine_labels.max()) == 99
    assert int(out.coarse_labels.max()) == 19
```

Register the `slow` marker so pytest doesn't warn. Create `pytest.ini` in repo root:

```ini
[pytest]
markers =
    slow: opt-in tests that need network / heavy resources
addopts = -m "not slow"
```

- [ ] **Step 6: Run; confirm slow test is collected but skipped by default**

Run: `pytest tests/test_loaders.py -v`
Expected: 4 PASS, 1 deselected (the `slow` one).

- [ ] **Step 7: Commit**

```bash
git add data/loaders.py tests/test_loaders.py pytest.ini
git commit -m "feat(data): replace CIFAKE stubs with CIFAR-100 loader"
```

---

### Task 5: Binary Task Construction

**Files:**
- Create: `data/tasks.py`
- Create: `tests/test_tasks.py`

**Interfaces:**
- Consumes: `Cifar100Split` from `data.loaders`, label helpers from `data.labels`.
- Produces:
  - `BinaryTask` dataclass: `images: np.ndarray (N, 32, 32, 3) uint8`, `binary_labels: np.ndarray (N,) int64 in {0,1}`, `class_counts: dict[int, int]`, `metadata: dict[str, object]`.
  - `make_binary_labels(labels: np.ndarray, positive_ids: Sequence[int]) -> np.ndarray` — returns int64 array of 0/1.
  - `class_counts(binary_labels: np.ndarray) -> dict[int, int]`.
  - `make_cifar100_binary_task(split_data: Cifar100Split, *, label_level: Literal["fine","coarse"], positive_label_names: Sequence[str], seed: int = 42) -> BinaryTask`.

- [ ] **Step 1: Write failing tests for `make_binary_labels` and `class_counts`**

Create `tests/test_tasks.py`:

```python
"""Tests for binary-task construction from CIFAR-100 splits."""

import numpy as np
import pytest

from data.loaders import Cifar100Split
from data.tasks import (
    BinaryTask,
    class_counts,
    make_binary_labels,
    make_cifar100_binary_task,
)


def test_make_binary_labels_marks_positive_ids():
    labels = np.array([0, 1, 2, 3, 1, 2], dtype=np.int64)
    out = make_binary_labels(labels, positive_ids=[1, 2])
    np.testing.assert_array_equal(out, np.array([0, 1, 1, 0, 1, 1], dtype=np.int64))
    assert out.dtype == np.int64


def test_make_binary_labels_empty_positive_ids_returns_all_zero():
    labels = np.array([0, 1, 2], dtype=np.int64)
    out = make_binary_labels(labels, positive_ids=[])
    np.testing.assert_array_equal(out, np.zeros(3, dtype=np.int64))


def test_class_counts_returns_int_keys_and_values():
    bin_labels = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    counts = class_counts(bin_labels)
    assert counts == {0: 2, 1: 3}
```

- [ ] **Step 2: Run; expect import / failure**

Run: `pytest tests/test_tasks.py -v`
Expected: collection error — `data.tasks` does not exist.

- [ ] **Step 3: Implement `make_binary_labels` and `class_counts`**

Create `data/tasks.py`:

```python
"""Binary-task construction over CIFAR-100 splits.

A binary task labels a chosen positive class (or set of classes) as 1 and
everything else as 0, at either the fine or coarse label level.
"""

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from data.labels import LabelLevel, get_label_ids
from data.loaders import Cifar100Split


@dataclass(frozen=True)
class BinaryTask:
    """Materialized binary task plus the metadata needed to reproduce it."""

    images: np.ndarray  # (N, 32, 32, 3) uint8
    binary_labels: np.ndarray  # (N,) int64 in {0, 1}
    class_counts: dict[int, int]
    metadata: dict = field(default_factory=dict)


def make_binary_labels(
    labels: np.ndarray, positive_ids: Sequence[int]
) -> np.ndarray:
    """Return a 0/1 int64 array where positives are members of `positive_ids`."""
    positive_set = set(int(i) for i in positive_ids)
    mask = np.isin(labels, list(positive_set)) if positive_set else np.zeros_like(
        labels, dtype=bool
    )
    return mask.astype(np.int64)


def class_counts(binary_labels: np.ndarray) -> dict[int, int]:
    """Return {0: n_neg, 1: n_pos}; missing classes appear with count 0."""
    values, counts = np.unique(binary_labels, return_counts=True)
    out = {0: 0, 1: 0}
    for v, c in zip(values.tolist(), counts.tolist()):
        out[int(v)] = int(c)
    return out


def make_cifar100_binary_task(
    split_data: Cifar100Split,
    *,
    label_level: LabelLevel,
    positive_label_names: Sequence[str],
    seed: int = 42,
) -> BinaryTask:
    """Build a binary task from a loaded CIFAR-100 split.

    The metadata block records the positive definition, negative definition,
    class counts, and seed so downstream code can log a reproducible task spec.
    """
    if label_level == "fine":
        source_labels = split_data.fine_labels
    elif label_level == "coarse":
        source_labels = split_data.coarse_labels
    else:
        raise ValueError(
            f"label_level must be 'fine' or 'coarse'; got {label_level!r}"
        )

    positive_ids = get_label_ids(label_level, positive_label_names)
    binary = make_binary_labels(source_labels, positive_ids=positive_ids)
    counts = class_counts(binary)

    metadata = {
        "label_level": label_level,
        "positive_label_names": list(positive_label_names),
        "positive_ids": positive_ids,
        "negative_definition": "all other classes",
        "n_total": int(binary.shape[0]),
        "class_counts": counts,
        "seed": seed,
    }
    return BinaryTask(
        images=split_data.images,
        binary_labels=binary,
        class_counts=counts,
        metadata=metadata,
    )
```

- [ ] **Step 4: Run; the first three tests should pass**

Run: `pytest tests/test_tasks.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Add failing tests for `make_cifar100_binary_task` (fine + coarse)**

Append to `tests/test_tasks.py`:

```python
def _synthetic_split(n=20):
    rng = np.random.default_rng(0)
    return Cifar100Split(
        images=rng.integers(0, 256, size=(n, 32, 32, 3), dtype=np.uint8),
        fine_labels=np.arange(n, dtype=np.int64) % 100,
        coarse_labels=np.arange(n, dtype=np.int64) % 20,
    )


def test_make_cifar100_binary_task_fine_cow_vs_rest():
    split = _synthetic_split(n=200)
    task = make_cifar100_binary_task(
        split,
        label_level="fine",
        positive_label_names=["cow"],
    )
    assert isinstance(task, BinaryTask)
    assert set(np.unique(task.binary_labels).tolist()).issubset({0, 1})
    assert task.metadata["label_level"] == "fine"
    assert task.metadata["positive_label_names"] == ["cow"]
    assert task.class_counts[1] > 0
    assert task.class_counts[0] + task.class_counts[1] == 200


def test_make_cifar100_binary_task_coarse_aquatic_vs_rest():
    split = _synthetic_split(n=100)
    task = make_cifar100_binary_task(
        split,
        label_level="coarse",
        positive_label_names=["aquatic_mammals"],
    )
    assert task.metadata["label_level"] == "coarse"
    assert task.class_counts[1] > 0
    assert task.class_counts[0] + task.class_counts[1] == 100


def test_make_cifar100_binary_task_rejects_unknown_level():
    split = _synthetic_split()
    with pytest.raises(ValueError):
        make_cifar100_binary_task(
            split,
            label_level="super",  # type: ignore[arg-type]
            positive_label_names=["cow"],
        )
```

- [ ] **Step 6: Run; expect all task tests to pass**

Run: `pytest tests/test_tasks.py -v`
Expected: 6 PASS.

- [ ] **Step 7: Commit**

```bash
git add data/tasks.py tests/test_tasks.py
git commit -m "feat(data): binary task construction for fine and coarse splits"
```

---

### Task 6: `make_pipeline` Building Two-View `tf.data.Dataset`s

**Files:**
- Modify: `data/loaders.py` (append `make_pipeline`)
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `BinaryTask` (or raw `images, labels` numpy arrays) and preprocessing primitives from Task 2.
- Produces:
  - `make_pipeline(images: np.ndarray, labels: np.ndarray, *, view: Literal["image","sequence"], batch_size: int = 32, shuffle: bool = False, cache: bool = False, prefetch: bool = True, shuffle_buffer: int = 1024, seed: int | None = None) -> tf.data.Dataset`.
  - Yielded elements: `(images: tf.float32, labels: tf.int64)`; image batches shaped `(B, 32, 32, 3)` and sequence batches shaped `(B, 32, 96)`. Labels are unchanged 1-D int64.

- [ ] **Step 1: Write failing pipeline tests**

Create `tests/test_pipeline.py`:

```python
"""Tests for data.loaders.make_pipeline view contract."""

import numpy as np
import pytest
import tensorflow as tf

from data.loaders import make_pipeline


@pytest.fixture
def small_arrays():
    images = (np.arange(8 * 32 * 32 * 3, dtype=np.uint8) % 256).reshape(
        8, 32, 32, 3
    )
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    return images, labels


def test_image_view_batch_shape(small_arrays):
    images, labels = small_arrays
    ds = make_pipeline(images, labels, view="image", batch_size=4)
    x, y = next(iter(ds))
    assert tuple(x.shape) == (4, 32, 32, 3)
    assert tuple(y.shape) == (4,)
    assert x.dtype == tf.float32
    assert y.dtype == tf.int64


def test_image_view_pixels_normalized_to_unit_range(small_arrays):
    images, labels = small_arrays
    ds = make_pipeline(images, labels, view="image", batch_size=4)
    x, _ = next(iter(ds))
    assert float(tf.reduce_min(x).numpy()) >= 0.0
    assert float(tf.reduce_max(x).numpy()) <= 1.0


def test_sequence_view_batch_shape(small_arrays):
    images, labels = small_arrays
    ds = make_pipeline(images, labels, view="sequence", batch_size=2)
    x, y = next(iter(ds))
    assert tuple(x.shape) == (2, 32, 96)
    assert tuple(y.shape) == (2,)
    assert x.dtype == tf.float32


def test_shuffle_with_seed_is_deterministic(small_arrays):
    images, labels = small_arrays
    a = list(make_pipeline(images, labels, view="image", batch_size=8, shuffle=True, seed=7))
    b = list(make_pipeline(images, labels, view="image", batch_size=8, shuffle=True, seed=7))
    np.testing.assert_array_equal(a[0][1].numpy(), b[0][1].numpy())


def test_unknown_view_raises(small_arrays):
    images, labels = small_arrays
    with pytest.raises(ValueError):
        make_pipeline(images, labels, view="other", batch_size=2)  # type: ignore[arg-type]
```

- [ ] **Step 2: Run; expect ImportError**

Run: `pytest tests/test_pipeline.py -v`
Expected: collection error — `make_pipeline` is gone (we replaced `data/loaders.py` in Task 4).

- [ ] **Step 3: Append `make_pipeline` to `data/loaders.py`**

Append to `data/loaders.py`:

```python
View = Literal["image", "sequence"]


def make_pipeline(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    view: View,
    batch_size: int = 32,
    shuffle: bool = False,
    cache: bool = False,
    prefetch: bool = True,
    shuffle_buffer: int = 1024,
    seed: Optional[int] = None,
) -> tf.data.Dataset:
    """Build a `tf.data.Dataset` yielding (images, labels) in the chosen view.

    Image view yields ``(B, 32, 32, 3)`` float32 in [0, 1]; sequence view yields
    ``(B, 32, 96)`` float32 in [0, 1]. Shuffle is deterministic when ``seed`` is
    given. Cache is in-memory and only useful for splits that comfortably fit.
    """
    if view not in ("image", "sequence"):
        raise ValueError(f"view must be 'image' or 'sequence'; got {view!r}")
    if images.ndim != 4 or images.shape[1:] != (32, 32, 3):
        raise ValueError(
            f"images must have shape (N, 32, 32, 3); got {images.shape}"
        )
    if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
        raise ValueError(
            f"labels must be 1-D with len == N; got {labels.shape}"
        )

    images_f32 = images.astype(np.float32, copy=False) / 255.0
    if view == "sequence":
        images_f32 = images_f32.reshape(images_f32.shape[0], 32, 96)

    labels_i64 = labels.astype(np.int64, copy=False)

    ds = tf.data.Dataset.from_tensor_slices((images_f32, labels_i64))
    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(
            buffer_size=shuffle_buffer,
            seed=seed,
            reshuffle_each_iteration=False,
        )
    ds = ds.batch(batch_size, drop_remainder=False)
    if prefetch:
        ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
```

- [ ] **Step 4: Run pipeline tests; expect all to pass**

Run: `pytest tests/test_pipeline.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Run the full suite; expect a green tree**

Run: `pytest -v`
Expected: all tests across `test_smoke`, `test_preprocessing`, `test_labels`, `test_loaders`, `test_tasks`, `test_pipeline` PASS; the `slow` test is deselected.

- [ ] **Step 6: Commit**

```bash
git add data/loaders.py tests/test_pipeline.py
git commit -m "feat(data): tf.data pipeline with image and sequence views"
```

---

### Task 7: Package Exports, End-to-End Smoke Check, Acceptance

**Files:**
- Modify: `data/__init__.py`
- Create: `tests/test_acceptance.py`
- Modify: `data/README.md` (optional — only if existing wording becomes stale)

**Interfaces:**
- Consumes: all prior tasks.
- Produces: public `data` API surfacing the names the spec proposes.

- [ ] **Step 1: Rewrite `data/__init__.py` to export the new API**

Replace `data/__init__.py`:

```python
"""Data module: CIFAR-100 loaders, label metadata, binary tasks, pipelines."""

from data.labels import (
    CIFAR100_COARSE_LABEL_NAMES,
    CIFAR100_FINE_LABEL_NAMES,
    get_cifar100_label_names,
    get_label_ids,
)
from data.loaders import Cifar100Split, load_cifar100, make_pipeline
from data.preprocessing import (
    apply_row_masking,
    normalize_images,
    to_image,
    to_sequence,
)
from data.tasks import BinaryTask, class_counts, make_binary_labels, make_cifar100_binary_task

__all__ = [
    "BinaryTask",
    "CIFAR100_COARSE_LABEL_NAMES",
    "CIFAR100_FINE_LABEL_NAMES",
    "Cifar100Split",
    "apply_row_masking",
    "class_counts",
    "get_cifar100_label_names",
    "get_label_ids",
    "load_cifar100",
    "make_binary_labels",
    "make_cifar100_binary_task",
    "make_pipeline",
    "normalize_images",
    "to_image",
    "to_sequence",
]
```

- [ ] **Step 2: Write the end-to-end acceptance test**

Create `tests/test_acceptance.py`:

```python
"""End-to-end acceptance test for the CIFAR-100 data pipeline.

Walks through the spec's acceptance criteria with a synthetic CIFAR-100
stand-in (no network). Verifies:
  - image batch shape (B, 32, 32, 3)
  - sequence batch shape (B, 32, 96)
  - binary labels are 0/1
  - class counts are visible for the selected task
"""

import numpy as np

from data import (
    BinaryTask,
    Cifar100Split,
    load_cifar100,
    make_cifar100_binary_task,
    make_pipeline,
)


def _fake_loader_factory(synthetic_cifar100):
    (train, test) = synthetic_cifar100

    def fake(label_mode: str = "fine"):
        x_tr, yf_tr, yc_tr = train
        x_te, yf_te, yc_te = test
        y_tr = yf_tr if label_mode == "fine" else yc_tr
        y_te = yf_te if label_mode == "fine" else yc_te
        return (x_tr, y_tr), (x_te, y_te)

    return fake


def test_acceptance_fine_class_image_view(synthetic_cifar100):
    loader = _fake_loader_factory(synthetic_cifar100)
    split = load_cifar100("train", _loader=loader)
    assert isinstance(split, Cifar100Split)

    task = make_cifar100_binary_task(
        split, label_level="fine", positive_label_names=["cow"]
    )
    assert isinstance(task, BinaryTask)
    # binary labels are 0/1
    assert set(np.unique(task.binary_labels).tolist()).issubset({0, 1})
    # class counts visible
    assert task.class_counts.keys() == {0, 1}
    assert task.class_counts[0] + task.class_counts[1] == task.binary_labels.shape[0]

    ds = make_pipeline(
        task.images, task.binary_labels, view="image", batch_size=8
    )
    x, y = next(iter(ds))
    assert tuple(x.shape) == (8, 32, 32, 3)
    assert tuple(y.shape) == (8,)


def test_acceptance_coarse_superclass_sequence_view(synthetic_cifar100):
    loader = _fake_loader_factory(synthetic_cifar100)
    split = load_cifar100("train", _loader=loader)

    task = make_cifar100_binary_task(
        split,
        label_level="coarse",
        positive_label_names=["aquatic_mammals"],
    )
    assert set(np.unique(task.binary_labels).tolist()).issubset({0, 1})
    assert task.class_counts[0] + task.class_counts[1] == task.binary_labels.shape[0]

    ds = make_pipeline(
        task.images, task.binary_labels, view="sequence", batch_size=4
    )
    x, y = next(iter(ds))
    assert tuple(x.shape) == (4, 32, 96)
    assert tuple(y.shape) == (4,)
```

- [ ] **Step 3: Run the full suite green**

Run: `pytest -v`
Expected: every test across `test_smoke`, `test_preprocessing`, `test_labels`, `test_loaders`, `test_tasks`, `test_pipeline`, `test_acceptance` PASS; one `slow` test deselected.

- [ ] **Step 4: Confirm no `notebooks/` change crept in**

Run: `git status --porcelain notebooks/`
Expected: empty output.

- [ ] **Step 5: Confirm import-time has no side effects**

Run:
```bash
python -c "import time; t=time.time(); import data; print(f'imported in {time.time()-t:.3f}s, exports={sorted(data.__all__)}')"
```
Expected: import completes well under a second (no Keras dataset download); `exports=` lists every name from `__all__`.

- [ ] **Step 6: Commit**

```bash
git add data/__init__.py tests/test_acceptance.py
git commit -m "feat(data): public exports + end-to-end CIFAR-100 acceptance test"
```

---

## Spec Coverage Self-Check

- Load CIFAR-100 with both fine and coarse → Task 4 (`load_cifar100` + `Cifar100Split`).
- Label-name mappings for 100 fine + 20 coarse → Task 3.
- Binary labels for fine-vs-rest and coarse-vs-rest → Tasks 3 + 5.
- Class counts exposed → Task 5 (`class_counts`, `BinaryTask.class_counts`, `BinaryTask.metadata`).
- Image view `(B, 32, 32, 3)` and sequence view `(B, 32, 96)` → Task 6.
- `tf.data` with batch, shuffle, cache, prefetch → Task 6.
- No download at import time → Task 7 step 5 verifies via timed import; Task 4 confines downloads to `load_cifar100`.
- `data/preprocessing.py` implements normalization, row-as-timestep, optional masking → Task 2.
- `data/loaders.py` no longer references the prior dataset → Task 4 rewrites it whole.
- Fine task (`cow` vs rest) buildable → Task 7 acceptance test.
- Superclass task (`aquatic_mammals` vs rest) buildable → Task 7 acceptance test.
- Smoke check verifying shapes, 0/1 labels, class counts → Task 7 acceptance test.
