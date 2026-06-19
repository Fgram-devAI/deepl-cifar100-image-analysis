# Claude Implementation Plan: CIFAR-100 Data Pipeline

Branch: `feat/cifar100-data-pipeline`

## Working Principles

- Keep this branch focused on data loading and preprocessing only.
- Prefer simple TensorFlow/Keras APIs before adding new dependencies.
- Preserve the two-view contract:
  - sequential models: `(batch, 32, 96)`;
  - image models: `(batch, 32, 32, 3)`.
- Keep public functions small, typed, and documented.
- Add smoke checks before broader training code.

## Step Plan

1. **Read current data stubs.**
   - Inspect `data/loaders.py`, `data/preprocessing.py`, and `data/__init__.py`.
   - Remove stale references to the previous dataset direction.

2. **Implement preprocessing first.**
   - Normalize images to `float32`.
   - Implement `to_image`.
   - Implement `to_sequence` as reshape from `(N, 32, 32, 3)` to `(N, 32, 96)`.
   - Implement row masking with an explicit sentinel policy.

3. **Add CIFAR-100 label metadata.**
   - Define fine class names and coarse superclass names.
   - Add helpers for label-name to label-id lookup.
   - Keep mappings explicit and easy to inspect.

4. **Implement CIFAR-100 loading.**
   - Load train/test splits explicitly.
   - Return images with fine and coarse labels available.
   - Avoid loading/downloading at import time.

5. **Implement binary task construction.**
   - Support fine class vs. rest.
   - Support coarse superclass vs. rest.
   - Return binary labels and class counts.

6. **Implement `tf.data` pipelines.**
   - Support image and sequence views.
   - Support shuffle, batch, cache, and prefetch options.
   - Keep defaults small and Colab-safe.

7. **Add smoke verification.**
   - Prefer a lightweight script or test using a small subset.
   - Verify shapes, dtypes, labels, and counts.

8. **Update data docs.**
   - Document the task helpers and example task definitions.
   - Keep notebooks independent from local data helpers.

## Suggested First Tasks

- Start with `cow` vs. rest using fine labels.
- Then add `aquatic mammals` vs. rest using coarse labels.
- Use a small subset to keep smoke checks fast.

## Completion Checklist

- [ ] No stale previous-dataset wording in `data/`.
- [ ] Fine label mapping exists.
- [ ] Coarse label mapping exists.
- [ ] Fine class binary task works.
- [ ] Superclass binary task works.
- [ ] Image view pipeline works.
- [ ] Sequence view pipeline works.
- [ ] Smoke check passes.
- [ ] Data docs reflect the implemented API.
