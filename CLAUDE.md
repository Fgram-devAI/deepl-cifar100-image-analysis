# CLAUDE.md — Image-as-Sequence: Sequential vs Attention Architectures for Vision

> Persistent context for coding agents working in this repo. Read this before writing or
> changing code. The design decisions below are **locked** (settled in the design phase) —
> do not re-litigate them; implement against them.

## 1. What this project is

A deep-learning evaluation framework that benchmarks how fundamentally different neural
architectures process the **same** computer-vision data. The central comparison is
**sequential processing (RNN family) vs. parallel attention (Transformer) vs. transfer
learning (CNN baseline)**, with images fed to the sequential models **row-by-row as a time
series**.

It is a one-week, single-GPU academic project. Rigor and clean comparisons matter more than
state-of-the-art accuracy.

## 2. Stack & environment

- **Framework: TensorFlow / Keras.** Chosen because (a) the `Masking` layer is a hard project
  requirement and is native to Keras, and (b) the RNN/LSTM/Bi-LSTM core is the bulk of the work
  and is cleanest in Keras. *This is the one swappable decision — if the stack changes to
  PyTorch + timm, this file and the module layout change with it.*
- **Python 3.11.**
- **Hardware target: a single Colab T4 (16 GB).** All code must run there without OOM.
- Pin dependencies loosely and **match Colab's preinstalled TensorFlow** — reinstalling TF on
  Colab is slow and frequently breaks CUDA. See `requirements.txt`.

## 3. Hard constraints (non-negotiable)

1. **No OOM on a T4, ever.** Default to small batches and `tf.data` streaming, not in-memory
   arrays for the full set.
2. **Fast iteration.** A single ablation run should be minutes, not hours. Use the subsampled
   training set (Section 5) for sweeps; reserve the full set for final canonical runs only.
3. **Sequence length is fixed at T = 32.** Images are 32×32, processed one row per timestep.
   Do not introduce resolutions that raise T — the vanilla RNN baseline collapses past ~30
   steps, which would break the core comparison.
4. **Never train, validate, or tune on the Midjourney OOD set.** It is final-eval-only.

## 4. Core design decisions (locked)

- **Row-as-timestep tensor shape.** A `32×32×3` image becomes a sequence of shape
  `(T=32, features=96)` where each timestep is one flattened RGB row (`32 px × 3 ch`). Keep this
  contract in `data/preprocessing.py`; every sequential model consumes `(batch, 32, 96)`.
- **Primary dataset: CIFAKE** (120k images, native 32×32; real = CIFAR-10, fake = SD v1.4).
- **OOD test set: Midjourney v6 CIFAKE-inspired** (~4k, 32×32). Train-on-CIFAKE → test-on-MJ to
  produce a generator-era generalization study.
- **Dual task on one pipeline** (this satisfies the loss-comparison requirement without a second
  dataset):
  - **Binary** real-vs-fake → single logit + **sigmoid + binary cross-entropy**.
  - **10-class** object classification on the *real* (CIFAR-10) subset → **softmax + categorical
    cross-entropy**.
- **Masking experiment.** Simulate "missing rows" by zeroing/sentinel-ing random rows, then let a
  `Masking` layer skip them. Used to observe how temporal architectures tolerate corruption.

## 5. Data handling

- Place downloaded data under `data/raw/` (gitignored). See `data/README.md` for Kaggle
  download steps. Expected CIFAKE Kaggle slug:
  `birdy654/cifake-real-and-ai-generated-synthetic-images` — **verify before relying on it.**
- **Ablation subset:** balanced ~15–25k images, fixed seed, cached. All sweeps run on this.
- **Final runs:** full 120k, one canonical run per architecture.
- Loaders return `tf.data.Dataset`. Provide both the `(batch, 32, 96)` sequential view and the
  `(batch, 32, 32, 3)` image view (the transfer/CNN branch needs the image view).

### Synthetic newer-generator extension

- Optional extension: create a capped synthetic set of **at most 500 images** using selected
  regions from real dataset images as the prompt source.
- The intended workflow is: select/crop a specific region of a real image, run a vision model over
  that region to produce a structured description, convert the description into a generation
  prompt, regenerate the scene with a newer image model/provider such as a Higgsfield subscription,
  then resize/store the result in the same 32×32-compatible pipeline.
- Keep this extension separate from the core CIFAKE/Midjourney benchmark. Use explicit manifests
  that record source image id, crop box or mask, original label, generated label, prompt, provider,
  model/version, seed if available, and generation date.
- Preferred first use: treat the generated images as an **evaluation-only newer-generator OOD
  set**. Train on CIFAKE only, then test on this synthetic set to measure whether detectors
  generalize to newer generated imagery.
- Optional second use: a separate augmentation experiment where generated images are added to the
  training set. Do not train on and test on the same generated images; split manifests clearly
  into `synthetic_train_aug` and `synthetic_eval` if both are used.
- Do not let this extension compromise the locked core comparison: sequential models still consume
  `(batch, 32, 96)`, image models still consume `(batch, 32, 32, 3)`, and the Midjourney OOD set
  remains final-eval-only.

## 6. Models to implement (`models/`)

- `sequential.py` — `build_rnn`, `build_lstm`, `build_bilstm`. Each accepts `activation`
  (`"tanh"` | `"relu"`), `use_masking` (bool), and `head` (`"binary"` | `"multiclass"`).
  Input → optional `Masking` → recurrent core → dense head.
- `vit.py` — from-scratch small ViT: patch embedding, a single-head `Attention` layer (for the
  requirement) **and** `MultiHeadAttention`, positional embeddings, transformer encoder blocks,
  classification head. Keep it tiny (e.g. patch 4×4 → 64 patches, 1–2 blocks, dim ≤ 128) so it
  fits the T4 and the week.
- `transfer.py` — `MobileNetV3Small` with **frozen backbone**, new head, as the upper benchmark.

## 7. Experiment matrix (keep runs aligned to this)

| Axis | Values |
|---|---|
| Architecture | vanilla RNN, LSTM, Bi-LSTM, scratch-ViT, MobileNetV3 (transfer) |
| Hidden activation (RNN family) | `tanh` vs `relu` |
| Output head / loss | sigmoid+BCE (binary) vs softmax+categorical CE (10-class) |
| Masking | off vs on (with row-corruption) |
| Final | OOD eval on Midjourney v6 for the best binary models |

Run sweeps on the subset; promote winners to full-data runs; finish with the OOD eval.

## 8. Gotchas — must remember

- **Masking sentinel collision.** `Masking(mask_value=0.0)` will wrongly skip any *real* row that
  happens to be all-zeros after normalization. Choose the mask sentinel so it cannot collide with
  valid normalized pixels (e.g. mask rows to `0.0` **before** scaling to a strictly-positive
  range, or use an out-of-range sentinel and matching `mask_value`). Document the choice in code.
- **ReLU in a vanilla RNN explodes.** When `activation="relu"` on the recurrent core, enable
  gradient clipping (`clipnorm`) or it will NaN. Report this as a finding, not a bug.
- **MobileNetV3 hates 32×32.** Pretrained convnets underperform on tiny inputs. In the transfer
  branch (which is *not* row-as-timestep), resize 32×32 up to the backbone's comfort zone (e.g.
  96×96) before the backbone. This does not affect the sequential models' T=32 constraint.
- **Determinism.** Set and log a global seed per run; results must be reproducible for the report.

## 9. Commands

```bash
# Setup (Colab: skip the TF line, use preinstalled)
pip install -r requirements.txt

# Single training run, config-driven
python -m training.train --config configs/lstm.yaml

# Full ablation sweep (subset)
python -m experiments.run_ablations

# OOD evaluation of a trained binary model
python -m evaluation.ood_eval --weights results/<run>/weights.h5
```

## 10. Conventions

- **Config-driven** (`configs/*.yaml`); no hard-coded hyperparameters in model/training code.
- Every run writes to `results/<run_name>/` (config snapshot, metrics JSON, training curves).
- Prefer `tf.data` pipelines over loading full arrays into memory.
- Type hints + short docstrings on public functions; keep modules single-responsibility.
