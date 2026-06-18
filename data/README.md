# Data Module

## CIFAKE Dataset

CIFAKE is a dataset of 120k 32×32 images: real images from CIFAR-10 and synthetic images from Stable Diffusion v1.4.

### Download

1. Go to [Kaggle](https://www.kaggle.com/) and set up your API credentials (`~/.kaggle/kaggle.json`).
2. Download CIFAKE using:
   ```bash
   kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images -p data/raw/
   unzip data/raw/cifake-real-and-ai-generated-synthetic-images.zip -d data/raw/
   ```
3. Expected structure:
   ```
   data/raw/
   └── cifake/
       ├── REAL/      (CIFAR-10 real images)
       └── FAKE/      (Stable Diffusion synthetic images)
   ```

## Midjourney OOD Test Set

The Midjourney v6 subset (~4k CIFAKE-inspired images) is used **for evaluation only**. Do **not** train, validate, or tune on this set.

### Download

```bash
# TODO: Add download link/procedure for MJ v6 OOD set
# Expected structure:
# data/raw/midjourney_ood/
#   └── images/  (4k 32×32 images)
```

## Preprocessing

- Images are normalized to [0, 1].
- Reshaped from (32, 32, 3) to (32, 96) for sequential models (row-as-timestep).
- Image view (32, 32, 3) is preserved for the CNN transfer branch.

## Ablation Subset

Sweeps run on a balanced ~15–25k-image subset (configurable). The full 120k set is reserved for final canonical runs.
