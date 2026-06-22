# Notebooks

Standalone Colab notebooks live here.

Notebooks should not import local modules from `data/`, `models/`, `training/`, or `evaluation/`.
Each notebook should include the CIFAR-100 loading, preprocessing, model definitions, training,
evaluation, and plotting needed to run independently in Colab.

- [01 CIFAR-100 data exploration](01_cifar100_data_exploration.ipynb)
- [02 Baseline CNN training](02_baseline_cnn_training.ipynb)
- [03 ResNet-family transfer learning](03_resnet_family_transfer_learning.ipynb)
- [04 EfficientNetB0 coarse transfer learning](04_efficientnet_b0_transfer_learning.ipynb)
- [05 EfficientNetB0 fine transfer learning](05_efficientnet_b0_fine_transfer_learning.ipynb)
- [06 MobileNetV3 coarse frozen/unfrozen transfer learning](06_mobilenetv3_coarse_frozen_unfrozen.ipynb)
- [07 MobileNetV3 fine frozen/unfrozen transfer learning](07_mobilenetv3_fine_frozen_unfrozen.ipynb)
