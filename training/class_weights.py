"""Class-weight computation for imbalanced binary tasks."""

import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def compute_balanced_class_weights(labels: np.ndarray) -> dict[int, float]:
    """Return sklearn's 'balanced' class weights for binary labels.

    When only one class is present the inverse-frequency formula is
    undefined, so we return unit weights — the caller should also log that
    the imbalance strategy was effectively disabled for that split.
    """
    labels = np.asarray(labels).reshape(-1)
    classes = np.unique(labels)
    if classes.size < 2:
        return {0: 1.0, 1: 1.0}
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=labels
    )
    return {int(c): float(w) for c, w in zip(classes, weights)}
