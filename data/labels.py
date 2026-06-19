"""CIFAR-100 label-name metadata (fine + coarse) and lookup helpers.

Names follow the canonical CIFAR-100 ordering used by
``tf.keras.datasets.cifar100.load_data``. Underscores join multi-word coarse
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

# Friendly synonyms — CIFAR-100 fine id 19 is officially "cattle"; the spec uses
# "cow" as a task name, so callers may pass either.
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
