"""Preprocessing dan augmentasi data."""

from tensorflow import keras
from tensorflow.keras import layers


def build_data_augmentation() -> keras.Sequential:
    """Membuat layer augmentasi sesuai notebook awal."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
        ],
        name="data_augmentation",
    )
