"""Arsitektur CNN kustom dari notebook awal."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from preprocessing.transforms import build_data_augmentation


def build_custom_cnn(
    image_size: int,
    num_classes: int,
    model_name: str = "CNN_RiceDiseaseClassifier",
) -> keras.Model:
    """Membangun model CNN kustom.

    Model mempertahankan desain notebook dengan tambahan Dropout:
    Conv2D(16) -> MaxPooling -> Dropout ->
    Conv2D(32) -> MaxPooling -> Dropout ->
    Conv2D(64) -> MaxPooling -> Dropout ->
    Flatten -> Dense(128) -> Dropout -> Dense(num_classes).
    Output tidak memakai Softmax eksplisit sehingga loss memakai
    SparseCategoricalCrossentropy(from_logits=True).
    """
    data_augmentation = build_data_augmentation()

    model = keras.Sequential(
        [
            layers.InputLayer(input_shape=(image_size, image_size, 3)),
            layers.Rescaling(1.0 / 255),
            data_augmentation,
            layers.Conv2D(16, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.10),
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.15),
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.20),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.50),
            layers.Dense(num_classes, name="output"),
        ],
        name=model_name,
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model
