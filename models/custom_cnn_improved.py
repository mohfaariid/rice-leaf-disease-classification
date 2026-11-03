"""Arsitektur CNN kustom improved untuk eksperimen paper 6 kelas."""

import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_improved_data_augmentation() -> keras.Sequential:
    """Augmentasi yang lebih kaya tetapi masih wajar untuk citra daun padi."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.15),
        ],
        name="improved_data_augmentation",
    )


def conv_block(filters: int, dropout_rate: float) -> keras.Sequential:
    """Blok konvolusi kecil untuk ekstraksi fitur bertahap."""
    return keras.Sequential(
        [
            layers.Conv2D(filters, (3, 3), padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Conv2D(filters, (3, 3), padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(),
            layers.Dropout(dropout_rate),
        ]
    )


def build_custom_cnn_improved(
    image_size: int,
    num_classes: int,
    model_name: str = "custom_cnn_improved",
) -> keras.Model:
    """Membangun CNN kustom yang lebih stabil untuk dataset 6 kelas.

    Perbaikan utama dibanding CNN awal:
    - Blok konvolusi dibuat lebih dalam.
    - BatchNormalization membantu training lebih stabil.
    - Dropout mengurangi risiko overfitting.
    - GlobalAveragePooling2D menggantikan Flatten agar parameter lebih efisien.
    - Learning rate dibuat lebih kecil agar pembelajaran tidak terlalu agresif.
    """
    learning_rate = float(os.getenv("RICE_LEARNING_RATE", "0.0005"))

    model = keras.Sequential(
        [
            layers.InputLayer(input_shape=(image_size, image_size, 3)),
            layers.Rescaling(1.0 / 255),
            build_improved_data_augmentation(),
            conv_block(32, 0.15),
            conv_block(64, 0.20),
            conv_block(128, 0.30),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.40),
            layers.Dense(num_classes, name="output"),
        ],
        name=model_name,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model
