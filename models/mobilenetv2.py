"""Arsitektur MobileNetV2 sebagai model pembanding."""

import os

from tensorflow import keras
from tensorflow.keras import layers

from preprocessing.transforms import build_data_augmentation


def build_mobilenetv2(image_size: int, num_classes: int) -> keras.Model:
    """Membangun MobileNetV2 transfer learning."""
    base_model = keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = build_data_augmentation()(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs, outputs, name="MobileNetV2_RiceDiseaseClassifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_mobilenetv2_base_model(model: keras.Model) -> keras.Model:
    """Mengambil base model MobileNetV2 dari model transfer learning."""
    for layer in model.layers:
        if isinstance(layer, keras.Model) and layer.name.startswith("mobilenetv2"):
            return layer
    raise ValueError("Base model MobileNetV2 tidak ditemukan pada model.")


def prepare_mobilenetv2_fine_tuning(
    model: keras.Model,
    trainable_layers: int | None = None,
    learning_rate: float | None = None,
) -> keras.Model:
    """Membuka sebagian layer akhir MobileNetV2 unppppppppppp-tuk fine-tuning."""
    if trainable_layers is None:
        trainable_layers = int(os.getenv("RICE_FINE_TUNE_LAYERS", "30"))
    if learning_rate is None:
        learning_rate = float(os.getenv("RICE_FINE_TUNE_LR", "0.00001"))

    base_model = get_mobilenetv2_base_model(model)
    base_model.trainable = True

    freeze_until = max(0, len(base_model.layers) - trainable_layers)
    for index, layer in enumerate(base_model.layers):
        layer.trainable = index >= freeze_until
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("=" * 70)
    print("KONFIGURASI FINE-TUNING MOBILENETV2")
    print("=" * 70)
    print(f"Total layer base model     : {len(base_model.layers)}")
    print(f"Layer akhir yang dibuka    : {trainable_layers}")
    print(f"Learning rate fine-tuning  : {learning_rate}")
    print("BatchNormalization         : tetap frozen")
    print("=" * 70)
    return model
