"""Prediktor untuk gambar daun padi baru."""

from pathlib import Path

import numpy as np
import tensorflow as tf

from configuration.settings import SETTINGS


def load_image_for_prediction(image_path: str | Path, image_size: int = SETTINGS.image_size):
    """Memuat satu gambar dan mengubahnya menjadi batch tensor."""
    image = tf.keras.utils.load_img(image_path, target_size=(image_size, image_size))
    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


class RiceDiseasePredictor:
    """Wrapper sederhana untuk memuat model dan melakukan prediksi."""

    def __init__(self, model_path: str | Path, class_names: list[str], image_size: int = SETTINGS.image_size):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = class_names
        self.image_size = image_size

    def predict(self, image_path: str | Path) -> dict:
        """Mengembalikan label prediksi dan confidence score."""
        image_batch = load_image_for_prediction(image_path, self.image_size)
        predictions = self.model.predict(image_batch, verbose=0)
        scores = predictions[0]
        if not np.isclose(np.sum(scores), 1.0, atol=1e-3):
            scores = tf.nn.softmax(scores).numpy()

        predicted_index = int(np.argmax(scores))
        predicted_label = self.class_names[predicted_index]
        return {
            "image_path": str(image_path),
            "predicted_index": predicted_index,
            "predicted_label": predicted_label,
            "confidence": float(scores[predicted_index]),
        }
