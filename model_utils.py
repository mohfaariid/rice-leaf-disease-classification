"""Utilitas inferensi model deep learning dan Explainable AI (Grad-CAM)."""

import os
import time
import numpy as np
from PIL import Image, ImageOps
import matplotlib
import matplotlib.cm as cm
import streamlit as st

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import tensorflow as tf

from constants import (
    IMAGE_SIZE,
    MOBILENET_PATH,
    CUSTOM_CNN_PATH,
)


@st.cache_resource(show_spinner="Memuat model MobileNetV2 (Best Weights)...")
def load_mobilenet_model() -> tf.keras.Model:
    """Memuat model terbaik MobileNetV2 Fine-tuning 50 Layers."""
    if not MOBILENET_PATH.exists():
        raise FileNotFoundError(f"Berkas model MobileNetV2 tidak ditemukan: {MOBILENET_PATH}")
    return tf.keras.models.load_model(MOBILENET_PATH, compile=False)


@st.cache_resource(show_spinner="Memuat model CNN Kustom...")
def load_custom_cnn_model() -> tf.keras.Model | None:
    """Memuat model pembanding CNN Kustom (Dropout)."""
    if not CUSTOM_CNN_PATH.exists():
        return None
    try:
        return tf.keras.models.load_model(CUSTOM_CNN_PATH, compile=False)
    except Exception:
        return None


def run_prediction(image: Image.Image, model: tf.keras.Model) -> tuple[np.ndarray, float]:
    """Mengolah citra dan melakukan prediksi dengan pencatatan waktu inferensi."""
    prepared = ImageOps.exif_transpose(image).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    image_array = np.asarray(prepared, dtype=np.float32)
    input_tensor = np.expand_dims(image_array, axis=0)

    t_start = time.perf_counter()
    raw_scores = model.predict(input_tensor, verbose=0)[0]
    latency_ms = (time.perf_counter() - t_start) * 1000.0

    if not np.isclose(np.sum(raw_scores), 1.0, atol=1e-3):
        scores = tf.nn.softmax(raw_scores).numpy()
    else:
        scores = raw_scores

    return scores.astype(float), latency_ms


def generate_gradcam_overlay(model: tf.keras.Model, image: Image.Image, class_idx: int) -> Image.Image | None:
    """Menghasilkan visualisasi Explainable AI (Grad-CAM) untuk MobileNetV2."""
    try:
        prepared = ImageOps.exif_transpose(image).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        image_array = np.asarray(prepared, dtype=np.float32)
        input_tensor = np.expand_dims(image_array, axis=0)

        base_model = [l for l in model.layers if "mobilenetv2" in l.name][0]
        gap = [l for l in model.layers if "global_average_pooling" in l.name][0]
        dense = [l for l in model.layers if "output" in l.name][0]
        conv_layer = base_model.get_layer("out_relu")

        grad_model = tf.keras.models.Model(
            inputs=[base_model.input],
            outputs=[conv_layer.output, base_model.output],
        )

        preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(np.copy(input_tensor))
        with tf.GradientTape() as tape:
            conv_outputs, _ = grad_model(preprocessed)
            tape.watch(conv_outputs)
            pooled = gap(conv_outputs)
            preds = dense(pooled)
            class_channel = preds[:, class_idx]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)[0]
        heatmap = np.maximum(heatmap, 0)
        denom = np.max(heatmap)
        if denom > 0:
            heatmap /= denom

        heatmap = np.uint8(255 * heatmap)
        cmap = matplotlib.colormaps["jet"]
        jet_colors = cmap(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap_img = Image.fromarray(np.uint8(jet_heatmap * 255)).resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
        )

        orig_img = Image.fromarray(np.uint8(image_array))
        blended = Image.blend(orig_img, jet_heatmap_img, alpha=0.42)
        return blended
    except Exception:
        return None
