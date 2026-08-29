"""Metrik evaluasi model."""

import numpy as np
from sklearn.metrics import classification_report


def collect_predictions(model, dataset):
    """Mengumpulkan label asli dan prediksi model dari dataset."""
    all_predictions = []
    all_labels = []

    for images, labels in dataset:
        classifications = model(images)
        predicted_batch = [np.argmax(clf) for clf in classifications.numpy()]

        all_predictions.extend(predicted_batch)
        all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_predictions)


def build_classification_report(y_true, y_pred, class_names):
    """Membuat classification report sklearn."""
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )


def build_classification_report_dict(y_true, y_pred, class_names):
    """Membuat classification report dalam bentuk dictionary."""
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
