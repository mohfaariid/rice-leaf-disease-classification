"""Callback training."""

from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score


class MetricsPerEpochCallback(tf.keras.callbacks.Callback):
    """Menampilkan precision, recall, dan F1-score validation setiap epoch."""

    def __init__(self, val_dataset):
        super().__init__()
        self.val_dataset = val_dataset

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred = [], []
        for images, labels in self.val_dataset:
            preds = self.model.predict(images, verbose=0)
            y_pred.extend(np.argmax(preds, axis=1))
            y_true.extend(labels.numpy())

        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        print(f"   val_precision: {precision:.4f} | val_recall: {recall:.4f} | val_f1: {f1:.4f}")


def build_training_callbacks(
    val_dataset,
    checkpoint_path: str | Path | None = None,
    latest_weights_path: str | Path | None = None,
    history_log_path: str | Path | None = None,
    monitor: str = "val_loss",
    patience: int = 12,
    csv_append: bool = False,
):
    """Membuat callback training dengan early stopping dan checkpoint.

    Checkpoint bobot terakhir membuat training bisa dilanjutkan tanpa memakai
    BackupAndRestore yang rawan file lock pada Windows.
    """
    callbacks = [MetricsPerEpochCallback(val_dataset)]

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor=monitor,
                mode="min",
                save_best_only=True,
                verbose=1,
            )
        )

    if latest_weights_path is not None:
        latest_weights_path = Path(latest_weights_path)
        latest_weights_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(latest_weights_path),
                save_weights_only=True,
                save_best_only=False,
                verbose=0,
            )
        )

    if history_log_path is not None:
        history_log_path = Path(history_log_path)
        history_log_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.CSVLogger(
                filename=str(history_log_path),
                append=csv_append,
            )
        )

    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="min",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        )
    )
    return callbacks
