"""Visualisasi training dan evaluasi."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from configuration.paths import REPORT_DIR, ensure_project_dirs


def finish_plot(output_path):
    """Menyimpan plot dan menutup figure agar tidak menggantung di terminal."""
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if os.getenv("RICE_SHOW_PLOTS", "0") == "1":
        plt.show()
    else:
        plt.close()


def plot_training_history(
    history,
    epochs: int,
    output_path: str | Path | None = None,
    title: str = "Performa Model CNN - Penyakit Padi",
):
    """Membuat grafik accuracy dan loss seperti notebook awal."""
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epoch_range = range(min(epochs, len(acc)))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epoch_range, acc, label="Akurasi Training", color="blue", linewidth=2)
    plt.plot(epoch_range, val_acc, label="Akurasi Validasi", color="orange", linewidth=2, linestyle="--")
    plt.legend(loc="lower right")
    plt.title("Grafik Akurasi Training vs Validasi", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Akurasi")
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epoch_range, loss, label="Loss Training", color="blue", linewidth=2)
    plt.plot(epoch_range, val_loss, label="Loss Validasi", color="orange", linewidth=2, linestyle="--")
    plt.legend(loc="upper right")
    plt.title("Grafik Loss Training vs Validasi", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path is None:
        ensure_project_dirs()
        output_path = REPORT_DIR / "grafik_pelatihan.png"
    finish_plot(output_path)


def plot_sample_predictions(model, test_ds, class_names, output_path: str | Path | None = None):
    """Menampilkan contoh prediksi pada satu batch test set."""
    plt.figure(figsize=(15, 15))

    for images, labels in test_ds.take(1):
        classifications = model(images)

        for i in range(min(9, len(images))):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))

            prediction_index = np.argmax(classifications[i])
            prediction_name = class_names[prediction_index]
            actual_name = class_names[int(labels[i].numpy())]
            color = "green" if prediction_name == actual_name else "red"

            plt.title(
                f"Prediksi : {prediction_name}\nAsli     : {actual_name}",
                color=color,
                fontsize=9,
            )
            plt.axis("off")

    plt.suptitle("Hasil Prediksi Model", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if output_path is None:
        ensure_project_dirs()
        output_path = REPORT_DIR / "hasil_prediksi.png"
    finish_plot(output_path)


def plot_dataset_samples(dataset, class_names, output_path: str | Path | None = None):
    """Menampilkan 9 sampel gambar dari dataset training."""
    plt.figure(figsize=(10, 10))

    for images, labels in dataset.take(1):
        for i in range(min(9, len(images))):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[int(labels[i].numpy())])
            plt.axis("off")

    plt.suptitle("Sampel Gambar Dataset Penyakit Padi", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path is None:
        ensure_project_dirs()
        output_path = REPORT_DIR / "sampel_dataset.png"
    finish_plot(output_path)


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    output_path: str | Path | None = None,
    title: str = "Evaluasi Confusion Matrix",
):
    """Membuat confusion matrix jumlah dan persentase."""
    conf_matrix = confusion_matrix(y_true, y_pred)
    conf_norm = conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
    conf_norm = np.nan_to_num(conf_norm)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    tick_marks = np.arange(len(class_names))

    axes[0].imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    axes[0].set_title("Confusion Matrix (Jumlah)", fontweight="bold")
    axes[0].set_xticks(tick_marks)
    axes[0].set_xticklabels(class_names, rotation=45, ha="right")
    axes[0].set_yticks(tick_marks)
    axes[0].set_yticklabels(class_names)
    axes[0].set_xlabel("Prediksi")
    axes[0].set_ylabel("Label Asli")

    thresh = conf_matrix.max() / 2.0
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            axes[0].text(
                j,
                i,
                format(conf_matrix[i, j], "d"),
                ha="center",
                va="center",
                color="white" if conf_matrix[i, j] > thresh else "black",
                fontsize=12,
                fontweight="bold",
            )

    axes[1].imshow(conf_norm, interpolation="nearest", cmap=plt.cm.Blues)
    axes[1].set_title("Confusion Matrix (Persentase)", fontweight="bold")
    axes[1].set_xticks(tick_marks)
    axes[1].set_xticklabels(class_names, rotation=45, ha="right")
    axes[1].set_yticks(tick_marks)
    axes[1].set_yticklabels(class_names)
    axes[1].set_xlabel("Prediksi")
    axes[1].set_ylabel("Label Asli")

    for i in range(conf_norm.shape[0]):
        for j in range(conf_norm.shape[1]):
            axes[1].text(
                j,
                i,
                f"{conf_norm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if conf_norm[i, j] > 0.5 else "black",
                fontsize=12,
                fontweight="bold",
            )

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if output_path is None:
        ensure_project_dirs()
        output_path = REPORT_DIR / "confusion_matrix.png"
    finish_plot(output_path)


def plot_model_comparison(comparison_df, output_path: str | Path | None = None):
    """Membuat grafik perbandingan accuracy, precision, recall, dan F1."""
    x = np.arange(len(comparison_df["model"]))
    width = 0.2

    plt.figure(figsize=(10, 5))
    plt.bar(x - width * 1.5, comparison_df["accuracy"], width, label="Accuracy")
    plt.bar(x - width * 0.5, comparison_df["precision_macro"], width, label="Precision Macro")
    plt.bar(x + width * 0.5, comparison_df["recall_macro"], width, label="Recall Macro")
    plt.bar(x + width * 1.5, comparison_df["f1_macro"], width, label="F1 Macro")

    plt.xticks(x, comparison_df["model"])
    plt.ylim([0, 1])
    plt.ylabel("Skor")
    plt.title("Perbandingan Performa Model pada Test Set")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if output_path is None:
        ensure_project_dirs()
        output_path = REPORT_DIR / "grafik_perbandingan_model.png"
    finish_plot(output_path)
