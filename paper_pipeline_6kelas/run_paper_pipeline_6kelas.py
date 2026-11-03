"""Pipeline paper CNN kustom 6 kelas tanpa Tungro.

Pipeline ini sengaja dibuat terpisah dari pipeline skripsi supaya output paper
tidak bercampur dengan eksperimen 7 kelas original/balanced.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers


PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent
DATASET_DIR = PROJECT_ROOT / "data" / "rice_leaf_aug_6_no_tungro"
OUTPUT_DIR = PIPELINE_DIR / "outputs"
REPORT_DIR = PIPELINE_DIR / "reports"

CLASS_NAMES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf scald",
    "Sheath Blight",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

IMAGE_SIZE = int(os.getenv("PAPER_IMAGE_SIZE", "256"))
BATCH_SIZE = int(os.getenv("PAPER_BATCH_SIZE", "32"))
EPOCHS = int(os.getenv("PAPER_EPOCHS", "100"))
LEARNING_RATE = float(os.getenv("PAPER_LEARNING_RATE", "0.001"))
SEED = int(os.getenv("PAPER_SEED", "123"))
VALIDATION_TEST_SPLIT = float(os.getenv("PAPER_VALIDATION_TEST_SPLIT", "0.30"))
EARLY_STOPPING_PATIENCE = int(os.getenv("PAPER_EARLY_STOPPING_PATIENCE", "12"))


def ensure_dirs() -> None:
    """Membuat folder output dan report."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def collect_image_paths_and_labels() -> tuple[list[str], list[int]]:
    """Mengambil semua path citra dan label untuk 6 kelas tanpa Tungro."""
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATASET_DIR}")

    image_paths: list[str] = []
    labels: list[int] = []

    for label_index, class_name in enumerate(CLASS_NAMES):
        class_dir = DATASET_DIR / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Folder kelas tidak ditemukan: {class_dir}")

        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(str(image_path))
                labels.append(label_index)

    if not image_paths:
        raise ValueError(f"Tidak ada citra yang ditemukan di: {DATASET_DIR}")

    return image_paths, labels


def save_dataset_distribution(labels: list[int]) -> Path:
    """Menyimpan distribusi kelas dataset 6 kelas."""
    rows = []
    for label_index, class_name in enumerate(CLASS_NAMES):
        rows.append(
            {
                "class_name": class_name,
                "count": int(np.sum(np.asarray(labels) == label_index)),
            }
        )

    distribution_df = pd.DataFrame(rows)
    output_path = REPORT_DIR / "dataset_distribution_6kelas.csv"
    distribution_df.to_csv(output_path, index=False)
    return output_path


def split_dataset(image_paths: list[str], labels: list[int]):
    """Membagi dataset menjadi train, validation, dan test secara stratified."""
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths,
        labels,
        test_size=VALIDATION_TEST_SPLIT,
        random_state=SEED,
        stratify=labels,
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.5,
        random_state=SEED,
        stratify=temp_labels,
    )
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def load_and_resize_image(image_path, label):
    """Membaca citra dari path dan resize ke ukuran input model."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32)
    return image, label


def build_dataset(image_paths: list[str], labels: list[int], shuffle: bool) -> tf.data.Dataset:
    """Membuat tf.data.Dataset dari path citra dan label."""
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(1000, len(image_paths)),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(load_and_resize_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def build_custom_cnn() -> keras.Model:
    """Membangun CNN kustom dasar untuk paper 6 kelas."""
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
        ],
        name="data_augmentation",
    )

    model = keras.Sequential(
        [
            layers.InputLayer(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
            layers.Rescaling(1.0 / 255),
            data_augmentation,
            layers.Conv2D(16, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(len(CLASS_NAMES), name="output"),
        ],
        name="paper_custom_cnn_6kelas_tanpa_tungro",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def build_callbacks() -> list[keras.callbacks.Callback]:
    """Membuat callback training."""
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(OUTPUT_DIR / "best_model_val_loss.keras"),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(OUTPUT_DIR / "latest_weights.weights.h5"),
            save_weights_only=True,
            save_best_only=False,
            verbose=0,
        ),
        keras.callbacks.CSVLogger(
            filename=str(REPORT_DIR / "training_history.csv"),
            append=False,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
    ]


def plot_training_history(history: keras.callbacks.History) -> Path:
    """Menyimpan grafik training dan validation accuracy/loss."""
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epoch_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_range, acc, label="Training Accuracy", linewidth=2)
    plt.plot(epoch_range, val_acc, label="Validation Accuracy", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    plt.grid(alpha=0.3)
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epoch_range, loss, label="Training Loss", linewidth=2)
    plt.plot(epoch_range, val_loss, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.title("Loss")

    plt.suptitle("Training CNN Kustom 6 Kelas Tanpa Tungro", fontweight="bold")
    plt.tight_layout()

    output_path = REPORT_DIR / "training_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def collect_predictions(model: keras.Model, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Mengambil label asli dan prediksi model."""
    y_true = []
    y_pred = []
    for images, labels in dataset:
        logits = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(logits, axis=1))
        y_true.extend(labels.numpy())
    return np.asarray(y_true), np.asarray(y_pred)


def evaluate_and_save(model: keras.Model, test_ds: tf.data.Dataset) -> dict:
    """Evaluasi model dan simpan classification report serta confusion matrix."""
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    y_true, y_pred = collect_predictions(model, test_ds)

    report_text = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)
    report_path = REPORT_DIR / "classification_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )

    conf_matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=CLASS_NAMES)
    display.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title("Confusion Matrix CNN Kustom 6 Kelas")
    plt.tight_layout()
    confusion_path = REPORT_DIR / "confusion_matrix.png"
    plt.savefig(confusion_path, dpi=150, bbox_inches="tight")
    plt.close()

    summary = {
        "model": "paper_custom_cnn_6kelas_tanpa_tungro",
        "test_loss": float(test_loss),
        "test_accuracy_model_evaluate": float(test_accuracy),
        "accuracy_classification_report": float(report_dict["accuracy"]),
        "precision_macro": float(report_dict["macro avg"]["precision"]),
        "recall_macro": float(report_dict["macro avg"]["recall"]),
        "f1_macro": float(report_dict["macro avg"]["f1-score"]),
        "precision_weighted": float(report_dict["weighted avg"]["precision"]),
        "recall_weighted": float(report_dict["weighted avg"]["recall"]),
        "f1_weighted": float(report_dict["weighted avg"]["f1-score"]),
        "classification_report_path": str(report_path),
        "confusion_matrix_path": str(confusion_path),
    }
    pd.DataFrame([summary]).to_csv(REPORT_DIR / "summary_metrics.csv", index=False)
    return summary


def save_experiment_config(metadata: dict) -> Path:
    """Menyimpan konfigurasi eksperimen."""
    config = {
        "dataset_dir": str(DATASET_DIR),
        "class_names": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_max": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "loss": "SparseCategoricalCrossentropy(from_logits=True)",
        "optimizer": "Adam",
        "early_stopping_monitor": "val_loss",
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "metadata": metadata,
    }
    output_path = REPORT_DIR / "experiment_config.json"
    output_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return output_path


def run() -> dict:
    """Menjalankan pipeline lengkap paper."""
    ensure_dirs()
    tf.keras.utils.set_random_seed(SEED)

    print("=" * 70)
    print("PIPELINE PAPER: CNN KUSTOM 6 KELAS TANPA TUNGRO")
    print("=" * 70)
    print(f"Dataset      : {DATASET_DIR}")
    print(f"Epoch max    : {EPOCHS}")
    print(f"Image size   : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size   : {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")

    image_paths, labels = collect_image_paths_and_labels()
    distribution_path = save_dataset_distribution(labels)
    train_data, val_data, test_data = split_dataset(image_paths, labels)

    train_paths, train_labels = train_data
    val_paths, val_labels = val_data
    test_paths, test_labels = test_data

    metadata = {
        "total_images": len(image_paths),
        "train_images": len(train_paths),
        "validation_images": len(val_paths),
        "test_images": len(test_paths),
        "num_classes": len(CLASS_NAMES),
    }
    config_path = save_experiment_config(metadata)

    print("=" * 70)
    print("DATASET")
    print("=" * 70)
    print(f"Total images      : {metadata['total_images']}")
    print(f"Training images   : {metadata['train_images']}")
    print(f"Validation images : {metadata['validation_images']}")
    print(f"Testing images    : {metadata['test_images']}")
    print(f"Distribusi dataset: {distribution_path}")
    print(f"Config eksperimen : {config_path}")

    train_ds = build_dataset(train_paths, train_labels, shuffle=True)
    val_ds = build_dataset(val_paths, val_labels, shuffle=False)
    test_ds = build_dataset(test_paths, test_labels, shuffle=False)

    model = build_custom_cnn()
    model.summary()

    print("=" * 70)
    print("TRAINING")
    print("=" * 70)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=build_callbacks(),
        verbose=2,
    )

    best_epoch_index = int(np.argmin(history.history["val_loss"]))
    best_epoch = best_epoch_index + 1
    best_val_loss = float(history.history["val_loss"][best_epoch_index])
    best_val_accuracy = float(history.history["val_accuracy"][best_epoch_index])

    if (OUTPUT_DIR / "best_model_val_loss.keras").exists():
        model = keras.models.load_model(OUTPUT_DIR / "best_model_val_loss.keras")
        print(f"Model terbaik dimuat ulang dari: {OUTPUT_DIR / 'best_model_val_loss.keras'}")

    final_model_path = OUTPUT_DIR / "final_model.keras"
    model.save(final_model_path)
    curve_path = plot_training_history(history)
    summary = evaluate_and_save(model, test_ds)

    summary.update(
        {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_accuracy": best_val_accuracy,
            "actual_epochs_ran": len(history.history["loss"]),
            "final_model_path": str(final_model_path),
            "best_model_path": str(OUTPUT_DIR / "best_model_val_loss.keras"),
            "training_curve_path": str(curve_path),
        }
    )
    pd.DataFrame([summary]).to_csv(REPORT_DIR / "summary_metrics.csv", index=False)

    print("=" * 70)
    print("RINGKASAN HASIL")
    print("=" * 70)
    print(f"Epoch yang berjalan     : {summary['actual_epochs_ran']}")
    print(f"Epoch terbaik val_loss  : {summary['best_epoch']}")
    print(f"Best val_loss           : {summary['best_val_loss']:.4f}")
    print(f"Best val_accuracy       : {summary['best_val_accuracy']:.4f}")
    print(f"Test loss               : {summary['test_loss']:.4f}")
    print(f"Test accuracy evaluate  : {summary['test_accuracy_model_evaluate']:.4f}")
    print(f"Test accuracy report    : {summary['accuracy_classification_report']:.4f}")
    print(f"F1 macro                : {summary['f1_macro']:.4f}")
    print(f"Summary CSV             : {REPORT_DIR / 'summary_metrics.csv'}")
    return summary


if __name__ == "__main__":
    run()
