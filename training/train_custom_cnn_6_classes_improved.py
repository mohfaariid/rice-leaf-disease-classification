"""Pipeline CNN kustom improved untuk 6 kelas daun padi tanpa Tungro."""

from dataclasses import replace
from pathlib import Path

import pandas as pd
import tensorflow as tf

from configuration.paths import OUTPUT_DIR, REPORT_DIR, ensure_project_dirs
from configuration.settings import SETTINGS
from evaluation.compare import summarize_model
from evaluation.evaluate import evaluate_model
from evaluation.plots import plot_confusion_matrix, plot_training_history
from models.custom_cnn_improved import build_custom_cnn_improved
from preprocessing.dataset_6_classes import (
    EXCLUDED_CLASS,
    SIX_CLASS_NAMES,
    prepare_6_class_datasets,
)
from training.callbacks import MetricsPerEpochCallback
from training.trainer import save_model


EXPERIMENT_NAME = "custom_cnn_improved_6kelas_tanpa_tungro"


def build_6_class_settings():
    """Membuat konfigurasi khusus untuk eksperimen improved 6 kelas."""
    return replace(
        SETTINGS,
        class_names=SIX_CLASS_NAMES,
        model_name=EXPERIMENT_NAME,
    )


def build_improved_callbacks(val_ds, checkpoint_path: Path):
    """Membuat callback untuk memilih model terbaik dan menjaga training stabil."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        MetricsPerEpochCallback(val_ds),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def train_improved_model(model, train_ds, val_ds, settings, checkpoint_path: Path):
    """Melatih CNN improved dengan callback model terbaik."""
    callbacks = build_improved_callbacks(val_ds, checkpoint_path)
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=settings.epochs,
        callbacks=callbacks,
        verbose=2,
    )


def save_experiment_summary(summary: dict, metadata: dict, output_path: Path) -> Path:
    """Menyimpan ringkasan hasil eksperimen CNN improved."""
    row = {
        "experiment": EXPERIMENT_NAME,
        "model": summary["model"],
        "num_classes": metadata["num_classes"],
        "excluded_class": EXCLUDED_CLASS,
        "accuracy": summary["accuracy"],
        "precision_macro": summary["precision_macro"],
        "recall_macro": summary["recall_macro"],
        "f1_macro": summary["f1_macro"],
        "precision_weighted": summary["precision_weighted"],
        "recall_weighted": summary["recall_weighted"],
        "f1_weighted": summary["f1_weighted"],
    }
    df = pd.DataFrame([row])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def print_class_setup(metadata: dict) -> None:
    """Menampilkan kelas yang digunakan dalam eksperimen."""
    print("Kelas yang digunakan:")
    for index, class_name in enumerate(metadata["class_names"], start=1):
        print(f"{index}. {class_name}")
    print(f"Kelas yang dikeluarkan: {EXCLUDED_CLASS}")


def run():
    """Menjalankan training dan evaluasi CNN improved untuk 6 kelas."""
    ensure_project_dirs()
    settings = build_6_class_settings()
    checkpoint_path = OUTPUT_DIR / f"{EXPERIMENT_NAME}_best.keras"

    print("=" * 70)
    print("EKSPERIMEN PAPER: CNN KUSTOM IMPROVED 6 KELAS TANPA TUNGRO")
    print("=" * 70)
    print("Tujuan: meningkatkan performa CNN kustom pada dataset 6 kelas seimbang.")
    print("Perbaikan: augmentasi lebih kaya, batch normalization, dropout, GAP, checkpoint.")

    print("=" * 70)
    print("1. LOAD DATASET 6 KELAS")
    print("=" * 70)
    train_ds, val_ds, test_ds, metadata = prepare_6_class_datasets(settings)
    class_names = metadata["class_names"]
    num_classes = metadata["num_classes"]
    print_class_setup(metadata)

    print("=" * 70)
    print("2. SPLIT TRAIN/VALIDATION/TEST")
    print("=" * 70)
    print(f"Training batches   : {metadata['train_batches']}")
    print(f"Validation batches : {metadata['val_batches']}")
    print(f"Testing batches    : {metadata['test_batches']}")
    print(f"Training images    : {metadata['train_images']}")
    print(f"Validation images  : {metadata['val_images']}")
    print(f"Testing images     : {metadata['test_images']}")

    print("=" * 70)
    print("3. AUGMENTASI DAN PREPROCESSING")
    print("=" * 70)
    print("Augmentasi: RandomFlip, RandomRotation, RandomZoom")
    print("Preprocessing CNN Kustom: Rescaling 1/255")

    print("=" * 70)
    print("4. MODEL: CNN KUSTOM IMPROVED")
    print("=" * 70)
    model = build_custom_cnn_improved(
        image_size=settings.image_size,
        num_classes=num_classes,
        model_name=EXPERIMENT_NAME,
    )
    model.summary()

    print("=" * 70)
    print("5. TRAINING CNN KUSTOM IMPROVED")
    print("=" * 70)
    print(f"Epochs       : {settings.epochs}")
    print(f"Image size   : {settings.image_size}x{settings.image_size}")
    print(f"Batch size   : {settings.batch_size}")
    print(f"Best checkpoint: {checkpoint_path}")
    history = train_improved_model(model, train_ds, val_ds, settings, checkpoint_path)
    model_path = save_model(model, OUTPUT_DIR / f"{EXPERIMENT_NAME}.keras")

    plot_training_history(
        history,
        len(history.history["accuracy"]),
        REPORT_DIR / f"grafik_pelatihan_{EXPERIMENT_NAME}.png",
        title="Performa Training - CNN Kustom Improved 6 Kelas Tanpa Tungro",
    )

    print("=" * 70)
    print("6. EVALUASI CNN KUSTOM IMPROVED")
    print("=" * 70)
    evaluation = evaluate_model(
        model,
        test_ds,
        class_names,
        EXPERIMENT_NAME,
        make_prediction_plot=True,
        make_confusion_plot=False,
    )
    summary = summarize_model(model, test_ds, class_names, EXPERIMENT_NAME)

    print("=" * 70)
    print("7. CONFUSION MATRIX")
    print("=" * 70)
    plot_confusion_matrix(
        evaluation["y_true"],
        evaluation["y_pred"],
        class_names,
        REPORT_DIR / f"confusion_matrix_{EXPERIMENT_NAME}.png",
    )

    print("=" * 70)
    print("8. RINGKASAN HASIL UNTUK PAPER")
    print("=" * 70)
    summary_path = save_experiment_summary(
        summary,
        metadata,
        REPORT_DIR / f"ringkasan_{EXPERIMENT_NAME}.csv",
    )
    print(f"Model akhir disimpan ke        : {model_path}")
    print(f"Best checkpoint disimpan ke    : {checkpoint_path}")
    print(f"Classification report disimpan : {evaluation['report_path']}")
    print(f"Ringkasan CSV disimpan         : {summary_path}")
    print(f"Accuracy                       : {summary['accuracy']:.4f}")
    print(f"Precision macro                : {summary['precision_macro']:.4f}")
    print(f"Recall macro                   : {summary['recall_macro']:.4f}")
    print(f"F1-score macro                 : {summary['f1_macro']:.4f}")

    return {
        "model": model,
        "model_path": model_path,
        "checkpoint_path": checkpoint_path,
        "evaluation": evaluation,
        "summary": summary,
        "summary_path": summary_path,
        "metadata": metadata,
    }


if __name__ == "__main__":
    run()
