"""Menjalankan ulang hanya CNN Kustom Dropout untuk dataset original dan balanced."""

from dataclasses import replace
from pathlib import Path
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tensorflow import keras

from configuration.paths import OUTPUT_DIR, REPORT_DIR, ensure_project_dirs
from configuration.settings import SETTINGS
from evaluation.compare import build_comparison_table, save_comparison_table, summarize_model
from evaluation.evaluate import evaluate_model
from evaluation.plots import plot_confusion_matrix, plot_training_history
from models.custom_cnn import build_custom_cnn
from preprocessing.dataset import prepare_datasets
from preprocessing.distribution import (
    collect_class_distribution,
    print_class_distribution,
    save_class_distribution,
)
from training.trainer import save_model, train_model


BALANCED_EXPERIMENT_NAME = os.getenv("RICE_BALANCED_EXPERIMENT_NAME", "balanced_quality_hybrid_653")
BALANCED_DATASET_PATH = Path(
    os.getenv(
        "RICE_BALANCED_DATASET_PATH",
        str(PROJECT_ROOT / "data" / "rice_leaf_aug_7_balanced_quality_hybrid_653"),
    )
)


def build_scenarios():
    """Membuat daftar skenario CNN Dropout yang akan dijalankan."""
    return [
        {
            "name": "original_7kelas",
            "display_name": "Dataset original 7 kelas",
            "settings": SETTINGS,
        },
        {
            "name": BALANCED_EXPERIMENT_NAME,
            "display_name": "Dataset balanced revisi quality hybrid",
            "settings": replace(
                SETTINGS,
                dataset_path=str(BALANCED_DATASET_PATH),
                model_name=BALANCED_EXPERIMENT_NAME,
            ),
        },
    ]


def train_and_evaluate_cnn_dropout(scenario):
    """Melatih dan mengevaluasi CNN Kustom Dropout pada satu skenario."""
    settings = scenario["settings"]
    scenario_name = scenario["name"]
    model_name = f"custom_cnn_dropout_{scenario_name}"

    print("=" * 70)
    print(f"SKENARIO CNN DROPOUT: {scenario['display_name']}")
    print("=" * 70)
    print(f"Dataset path: {settings.dataset_path}")
    print(f"Epochs      : {settings.epochs}")
    print(f"Image size  : {settings.image_size}x{settings.image_size}")
    print(f"Batch size  : {settings.batch_size}")

    distribution_rows = collect_class_distribution(settings.dataset_path)
    print_class_distribution(distribution_rows, scenario["display_name"])
    distribution_path = save_class_distribution(
        distribution_rows,
        REPORT_DIR / f"distribusi_dataset_{scenario_name}_cnn_dropout.csv",
    )
    print(f"Distribusi dataset disimpan ke: {distribution_path}")

    train_ds, val_ds, test_ds, metadata = prepare_datasets(settings)
    class_names = metadata["class_names"]
    num_classes = metadata["num_classes"]

    print("=" * 70)
    print("ARSITEKTUR CNN KUSTOM DROPOUT")
    print("=" * 70)
    model = build_custom_cnn(
        image_size=settings.image_size,
        num_classes=num_classes,
        model_name=model_name,
    )
    model.summary()

    checkpoint_path = OUTPUT_DIR / f"{model_name}_best_val_loss.keras"
    latest_weights_path = OUTPUT_DIR / "latest_checkpoints" / model_name / "latest.weights.h5"
    history_log_path = REPORT_DIR / f"history_{model_name}.csv"
    if history_log_path.exists():
        history_log_path.unlink()

    print("=" * 70)
    print("TRAINING CNN KUSTOM DROPOUT")
    print("=" * 70)
    history = train_model(
        model,
        train_ds,
        val_ds,
        settings,
        checkpoint_path=checkpoint_path,
        latest_weights_path=latest_weights_path,
        history_log_path=history_log_path,
        epochs=settings.epochs,
    )

    if checkpoint_path.exists():
        model = keras.models.load_model(checkpoint_path)
        print(f"Model terbaik berdasarkan val_loss dimuat dari: {checkpoint_path}")

    final_model_path = save_model(model, OUTPUT_DIR / f"{model_name}.keras")
    training_curve_path = REPORT_DIR / f"grafik_pelatihan_{model_name}.png"
    plot_training_history(
        history,
        settings.epochs,
        training_curve_path,
        title=f"Performa Training - {model_name}",
    )

    print("=" * 70)
    print("EVALUASI CNN KUSTOM DROPOUT")
    print("=" * 70)
    evaluation = evaluate_model(
        model,
        test_ds,
        class_names,
        model_name,
        make_prediction_plot=True,
        make_confusion_plot=False,
    )
    summary = summarize_model(model, test_ds, class_names, model_name)
    confusion_matrix_path = REPORT_DIR / f"confusion_matrix_{model_name}.png"
    plot_confusion_matrix(
        evaluation["y_true"],
        evaluation["y_pred"],
        class_names,
        confusion_matrix_path,
    )

    summary.update(
        {
            "scenario": scenario_name,
            "num_classes": num_classes,
            "train_batches": metadata["train_batches"],
            "validation_batches": metadata["val_batches"],
            "test_batches": metadata["test_batches"],
            "best_model_path": str(checkpoint_path),
            "final_model_path": str(final_model_path),
            "history_log_path": str(history_log_path),
            "training_curve_path": str(training_curve_path),
            "confusion_matrix_path": str(confusion_matrix_path),
        }
    )
    return summary


def main():
    """Menjalankan semua skenario CNN Dropout."""
    ensure_project_dirs()
    results = []
    for scenario in build_scenarios():
        results.append(train_and_evaluate_cnn_dropout(scenario))

    comparison_df = build_comparison_table(results)
    comparison_path = save_comparison_table(
        comparison_df,
        filename="tabel_perbandingan_cnn_dropout_original_vs_balanced.csv",
    )
    print("=" * 70)
    print("REKAP CNN KUSTOM DROPOUT")
    print("=" * 70)
    print(comparison_df)
    print(f"Tabel rekap disimpan ke: {comparison_path}")


if __name__ == "__main__":
    main()
