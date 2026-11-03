"""Melanjutkan training CNN Kustom Dropout untuk dataset balanced Tungro."""

from dataclasses import replace
from pathlib import Path
import csv
import sys

from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configuration.paths import OUTPUT_DIR, REPORT_DIR, ensure_project_dirs
from configuration.settings import SETTINGS
from evaluation.compare import save_comparison_table, summarize_model
from evaluation.evaluate import evaluate_model
from evaluation.plots import plot_confusion_matrix, plot_training_history
from models.custom_cnn import build_custom_cnn
from preprocessing.dataset import prepare_datasets
from training.trainer import save_model, train_model


EXPERIMENT_NAME = "custom_cnn_dropout_balanced_tungro_636"
BALANCED_DATASET_PATH = PROJECT_ROOT / "data" / "rice_leaf_aug_7_balanced_tungro_636"


def count_completed_epochs(history_log_path: Path) -> int:
    """Menghitung epoch yang sudah selesai dari CSVLogger."""
    if not history_log_path.exists():
        return 0
    with history_log_path.open("r", newline="", encoding="utf-8") as csv_file:
        return sum(1 for _ in csv.DictReader(csv_file))


def main():
    """Resume training CNN Dropout balanced, lalu evaluasi pada test set."""
    ensure_project_dirs()
    settings = replace(
        SETTINGS,
        dataset_path=str(BALANCED_DATASET_PATH),
        model_name=EXPERIMENT_NAME,
    )

    print("=" * 70)
    print("RESUME CNN KUSTOM DROPOUT - DATASET BALANCED TUNGRO 636")
    print("=" * 70)
    print(f"Dataset path : {settings.dataset_path}")
    print(f"Epoch maksimum: {settings.epochs}")
    print(f"Image size   : {settings.image_size}x{settings.image_size}")
    print(f"Batch size   : {settings.batch_size}")

    train_ds, val_ds, test_ds, metadata = prepare_datasets(settings)
    class_names = metadata["class_names"]
    num_classes = metadata["num_classes"]

    model = build_custom_cnn(
        image_size=settings.image_size,
        num_classes=num_classes,
        model_name=EXPERIMENT_NAME,
    )

    best_checkpoint_path = OUTPUT_DIR / f"{EXPERIMENT_NAME}_best_val_loss.keras"
    latest_weights_path = OUTPUT_DIR / "latest_checkpoints" / EXPERIMENT_NAME / "latest.weights.h5"
    history_log_path = REPORT_DIR / f"history_{EXPERIMENT_NAME}.csv"
    initial_epoch = count_completed_epochs(history_log_path)

    print(f"History log       : {history_log_path}")
    print(f"Epoch selesai     : {initial_epoch}")
    print(f"Checkpoint terbaik: {best_checkpoint_path}")
    print(f"Checkpoint resume : {latest_weights_path}")

    if initial_epoch > 0 and latest_weights_path.exists():
        model.load_weights(latest_weights_path)
        print(f"Training dilanjutkan dari bobot terakhir: {latest_weights_path}")
    elif best_checkpoint_path.exists():
        model = keras.models.load_model(best_checkpoint_path)
        print(f"Training dilanjutkan dari checkpoint terbaik: {best_checkpoint_path}")
    else:
        print("Checkpoint belum ditemukan. Training dimulai dari awal.")

    history = train_model(
        model,
        train_ds,
        val_ds,
        settings,
        checkpoint_path=best_checkpoint_path,
        latest_weights_path=latest_weights_path,
        history_log_path=history_log_path,
        initial_epoch=initial_epoch,
        epochs=settings.epochs,
        csv_append=initial_epoch > 0,
    )

    if best_checkpoint_path.exists():
        model = keras.models.load_model(best_checkpoint_path)
        print(f"Model terbaik berdasarkan val_loss dimuat dari: {best_checkpoint_path}")

    final_model_path = save_model(model, OUTPUT_DIR / f"{EXPERIMENT_NAME}.keras")
    training_curve_path = REPORT_DIR / f"grafik_pelatihan_{EXPERIMENT_NAME}.png"
    plot_training_history(
        history,
        settings.epochs,
        training_curve_path,
        title=f"Performa Training - {EXPERIMENT_NAME}",
    )

    evaluation = evaluate_model(
        model,
        test_ds,
        class_names,
        EXPERIMENT_NAME,
        make_prediction_plot=True,
        make_confusion_plot=False,
    )
    summary = summarize_model(model, test_ds, class_names, EXPERIMENT_NAME)
    confusion_matrix_path = REPORT_DIR / f"confusion_matrix_{EXPERIMENT_NAME}.png"
    plot_confusion_matrix(
        evaluation["y_true"],
        evaluation["y_pred"],
        class_names,
        confusion_matrix_path,
    )

    comparison_path = save_comparison_table(
        __import__("pandas").DataFrame([summary]),
        filename="tabel_hasil_cnn_dropout_balanced_tungro_636.csv",
    )

    print("=" * 70)
    print("HASIL AKHIR CNN KUSTOM DROPOUT BALANCED")
    print("=" * 70)
    print(summary)
    print(f"Model final          : {final_model_path}")
    print(f"Tabel hasil          : {comparison_path}")
    print(f"Grafik pelatihan     : {training_curve_path}")
    print(f"Confusion matrix     : {confusion_matrix_path}")


if __name__ == "__main__":
    main()
