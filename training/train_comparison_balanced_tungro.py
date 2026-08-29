"""Pipeline perbandingan model pada dataset 7 kelas balanced Tungro."""

import csv
import os
from dataclasses import replace
from pathlib import Path

from tensorflow import keras

from configuration.paths import OUTPUT_DIR, REPORT_DIR, ensure_project_dirs
from configuration.settings import SETTINGS
from evaluation.compare import (
    build_comparison_table,
    get_best_model_summary,
    save_comparison_table,
    summarize_model,
)
from evaluation.evaluate import evaluate_model
from evaluation.plots import plot_confusion_matrix, plot_model_comparison, plot_training_history
from models.custom_cnn import build_custom_cnn
from models.mobilenetv2 import build_mobilenetv2, prepare_mobilenetv2_fine_tuning
from preprocessing.dataset import prepare_datasets
from preprocessing.distribution import (
    collect_class_distribution,
    print_class_distribution,
    save_class_distribution,
)
from training.trainer import save_model, train_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = os.getenv("RICE_BALANCED_EXPERIMENT_NAME", "balanced_quality_hybrid_653")
BALANCED_DATASET_PATH = Path(
    os.getenv(
        "RICE_BALANCED_DATASET_PATH",
        str(PROJECT_ROOT / "data" / "rice_leaf_aug_7_balanced_quality_hybrid_653"),
    )
)


def build_balanced_settings():
    """Membuat konfigurasi eksperimen dataset balanced Tungro."""
    return replace(
        SETTINGS,
        dataset_path=str(BALANCED_DATASET_PATH),
        model_name=EXPERIMENT_NAME,
    )


def get_completed_epochs(history_log_path):
    """Menghitung jumlah epoch yang sudah selesai dari CSV history."""
    if not history_log_path.exists():
        return 0

    with history_log_path.open("r", newline="", encoding="utf-8") as csv_file:
        return sum(1 for _ in csv.DictReader(csv_file))


def should_resume_training():
    """Menentukan apakah training dilanjutkan dari history lama."""
    return os.getenv("RICE_RESUME_TRAINING", "0") == "1"


def train_single_model(model, model_name, train_ds, val_ds, settings):
    """Melatih, menyimpan, dan memvisualisasikan training satu model."""
    print("=" * 70)
    print(f"TRAINING MODEL: {model_name}")
    print(f"Dataset   : {EXPERIMENT_NAME}")
    epochs = int(os.getenv("RICE_CURRENT_MAX_EPOCHS", str(settings.epochs)))
    resume_training = should_resume_training()

    print(f"Epochs    : {epochs}")
    print(f"Image size: {settings.image_size}x{settings.image_size}")
    print(f"Batch size: {settings.batch_size}")
    print(f"Resume    : {resume_training}")
    print("=" * 70)
    model.summary()

    best_checkpoint_path = OUTPUT_DIR / f"{model_name}_best_val_loss.keras"
    latest_weights_path = OUTPUT_DIR / "latest_checkpoints" / model_name / "latest.weights.h5"
    history_log_path = REPORT_DIR / f"history_{model_name}.csv"
    if not resume_training and history_log_path.exists():
        history_log_path.unlink()
    initial_epoch = get_completed_epochs(history_log_path) if resume_training else 0

    print(f"Checkpoint terbaik : {best_checkpoint_path}")
    print(f"Checkpoint resume  : {latest_weights_path}")
    print(f"Log history epoch  : {history_log_path}")
    print(f"Initial epoch      : {initial_epoch}")

    if resume_training and initial_epoch > 0 and latest_weights_path.exists():
        model.load_weights(latest_weights_path)
        print(f"Training dilanjutkan dari bobot terakhir: {latest_weights_path}")
    elif resume_training and initial_epoch > 0 and best_checkpoint_path.exists():
        model = keras.models.load_model(best_checkpoint_path)
        print(f"Training dilanjutkan dari checkpoint terbaik: {best_checkpoint_path}")

    history = train_model(
        model,
        train_ds,
        val_ds,
        settings,
        checkpoint_path=best_checkpoint_path,
        latest_weights_path=latest_weights_path,
        history_log_path=history_log_path,
        initial_epoch=initial_epoch,
        epochs=epochs,
        csv_append=resume_training,
    )

    if best_checkpoint_path.exists():
        model = keras.models.load_model(best_checkpoint_path)
        print(f"Model terbaik berdasarkan val_loss dimuat dari: {best_checkpoint_path}")

    save_path = save_model(model, OUTPUT_DIR / f"{model_name}.keras")

    plot_training_history(
        history,
        epochs,
        REPORT_DIR / f"grafik_pelatihan_{model_name}.png",
        title=f"Performa Training - {model_name}",
    )

    return {
        "model_name": model_name,
        "model": model,
        "history": history,
        "save_path": save_path,
    }


def train_single_model_with_epochs(model, model_name, train_ds, val_ds, settings, max_epochs):
    """Melatih model dengan jumlah epoch maksimum sementara."""
    previous_value = os.environ.get("RICE_CURRENT_MAX_EPOCHS")
    os.environ["RICE_CURRENT_MAX_EPOCHS"] = str(max_epochs)
    try:
        return train_single_model(model, model_name, train_ds, val_ds, settings)
    finally:
        if previous_value is None:
            os.environ.pop("RICE_CURRENT_MAX_EPOCHS", None)
        else:
            os.environ["RICE_CURRENT_MAX_EPOCHS"] = previous_value


def train_mobilenetv2_with_fine_tuning(model, model_name, train_ds, val_ds, settings):
    """Melatih MobileNetV2 dengan feature extraction lalu fine-tuning."""
    feature_epochs = int(os.getenv("RICE_FEATURE_EPOCHS", str(settings.epochs)))
    fine_tune_epochs = int(os.getenv("RICE_FINE_TUNE_EPOCHS", "30"))

    print("=" * 70)
    print("MOBILENETV2 TAHAP 1: FEATURE EXTRACTION")
    print("=" * 70)
    feature_training = train_single_model_with_epochs(
        model,
        f"{model_name}_feature_extraction",
        train_ds,
        val_ds,
        settings,
        feature_epochs,
    )

    print("=" * 70)
    print("MOBILENETV2 TAHAP 2: FINE-TUNING")
    print("=" * 70)
    fine_tune_model = prepare_mobilenetv2_fine_tuning(feature_training["model"])
    fine_tune_training = train_single_model_with_epochs(
        fine_tune_model,
        f"{model_name}_fine_tuning",
        train_ds,
        val_ds,
        settings,
        fine_tune_epochs,
    )
    final_save_path = save_model(fine_tune_training["model"], OUTPUT_DIR / f"{model_name}.keras")

    return {
        "model_name": model_name,
        "model": fine_tune_training["model"],
        "history": fine_tune_training["history"],
        "feature_history": feature_training["history"],
        "fine_tune_history": fine_tune_training["history"],
        "save_path": final_save_path,
        "feature_save_path": feature_training["save_path"],
        "fine_tune_save_path": fine_tune_training["save_path"],
    }


def evaluate_single_model(model, model_name, test_ds, class_names):
    """Mengevaluasi satu model pada test set."""
    evaluation_result = evaluate_model(
        model,
        test_ds,
        class_names,
        model_name,
        make_prediction_plot=True,
        make_confusion_plot=False,
    )
    summary = summarize_model(model, test_ds, class_names, model_name)

    return {
        "evaluation": evaluation_result,
        "summary": summary,
    }


def run():
    """Menjalankan pipeline perbandingan pada dataset balanced revisi."""
    ensure_project_dirs()
    settings = build_balanced_settings()

    print("=" * 70)
    print("EKSPERIMEN SKRIPSI: DATASET 7 KELAS BALANCED REVISI")
    print("=" * 70)
    print(f"Dataset path: {settings.dataset_path}")
    print("Tujuan: membandingkan CNN Kustom dan MobileNetV2 pada dataset balanced revisi.")

    print("=" * 70)
    print("DISTRIBUSI KELAS DATASET BALANCED REVISI")
    print("=" * 70)
    distribution_rows = collect_class_distribution(settings.dataset_path)
    print_class_distribution(distribution_rows, "Distribusi dataset 7 kelas balanced revisi")
    distribution_path = save_class_distribution(
        distribution_rows,
        REPORT_DIR / f"distribusi_dataset_{EXPERIMENT_NAME}.csv",
    )
    print(f"Distribusi dataset balanced disimpan ke: {distribution_path}")

    print("=" * 70)
    print("1. LOAD DATASET")
    print("=" * 70)
    train_ds, val_ds, test_ds, metadata = prepare_datasets(settings)
    class_names = metadata["class_names"]
    num_classes = metadata["num_classes"]

    if num_classes != 7:
        raise ValueError(f"Dataset balanced seharusnya 7 kelas, tetapi terbaca {num_classes}: {class_names}")

    print("=" * 70)
    print("2. SPLIT TRAIN/VALIDATION/TEST")
    print("=" * 70)
    print(f"Training batches   : {metadata['train_batches']}")
    print(f"Validation batches : {metadata['val_batches']}")
    print(f"Testing batches    : {metadata['test_batches']}")
    print(f"Class names        : {class_names}")

    print("=" * 70)
    print("3. AUGMENTASI DAN PREPROCESSING")
    print("=" * 70)
    print("Augmentasi: RandomFlip dan RandomRotation")
    print("Preprocessing CNN Kustom: Rescaling 1/255")
    print("Preprocessing MobileNetV2: preprocess_input MobileNetV2")

    print("=" * 70)
    print("4. MODEL 1: CNN KUSTOM")
    print("=" * 70)
    custom_model_name = f"custom_cnn_{EXPERIMENT_NAME}"
    custom_model_path = OUTPUT_DIR / f"{custom_model_name}.keras"
    skip_custom_training = os.getenv("RICE_SKIP_CUSTOM_CNN_TRAINING", "0") == "1"
    if skip_custom_training and custom_model_path.exists():
        print(f"Training CNN Kustom dilewati. Model dimuat dari: {custom_model_path}")
        custom_cnn = keras.models.load_model(custom_model_path)
        custom_training = {
            "model_name": custom_model_name,
            "model": custom_cnn,
            "history": None,
            "save_path": custom_model_path,
        }
    else:
        custom_cnn = build_custom_cnn(
            image_size=settings.image_size,
            num_classes=num_classes,
            model_name=custom_model_name,
        )

        print("=" * 70)
        print("5. TRAINING CNN KUSTOM")
        print("=" * 70)
        custom_training = train_single_model(
            custom_cnn,
            custom_model_name,
            train_ds,
            val_ds,
            settings,
        )

    print("=" * 70)
    print("6. EVALUASI CNN KUSTOM")
    print("=" * 70)
    custom_evaluation = evaluate_single_model(custom_training["model"], custom_model_name, test_ds, class_names)
    custom_result = {**custom_training, **custom_evaluation}

    print("=" * 70)
    print("7. MODEL 2: MOBILENETV2")
    print("=" * 70)
    mobilenet_model_name = f"mobilenetv2_{EXPERIMENT_NAME}"
    mobilenetv2 = build_mobilenetv2(
        image_size=settings.image_size,
        num_classes=num_classes,
    )

    print("=" * 70)
    print("8. TRAINING MOBILENETV2")
    print("=" * 70)
    mobilenet_training = train_mobilenetv2_with_fine_tuning(
        mobilenetv2,
        mobilenet_model_name,
        train_ds,
        val_ds,
        settings,
    )

    print("=" * 70)
    print("9. EVALUASI MOBILENETV2")
    print("=" * 70)
    mobilenet_evaluation = evaluate_single_model(mobilenet_training["model"], mobilenet_model_name, test_ds, class_names)
    mobilenet_result = {**mobilenet_training, **mobilenet_evaluation}

    results = [custom_result, mobilenet_result]

    print("=" * 70)
    print("10. TABEL PERBANDINGAN HASIL")
    print("=" * 70)
    comparison_df = build_comparison_table([result["summary"] for result in results])
    comparison_path = save_comparison_table(
        comparison_df,
        filename=f"tabel_perbandingan_model_{EXPERIMENT_NAME}.csv",
    )
    plot_model_comparison(
        comparison_df,
        REPORT_DIR / f"grafik_perbandingan_model_{EXPERIMENT_NAME}.png",
    )

    print(comparison_df)
    print(f"\nTabel disimpan ke: {comparison_path}")

    print("=" * 70)
    print("11. CONFUSION MATRIX MASING-MASING MODEL")
    print("=" * 70)
    for result in results:
        model_name = result["model_name"]
        evaluation = result["evaluation"]
        plot_confusion_matrix(
            evaluation["y_true"],
            evaluation["y_pred"],
            class_names,
            REPORT_DIR / f"confusion_matrix_{model_name}.png",
            title=f"Evaluasi Confusion Matrix - {model_name}",
        )

    print("=" * 70)
    print("12. KESIMPULAN MODEL TERBAIK")
    print("=" * 70)
    best_model = get_best_model_summary(comparison_df)
    print(f"Model terbaik berdasarkan F1-score macro: {best_model['model']}")
    print(f"Accuracy        : {best_model['accuracy']:.4f}")
    print(f"Precision macro : {best_model['precision_macro']:.4f}")
    print(f"Recall macro    : {best_model['recall_macro']:.4f}")
    print(f"F1-score macro  : {best_model['f1_macro']:.4f}")

    return {
        "results": results,
        "comparison_df": comparison_df,
        "best_model": best_model,
        "metadata": metadata,
        "distribution_path": distribution_path,
    }


if __name__ == "__main__":
    run()
