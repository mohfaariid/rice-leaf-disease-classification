"""Evaluasi model pada test set."""

from pathlib import Path

import tensorflow as tf

from configuration.paths import MODEL_PATH, REPORT_DIR, ensure_project_dirs
from evaluation.metrics import build_classification_report, collect_predictions
from evaluation.plots import plot_confusion_matrix, plot_sample_predictions
from preprocessing.dataset import prepare_datasets


def evaluate_model(
    model,
    test_ds,
    class_names,
    model_name: str = "model",
    make_prediction_plot: bool = True,
    make_confusion_plot: bool = True,
):
    """Menjalankan evaluasi final, prediksi, report, dan confusion matrix."""
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    y_true, y_pred = collect_predictions(model, test_ds)
    report = build_classification_report(y_true, y_pred, class_names)

    print("=" * 60)
    print(f"EVALUASI FINAL PADA TEST SET - {model_name}")
    print("=" * 60)
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_acc * 100:.2f}%")
    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)

    ensure_project_dirs()
    report_path = REPORT_DIR / f"classification_report_{model_name}.txt"
    report_path.write_text(report, encoding="utf-8")

    if make_prediction_plot:
        plot_sample_predictions(model, test_ds, class_names, REPORT_DIR / f"hasil_prediksi_{model_name}.png")

    if make_confusion_plot:
        plot_confusion_matrix(
            y_true,
            y_pred,
            class_names,
            REPORT_DIR / f"confusion_matrix_{model_name}.png",
            title=f"Evaluasi Confusion Matrix - {model_name}",
        )

    return {
        "model": model_name,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "report": report,
        "report_path": report_path,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def run(model_path: str | Path = MODEL_PATH):
    """Memuat model tersimpan dan mengevaluasinya pada test set."""
    _, _, test_ds, metadata = prepare_datasets()
    model = tf.keras.models.load_model(model_path)
    model_name = Path(model_path).stem
    return evaluate_model(model, test_ds, metadata["class_names"], model_name)


if __name__ == "__main__":
    run()
