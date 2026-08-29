"""Perbandingan hasil evaluasi beberapa model."""

import pandas as pd

from configuration.paths import REPORT_DIR, ensure_project_dirs
from evaluation.metrics import build_classification_report_dict, collect_predictions


def summarize_model(model, dataset, class_names, model_name: str) -> dict:
    """Meringkas metrik model dalam format tabel."""
    y_true, y_pred = collect_predictions(model, dataset)
    report = build_classification_report_dict(y_true, y_pred, class_names)

    return {
        "model": model_name,
        "accuracy": report["accuracy"],
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"],
        "precision_weighted": report["weighted avg"]["precision"],
        "recall_weighted": report["weighted avg"]["recall"],
        "f1_weighted": report["weighted avg"]["f1-score"],
    }


def build_comparison_table(results: list[dict]) -> pd.DataFrame:
    """Membuat tabel perbandingan model."""
    return pd.DataFrame(results).sort_values(by="f1_macro", ascending=False)


def save_comparison_table(comparison_df: pd.DataFrame, filename: str = "tabel_perbandingan_model.csv"):
    """Menyimpan tabel perbandingan model ke folder reports."""
    ensure_project_dirs()
    output_path = REPORT_DIR / filename
    comparison_df.to_csv(output_path, index=False)
    return output_path


def get_best_model_summary(comparison_df: pd.DataFrame) -> dict:
    """Menentukan model terbaik berdasarkan F1-score macro."""
    best_model = comparison_df.iloc[0]
    return {
        "model": best_model["model"],
        "accuracy": best_model["accuracy"],
        "precision_macro": best_model["precision_macro"],
        "recall_macro": best_model["recall_macro"],
        "f1_macro": best_model["f1_macro"],
    }
