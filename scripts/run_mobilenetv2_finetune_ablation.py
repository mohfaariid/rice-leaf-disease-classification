"""Percobaan ablasi jumlah layer akhir MobileNetV2 yang di-unfreeze."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configuration.paths import OUTPUT_DIR, REPORT_DIR, ensure_project_dirs
from configuration.settings import SETTINGS
from evaluation.compare import build_comparison_table, save_comparison_table, summarize_model
from evaluation.evaluate import evaluate_model
from evaluation.plots import plot_training_history
from models.mobilenetv2 import get_mobilenetv2_base_model, prepare_mobilenetv2_fine_tuning
from preprocessing.dataset import prepare_datasets
from training.trainer import save_model, train_model


DEFAULT_EXPERIMENT_NAME = os.getenv("RICE_ABLATION_EXPERIMENT_NAME", "balanced_quality_hybrid_653")
DEFAULT_DATASET_PATH = Path(
    os.getenv(
        "RICE_ABLATION_DATASET_PATH",
        str(PROJECT_ROOT / "data" / "rice_leaf_aug_7_balanced_quality_hybrid_653"),
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablasi fine-tuning MobileNetV2.")
    parser.add_argument(
        "--layers",
        default=os.getenv("RICE_ABLATION_LAYERS", "0,10,20,30,50"),
        help="Daftar jumlah layer akhir yang dibuka, dipisahkan koma.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(os.getenv("RICE_ABLATION_EPOCHS", os.getenv("RICE_FINE_TUNE_EPOCHS", "30"))),
        help="Epoch maksimum tiap konfigurasi fine-tuning.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=float(os.getenv("RICE_FINE_TUNE_LR", "0.00001")),
    )
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--feature-model-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=os.getenv("RICE_ABLATION_RESUME", "1") == "1",
        help="Lanjutkan training dari latest weights dan history jika tersedia.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def parse_layer_counts(raw_layers: str) -> list[int]:
    layer_counts = []
    for item in raw_layers.split(","):
        item = item.strip()
        if item:
            layer_counts.append(int(item))
    if not layer_counts:
        raise ValueError("Daftar layer ablasi kosong.")
    return layer_counts


def get_completed_epochs(history_log_path: Path) -> int:
    if not history_log_path.exists():
        return 0
    with history_log_path.open("r", newline="", encoding="utf-8") as csv_file:
        return sum(1 for _ in csv.DictReader(csv_file))


def get_best_val_loss_and_epoch(history_log_path: Path) -> tuple[float | None, int | None, int]:
    if not history_log_path.exists():
        return None, None, 0
    history_df = pd.read_csv(history_log_path)
    if history_df.empty or "val_loss" not in history_df.columns:
        return None, None, len(history_df)
    best_idx = history_df["val_loss"].idxmin()
    best_epoch = int(history_df.loc[best_idx, "epoch"]) + 1
    best_val_loss = float(history_df.loc[best_idx, "val_loss"])
    return best_val_loss, best_epoch, len(history_df)


def count_trainable_params(model: keras.Model) -> tuple[int, int]:
    trainable = int(sum(keras.backend.count_params(weight) for weight in model.trainable_weights))
    non_trainable = int(sum(keras.backend.count_params(weight) for weight in model.non_trainable_weights))
    return trainable, non_trainable


def evaluate_feature_extraction_baseline(model, experiment_name, test_ds, class_names) -> dict:
    model_name = f"mobilenetv2_{experiment_name}_ablation_0_layers"
    evaluation = evaluate_model(
        model,
        test_ds,
        class_names,
        model_name,
        make_prediction_plot=False,
        make_confusion_plot=False,
    )
    summary = summarize_model(model, test_ds, class_names, model_name)
    trainable, non_trainable = count_trainable_params(model)
    summary.update(
        {
            "unfrozen_layers": 0,
            "best_val_loss": None,
            "best_epoch": None,
            "epochs_ran": 0,
            "trainable_params": trainable,
            "non_trainable_params": non_trainable,
            "test_loss": evaluation["test_loss"],
        }
    )
    return summary


def run_fine_tuning_ablation(
    layer_count: int,
    experiment_name: str,
    feature_model_path: Path,
    train_ds,
    val_ds,
    test_ds,
    class_names,
    settings,
    epochs: int,
    learning_rate: float,
    skip_existing: bool,
    resume: bool,
) -> dict:
    model_name = f"mobilenetv2_{experiment_name}_ablation_{layer_count}_layers"
    checkpoint_path = OUTPUT_DIR / f"{model_name}_best_val_loss.keras"
    latest_weights_path = OUTPUT_DIR / "latest_checkpoints" / model_name / "latest.weights.h5"
    history_log_path = REPORT_DIR / f"history_{model_name}.csv"
    final_model_path = OUTPUT_DIR / f"{model_name}.keras"

    completed_epochs = get_completed_epochs(history_log_path)
    if skip_existing and final_model_path.exists() and history_log_path.exists():
        print(f"[SKIP] {model_name} sudah ada, langsung evaluasi.")
        model = keras.models.load_model(final_model_path)
    elif resume and completed_epochs >= epochs and final_model_path.exists():
        print(f"[RESUME] {model_name} sudah selesai {completed_epochs}/{epochs} epoch, langsung evaluasi.")
        model = keras.models.load_model(final_model_path)
    else:
        if history_log_path.exists() and not resume:
            history_log_path.unlink()
            completed_epochs = 0

        print("=" * 70)
        print(f"ABLASI FINE-TUNING: {layer_count} LAYER AKHIR DIBUKA")
        print("=" * 70)
        model = keras.models.load_model(feature_model_path)
        model = prepare_mobilenetv2_fine_tuning(
            model,
            trainable_layers=layer_count,
            learning_rate=learning_rate,
        )
        trainable, non_trainable = count_trainable_params(model)
        base_model = get_mobilenetv2_base_model(model)
        print(f"Base MobileNetV2 layer total : {len(base_model.layers)}")
        print(f"Trainable params             : {trainable:,}")
        print(f"Non-trainable params         : {non_trainable:,}")
        print(f"Resume                       : {resume}")
        print(f"Initial epoch                : {completed_epochs if resume else 0}")

        initial_epoch = completed_epochs if resume else 0
        csv_append = resume and completed_epochs > 0
        if resume and completed_epochs > 0:
            if latest_weights_path.exists():
                model.load_weights(latest_weights_path)
                print(f"Training dilanjutkan dari latest weights: {latest_weights_path}")
            elif checkpoint_path.exists():
                model = keras.models.load_model(checkpoint_path)
                print(f"Latest weights tidak ada, memakai checkpoint terbaik: {checkpoint_path}")
            else:
                print("Checkpoint resume belum tersedia, training dimulai ulang dari feature model.")
                initial_epoch = 0
                csv_append = False

        history = train_model(
            model,
            train_ds,
            val_ds,
            settings,
            checkpoint_path=checkpoint_path,
            latest_weights_path=latest_weights_path,
            history_log_path=history_log_path,
            initial_epoch=initial_epoch,
            epochs=epochs,
            csv_append=csv_append,
        )

        if checkpoint_path.exists():
            model = keras.models.load_model(checkpoint_path)
            print(f"Model terbaik dimuat dari: {checkpoint_path}")

        save_model(model, final_model_path)
        plot_training_history(
            history,
            epochs,
            REPORT_DIR / f"grafik_pelatihan_{model_name}.png",
            title=f"Performa Training - {model_name}",
        )

    evaluation = evaluate_model(
        model,
        test_ds,
        class_names,
        model_name,
        make_prediction_plot=False,
        make_confusion_plot=False,
    )
    summary = summarize_model(model, test_ds, class_names, model_name)
    trainable, non_trainable = count_trainable_params(model)
    best_val_loss, best_epoch, epochs_ran = get_best_val_loss_and_epoch(history_log_path)
    summary.update(
        {
            "unfrozen_layers": layer_count,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "epochs_ran": epochs_ran,
            "trainable_params": trainable,
            "non_trainable_params": non_trainable,
            "test_loss": evaluation["test_loss"],
        }
    )
    return summary


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    feature_model_path = args.feature_model_path
    if feature_model_path is None:
        feature_best_path = OUTPUT_DIR / f"mobilenetv2_{args.experiment_name}_feature_extraction_best_val_loss.keras"
        feature_model_path = OUTPUT_DIR / f"mobilenetv2_{args.experiment_name}_feature_extraction.keras"
        if feature_best_path.exists():
            feature_model_path = feature_best_path

    if not feature_model_path.exists():
        raise FileNotFoundError(f"Model feature extraction tidak ditemukan: {feature_model_path}")

    settings = replace(
        SETTINGS,
        dataset_path=str(args.dataset_path),
        model_name=args.experiment_name,
    )
    train_ds, val_ds, test_ds, metadata = prepare_datasets(settings)
    class_names = metadata["class_names"]
    layer_counts = parse_layer_counts(args.layers)

    print("=" * 70)
    print("PERC0BAAN ABLASI FINE-TUNING MOBILENETV2")
    print("=" * 70)
    print(f"Dataset              : {settings.dataset_path}")
    print(f"Experiment name      : {args.experiment_name}")
    print(f"Feature model        : {feature_model_path}")
    print(f"Layer counts         : {layer_counts}")
    print(f"Epoch maksimum       : {args.epochs}")
    print(f"Learning rate        : {args.learning_rate}")
    print(f"Class names          : {class_names}")

    results = []
    if 0 in layer_counts:
        feature_model = keras.models.load_model(feature_model_path)
        results.append(evaluate_feature_extraction_baseline(feature_model, args.experiment_name, test_ds, class_names))
        keras.backend.clear_session()

    for layer_count in layer_counts:
        if layer_count == 0:
            continue
        results.append(
            run_fine_tuning_ablation(
                layer_count,
                args.experiment_name,
                feature_model_path,
                train_ds,
                val_ds,
                test_ds,
                class_names,
                settings,
                args.epochs,
                args.learning_rate,
                args.skip_existing,
                args.resume,
            )
        )
        keras.backend.clear_session()

    comparison_df = build_comparison_table(results)
    comparison_path = save_comparison_table(
        comparison_df,
        filename=f"tabel_ablation_mobilenetv2_{args.experiment_name}.csv",
    )
    print("=" * 70)
    print("HASIL ABLASI")
    print("=" * 70)
    print(comparison_df)
    print(f"Tabel ablasi disimpan ke: {comparison_path}")


if __name__ == "__main__":
    main()
