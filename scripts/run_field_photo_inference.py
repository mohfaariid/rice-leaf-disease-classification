"""Run field-photo inference with the final CNN and MobileNetV2 models."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIELD_DIR = PROJECT_ROOT / "data" / "foto_lapangan"
REPORTS_DIR = PROJECT_ROOT / "reports"

IMAGE_SIZE = 256
CLASS_NAMES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf scald",
    "Sheath Blight",
    "Tungro",
]

MODELS = [
    (
        "CNN Kustom + Dropout",
        PROJECT_ROOT / "outputs" / "custom_cnn_balanced_quality_hybrid_653_best_val_loss.keras",
    ),
    (
        "MobileNetV2 fine-tuning 50 layer",
        PROJECT_ROOT
        / "outputs"
        / "mobilenetv2_balanced_quality_hybrid_653_ablation_50_layers_best_val_loss.keras",
    ),
]


def load_image_batch(image_path: Path) -> np.ndarray:
    image = tf.keras.utils.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    image_array = tf.keras.utils.img_to_array(image)
    return np.expand_dims(image_array, axis=0)


def normalize_scores(raw_scores: np.ndarray) -> np.ndarray:
    scores = raw_scores.astype("float64")
    if not np.isclose(np.sum(scores), 1.0, atol=1e-3):
        scores = tf.nn.softmax(scores).numpy()
    return scores


def predict_image(model: tf.keras.Model, image_path: Path) -> tuple[str, float, list[tuple[str, float]]]:
    raw_predictions = model.predict(load_image_batch(image_path), verbose=0)[0]
    scores = normalize_scores(raw_predictions)
    top_indices = np.argsort(scores)[::-1][:3]
    top3 = [(CLASS_NAMES[int(index)], float(scores[int(index)])) for index in top_indices]
    return top3[0][0], top3[0][1], top3


def format_top3(top3: list[tuple[str, float]]) -> str:
    return "; ".join(f"{label} ({score * 100:.2f}%)" for label, score in top3)


def save_visualization(rows: list[dict[str, str]], output_path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    for ax, row in zip(axes, rows):
        image = Image.open(row["path"])
        image = ImageOps.exif_transpose(image)
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(
            f"{row['file']}\n"
            f"CNN: {row['cnn_prediction']} ({row['cnn_confidence']})\n"
            f"MobileNetV2: {row['mobilenet_prediction']} ({row['mobilenet_confidence']})",
            fontsize=9,
        )

    for ax in axes[len(rows) :]:
        ax.axis("off")

    fig.suptitle("Hasil Prediksi Foto Lapangan", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        [
            path
            for path in FIELD_DIR.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
    )
    if not image_paths:
        raise FileNotFoundError(f"Tidak ada gambar di folder {FIELD_DIR}")

    loaded_models = [(name, tf.keras.models.load_model(path, compile=False)) for name, path in MODELS]

    rows: list[dict[str, str]] = []
    for image_path in image_paths:
        predictions: dict[str, tuple[str, float, list[tuple[str, float]]]] = {}
        for model_name, model in loaded_models:
            predictions[model_name] = predict_image(model, image_path)

        cnn_label, cnn_conf, cnn_top3 = predictions["CNN Kustom + Dropout"]
        mobile_label, mobile_conf, mobile_top3 = predictions["MobileNetV2 fine-tuning 50 layer"]

        rows.append(
            {
                "file": image_path.name,
                "path": str(image_path),
                "cnn_prediction": cnn_label,
                "cnn_confidence": f"{cnn_conf * 100:.2f}%",
                "cnn_top3": format_top3(cnn_top3),
                "mobilenet_prediction": mobile_label,
                "mobilenet_confidence": f"{mobile_conf * 100:.2f}%",
                "mobilenet_top3": format_top3(mobile_top3),
                "same_prediction": "Ya" if cnn_label == mobile_label else "Tidak",
            }
        )

    csv_path = REPORTS_DIR / "prediksi_foto_lapangan_2_model.csv"
    md_path = REPORTS_DIR / "prediksi_foto_lapangan_2_model.md"
    png_path = REPORTS_DIR / "hasil_prediksi_foto_lapangan_2_model.png"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as file:
        file.write("# Hasil Prediksi Foto Lapangan\n\n")
        file.write(
            "Pengujian ini menggunakan foto lapangan tanpa label ground truth. "
            "Hasil berikut digunakan sebagai pengujian tambahan secara kualitatif.\n\n"
        )
        file.write(
            "| No | File | Prediksi CNN Kustom + Dropout | Confidence CNN | "
            "Prediksi MobileNetV2 | Confidence MobileNetV2 | Prediksi Sama |\n"
        )
        file.write("|---:|---|---|---:|---|---:|---|\n")
        for index, row in enumerate(rows, start=1):
            file.write(
                f"| {index} | `{row['file']}` | {row['cnn_prediction']} | "
                f"{row['cnn_confidence']} | {row['mobilenet_prediction']} | "
                f"{row['mobilenet_confidence']} | {row['same_prediction']} |\n"
            )
        file.write("\n## Top-3 Prediction\n\n")
        file.write("| No | File | Top-3 CNN | Top-3 MobileNetV2 |\n")
        file.write("|---:|---|---|---|\n")
        for index, row in enumerate(rows, start=1):
            file.write(f"| {index} | `{row['file']}` | {row['cnn_top3']} | {row['mobilenet_top3']} |\n")

    save_visualization(rows, png_path)

    print(f"Jumlah foto diuji: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    print(f"Gambar: {png_path}")
    for row in rows:
        print(
            f"{row['file']} | CNN={row['cnn_prediction']} ({row['cnn_confidence']}) | "
            f"MobileNetV2={row['mobilenet_prediction']} ({row['mobilenet_confidence']})"
        )


if __name__ == "__main__":
    main()
