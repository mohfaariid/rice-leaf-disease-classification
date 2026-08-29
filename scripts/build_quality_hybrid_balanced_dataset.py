"""Membangun dataset balanced dengan quality-based filtering dan augmentasi.

Alur revisi:
1. Hitung skor kualitas citra dari ketajaman, kontras, dan pencahayaan.
2. Untuk kelas mayoritas, pilih citra terbaik berdasarkan skor kualitas.
3. Untuk kelas minoritas, pertahankan citra asli lalu tambah citra augmentasi
   sampai jumlah tiap kelas sama.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT.parent / "Rice_Leaf_AUG"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "rice_leaf_aug_7_balanced_quality_hybrid_653"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "quality_hybrid_balancing_report.csv"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "reports" / "quality_hybrid_balancing_summary.md"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ImageQuality:
    path: Path
    class_name: str
    width: int
    height: int
    brightness: float
    contrast: float
    sharpness: float
    quality_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Buat dataset balanced revisi dengan quality-based filtering dan augmentasi.",
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="Jumlah target per kelas. Default: jumlah terbesar dari kelas non-mayoritas.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Hapus folder output jika sudah ada.",
    )
    return parser.parse_args()


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def variance_of_laplacian(gray: np.ndarray) -> float:
    center = gray[1:-1, 1:-1] * 4
    neighbors = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    return float(np.var(center - neighbors))


def compute_quality(path: Path, class_name: str) -> ImageQuality | None:
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            width, height = image.size
            gray = np.asarray(image.convert("L"), dtype=np.float32)
    except Exception as exc:
        print(f"[SKIP] Gagal membaca {path}: {exc}")
        return None

    brightness = float(gray.mean())
    contrast = float(gray.std())
    sharpness = variance_of_laplacian(gray)

    sharpness_score = min(1.0, np.log1p(sharpness) / np.log1p(1000.0))
    contrast_score = min(1.0, contrast / 64.0)
    brightness_score = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)
    resolution_score = min(1.0, (width * height) / float(224 * 224))

    quality_score = (
        0.55 * sharpness_score
        + 0.20 * contrast_score
        + 0.15 * brightness_score
        + 0.10 * resolution_score
    )

    return ImageQuality(
        path=path,
        class_name=class_name,
        width=width,
        height=height,
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
        quality_score=float(quality_score),
    )


def load_quality_rows(source_dir: Path) -> dict[str, list[ImageQuality]]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Folder sumber tidak ditemukan: {source_dir}")

    class_dirs = sorted(path for path in source_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"Tidak ada folder kelas di: {source_dir}")

    quality_by_class: dict[str, list[ImageQuality]] = {}
    for class_dir in class_dirs:
        rows = []
        for image_path in list_images(class_dir):
            quality = compute_quality(image_path, class_dir.name)
            if quality is not None:
                rows.append(quality)
        quality_by_class[class_dir.name] = sorted(
            rows,
            key=lambda item: (item.quality_score, item.sharpness),
            reverse=True,
        )
    return quality_by_class


def choose_target_count(quality_by_class: dict[str, list[ImageQuality]], target_count: int | None) -> int:
    if target_count is not None:
        return target_count

    counts = {class_name: len(rows) for class_name, rows in quality_by_class.items()}
    majority_class = max(counts, key=counts.get)
    non_majority_counts = [count for class_name, count in counts.items() if class_name != majority_class]
    return max(non_majority_counts)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    project_data_dir = (PROJECT_ROOT / "data").resolve()
    resolved_output = output_dir.resolve()

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Folder output sudah ada: {output_dir}. Gunakan --overwrite.")
        if project_data_dir not in resolved_output.parents:
            raise ValueError(f"Output tidak berada di folder data proyek: {resolved_output}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def copy_selected_images(selected: list[ImageQuality], destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for item in selected:
        shutil.copy2(item.path, destination_dir / item.path.name)


def augment_image(source_path: Path, destination_path: Path, rng: random.Random) -> None:
    with Image.open(source_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")

    if rng.random() < 0.5:
        image = ImageOps.mirror(image)

    angle = rng.uniform(-18, 18)
    image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(0, 0, 0))

    width, height = image.size
    zoom = rng.uniform(0.90, 1.00)
    crop_w = max(1, int(width * zoom))
    crop_h = max(1, int(height * zoom))
    left = rng.randint(0, max(0, width - crop_w))
    top = rng.randint(0, max(0, height - crop_h))
    image = image.crop((left, top, left + crop_w, top + crop_h)).resize((width, height), Image.Resampling.BICUBIC)

    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.90, 1.10))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.90, 1.10))
    image.save(destination_path, format="JPEG", quality=95)


def build_dataset(
    quality_by_class: dict[str, list[ImageQuality]],
    output_dir: Path,
    report_path: Path,
    summary_path: Path,
    target_count: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    with report_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "class_name",
                "source_file",
                "output_file",
                "action",
                "width",
                "height",
                "brightness",
                "contrast",
                "sharpness",
                "quality_score",
            ],
        )
        writer.writeheader()

        for class_name, rows in quality_by_class.items():
            class_output_dir = output_dir / class_name
            selected = rows[:target_count]
            rejected = rows[target_count:]
            copy_selected_images(selected, class_output_dir)

            for item in selected:
                writer.writerow(build_report_row(item, item.path.name, "selected_original"))

            for item in rejected:
                writer.writerow(build_report_row(item, "", "excluded_lower_quality"))

            augmented_count = max(0, target_count - len(selected))
            for index in range(augmented_count):
                source_item = selected[index % len(selected)]
                output_name = f"aug_quality_hybrid_{index + 1:04d}_{source_item.path.stem}.jpg"
                augment_image(source_item.path, class_output_dir / output_name, rng)
                writer.writerow(build_report_row(source_item, output_name, "augmented_minority"))

            summary_rows.append(
                {
                    "class_name": class_name,
                    "original_count": len(rows),
                    "selected_original": len(selected),
                    "excluded_lower_quality": len(rejected),
                    "augmented": augmented_count,
                    "final_count": len(selected) + augmented_count,
                }
            )

    write_summary(summary_path, output_dir, target_count, summary_rows, report_path)


def build_report_row(item: ImageQuality, output_file: str, action: str) -> dict[str, str | float | int]:
    return {
        "class_name": item.class_name,
        "source_file": str(item.path),
        "output_file": output_file,
        "action": action,
        "width": item.width,
        "height": item.height,
        "brightness": round(item.brightness, 4),
        "contrast": round(item.contrast, 4),
        "sharpness": round(item.sharpness, 4),
        "quality_score": round(item.quality_score, 6),
    }


def write_summary(
    summary_path: Path,
    output_dir: Path,
    target_count: int,
    rows: list[dict[str, int | str]],
    report_path: Path,
) -> None:
    lines = [
        "# Ringkasan Dataset Balanced Quality Hybrid",
        "",
        f"Folder output: `{output_dir}`",
        f"Target jumlah per kelas: {target_count}",
        f"Report detail: `{report_path}`",
        "",
        "| Kelas | Data awal | Asli dipakai | Dikeluarkan | Augmentasi | Total akhir |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {class_name} | {original_count} | {selected_original} | "
            "{excluded_lower_quality} | {augmented} | {final_count} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Metode: quality-based filtering digunakan saat memilih citra asli yang dipertahankan. "
            "Kelas dengan jumlah data lebih banyak dari target dikurangi berdasarkan skor kualitas citra. "
            "Kelas dengan jumlah data kurang dari target ditambah menggunakan augmentasi citra.",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    quality_by_class = load_quality_rows(args.source_dir)
    target_count = choose_target_count(quality_by_class, args.target_count)
    prepare_output_dir(args.output_dir, args.overwrite)
    build_dataset(
        quality_by_class,
        args.output_dir,
        args.report_path,
        args.summary_path,
        target_count,
        args.seed,
    )
    print(f"Dataset balanced revisi dibuat di: {args.output_dir}")
    print(f"Target per kelas: {target_count}")
    print(f"Report detail: {args.report_path}")
    print(f"Ringkasan: {args.summary_path}")


if __name__ == "__main__":
    main()
