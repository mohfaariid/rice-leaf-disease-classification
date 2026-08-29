"""Utilitas untuk menghitung dan menampilkan distribusi kelas dataset."""

from __future__ import annotations

import csv
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_class_distribution(dataset_path: str | Path) -> list[dict[str, int | str]]:
    """Mengambil jumlah citra pada setiap folder kelas."""
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {dataset_dir}")

    rows: list[dict[str, int | str]] = []
    for class_dir in sorted([path for path in dataset_dir.iterdir() if path.is_dir()]):
        image_count = sum(
            1
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        rows.append({"class_name": class_dir.name, "image_count": image_count})

    if not rows:
        raise ValueError(f"Tidak ada folder kelas di dalam dataset: {dataset_dir}")

    return rows


def print_class_distribution(rows: list[dict[str, int | str]], title: str) -> None:
    """Menampilkan distribusi kelas ke terminal."""
    total_images = sum(int(row["image_count"]) for row in rows)
    max_name_length = max(len(str(row["class_name"])) for row in rows)

    print(title)
    print("-" * 70)
    print(f"{'Kelas'.ljust(max_name_length)} | Jumlah Citra")
    print("-" * 70)
    for row in rows:
        print(f"{str(row['class_name']).ljust(max_name_length)} | {int(row['image_count']):>12}")
    print("-" * 70)
    print(f"{'Total'.ljust(max_name_length)} | {total_images:>12}")


def save_class_distribution(
    rows: list[dict[str, int | str]],
    output_path: str | Path,
) -> Path:
    """Menyimpan distribusi kelas ke file CSV."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["class_name", "image_count"])
        writer.writeheader()
        writer.writerows(rows)

    return output_file
