"""Main pipeline untuk eksperimen dataset 7 kelas balanced revisi."""

from training.train_comparison_balanced_tungro import run as run_balanced_pipeline


PIPELINE_STEPS = [
    "Load dataset 7 kelas balanced revisi",
    "Split train/validation/test",
    "Augmentasi dan preprocessing",
    "Model 1: CNN Kustom",
    "Training CNN Kustom",
    "Evaluasi CNN Kustom",
    "Model 2: MobileNetV2",
    "Training MobileNetV2",
    "Evaluasi MobileNetV2",
    "Tabel perbandingan hasil balanced",
    "Confusion matrix masing-masing model",
    "Kesimpulan model terbaik",
]


def print_pipeline_steps() -> None:
    """Menampilkan urutan pipeline sebelum proses dijalankan."""
    print("=" * 70)
    print("URUTAN PIPELINE SKRIPSI - DATASET BALANCED TUNGRO")
    print("=" * 70)
    for index, step in enumerate(PIPELINE_STEPS, start=1):
        print(f"{index}. {step}")
    print("=" * 70)


def main():
    """Menjalankan pipeline balanced revisi."""
    print_pipeline_steps()
    return run_balanced_pipeline()


if __name__ == "__main__":
    main()
