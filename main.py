"""Main pipeline untuk perbandingan CNN Kustom dan MobileNetV2.

Urutan eksekusi ketika file ini dijalankan:
1. Load dataset
2. Split train/validation/test
3. Augmentasi dan preprocessing
4. Model 1: CNN Kustom
5. Training CNN Kustom
6. Evaluasi CNN Kustom
7. Model 2: MobileNetV2
8. Training MobileNetV2
9. Evaluasi MobileNetV2
10. Tabel perbandingan hasil
11. Confusion matrix masing-masing model
12. Kesimpulan model terbaik
"""

from training.train_comparison import run as run_comparison_pipeline


PIPELINE_STEPS = [
    "Load dataset",
    "Split train/validation/test",
    "Augmentasi dan preprocessing",
    "Model 1: CNN Kustom",
    "Training CNN Kustom",
    "Evaluasi CNN Kustom",
    "Model 2: MobileNetV2",
    "Training MobileNetV2",
    "Evaluasi MobileNetV2",
    "Tabel perbandingan hasil",
    "Confusion matrix masing-masing model",
    "Kesimpulan model terbaik",
]


def print_pipeline_steps() -> None:
    """Menampilkan urutan pipeline sebelum proses dijalankan."""
    print("=" * 70)
    print("URUTAN PIPELINE PENELITIAN")
    print("=" * 70)
    for index, step in enumerate(PIPELINE_STEPS, start=1):
        print(f"{index}. {step}")
    print("=" * 70)


def main():
    """Menjalankan pipeline lengkap perbandingan dua model."""
    print_pipeline_steps()
    return run_comparison_pipeline()


if __name__ == "__main__":
    main()
