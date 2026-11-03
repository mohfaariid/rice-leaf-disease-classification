"""Main pipeline CNN kustom untuk 6 kelas tanpa Tungro.

Urutan eksekusi ketika file ini dijalankan:
1. Load dataset 6 kelas
2. Split train/validation/test
3. Augmentasi dan preprocessing
4. Model: CNN Kustom
5. Training CNN Kustom
6. Evaluasi CNN Kustom
7. Confusion matrix
8. Ringkasan hasil untuk paper
"""

from training.train_custom_cnn_6_classes import run as run_custom_cnn_6_classes


PIPELINE_STEPS = [
    "Load dataset 6 kelas tanpa Tungro",
    "Split train/validation/test",
    "Augmentasi dan preprocessing",
    "Model: CNN Kustom",
    "Training CNN Kustom",
    "Evaluasi CNN Kustom",
    "Confusion matrix",
    "Ringkasan hasil untuk paper",
]


def print_pipeline_steps() -> None:
    """Menampilkan urutan pipeline sebelum proses dijalankan."""
    print("=" * 70)
    print("URUTAN PIPELINE PAPER")
    print("=" * 70)
    for index, step in enumerate(PIPELINE_STEPS, start=1):
        print(f"{index}. {step}")
    print("=" * 70)


def main():
    """Menjalankan pipeline CNN kustom 6 kelas tanpa Tungro."""
    print_pipeline_steps()
    return run_custom_cnn_6_classes()


if __name__ == "__main__":
    main()
