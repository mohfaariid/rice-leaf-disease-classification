"""Main pipeline CNN kustom improved untuk 6 kelas tanpa Tungro."""

from training.train_custom_cnn_6_classes_improved import run as run_improved_pipeline


PIPELINE_STEPS = [
    "Load dataset 6 kelas tanpa Tungro",
    "Split train/validation/test",
    "Augmentasi dan preprocessing improved",
    "Model: CNN Kustom Improved",
    "Training CNN Kustom Improved",
    "Evaluasi CNN Kustom Improved",
    "Confusion matrix",
    "Ringkasan hasil untuk paper",
]


def print_pipeline_steps() -> None:
    """Menampilkan urutan pipeline sebelum proses dijalankan."""
    print("=" * 70)
    print("URUTAN PIPELINE PAPER - CNN KUSTOM IMPROVED")
    print("=" * 70)
    for index, step in enumerate(PIPELINE_STEPS, start=1):
        print(f"{index}. {step}")
    print("=" * 70)


def main():
    """Menjalankan pipeline CNN kustom improved 6 kelas tanpa Tungro."""
    print_pipeline_steps()
    return run_improved_pipeline()


if __name__ == "__main__":
    main()
