"""Training MobileNetV2 transfer learning."""

from configuration.paths import OUTPUT_DIR, ensure_project_dirs
from configuration.settings import SETTINGS
from models.mobilenetv2 import build_mobilenetv2
from preprocessing.dataset import prepare_datasets
from training.trainer import save_model, train_model


def run():
    """Menjalankan training MobileNetV2 dari awal sampai simpan model."""
    ensure_project_dirs()
    train_ds, val_ds, test_ds, metadata = prepare_datasets(SETTINGS)

    model = build_mobilenetv2(
        image_size=SETTINGS.image_size,
        num_classes=metadata["num_classes"],
    )
    model.summary()

    history = train_model(model, train_ds, val_ds, SETTINGS)
    save_path = save_model(model, OUTPUT_DIR / "rice_disease_mobilenetv2.keras")

    return {
        "model": model,
        "history": history,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "metadata": metadata,
        "save_path": save_path,
    }


if __name__ == "__main__":
    run()
