"""Training CNN kustom."""

from configuration.paths import MODEL_PATH, ensure_project_dirs
from configuration.settings import SETTINGS
from models.custom_cnn import build_custom_cnn
from preprocessing.dataset import prepare_datasets
from training.trainer import save_model, train_model


def run():
    """Menjalankan training CNN kustom dari awal sampai simpan model."""
    ensure_project_dirs()
    train_ds, val_ds, test_ds, metadata = prepare_datasets(SETTINGS)

    model = build_custom_cnn(
        image_size=SETTINGS.image_size,
        num_classes=metadata["num_classes"],
        model_name=SETTINGS.model_name,
    )
    model.summary()

    history = train_model(model, train_ds, val_ds, SETTINGS)
    save_path = save_model(model, MODEL_PATH)

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
