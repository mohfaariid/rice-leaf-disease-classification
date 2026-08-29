"""Utilitas training model."""

from pathlib import Path

from configuration.paths import MODEL_PATH, ensure_project_dirs
from configuration.settings import SETTINGS, ProjectSettings
from training.callbacks import build_training_callbacks


def train_model(
    model,
    train_ds,
    val_ds,
    settings: ProjectSettings = SETTINGS,
    checkpoint_path: str | Path | None = None,
    latest_weights_path: str | Path | None = None,
    history_log_path: str | Path | None = None,
    initial_epoch: int = 0,
    epochs: int | None = None,
    csv_append: bool = False,
):
    """Melatih model dengan dataset training dan validation."""
    callbacks = build_training_callbacks(
        val_ds,
        checkpoint_path=checkpoint_path,
        latest_weights_path=latest_weights_path,
        history_log_path=history_log_path,
        csv_append=csv_append,
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs or settings.epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=2,
    )
    return history


def save_model(model, save_path: str | Path = MODEL_PATH) -> Path:
    """Menyimpan model terlatih."""
    ensure_project_dirs()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    return save_path
