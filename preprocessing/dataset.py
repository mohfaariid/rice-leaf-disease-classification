"""Loader dataset penyakit daun padi."""

from pathlib import Path

import tensorflow as tf

from configuration.settings import SETTINGS, ProjectSettings
from preprocessing.split import count_batches, split_validation_test


def load_train_and_val_test(settings: ProjectSettings = SETTINGS):
    """Memuat training set dan gabungan validation-test dari direktori gambar."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        settings.dataset_path,
        validation_split=settings.validation_split,
        subset="training",
        seed=settings.seed,
        image_size=(settings.image_size, settings.image_size),
        batch_size=settings.batch_size,
    )

    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        settings.dataset_path,
        validation_split=settings.validation_split,
        subset="validation",
        seed=settings.seed,
        image_size=(settings.image_size, settings.image_size),
        batch_size=settings.batch_size,
    )

    return train_ds, val_test_ds


def prepare_datasets(settings: ProjectSettings = SETTINGS):
    """Memuat, membagi, dan mengoptimalkan dataset seperti alur notebook."""
    train_ds, val_test_ds = load_train_and_val_test(settings)
    class_names = train_ds.class_names
    val_ds, test_ds = split_validation_test(val_test_ds)

    train_batches = count_batches(train_ds)
    val_batches = count_batches(val_ds)
    test_batches = count_batches(test_ds)

    train_ds, val_ds, test_ds = optimize_datasets(train_ds, val_ds, test_ds, settings)

    metadata = {
        "class_names": class_names,
        "num_classes": len(class_names),
        "train_batches": train_batches,
        "val_batches": val_batches,
        "test_batches": test_batches,
        "estimated_train_images": train_batches * settings.batch_size,
        "estimated_val_images": val_batches * settings.batch_size,
        "estimated_test_images": test_batches * settings.batch_size,
    }
    return train_ds, val_ds, test_ds, metadata


def optimize_datasets(train_ds, val_ds, test_ds, settings: ProjectSettings = SETTINGS):
    """Menerapkan cache, shuffle, dan prefetch."""
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(settings.shuffle_buffer).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)
    test_ds = test_ds.cache().prefetch(buffer_size=autotune)
    return train_ds, val_ds, test_ds


def list_class_dirs(dataset_path: str | Path = SETTINGS.dataset_path) -> list[Path]:
    """Mengembalikan daftar folder kelas pada dataset."""
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {dataset_dir}")

    class_dirs = [p for p in dataset_dir.iterdir() if p.is_dir()]
    if not class_dirs:
        raise ValueError(f"Tidak ada subfolder kelas di: {dataset_dir}")

    return sorted(class_dirs)
