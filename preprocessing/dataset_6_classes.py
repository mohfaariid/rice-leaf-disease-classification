"""Loader dataset khusus eksperimen 6 kelas tanpa Tungro."""

from pathlib import Path

import tensorflow as tf
from sklearn.model_selection import train_test_split

from configuration.settings import SETTINGS, ProjectSettings
from preprocessing.dataset import list_class_dirs
from preprocessing.split import count_batches


EXCLUDED_CLASS = "Tungro"
SIX_CLASS_NAMES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf scald",
    "Sheath Blight",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def collect_image_paths_and_labels(settings: ProjectSettings = SETTINGS):
    """Mengambil path gambar dan label untuk 6 kelas tanpa Tungro."""
    dataset_dir = Path(settings.dataset_path)
    class_dirs = list_class_dirs(dataset_dir)
    available_classes = {class_dir.name for class_dir in class_dirs}

    missing_classes = [class_name for class_name in SIX_CLASS_NAMES if class_name not in available_classes]
    if missing_classes:
        raise ValueError(f"Kelas berikut tidak ditemukan di dataset: {missing_classes}")

    image_paths = []
    labels = []
    for label_index, class_name in enumerate(SIX_CLASS_NAMES):
        class_dir = dataset_dir / class_name
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(str(image_path))
                labels.append(label_index)

    if not image_paths:
        raise ValueError(f"Tidak ada gambar yang ditemukan di {dataset_dir}")

    return image_paths, labels, SIX_CLASS_NAMES


def split_paths_and_labels(image_paths, labels, settings: ProjectSettings = SETTINGS):
    """Membagi path gambar menjadi train, validation, dan test secara stratified."""
    test_val_size = settings.validation_split
    val_size_from_temp = 0.5

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths,
        labels,
        test_size=test_val_size,
        random_state=settings.seed,
        stratify=labels,
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=val_size_from_temp,
        random_state=settings.seed,
        stratify=temp_labels,
    )

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def load_and_resize_image(image_path, label, settings: ProjectSettings = SETTINGS):
    """Membaca file gambar dan mengubah ukurannya sesuai konfigurasi."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [settings.image_size, settings.image_size])
    image = tf.cast(image, tf.float32)
    return image, label


def build_dataset(image_paths, labels, settings: ProjectSettings = SETTINGS, shuffle: bool = False):
    """Membuat tf.data.Dataset dari path gambar dan label."""
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(settings.shuffle_buffer, len(image_paths)),
            seed=settings.seed,
            reshuffle_each_iteration=True,
        )
    dataset = dataset.map(
        lambda image_path, label: load_and_resize_image(image_path, label, settings),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(settings.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def prepare_6_class_datasets(settings: ProjectSettings = SETTINGS):
    """Memuat, membagi, dan mengoptimalkan dataset 6 kelas tanpa Tungro."""
    image_paths, labels, class_names = collect_image_paths_and_labels(settings)
    train_data, val_data, test_data = split_paths_and_labels(image_paths, labels, settings)

    train_paths, train_labels = train_data
    val_paths, val_labels = val_data
    test_paths, test_labels = test_data

    train_ds = build_dataset(train_paths, train_labels, settings, shuffle=True)
    val_ds = build_dataset(val_paths, val_labels, settings, shuffle=False)
    test_ds = build_dataset(test_paths, test_labels, settings, shuffle=False)

    metadata = {
        "class_names": class_names,
        "num_classes": len(class_names),
        "excluded_class": EXCLUDED_CLASS,
        "total_images": len(image_paths),
        "train_images": len(train_paths),
        "val_images": len(val_paths),
        "test_images": len(test_paths),
        "train_batches": count_batches(train_ds),
        "val_batches": count_batches(val_ds),
        "test_batches": count_batches(test_ds),
    }
    return train_ds, val_ds, test_ds, metadata
