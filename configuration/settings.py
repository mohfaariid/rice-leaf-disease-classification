"""Pengaturan umum eksperimen."""

from dataclasses import dataclass, field
import os


@dataclass(frozen=True)
class ProjectSettings:
    """Konfigurasi utama proyek klasifikasi penyakit daun padi."""

    dataset_path: str = r"C:/Users/farid/Downloads/archive (5)/Rice_Leaf_AUG"
    image_size: int = int(os.getenv("RICE_IMAGE_SIZE", "256"))
    batch_size: int = int(os.getenv("RICE_BATCH_SIZE", "32"))
    epochs: int = int(os.getenv("RICE_EPOCHS", "20"))
    seed: int = 123
    validation_split: float = 0.30
    shuffle_buffer: int = 1000
    model_name: str = "CNN_RiceDiseaseClassifier"
    output_dir: str = "outputs"
    report_dir: str = "reports"
    class_names: list[str] = field(default_factory=list)


SETTINGS = ProjectSettings()
