"""Definisi path proyek."""

from pathlib import Path

from configuration.settings import SETTINGS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = Path(SETTINGS.dataset_path)
OUTPUT_DIR = PROJECT_ROOT / SETTINGS.output_dir
REPORT_DIR = PROJECT_ROOT / SETTINGS.report_dir
MODEL_PATH = OUTPUT_DIR / "rice_disease_model.keras"


def ensure_project_dirs() -> None:
    """Membuat folder output dan report jika belum ada."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def validate_dataset_dir(dataset_dir: Path = DATASET_DIR) -> list[Path]:
    """Validasi folder dataset dan kembalikan daftar folder kelas."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {dataset_dir}")

    class_dirs = [p for p in dataset_dir.iterdir() if p.is_dir()]
    if not class_dirs:
        raise ValueError(f"Tidak ada subfolder kelas di: {dataset_dir}")

    return sorted(class_dirs)
