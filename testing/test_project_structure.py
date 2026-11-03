"""Pengujian ringan struktur proyek."""

from pathlib import Path


def test_core_directories_exist():
    project_root = Path(__file__).resolve().parents[1]
    expected_dirs = [
        "configuration",
        "data",
        "preprocessing",
        "models",
        "training",
        "evaluation",
        "inference",
        "scripts",
    ]

    for dirname in expected_dirs:
        assert (project_root / dirname).exists()
