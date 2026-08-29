"""Script command line untuk menjalankan perbandingan dua model."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.train_comparison import run


if __name__ == "__main__":
    run()
