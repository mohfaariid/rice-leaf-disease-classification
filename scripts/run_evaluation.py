"""Script command line untuk menjalankan evaluasi."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configuration.paths import MODEL_PATH
from evaluation.evaluate import run


def main():
    parser = argparse.ArgumentParser(description="Evaluasi model penyakit daun padi.")
    parser.add_argument("--model-path", default=str(MODEL_PATH), help="Path model .keras.")
    args = parser.parse_args()
    run(args.model_path)


if __name__ == "__main__":
    main()
