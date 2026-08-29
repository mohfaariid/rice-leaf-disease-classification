"""Script command line untuk menjalankan inference."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configuration.paths import MODEL_PATH
from inference.predict import run


def main():
    parser = argparse.ArgumentParser(description="Prediksi penyakit daun padi.")
    parser.add_argument("image_path", help="Path gambar yang akan diprediksi.")
    parser.add_argument("--model-path", default=str(MODEL_PATH), help="Path model .keras.")
    args = parser.parse_args()
    run(args.image_path, args.model_path)


if __name__ == "__main__":
    main()
