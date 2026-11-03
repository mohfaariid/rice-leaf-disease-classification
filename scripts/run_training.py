"""Script command line untuk menjalankan training."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.train_custom_cnn import run as run_custom_cnn
from training.train_comparison import run as run_comparison
from training.train_mobilenetv2 import run as run_mobilenetv2


def main():
    parser = argparse.ArgumentParser(description="Training model penyakit daun padi.")
    parser.add_argument(
        "--model",
        choices=["custom_cnn", "mobilenetv2", "comparison"],
        default="comparison",
        help="Model yang akan dilatih.",
    )
    args = parser.parse_args()

    if args.model == "custom_cnn":
        run_custom_cnn()
    elif args.model == "mobilenetv2":
        run_mobilenetv2()
    else:
        run_comparison()


if __name__ == "__main__":
    main()
