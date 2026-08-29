"""Registry model yang tersedia."""

from models.custom_cnn import build_custom_cnn
from models.mobilenetv2 import build_mobilenetv2


MODEL_REGISTRY = {
    "custom_cnn": build_custom_cnn,
    "mobilenetv2": build_mobilenetv2,
}


def get_model_builder(model_key: str):
    """Mengambil builder model berdasarkan nama."""
    try:
        return MODEL_REGISTRY[model_key]
    except KeyError as exc:
        available = ", ".join(MODEL_REGISTRY)
        raise ValueError(f"Model tidak dikenal: {model_key}. Pilihan: {available}") from exc
