"""Entry point prediksi gambar baru."""

from configuration.paths import MODEL_PATH
from inference.predictor import RiceDiseasePredictor
from preprocessing.dataset import prepare_datasets


def run(image_path: str, model_path=MODEL_PATH):
    """Menjalankan prediksi untuk satu gambar."""
    _, _, _, metadata = prepare_datasets()
    predictor = RiceDiseasePredictor(model_path, metadata["class_names"])
    result = predictor.predict(image_path)

    print("=" * 60)
    print("HASIL PREDIKSI")
    print("=" * 60)
    print(f"Gambar     : {result['image_path']}")
    print(f"Prediksi   : {result['predicted_label']}")
    print(f"Confidence : {result['confidence']:.4f}")
    print("=" * 60)

    return result
