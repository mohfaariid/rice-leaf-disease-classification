"""REST API Service untuk Klasifikasi Penyakit Daun Padi (7 Kelas) menggunakan FastAPI.

Dikembangkan berdasarkan penelitian skripsi:
Model: MobileNetV2 Fine-tuning 50 Layers (Balanced Quality-Hybrid Dataset)
Akurasi: 92.26% | Macro F1-Score: 92.17%
Penulis: Mohammad Farid Anshori (NIM: 22523124)
"""

from pathlib import Path
import os
import time
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from PIL import Image, ImageOps

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import tensorflow as tf
from chatbot import get_chat_response


# ==============================================================================
# INISIALISASI & KONFIGURASI
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "mobilenetv2_balanced_quality_hybrid_653_ablation_50_layers_best_val_loss.keras"
)
IMAGE_SIZE = 256

CLASS_NAMES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf Scald",
    "Sheath Blight",
    "Tungro",
]

CLASS_NAMES_ID = {
    "Bacterial Leaf Blight": "Hawar Daun Bakteri (Bacterial Leaf Blight)",
    "Brown Spot": "Bercak Cokelat (Brown Spot)",
    "Healthy Rice Leaf": "Daun Padi Sehat (Healthy Rice Leaf)",
    "Leaf Blast": "Blas Daun (Leaf Blast)",
    "Leaf Scald": "Hawar Bergaris / Scald (Leaf Scald)",
    "Sheath Blight": "Hawar Pelepah (Sheath Blight)",
    "Tungro": "Penyakit Tungro (Tungro)",
}

CLASS_ACTIONS_ID = {
    "Bacterial Leaf Blight": "Periksa petakan sekitar, atur drainase air, kurangi pemupukan Nitrogen berlebih, dan hindari perpindahan alat basah.",
    "Brown Spot": "Perbaiki kecukupan hara Kalium & Silika, sanitasi sisa jerami pasca-panen, dan jaga suplai pengairan sawah.",
    "Healthy Rice Leaf": "Kondisi fisiologis normal. Lanjutkan pengamatan rutin mingguan dan pertahankan pola pemupukan berimbang.",
    "Leaf Blast": "Waspadai penjalaran spora jamur lewat angin ke daun muda, kurangi dosis N, dan gunakan varietas tahan blas di musim berikutnya.",
    "Leaf Scald": "Perbaiki aerasi kanopi tanaman (misal jajar legowo) dan bersihkan gulma pematang yang menjadi inang jamur.",
    "Sheath Blight": "Periksa bagian pangkal pelepah dekat permukaan air dan lakukan pengeringan sawah secara berkala (intermittent).",
    "Tungro": "Pantau populasi wereng hijau (vektor virus) dan laporkan segera ke PPL/POPT setempat untuk tindakan eradikasi tanaman sakit.",
}

app = FastAPI(
    title="Rice Leaf Disease Classification API",
    description="REST API Machine Learning untuk deteksi dan diagnosis 7 kondisi penyakit daun padi secara real-time.",
    version="1.0.0",
)

# Aktifkan CORS agar API bisa diakses dari frontend web/mobile
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None


def get_model() -> tf.keras.Model:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model weight file not found: {MODEL_PATH}")
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _model


# ==============================================================================
# SCHEMAS
# ==============================================================================
class ClassPrediction(BaseModel):
    class_name: str
    class_name_id: str
    confidence: float
    confidence_percentage: str


class PredictionResponse(BaseModel):
    success: bool
    predicted_class: str
    predicted_class_id: str
    confidence: float
    confidence_percentage: str
    inference_time_ms: float
    recommendation: str
    top_3_predictions: list[ClassPrediction]


# ==============================================================================
# ENDPOINTS
# ==============================================================================
@app.get("/", tags=["Health Check"])
def root():
    return {
        "status": "online",
        "service": "Rice Leaf Disease Classification API",
        "version": "1.0.0",
        "author": "Mohammad Farid Anshori",
        "documentation": "/docs",
    }


@app.get("/metadata", tags=["Model Metadata"])
def get_metadata():
    return {
        "model_architecture": "MobileNetV2 Transfer Learning",
        "fine_tuning": "50 layers unfreezed",
        "input_resolution": f"{IMAGE_SIZE}x{IMAGE_SIZE}x3",
        "dataset_scenario": "Balanced Quality-Hybrid (653 images/class)",
        "accuracy": 0.9226,
        "macro_f1_score": 0.9217,
        "num_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_rice_leaf(file: UploadFile = File(...)):
    """Menerima berkas citra (JPG, PNG, WEBP), melakukan inferensi, dan mengembalikan hasil diagnosis."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Format berkas '{file.content_type}' tidak didukung. Harap unggah berkas citra (JPG, PNG, atau WEBP).",
        )

    try:
        content = await file.read()
        image = Image.open(BytesIO(content))

        # Preprocessing & Orientasi EXIF
        prepared = ImageOps.exif_transpose(image).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        image_array = np.asarray(prepared, dtype=np.float32)
        input_tensor = np.expand_dims(image_array, axis=0)

        # Inferensi Model
        model = get_model()
        t_start = time.perf_counter()
        raw_scores = model.predict(input_tensor, verbose=0)[0]
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        if not np.isclose(np.sum(raw_scores), 1.0, atol=1e-3):
            scores = tf.nn.softmax(raw_scores).numpy().astype(float)
        else:
            scores = raw_scores.astype(float)

        ranked_indices = np.argsort(scores)[::-1]
        top_idx = int(ranked_indices[0])
        top_label = CLASS_NAMES[top_idx]
        top_conf = float(scores[top_idx])

        top_3 = []
        for idx in ranked_indices[:3]:
            lbl = CLASS_NAMES[int(idx)]
            conf = float(scores[int(idx)])
            top_3.append(
                ClassPrediction(
                    class_name=lbl,
                    class_name_id=CLASS_NAMES_ID[lbl],
                    confidence=round(conf, 4),
                    confidence_percentage=f"{conf * 100:.2f}%",
                )
            )

        return PredictionResponse(
            success=True,
            predicted_class=top_label,
            predicted_class_id=CLASS_NAMES_ID[top_label],
            confidence=round(top_conf, 4),
            confidence_percentage=f"{top_conf * 100:.2f}%",
            inference_time_ms=round(latency_ms, 2),
            recommendation=CLASS_ACTIONS_ID[top_label],
            top_3_predictions=top_3,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal memproses citra untuk inferensi: {str(exc)}",
        )


class ChatRequest(BaseModel):
    query: str
    context_disease: str | None = None
    api_key: str | None = None


class ChatResponse(BaseModel):
    success: bool
    query: str
    response: str
    context_disease: str | None = None


@app.post("/chat", response_model=ChatResponse, tags=["AI Consultation"])
def chat_consultation(request: ChatRequest):
    """Konsultasi tanya-jawab dengan Asisten Pakar Padi AI (bekerja 100% offline via local knowledge engine atau online via Gemini)."""
    reply = get_chat_response(
        query=request.query,
        current_predicted_disease=request.context_disease,
        api_key=request.api_key,
    )
    return ChatResponse(
        success=True,
        query=request.query,
        response=reply,
        context_disease=request.context_disease,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
