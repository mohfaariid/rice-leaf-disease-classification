"""Unit & Integration Test untuk REST API Klasifikasi Penyakit Daun Padi."""

from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
from api import app

client = TestClient(app)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_health_check_endpoint():
    """Memverifikasi endpoint / mengembalikan status 200 dan payload valid."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "documentation" in data


def test_metadata_endpoint():
    """Memverifikasi endpoint /metadata mengembalikan konfigurasi arsitektur dan 7 kelas."""
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["num_classes"] == 7
    assert data["accuracy"] == 0.9226
    assert data["macro_f1_score"] == 0.9217
    assert "Tungro" in data["classes"]
    assert "Healthy Rice Leaf" in data["classes"]


def test_predict_endpoint_with_image():
    """Menguji inferensi langsung melalui endpoint /predict dengan citra uji lapangan."""
    sample_path = PROJECT_ROOT / "data" / "foto_lapangan" / "foto_001.jpeg"
    if not sample_path.exists():
        pytest.skip("Berkas foto sampel lapangan tidak ditemukan.")

    with open(sample_path, "rb") as img_file:
        response = client.post(
            "/predict",
            files={"file": ("foto_001.jpeg", img_file, "image/jpeg")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "predicted_class" in data
    assert "predicted_class_id" in data
    assert data["confidence"] > 0.0
    assert len(data["top_3_predictions"]) == 3
    assert data["inference_time_ms"] > 0.0
    assert "recommendation" in data


def test_predict_endpoint_invalid_file():
    """Memverifikasi bahwa unggahan non-gambar akan ditolak dengan HTTP 400."""
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"Bukan berkas gambar", "text/plain")},
    )
    assert response.status_code == 400
    assert "tidak didukung" in response.json()["detail"]


def test_chat_endpoint_consultation():
    """Memverifikasi endpoint /chat memberikan jawaban cerdas seputar penyakit dan agronomi."""
    payload = {
        "query": "Bagaimana cara penanganan penyakit hawar daun bakteri?",
        "context_disease": "Bacterial Leaf Blight",
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "Hawar Daun Bakteri" in data["response"] or "Nitrogen" in data["response"]

