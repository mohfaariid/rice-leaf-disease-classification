# 🌾 Rice Leaf Vision: End-to-End Rice Leaf Disease Classification Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Undergraduate Thesis Project (Skripsi Sarjana)**  
> **Title:** *Analisis Perbandingan Performa CNN Kustom dan MobileNetV2 pada Klasifikasi Penyakit Daun Padi Menggunakan Dataset Tidak Seimbang dan Seimbang*  
> **Author:** Mohammad Farid Anshori (NIM: 22523124)  
> **Advisor:** Arrie Kurniawardhani, S.Si., M.Kom.  
> **Institution:** Program Studi Informatika, Universitas Islam Indonesia (UII), 2026

---

## 📌 Executive Summary

**Rice Leaf Vision** is an end-to-end computer vision and decision support system designed to classify **seven conditions of rice leaf images** (*Oryza sativa*) using deep learning architectures: a **Custom CNN (trained from scratch)** and a **Fine-Tuned MobileNetV2 (Transfer Learning)**.

The research addresses the real-world challenge of **severe class imbalance** (e.g., Tungro disease over-representation) by proposing a **Balanced Quality-Hybrid strategy** (quality-based filtering on majority samples combined with targeted augmentation on minority classes). The project is deployed as both an **interactive Streamlit diagnostic web application** with **Explainable AI (Grad-CAM)** and a **production-ready FastAPI REST API**.

---

## 🔬 Key Research Highlights & Benchmark Results

### 1. Benchmark Evaluation (Thesis Table 4.10)
All models were evaluated on the identical independent test set (768 images) across multiple scenarios:

| Model Architecture | Dataset Scenario | Configuration | Accuracy | Precision Macro | Recall Macro | F1-Score Macro |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **CNN Kustom (Awal)** | Original (Imbalanced) | Baseline Scratch | 83.20% | 81.11% | 80.79% | 80.74% |
| **CNN Kustom + Dropout** | Original (Imbalanced) | Regularized Scratch | 86.98% | 85.41% | 85.22% | 85.10% |
| **MobileNetV2** | Original (Imbalanced) | Fine-Tuning 50 Layers | 87.89% | 86.08% | 85.96% | 85.87% |
| **CNN Kustom + Dropout** | Balanced Quality-Hybrid | Regularized Scratch | 76.34% | 75.97% | 76.07% | 75.72% |
| **MobileNetV2 (Champion)** | **Balanced Quality-Hybrid** | **Fine-Tuning 50 Layers** | **92.26%** | **92.14%** | **92.21%** | **92.17%** |

### 2. Core Scientific Findings
* **Champion Model:** **MobileNetV2 with 50-layer fine-tuning on the balanced quality-hybrid dataset** achieved the highest performance across all evaluation metrics (**92.26% Accuracy** and **92.17% Macro F1-Score**).
* **The Asymmetric Impact of Dataset Balancing:**
  * For **MobileNetV2**, quality-hybrid balancing yielded a major performance surge (**+4.37% Accuracy, +6.30% F1-Macro**), demonstrating that pre-trained weights benefit substantially from clean, balanced visual feature representations.
  * For **Custom CNN**, balancing led to a drop from 86.98% to 76.34%, proving that scratch models are heavily data-hungry and fragile when sample volume is pruned.
* **Computational Efficiency for Edge Deployment:**  
  MobileNetV2 reduced model storage by **76% (24.1 MB vs. 101.0 MB)** and utilized **3.7x fewer parameters (2.4M vs. 25.2M)** compared to the Custom CNN, rendering it ideal for mobile and field drone deployment.

---

## 🚀 Core Features

### 1. 🌾 Interactive Streamlit Web Application (`app.py`)
* **Flexible Multi-Input:** Supports local image file upload (`.jpg`, `.jpeg`, `.png`, `.webp`), **direct live camera capture** (`st.camera_input`), and a **built-in quick sample gallery** including field test photos.
* **Explainable AI (Grad-CAM Heatmaps):** Real-time activation mapping highlighting the exact leaf lesions responsible for the prediction, validating model attention against false background artifacts.
* **Side-by-Side Model Comparison (Head-to-Head):** Evaluates MobileNetV2 vs. Custom CNN concurrently, displaying real-time latency in milliseconds and prediction discrepancies.
* **🤖 Integrated AI Agricultural Consultation (Agri-Chatbot):** Context-aware conversational AI assistant (`chatbot.py`) answering agronomic questions regarding symptoms, balanced fertilizer dosage (N-P-K & Silica), biological/chemical treatments, resistant rice varieties, and thesis ML findings. Operates 100% offline via a local domain knowledge engine or with Google Gemini API.
* **Agricultural Action Guidance & Export:** Actionable recommendations and one-click diagnostic report download in `.txt` format.

### 2. ⚡ Production-Ready FastAPI REST Service (`api.py`)
* High-throughput inference API with endpoints:
  * `POST /predict`: Real-time image disease classification and agronomic guidance.
  * `POST /chat`: AI conversational consultation endpoint.
  * `GET /metadata`: Model metrics, parameters, and class ontology.
* Built-in CORS support for seamless integration with mobile apps (Flutter/React Native) or web dashboards.
* **Interactive API Documentation:** Available out-of-the-box via Swagger UI (`/docs`) and ReDoc (`/redoc`).

### 3. 🐳 Containerization & Testing
* **Docker Support:** Reproducible deployment container defined in [`Dockerfile`](Dockerfile).
* **Automated Unit Testing:** 5-point test suite implemented with `pytest` and `fastapi.testclient` verifying API integrity, chat endpoint, and inference pipeline.

---

## 📂 Project Structure

```text
rice_leaf_disease_project/
│
├── app.py                     # Streamlit Interactive Web Application (Grad-CAM & Head-to-Head)
├── api.py                     # FastAPI REST API Service (Production Backend)
├── chatbot.py                 # Hybrid AI Agronomic & Thesis Q&A Chatbot Engine
├── Dockerfile                 # Multi-stage Docker Container Configuration
├── requirements_web.txt       # Web and API dependencies
├── .gitignore                 # Safe git exclusion rules (filters files > 100MB)
├── .dockerignore              # Docker build optimization rules
│
├── models/                    # Neural Network Architectures
│   ├── custom_cnn.py          # Custom CNN architecture & dropout variants
│   ├── mobilenetv2.py         # MobileNetV2 transfer learning & layer-ablation builder
│   └── registry.py            # Model factory registry
│
├── preprocessing/             # Data Preparation Pipelines
│   ├── dataset.py             # TensorFlow tf.data dataset loader
│   ├── distribution.py        # Class distribution metrics
│   ├── split.py               # Train / Val / Test splitting
│   └── transforms.py          # Preprocessing & augmentation layers
│
├── training/                  # Model Training Pipelines
│   ├── trainer.py             # Core training loop with callbacks
│   ├── callbacks.py           # Early stopping, ModelCheckpoint, LR scheduler
│   └── train_mobilenetv2.py   # Specialized fine-tuning trainer
│
├── outputs/                   # Saved Models & Weights
│   └── mobilenetv2_balanced_quality_hybrid_653_ablation_50_layers_best_val_loss.keras (24.1 MB)
│
├── tests/                     # Automated Test Suite
│   └── test_api.py            # Pytest suite for FastAPI endpoints
│
└── data/                      # Data assets
    └── foto_lapangan/         # Real-world field test photography
```

---

## 🛠️ Quickstart Guide

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/mohfaariid/rice-leaf-disease-classification-cnn-mobilenetv2.git
cd rice_leaf_disease_project

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements_web.txt
```

### 2. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser and navigate to: **`http://localhost:8501`**

### 3. Launch the FastAPI Service
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
* **API Root:** `http://localhost:8000`
* **Interactive Swagger UI:** `http://localhost:8000/docs`

#### Sample API Request (`curl`):
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/foto_lapangan/foto_001.jpeg"
```

### 4. Run Automated Tests
```bash
pytest tests/test_api.py -v
```

### 5. Run via Docker
```bash
# Build docker image
docker build -t rice-leaf-vision .

# Run container (Exposing Streamlit on 8501)
docker run -p 8501:8501 rice-leaf-vision
```

---

## 🌿 Target Disease Classes

| No | Disease Class (English) | Nama Kondisi (Bahasa Indonesia) | Causal Agent |
| :-: | :--- | :--- | :--- |
| 1 | **Bacterial Leaf Blight** | Hawar Daun Bakteri | *Xanthomonas oryzae pv. oryzae* |
| 2 | **Brown Spot** | Bercak Cokelat | *Bipolaris oryzae* |
| 3 | **Healthy Rice Leaf** | Daun Padi Sehat | *Normal physiological status* |
| 4 | **Leaf Blast** | Blas Daun | *Magnaporthe oryzae* |
| 5 | **Leaf Scald** | Hawar Bergaris / Scald | *Microdochium albescens* |
| 6 | **Sheath Blight** | Hawar Pelepah | *Rhizoctonia solani* |
| 7 | **Tungro** | Penyakit Tungro | *RTBV & RTSV (Vector: Nephotettix virescens)* |

---

## 📜 Citation & Academic Acknowledgment

If you reference this implementation or research in your academic work, please cite:

```bibtex
@thesis{anshori2026riceleaf,
  author       = {Mohammad Farid Anshori},
  title        = {Analisis Perbandingan Performa CNN Kustom dan MobileNetV2 pada Klasifikasi Penyakit Daun Padi Menggunakan Dataset Tidak Seimbang dan Seimbang},
  school       = {Universitas Islam Indonesia},
  year         = {2026},
  type         = {Skripsi Sarjana},
  address      = {Yogyakarta, Indonesia}
}
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
