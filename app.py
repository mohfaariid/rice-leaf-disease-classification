"""Aplikasi Web Klasifikasi & Diagnosis Penyakit Daun Padi (7 Kelas).

Dikembangkan berdasarkan penelitian Tugas Akhir:
"Analisis Perbandingan Performa CNN Kustom dan MobileNetV2 pada Klasifikasi
Penyakit Daun Padi Menggunakan Dataset Tidak Seimbang dan Seimbang"
Oleh: Mohammad Farid Anshori (NIM: 22523124)
Program Studi Informatika - Universitas Islam Indonesia (2026)
"""

import streamlit as st

from constants import CSS_FILE
from model_utils import load_mobilenet_model, load_custom_cnn_model
from ui.tab_diagnosis import render_tab_diagnosis
from ui.tab_comparison import render_tab_comparison
from ui.tab_chatbot import render_tab_chatbot
from ui.tab_research import render_tab_research


# ==============================================================================
# PENGATURAN TAMPILAN STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Rice Leaf Vision - Klasifikasi Penyakit Daun Padi",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Muat Stylesheet Eksternal (static/style.css)
if CSS_FILE.exists():
    with open(CSS_FILE, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ==============================================================================
# HEADER HERO BANNER
# ==============================================================================
st.markdown(
    """
    <div class="hero">
        <span class="badge">Sistem Pendukung Keputusan Berbasis Deep Learning</span>
        <h1>Rice Leaf Vision</h1>
        <p>
            Sistem klasifikasi otomatis 7 kondisi daun padi berbasis kecerdasan buatan.
            Membandingkan performa <strong>MobileNetV2 Fine-Tuning</strong> (Model Terbaik Skripsi - Akurasi 92,26%) 
            dan <strong>CNN Kustom</strong> pada dataset seimbang quality-hybrid.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# INISIALISASI MODEL & TABS
# ==============================================================================
mob_model = load_mobilenet_model()
cnn_model = load_custom_cnn_model()

tab_diagnosa, tab_komparasi, tab_chat, tab_skripsi = st.tabs([
    "🌾 Diagnosis & Klasifikasi Daun",
    "⚖️ Komparasi Model (Head-to-Head)",
    "🤖 Konsultasi & Tanya Jawab AI",
    "📊 Ringkasan Riset & Metrik Skripsi",
])

with tab_diagnosa:
    selected_image = render_tab_diagnosis(mob_model, cnn_model)

with tab_komparasi:
    render_tab_comparison(selected_image, mob_model, cnn_model)

with tab_chat:
    render_tab_chatbot()

with tab_skripsi:
    render_tab_research()