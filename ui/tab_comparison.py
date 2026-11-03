"""Tab 2: Komparasi Head-to-Head Model (MobileNetV2 vs CNN Kustom)."""

import numpy as np
from PIL import Image
import streamlit as st

from constants import (
    CLASS_NAMES,
    CLASS_NAMES_ID,
)
from model_utils import run_prediction


def render_tab_comparison(selected_image: Image.Image | None, mob_model, cnn_model):
    """Merender Tab 2 untuk pengujian langsung kedua model secara bersamaan."""
    st.markdown('<div class="section-header">Komparasi Langsung: MobileNetV2 vs CNN Kustom</div>', unsafe_allow_html=True)
    st.write(
        "Fitur ini menguji citra daun padi secara bersamaan pada kedua model "
        "untuk melihat perbedaan prediksi, tingkat kepercayaan, dan waktu pemrosesan secara *real-time*."
    )

    if selected_image is None:
        st.info("💡 Silakan pilih atau unggah citra terlebih dahulu pada Tab **'🌾 Diagnosis & Klasifikasi Daun'** untuk menjalankan komparasi.")
        return

    scores_mob, lat_mob = run_prediction(selected_image, mob_model)
    top_mob = int(np.argmax(scores_mob))

    if cnn_model is not None:
        scores_cnn, lat_cnn = run_prediction(selected_image, cnn_model)
        top_cnn = int(np.argmax(scores_cnn))

    col_mob, col_cnn = st.columns(2, gap="large")

    with col_mob:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="res-label">🏆 MobileNetV2 (Fine-Tuning 50 Layers)</div>
                <div class="res-title">{CLASS_NAMES_ID[CLASS_NAMES[top_mob]]}</div>
                <div class="res-meta">Confidence: <strong>{scores_mob[top_mob]:.2%}</strong> · Latensi: <strong>{lat_mob:.1f} ms</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("**3 Prediksi Teratas MobileNetV2:**")
        ranked_mob = np.argsort(scores_mob)[::-1]
        for i in ranked_mob[:3]:
            st.write(f"- {CLASS_NAMES_ID[CLASS_NAMES[i]]}: **{scores_mob[i]:.2%}**")

    with col_cnn:
        if cnn_model is not None:
            st.markdown(
                f"""
                <div class="result-card-cnn">
                    <div class="res-label">🔬 CNN Kustom (Scratch + Dropout)</div>
                    <div class="res-title">{CLASS_NAMES_ID[CLASS_NAMES[top_cnn]]}</div>
                    <div class="res-meta">Confidence: <strong>{scores_cnn[top_cnn]:.2%}</strong> · Latensi: <strong>{lat_cnn:.1f} ms</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("**3 Prediksi Teratas CNN Kustom:**")
            ranked_cnn = np.argsort(scores_cnn)[::-1]
            for i in ranked_cnn[:3]:
                st.write(f"- {CLASS_NAMES_ID[CLASS_NAMES[i]]}: **{scores_cnn[i]:.2%}**")
        else:
            st.info(
                "💡 **Model Pembanding CNN Kustom Dikecualikan dari Git (101 MB)**\\n\\n"
                "MobileNetV2 (24,1 MB) tetap dapat dijalankan secara instan. "
                "Perbandingan metrik performa riset lengkap dapat dilihat pada tabel di bawah."
            )

    st.markdown("---")
    st.markdown('<div class="section-header">Perbandingan Karakteristik Arsitektur & Efisiensi</div>', unsafe_allow_html=True)
    st.table({
        "Karakteristik": [
            "Akurasi Pengujian (Balanced)",
            "F1-Score Macro",
            "Jumlah Parameter",
            "Ukuran Berkas Model",
            "Pendekatan Pembelajaran",
            "Keunggulan Utama",
        ],
        "MobileNetV2 (Fine-tuning 50 Layer)": [
            "92,26%",
            "92,17%",
            "2.421.319 (2,4M)",
            "24,1 MB",
            "Transfer Learning Pretrained ImageNet",
            "Akurasi tinggi, representasi fitur matang, tahan foto lapangan",
        ],
        "CNN Kustom (Dropout)": [
            "76,34%",
            "75,72%",
            "25.250.247 (25,2M)",
            "101,0 MB",
            "Pelatihan dari Nol (Scratch)",
            "Arsitektur konvolusi sederhana, tanpa dependensi bobot eksternal",
        ],
    })
