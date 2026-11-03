"""Tab 1: Diagnosis & Klasifikasi Daun Padi (Real-time & XAI Grad-CAM)."""

from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st

from constants import (
    CLASS_NAMES,
    CLASS_NAMES_ID,
    CLASS_GUIDANCE_ID,
    FIELD_PHOTOS_DIR,
    AUG_DATASET_DIR,
)
from model_utils import (
    run_prediction,
    generate_gradcam_overlay,
)


def render_tab_diagnosis(mob_model, cnn_model) -> Image.Image | None:
    """Merender Tab 1 untuk alur diagnosis, visualisasi Grad-CAM, dan panduan agronomi."""
    col_input, col_result = st.columns([1.1, 1.3], gap="large")

    with col_input:
        st.markdown('<div class="section-header">Pilih Sumber Citra Daun</div>', unsafe_allow_html=True)
        input_mode = st.radio(
            "Metode Input Citra",
            ["📁 Unggah Berkas", "📷 Kamera Langsung", "🖼️ Galeri Sampel Uji"],
            horizontal=True,
            label_visibility="collapsed",
        )

        selected_image: Image.Image | None = None
        image_name_caption = ""

        if input_mode == "📁 Unggah Berkas":
            uploaded_file = st.file_uploader(
                "Pilih berkas citra daun padi (JPG, JPEG, PNG, WEBP)",
                type=["jpg", "jpeg", "png", "webp"],
            )
            if uploaded_file is not None:
                selected_image = Image.open(uploaded_file)
                image_name_caption = uploaded_file.name

        elif input_mode == "📷 Kamera Langsung":
            camera_file = st.camera_input("Arahkan kamera ke permukaan helai daun padi")
            if camera_file is not None:
                selected_image = Image.open(camera_file)
                image_name_caption = "Foto Kamera Langsung"

        else:  # Galeri Sampel Uji
            sample_options = {
                "Pilih Sampel Uji...": None,
                "Bacterial Leaf Blight (Sampel Dataset)": AUG_DATASET_DIR / "Bacterial Leaf Blight" / "20231006_164208.jpg",
                "Brown Spot (Sampel Dataset)": AUG_DATASET_DIR / "Brown Spot" / "20231006_165550.jpg",
                "Healthy Rice Leaf (Daun Sehat)": AUG_DATASET_DIR / "Healthy Rice Leaf" / "20231006_163255.jpg",
                "Leaf Blast (Sampel Dataset)": AUG_DATASET_DIR / "Leaf Blast" / "aug_0_101.jpg",
                "Leaf Scald (Sampel Dataset)": AUG_DATASET_DIR / "Leaf scald" / "aug_0_0.jpg",
                "Sheath Blight (Sampel Dataset)": AUG_DATASET_DIR / "Sheath Blight" / "aug_0_100.jpg",
                "Tungro (Sampel Dataset)": AUG_DATASET_DIR / "Tungro" / "TUNGRO1_001.jpg",
                "Foto Lapangan 001 (Uji Lapangan)": FIELD_PHOTOS_DIR / "foto_001.jpeg",
                "Foto Lapangan 002 (Uji Lapangan)": FIELD_PHOTOS_DIR / "foto_002.jpeg",
            }
            chosen_sample = st.selectbox("Pilih salah satu contoh citra:", list(sample_options.keys()))
            if chosen_sample and sample_options[chosen_sample] and Path(sample_options[chosen_sample]).exists():
                selected_image = Image.open(sample_options[chosen_sample])
                image_name_caption = chosen_sample

        st.markdown('<div class="section-header">Pengaturan Model & Visualisasi</div>', unsafe_allow_html=True)
        selected_model_name = st.selectbox(
            "Model Klasifikasi Aktif",
            [
                "MobileNetV2 (Model Terbaik - Fine-Tuned 50 Layers)",
                "CNN Kustom (Dropout - Balanced Quality-Hybrid)",
            ],
        )
        show_gradcam = st.checkbox("🔍 Tampilkan Peta Fokus AI (Grad-CAM Heatmap)", value=True)
        st.caption("ℹ️ *Catatan: Grad-CAM memperlihatkan area piksel bercak/lesi yang paling memengaruhi keputusan model.*")

    with col_result:
        if selected_image is None:
            st.markdown('<div class="section-header">Ruang Hasil Analisis</div>', unsafe_allow_html=True)
            st.info("👈 Silakan unggah foto daun, gunakan kamera, atau pilih salah satu contoh citra sampel di sebelah kiri.")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown('<div class="stat-box"><div class="stat-val">92,26%</div><div class="stat-desc">Akurasi Terbaik</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="stat-box"><div class="stat-val">92,17%</div><div class="stat-desc">Macro F1-Score</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown('<div class="stat-box"><div class="stat-val">7 Kelas</div><div class="stat-desc">Kondisi Daun Padi</div></div>', unsafe_allow_html=True)
        else:
            img_col, cam_col = st.columns(2)
            with img_col:
                st.image(selected_image, caption=f"Citra Masukan: {image_name_caption}", use_column_width=True)

            active_model = mob_model if "MobileNetV2" in selected_model_name else cnn_model
            if active_model is None:
                st.warning("⚠️ Berkas bobot model CNN Kustom tidak ditemukan di disk (dikecualikan dari Git karena ukuran 101 MB). Silakan gunakan model **MobileNetV2** pada pilihan di sebelah kiri.")
                return selected_image
            scores, latency = run_prediction(selected_image, active_model)
            ranked_indices = np.argsort(scores)[::-1]
            top_idx = int(ranked_indices[0])
            top_label = CLASS_NAMES[top_idx]
            top_label_id = CLASS_NAMES_ID[top_label]
            top_conf = float(scores[top_idx])

            st.session_state["active_disease"] = top_label
            st.session_state["active_conf"] = top_conf
            st.session_state["selected_image"] = selected_image

            if show_gradcam and "MobileNetV2" in selected_model_name:
                gradcam_img = generate_gradcam_overlay(active_model, selected_image, top_idx)
                with cam_col:
                    if gradcam_img is not None:
                        st.image(gradcam_img, caption="Grad-CAM: Area Fokus Visual AI", use_column_width=True)
                    else:
                        st.caption("Grad-CAM tidak dapat dibuat untuk format ini.")

            card_class = "result-card" if "MobileNetV2" in selected_model_name else "result-card-cnn"
            st.markdown(
                f"""
                <div class="{card_class}">
                    <div class="res-label">Hasil Prediksi ({selected_model_name.split()[0]})</div>
                    <div class="res-title">{top_label_id}</div>
                    <div class="res-meta">Tingkat Keyakinan (Confidence): <strong>{top_conf:.2%}</strong> · Latensi: <strong>{latency:.1f} ms</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown('<div class="section-header">Distribusi 3 Probabilitas Teratas</div>', unsafe_allow_html=True)
            for idx in ranked_indices[:3]:
                lbl = CLASS_NAMES_ID[CLASS_NAMES[int(idx)]]
                sc = float(scores[int(idx)])
                st.write(f"**{lbl}** · {sc:.2%}")
                st.progress(sc)

            guidance = CLASS_GUIDANCE_ID[top_label]
            with st.expander("🌾 Penjelasan Gejala & Anjuran Tindakan Lapangan", expanded=True):
                st.markdown(f"**Analisis Gejala:** {guidance['analysis']}")
                st.markdown("**Langkah Penanganan Direkomendasikan:**")
                for act in guidance["actions"]:
                    st.markdown(f"- {act}")
                st.warning(f"⚠️ **Penting untuk Dihindari:** {guidance['avoid']}")

            report_text = f"""LAPORAN HASIL DIAGNOSIS PENYAKIT DAUN PADI
==============================================
Sumber Berkas   : {image_name_caption}
Model Penguji   : {selected_model_name}
Waktu Inferensi : {latency:.2f} ms

HASIL DETEKSI:
----------------------------------------------
Penyakit / Kondisi : {top_label_id}
Tingkat Keyakinan  : {top_conf:.2%}

3 PROBABILITAS TERATAS:
1. {CLASS_NAMES_ID[CLASS_NAMES[int(ranked_indices[0])]]}: {scores[ranked_indices[0]]:.2%}
2. {CLASS_NAMES_ID[CLASS_NAMES[int(ranked_indices[1])]]}: {scores[ranked_indices[1]]:.2%}
3. {CLASS_NAMES_ID[CLASS_NAMES[int(ranked_indices[2])]]}: {scores[ranked_indices[2]]:.2%}

ANJURAN TINDAKAN LAPANGAN:
{chr(10).join(['- ' + a for a in guidance['actions']])}

CATATAN PENCEGAHAN:
{guidance['avoid']}
==============================================
Catatan: Hasil prediksi ini merupakan bantuan skrining citra digital computer vision dan bukan diagnosis mutlak hayati.
"""
            st.download_button(
                label="📥 Unduh Ringkasan Laporan Diagnosis (.txt)",
                data=report_text,
                file_name=f"diagnosis_{top_label.replace(' ', '_').lower()}.txt",
                mime="text/plain",
            )
            st.info("💡 **Ingin konsultasi atau bertanya seputar penyakit ini?** Buka tab **'🤖 Konsultasi & Tanya Jawab AI'** di atas untuk berdiskusi dengan Asisten Pakar Padi mengenai obat, dosis pupuk, varietas tahan, atau alasan diagnosa AI!")

    return selected_image
