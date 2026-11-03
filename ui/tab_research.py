"""Tab 4: Ringkasan Riset & Metrik Skripsi (Tabel 4.10 & Temuan Utama)."""

import streamlit as st


def render_tab_research():
    """Merender Tab 4 yang merangkum hasil eksperimen naskah Laporan Tugas Akhir."""
    st.markdown('<div class="section-header">Rekapitulasi Hasil Eksperimen Skripsi (Tabel 4.10)</div>', unsafe_allow_html=True)
    st.write(
        "Tabel di bawah ini merangkum seluruh skenario eksperimen utama yang dievaluasi "
        "pada naskah Laporan Tugas Akhir (*2026_07_22523124_LapTA.pdf*):"
    )

    st.table({
        "Model": [
            "CNN Kustom Awal",
            "CNN Kustom + Dropout",
            "MobileNetV2 (50 Layer)",
            "CNN Kustom + Dropout",
            "MobileNetV2 (50 Layer) [Terbaik]",
        ],
        "Skenario Dataset": [
            "Original (Tidak Seimbang)",
            "Original (Tidak Seimbang)",
            "Original (Tidak Seimbang)",
            "Balanced Quality-Hybrid",
            "Balanced Quality-Hybrid",
        ],
        "Akurasi": ["83,20%", "86,98%", "87,89%", "76,34%", "92,26%"],
        "Precision Macro": ["81,11%", "85,41%", "86,08%", "75,97%", "92,14%"],
        "Recall Macro": ["80,79%", "85,22%", "85,96%", "76,07%", "92,21%"],
        "F1-Score Macro": ["80,74%", "85,10%", "85,87%", "75,72%", "92,17%"],
    })

    st.markdown('<div class="section-header">Kesimpulan Pokok Penelitian</div>', unsafe_allow_html=True)
    st.markdown(
        """
        1. **Model Terbaik:** **MobileNetV2 Fine-tuning 50 Layer pada dataset Balanced Quality-Hybrid** mencapai performa tertinggi (**Akurasi 92,26%** dan **F1-Macro 92,17%**).
        2. **Dampak Strategi Penyeimbangan Data (Quality-Hybrid):**
           - Pada **MobileNetV2**, penyeimbangan kualitas hibrida mendongkrak performa (+4,37% akurasi & +6,30% F1-macro) karena representasi data visual lebih bersih dan seimbang.
           - Pada **CNN Kustom**, performa justru turun dari 86,98% ke 76,34%, membuktikan bahwa model yang dilatih dari nol (*scratch*) sangat rentan ketika jumlah sampel dipangkas.
        3. **Efisiensi Model:** MobileNetV2 memiliki ukuran bobot 4 kali lebih hemat (24,1 MB vs 101 MB) dan jumlah parameter 10 kali lebih sedikit daripada CNN Kustom berkat mekanisme *depthwise separable convolution*.
        """
    )
    st.caption("Disusun oleh: Mohammad Farid Anshori (NIM: 22523124) · Dosen Pembimbing: Arrie Kurniawardhani, S.Si., M.Kom. · UII 2026")
