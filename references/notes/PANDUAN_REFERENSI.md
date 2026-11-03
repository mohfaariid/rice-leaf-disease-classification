# Panduan Lengkap Referensi untuk Skripsi BAB I-V

## 📋 Ringkasan Pekerjaan

Telah dilakukan penyesuaian penomoran referensi pada draft skripsi BAB I-V dengan jurnal-jurnal yang tersedia di folder `references/journals/`. 

### Status: ✅ SELESAI

---

## 📁 File yang Telah Diperbarui

### 1. **draft_skripsi_bab_1_5.md** (Diperbarui)
   - **Perubahan**: Bagian "Daftar Pustaka Sementara" (baris 462-485) telah diganti dengan "Daftar Pustaka" yang terstruktur
   - **Jumlah Referensi**: 14 referensi (sebelumnya ada placeholder untuk 10)
   - **Format**: Sederhana (judul - jurnal/sumber - tahun)
   - **Lokasi**: Akhir document (BAB V - Penutup)

### 2. **REFERENSI_MAPPING.md** (File Baru)
   - **Isi**: Tabel pemetaan lengkap antara nomor citasi [1]-[14] dengan nama file PDF
   - **Fungsi**: Referensi silang untuk verifikasi
   - **Kegunaan**: Memudahkan menemukan file PDF untuk setiap referensi

### 3. **DAFTAR_PUSTAKA_IEEE.md** (File Baru)
   - **Isi**: Format IEEE lengkap untuk semua 14 referensi
   - **Fitur**: Tabel ringkasan, panduan penggunaan per bab, dan catatan penting
   - **Kegunaan**: Template untuk final submission dengan format resmi

---

## 📌 Pemetaan Referensi [1]-[14]

| No | PDF File | Topik |
|----|----------|-------|
| [1] | `18_Frontiers_Automatic_Diagnosis_Rice_Diseases_DL_2021.pdf` | Diagnosis otomatis dengan deep learning |
| [2] | `31_Frontiers_Image_Dataset_Config_Rice_Disease_CNN_2022.pdf` | Konfigurasi dataset CNN |
| [3] | `23_CoSIE_MobileNetV2_EfficientNetB0_Rice_Leaf_Disease_2026.pdf` | MobileNetV2 vs EfficientNetB0 |
| [4] | `24_Mandiri_MobileNetV2_MobileNetV3_Rice_Leaf_Disease_2025.pdf` | MobileNetV2 vs MobileNetV3 |
| [5] | `25_TIERS_Enhanced_CAE_CNN_Rice_Leaf_Disease_2025.pdf` | CNN arsitektur efisien |
| [6] | `27_JITK_Lightweight_CNN_Rice_Leaf_Disease_2026.pdf` | Lightweight CNN |
| [7] | `33_HSET_Lightweight_MobileNetV2_Paddy_Disease_2023.pdf` | MobileNetV2 lightweight |
| [8] | `37_JAIC_CNN_Transfer_Learning_Rice_Leaf_Disease_2025.pdf` | CNN + transfer learning |
| [9] | `13_IJISEM_Rice_Leaf_Disease_CNN.pdf` | Implementasi CNN |
| [10] | `35_JIU_UPGRIS_MobileNetV2_TFLite_Rice_Disease_2024.pdf` | MobileNetV2 + TFLite |
| [11] | `10_IJACSA_Explainable_Deep_Transfer_Learning_Rice_2024.pdf` | Transfer learning + explainable AI |
| [12] | `11_IJETER_Rice_Leaf_Disease_Deep_Learning_2020.pdf` | Deep learning daun padi |
| [13] | `22_JAIC_Multi_Class_Rice_Leaf_Disease_2025.pdf` | Multikelas classification |
| [14] | `08_arXiv_Rice_Disease_Recognition_Transfer_Learning_2023.pdf` | Transfer learning pengenalan |

---

## 🎯 Cara Menggunakan Daftar Pustaka

### Untuk Ketika Draft Sudah Final:

1. **Gunakan `DAFTAR_PUSTAKA_IEEE.md`** sebagai template
2. **Ekstrak metadata lengkap** dari setiap PDF (author, title, DOI, pages)
3. **Format sesuai petunjuk universitas** (IEEE, APA, Chicago, dll)
4. **Paste ke document skripsi final** di bagian Daftar Pustaka

### Untuk Verifikasi Referensi:

1. **Periksa `REFERENSI_MAPPING.md`** untuk menemukan file PDF
2. **Buka file PDF** dari `references/journals/` folder
3. **Extract metadata lengkap** jika belum ada di mapping

### Untuk Menulis BAB II (Tinjauan Pustaka):

Gunakan pengelompokan topik dari `DAFTAR_PUSTAKA_IEEE.md`:
- **Refs [1], [2]**: Pengantar foundational (Frontiers articles)
- **Refs [3]-[6]**: Diskusi arsitektur model
- **Refs [7], [8]**: Transfer learning dan lightweight approaches
- **Refs [9]-[14]**: Implementation details dan applications

---

## ⚠️ Hal-Hal Penting

### Sebelum Finalisasi:

1. **Extract Full Citation Details**
   - Buka setiap PDF di folder `references/journals/`
   - Catat nama author, full title, volume, issue, page numbers
   - Cari DOI jika tersedia

2. **Tentukan Format Final**
   - Sesuaikan dengan panduan universitas (IEEE recommended untuk teknik)
   - Template IEEE sudah disediakan di `DAFTAR_PUSTAKA_IEEE.md`

3. **Verifikasi Urutan Citasi**
   - Pastikan referensi muncul sesuai urutan dalam teks
   - [1]–[8] muncul di BAB II Sec 2.20
   - [9]–[14] juga di BAB II Sec 2.20

4. **Konsistensikan Path File**
   - Jika mengubah struktur folder, update path di dokumen

### Untuk Quality Assurance:

- ✅ Semua 14 referensi sudah dipetakan ke file PDF yang ada
- ✅ Citation numbering konsisten dalam teks dan daftar pustaka
- ✅ Format placeholder sudah disiapkan (simple + IEEE)
- ✅ Metadata dasar (title, source, year) sudah tersedia

---

## 🔍 Struktur File Terkait

```
rice_leaf_disease_project/
├── draft_skripsi_bab_1_5.md          ← SUDAH DIPERBARUI
├── REFERENSI_MAPPING.md              ← FILE BARU (pemetaan detail)
├── DAFTAR_PUSTAKA_IEEE.md            ← FILE BARU (format IEEE lengkap)
├── PANDUAN_REFERENSI.md              ← FILE INI
└── references/
    ├── journals/                      ← Folder berisi 25 file PDF
    │   ├── 08_arXiv_Rice_Disease_Recognition_Transfer_Learning_2023.pdf
    │   ├── 10_IJACSA_Explainable_Deep_Transfer_Learning_Rice_2024.pdf
    │   ├── 11_IJETER_Rice_Leaf_Disease_Deep_Learning_2020.pdf
    │   ├── 13_IJISEM_Rice_Leaf_Disease_CNN.pdf
    │   ├── 18_Frontiers_Automatic_Diagnosis_Rice_Diseases_DL_2021.pdf
    │   ├── 22_JAIC_Multi_Class_Rice_Leaf_Disease_2025.pdf
    │   ├── 23_CoSIE_MobileNetV2_EfficientNetB0_Rice_Leaf_Disease_2026.pdf
    │   ├── 24_Mandiri_MobileNetV2_MobileNetV3_Rice_Leaf_Disease_2025.pdf
    │   ├── 25_TIERS_Enhanced_CAE_CNN_Rice_Leaf_Disease_2025.pdf
    │   ├── 27_JITK_Lightweight_CNN_Rice_Leaf_Disease_2026.pdf
    │   ├── 31_Frontiers_Image_Dataset_Config_Rice_Disease_CNN_2022.pdf
    │   ├── 33_HSET_Lightweight_MobileNetV2_Paddy_Disease_2023.pdf
    │   ├── 35_JIU_UPGRIS_MobileNetV2_TFLite_Rice_Disease_2024.pdf
    │   ├── 37_JAIC_CNN_Transfer_Learning_Rice_Leaf_Disease_2025.pdf
    │   └── [11 file lainnya - belum digunakan]
    ├── references_index.csv            ← Index lengkap 25 jurnal
    └── README.md                       ← Info referensi
```

---

## 📝 Next Steps (Rekomendasi)

1. **Untuk Draft Pengajuan:**
   - Gunakan format sederhana di `draft_skripsi_bab_1_5.md` (sudah update)
   - Atau gunakan template IEEE dari `DAFTAR_PUSTAKA_IEEE.md`

2. **Untuk Final Submission:**
   - Ekstrak metadata lengkap dari setiap PDF
   - Format sesuai requirement universitas
   - Lakukan proof-reading dan cross-check

3. **Optional - Untuk Enhanced Metadata:**
   - Pertimbangkan extract metadata otomatis dari PDF metadata
   - Create BibTeX file (.bib) untuk kemudahan management
   - Use citation management tool (Zotero, Mendeley, dll)

---

## 📞 FAQ

**Q: Bagaimana jika saya ingin menambah/mengurangi referensi?**  
A: Tersedia 25 file PDF di folder `references/journals/`. Hanya 14 yang dipakai sekarang. Bisa swap dengan file lain sesuai kebutuhan, lalu update pemetaan di `REFERENSI_MAPPING.md`.

**Q: Format apa yang harus saya gunakan untuk daftar pustaka final?**  
A: Tergantung requirement universitas. Template IEEE sudah disediakan di `DAFTAR_PUSTAKA_IEEE.md`. Sesuaikan sesuai kebutuhan.

**Q: Bagaimana cara ekstrak author dan DOI dari PDF?**  
A: Buka PDF → lihat halaman pertama (biasanya ada title page dengan author) → cari DOI (biasanya di footer atau abstract page).

**Q: Apakah perlu download/extract metadata dari PDF secara programmatic?**  
A: Optional. Untuk skripsi ini, ekstrak manual dari PDF sudah cukup. Jika ingin otomasi, bisa gunakan library Python seperti `PyPDF2` atau `pdf2json`.

---

**Last Updated:** 2026-06-08  
**Status:** ✅ Ready for thesis writing

