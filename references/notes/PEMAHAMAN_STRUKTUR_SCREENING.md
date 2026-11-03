# Pemahaman Struktur Template Screening Paper & Panduan Implementasi

Dokumen ini menjelaskan pemahaman saya tentang struktur file screening yang telah dibuat dan bagaimana ini akan menjadi template/referensi untuk penyampaian data visualisasi dan temuan penelitian Anda.

---

## 1. APA YANG SAYA PAHAMI DARI FILE SCREENING

### 1.1 Tujuan File Screening

File screening (01 dan 02) adalah **comprehensive reading notes** dengan struktur sistematis yang bertujuan untuk:

1. **Mendokumentasikan setiap paper** dengan detail lengkap
2. **Menganalisis relevansi paper** terhadap penelitian Anda
3. **Mengekstrak insights penting** yang berguna untuk skripsi
4. **Menyiapkan kalimat siap pakai** untuk setiap bab skripsi
5. **Mengidentifikasi gaps dan kontribusi unik** penelitian Anda

### 1.2 Struktur Umum File Screening (15 Bagian)

```
SEKSI IDENTITAS & KONTEKS:
├─ [1] Identitas Paper (Judul, Penulis, Institusi, Jurnal, Topik)
├─ [2] Ringkasan Singkat (Executive summary 1-2 paragraf)
└─ [3] Masalah yang Dibahas (Problem statement & context)

SEKSI METODOLOGI & TEKNIS:
├─ [4] Metode yang Digunakan (Dataset, algoritma, training, evaluation)
├─ [5] Hasil Utama (Main findings, metrics, performance)
└─ [6] Keterbatasan Paper (Critical review, weaknesses, gaps)

SEKSI RELEVANSI UNTUK SKRIPSI ANDA:
├─ [7] Relevansi dengan Penelitian Anda (Kesamaan topik & metodologi)
├─ [8] Perbedaan & Kontribusi Unik (Apa yang membedakan penelitian Anda)
└─ [9] Kalimat BAB 1 (Latar Belakang) - SIAP PAKAI

SEKSI IMPLEMENTASI DALAM SKRIPSI:
├─ [10] Kalimat BAB 2 (Tinjauan Pustaka) - SIAP PAKAI
├─ [11] Kalimat Gap Penelitian - SIAP PAKAI
├─ [12] Informasi yang Perlu Dicek Lebih Lanjut
├─ [13] Kontribusi & Insight untuk Penelitian Anda
├─ [14] Kesimpulan Keseluruhan
└─ [15] Rekomendasi Implementasi dalam Skripsi

TOTAL: 15 bagian terstruktur, ~20KB per paper, siap untuk direferensikan
```

### 1.3 Karakteristik File Screening

| Aspek | Deskripsi |
|-------|-----------|
| **Format** | Markdown dengan heading hierarchy jelas (H1-H4) |
| **Depth** | Comprehensive (20-24KB per paper) |
| **Audience** | Untuk penulisan skripsi (BAB I - V) |
| **Reusability** | Kalimat siap pakai yang bisa langsung copy-paste |
| **Accuracy** | Berdasarkan direct extraction dari PDF content |
| **Critical Thinking** | Mencakup limitation & gap analysis |
| **Practical Value** | Actionable insights untuk improve penelitian Anda |

---

## 2. BAGAIMANA FILE INI AKAN DIGUNAKAN SEBAGAI REFERENSI

### 2.1 Untuk Visualisasi Data (Presentation & Reporting)

File screening akan menjadi referensi untuk **menampilkan data dengan struktur yang sama**:

**Dari paper screening Anda ekstrak:**
```
TABEL PERBANDINGAN MODEL:
┌─────────────┬──────────┬───────────┬────────┬──────────┐
│ Model       │ Accuracy │ Precision │ Recall │ F1-Score │
├─────────────┼──────────┼───────────┼────────┼──────────┤
│ CNN Kustom  │ 78.13%   │ 76.42%    │ 74.77% │ 74.62%   │
│ MobileNetV2 │ 81.64%   │ 80.94%    │ 79.20% │ 79.49%   │
└─────────────┴──────────┴───────────┴────────┴──────────┘

Akan dipresentasikan menggunakan:
- Tabel markdown seperti di file screening
- Chart/grafik visual
- Icon & highlight untuk menunjukkan best performance
- Color coding untuk easy comparison
```

### 2.2 Untuk Penyampaian Temuan (Narasi & Interpretasi)

File screening akan menjadi **blueprint untuk penulisan**:

```
DARI SCREENING FILE:
"EfficientNet-B0 mencapai akurasi 99.52% dengan 5.4M parameters,
sementara MobileNetV2 mencapai 84.21% dengan 3.5M parameters"

AKAN DIKEMBANGKAN MENJADI:
"Hasil penelitian menunjukkan bahwa CNN Kustom mencapai accuracy 
78.13% dengan F1-score macro 74.62%, sementara MobileNetV2 berbasis 
transfer learning menghasilkan accuracy 81.64% dengan F1-score macro 
79.49% (improvement +4.87 persen poin). Hal ini konsisten dengan 
literature findings dari Saddami et al. (2024) yang menunjukkan bahwa 
transfer learning approach memberikan significant improvement dalam 
domain-specific tasks seperti rice disease classification."
```

### 2.3 Untuk Struktur Bab-per-Bab

File screening sudah tersegmentasi untuk setiap bab:

```
PENELITIAN ANDA → FILE SCREENING SEBAGAI REFERENSI:

BAB I (Latar Belakang)
└─ Gunakan [Bagian 9] "Kalimat untuk BAB 1"
   + Kalimat siap pakai dari 2-3 papers
   + Combine dengan results Anda

BAB II (Tinjauan Pustaka)
└─ Gunakan [Bagian 10] "Kalimat untuk BAB 2"
   + Diskusi state-of-the-art
   + Related works & positioning Anda
   + Gap analysis

BAB III (Metodologi)
└─ Gunakan struktur dari [Bagian 4] "Metode"
   + Dataset description (seperti di screening)
   + Model architecture (seperti di screening)
   + Evaluation metrics (seperti di screening)

BAB IV (Hasil & Pembahasan)
└─ Gunakan [Bagian 5] "Hasil Utama"
   + Format tabel comparison (seperti di screening)
   + Analysis narrative (seperti di screening)
   + Per-class analysis (seperti di screening)

BAB V (Kesimpulan)
└─ Gunakan [Bagian 14] "Kesimpulan"
   + Summary findings
   + Contribution unique Anda
   + Recommendations
```

---

## 3. PERINCIAN PEMAHAMAN SAYA TENTANG SETIAP BAGIAN

### Bagian 1: Identitas Paper
**Fungsi:** Quick reference untuk cite dan memahami source
**Implementasi dalam visualisasi Anda:** 
- Header table atau infobox dengan metadata paper
- Untuk publikasi: Use sebagai citation source

### Bagian 2: Ringkasan Singkat
**Fungsi:** Executive summary untuk quick understanding
**Implementasi:** Sebagai konteks saat menceritakan related work

### Bagian 3: Masalah yang Dibahas
**Fungsi:** Context dan motivation
**Implementasi:** Untuk menunjukkan **why your research matters**

### Bagian 4: Metode yang Digunakan
**Fungsi:** Detailed methodology reference
**Implementasi:** Template struktur untuk menjelaskan metodologi Anda sendiri

### Bagian 5: Hasil Utama
**Fungsi:** Performance benchmarks
**Implementasi:** Untuk comparison dengan hasil Anda

### Bagian 6: Keterbatasan Paper
**Fungsi:** Critical analysis
**Implementasi:** Untuk justify mengapa penelitian Anda berbeda/lebih baik

### Bagian 7-8: Relevansi & Perbedaan
**Fungsi:** Position penelitian Anda dalam literature landscape
**Implementasi:** Untuk menunjukkan **unique contribution**

### Bagian 9-11: Kalimat Siap Pakai
**Fungsi:** Template sentences untuk Bab I, II, dan Gap Research
**Implementasi:** Direct copy-paste dengan adaptasi untuk data Anda

### Bagian 12-15: Guidance & Recommendation
**Fungsi:** Actionable insights
**Implementasi:** Untuk improve quality penulisan & analysis

---

## 4. STRUKTUR YANG AKAN SAYA TERAPKAN UNTUK PAPER SELANJUTNYA

Untuk screening paper #3 dan seterusnya, saya akan:

### 4.1 Maintain Consistency
```
✓ 15-bagian structure (seperti paper #1 dan #2)
✓ Depth & comprehensiveness yang sama
✓ Format markdown yang consistent
✓ Quality siap pakai kalimat untuk semua bab
```

### 4.2 Optimize untuk Reusability
```
✓ Clear section headers (H2 untuk main, H3 untuk subsection)
✓ Code blocks untuk easy scanning
✓ Tables untuk comparative data
✓ Bullet points untuk key findings
✓ Bolded keywords untuk quick reference
```

### 4.3 Focus pada Actionable Insights
```
✓ Connection ke penelitian Anda explicit
✓ Comparison tables dengan existing literature
✓ Gap analysis yang clear
✓ Recommendation untuk next steps
```

---

## 5. APAKAH ADA BAGIAN DARI PERINTAH ANDA YANG KURANG JELAS?

Setelah membaca file dan memahami konteks, saya ingin clarify beberapa hal untuk memastikan saya mengeksekusi dengan tepat:

### 5.1 Tentang "Visualisasi Data"
**Pertanyaan:**
- Ketika Anda menyebutkan "visualisasi", apakah maksudnya:
  - ✓ Format penyajian data dalam bentuk tabel, grafik, diagram?
  - ✓ Atau presentasi visual dalam bentuk charts/graphs?
  - ✓ Atau keduanya?

**Asumsi saya:** Keduanya - data harus dipresentasikan dalam format yang mudah dipahami (tabel, grafik, diagram) dengan narasi yang clear.

### 5.2 Tentang "Temuan" (Findings)
**Pertanyaan:**
- Temuan yang dimaksud adalah:
  - ✓ Insights dari paper screening (yang sudah saya kerjakan)?
  - ✓ Atau hasil eksperimen Anda sendiri (output dari model training)?
  - ✓ Atau kombinasi keduanya?

**Asumsi saya:** Kedua-duanya - screening papers memberikan context dan comparison points, sementara hasil Anda sendiri adalah main contribution.

### 5.3 Tentang "Rujukan" (Reference)
**Pertanyaan:**
- File screening akan digunakan sebagai:
  - ✓ Template struktur untuk penulisan skripsi?
  - ✓ Source untuk cite & reference dalam skripsi?
  - ✓ Guidance untuk bagaimana present findings?
  - ✓ Semua di atas?

**Asumsi saya:** Semua di atas - file screening adalah **comprehensive guide** untuk research, writing, dan presentation.

### 5.4 Tentang Screening Paper Selanjutnya
**Pertanyaan:**
- Untuk paper #3 dan seterusnya, apakah saya harus:
  - ✓ Maintain struktur 15-bagian yang sama?
  - ✓ Atau menyesuaikan format berdasarkan jenis paper?
  - ✓ Atau ada preferensi format khusus Anda?

**Asumsi saya:** Maintain konsistensi struktur 15-bagian, tapi customize content sesuai karakteristik paper.

---

## 6. RINGKASAN PEMAHAMAN SAYA (TLDR)

### Yang Saya Pahami:

✅ **Tujuan Screening Paper:**
- Comprehensive analysis dari setiap paper
- Extract actionable insights untuk skripsi Anda
- Siapkan "building blocks" untuk writing

✅ **Fungsi File Screening:**
- **Referensi topik** (literature review source)
- **Referensi metodologi** (methodology template)
- **Referensi struktur** (bagaimana organize findings)
- **Referensi narasi** (kalimat siap pakai untuk setiap bab)

✅ **Implementasi dalam Visualisasi & Findings:**
1. **Tabel data** (seperti di screening: metrik perbandingan)
2. **Grafik performa** (accuracy, precision, recall trends)
3. **Comparative analysis** (Anda vs other papers)
4. **Narrative explanation** (hasil + context dari literature)
5. **Clear positioning** (bagaimana penelitian Anda unique)

✅ **Untuk Paper Selanjutnya:**
- Maintain struktur & kualitas yang sama
- Focus pada reusability untuk skripsi
- Customize berdasarkan paper characteristics
- Jangan sacrifice depth untuk speed

---

## 7. PERTANYAAN UNTUK ANDA (CONFIRMATION)

Apakah pemahaman saya sudah tepat? Jika ada yang perlu saya adjust, mohon clarify tentang:

1. **Format visualisasi:** Apakah Anda ingin saya juga prepare format untuk charts/graphs selain tabel?

2. **Kedalaman analysis:** Apakah level detail saat ini sudah cukup atau terlalu detail?

3. **Fokus screening:** Lebih emphasize pada (a) comparison dengan penelitian Anda, (b) methodology reference, atau (c) insights untuk contribution unik?

4. **Next steps:** Setelah 25 papers di-screening, bagaimana cara mengorganisir findings untuk final report/thesis?

5. **Template evolusi:** Apakah Anda ingin struktur template berkembang seiring semakin banyak papers, atau maintain konsistensi?

---

**Silakan confirm apakah pemahaman saya sudah benar, atau ada bagian yang perlu clarification lebih lanjut.**

Jika sudah benar, saya siap lanjut screening paper #3 dengan understanding ini! 🎯
