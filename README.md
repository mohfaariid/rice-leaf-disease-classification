# Source Code Tugas Akhir

Nama: Mohammad Farid Anshori  
NIM: 22523124  
Program Studi: Informatika  
Judul: Analisis Perbandingan Performa CNN Kustom dan MobileNetV2 pada Klasifikasi Penyakit Daun Padi Menggunakan Dataset Tidak Seimbang dan Seimbang

## 1. Deskripsi Singkat

Folder ini berisi source code final yang digunakan dalam penelitian tugas akhir. Kode ini digunakan untuk membangun dataset balanced quality-hybrid, melatih model CNN Kustom dan MobileNetV2, menjalankan percobaan ablasi fine-tuning MobileNetV2, mengevaluasi model, serta melakukan pengujian menggunakan foto lapangan.

Kode yang disertakan hanya kode yang berkaitan dengan penelitian skripsi final tujuh kelas. File percobaan lain yang tidak digunakan dalam laporan akhir tidak disertakan agar struktur folder lebih ringkas.

Dataset, model hasil pelatihan, dan seluruh hasil eksperimen tidak disertakan penuh di dalam folder ini agar ukuran folder tetap ringan. Folder `data`, `outputs`, dan `reports` disediakan sebagai tempat penyimpanan jika kode akan dijalankan ulang.

## 2. Kelas Dataset

Penelitian ini menggunakan tujuh kelas citra daun padi:

```text
1. Bacterial Leaf Blight
2. Brown Spot
3. Healthy Rice Leaf
4. Leaf Blast
5. Leaf Scald
6. Sheath Blight
7. Tungro
```

Terdapat dua skenario dataset:

```text
1. Dataset original
   Dataset hasil penggabungan dari dua sumber dataset publik.
   Total data: 5.137 citra.

2. Dataset balanced quality-hybrid
   Dataset yang dibuat dengan quality-based filtering pada kelas Tungro
   dan augmentasi pada kelas minoritas.
   Total data: 4.571 citra, dengan 653 citra pada setiap kelas.
```

## 3. Struktur Folder

```text
source_code_final/
|-- configuration/        Konfigurasi path dan parameter dasar
|-- data/                 Tempat meletakkan dataset
|-- evaluation/           Kode evaluasi model, metrik, grafik, dan confusion matrix
|-- inference/            Kode prediksi citra
|-- models/               Arsitektur CNN Kustom dan MobileNetV2
|-- outputs/              Tempat menyimpan model hasil pelatihan
|-- preprocessing/        Kode pemuatan dataset, split data, dan transformasi
|-- reports/              Tempat menyimpan hasil evaluasi dan visualisasi
|-- scripts/              Script utama untuk menjalankan eksperimen
|-- training/             Kode pelatihan model dan callback
|-- main.py               Entry point pelatihan dasar
|-- main_balanced_tungro.py
|-- requirements.txt
`-- README.md
```

## 4. Kebutuhan Sistem dan Library

Kode dijalankan menggunakan Python dan TensorFlow. Library yang dibutuhkan tercantum pada file `requirements.txt`.

Instalasi library:

```bash
pip install -r requirements.txt
```

Isi utama `requirements.txt`:

```text
tensorflow==2.10.1
numpy==1.23.5
pandas
matplotlib
scikit-learn
pyyaml
```

## 5. Persiapan Dataset

Letakkan dataset pada folder `data` dengan struktur folder berdasarkan nama kelas. Contoh struktur:

```text
data/
|-- rice_leaf_7_original/
|   |-- Bacterial Leaf Blight/
|   |-- Brown Spot/
|   |-- Healthy Rice Leaf/
|   |-- Leaf Blast/
|   |-- Leaf Scald/
|   |-- Sheath Blight/
|   `-- Tungro/
`-- rice_leaf_aug_7_balanced_quality_hybrid_653/
    |-- Bacterial Leaf Blight/
    |-- Brown Spot/
    |-- Healthy Rice Leaf/
    |-- Leaf Blast/
    |-- Leaf Scald/
    |-- Sheath Blight/
    `-- Tungro/
```

Jika dataset balanced quality-hybrid belum tersedia, dataset tersebut dapat dibuat menggunakan script `scripts/build_quality_hybrid_balanced_dataset.py`.

## 6. File Model

Arsitektur model berada pada folder `models`.

```text
models/custom_cnn.py
```

File tersebut berisi arsitektur CNN Kustom yang digunakan sebagai model baseline. CNN Kustom dilatih dari awal tanpa bobot pretrained.

```text
models/mobilenetv2.py
```

File tersebut berisi arsitektur MobileNetV2 berbasis transfer learning. MobileNetV2 menggunakan bobot pretrained ImageNet, bagian klasifikasi bawaan tidak digunakan, kemudian ditambahkan layer klasifikasi baru untuk tujuh kelas daun padi.

## 7. Script Utama

### 7.1 Membuat Dataset Balanced Quality-Hybrid

Script:

```text
scripts/build_quality_hybrid_balanced_dataset.py
```

Fungsi:

```text
Membuat dataset balanced quality-hybrid dengan cara:
1. Menghitung skor kualitas citra.
2. Mengurangi citra kelas Tungro berdasarkan quality-based filtering.
3. Menambahkan citra pada kelas minoritas menggunakan augmentasi.
4. Membuat jumlah akhir setiap kelas menjadi 653 citra.
```

Cara menjalankan:

```bash
python scripts/build_quality_hybrid_balanced_dataset.py
```

### 7.2 Melatih CNN Kustom dengan Dropout

Script:

```text
scripts/run_cnn_dropout_original_balanced.py
```

Fungsi:

```text
Melatih dan mengevaluasi CNN Kustom dengan Dropout pada dua skenario dataset:
1. Dataset original.
2. Dataset balanced quality-hybrid.
```

Cara menjalankan:

```bash
python scripts/run_cnn_dropout_original_balanced.py
```

### 7.3 Melatih MobileNetV2 dengan Percobaan Ablasi Fine-tuning

Script:

```text
scripts/run_mobilenetv2_finetune_ablation.py
```

Fungsi:

```text
Menjalankan percobaan MobileNetV2 dengan beberapa skenario jumlah layer akhir
yang dibuka pada tahap fine-tuning, yaitu 0, 10, 20, 30, dan 50 layer.
```

Cara menjalankan:

```bash
python scripts/run_mobilenetv2_finetune_ablation.py
```

Contoh menjalankan hanya konfigurasi 50 layer:

```bash
python scripts/run_mobilenetv2_finetune_ablation.py --layers 50
```

### 7.4 Pengujian Foto Lapangan

Script:

```text
scripts/run_field_photo_inference.py
```

Fungsi:

```text
Melakukan prediksi terhadap foto daun padi yang diambil langsung dari lapangan
menggunakan dua model final:
1. CNN Kustom dengan Dropout.
2. MobileNetV2 fine-tuning 50 layer.
```

Cara menjalankan:

```bash
python scripts/run_field_photo_inference.py
```

Catatan:

```text
Pengujian foto lapangan bersifat kualitatif karena foto tidak memiliki label
ground truth yang terverifikasi.
```

## 8. Urutan Menjalankan Eksperimen

Urutan menjalankan eksperimen dari awal:

```text
1. Siapkan dataset original pada folder data.
2. Jalankan script pembuatan dataset balanced quality-hybrid.
3. Jalankan training CNN Kustom dengan Dropout.
4. Jalankan training MobileNetV2 dan percobaan ablasi fine-tuning.
5. Jalankan evaluasi dan periksa hasil pada folder reports.
6. Jalankan pengujian foto lapangan jika model final sudah tersedia.
```

Perintah ringkas:

```bash
python scripts/build_quality_hybrid_balanced_dataset.py
python scripts/run_cnn_dropout_original_balanced.py
python scripts/run_mobilenetv2_finetune_ablation.py
python scripts/run_field_photo_inference.py
```

## 9. Hasil Keluaran

Model hasil pelatihan disimpan pada folder:

```text
outputs/
```

Hasil evaluasi dan visualisasi disimpan pada folder:

```text
reports/
```

Contoh hasil keluaran:

```text
classification_report_custom_cnn_dropout_original.txt
classification_report_custom_cnn_balanced_quality_hybrid_653.txt
classification_report_mobilenetv2_original_ablation_50_layers.txt
classification_report_mobilenetv2_balanced_quality_hybrid_653_ablation_50_layers.txt
prediksi_foto_lapangan_2_model.csv
hasil_prediksi_foto_lapangan_2_model.png
```

## 10. Ringkasan Hasil Akhir Penelitian

Model terbaik pada penelitian ini adalah MobileNetV2 fine-tuning 50 layer pada dataset balanced quality-hybrid.

Hasil evaluasi model terbaik:

```text
Accuracy       : 92,26%
Recall macro   : 92,21%
F1-score macro : 92,17%
```

MobileNetV2 fine-tuning 50 layer dipilih karena memperoleh performa evaluasi terbaik dibandingkan CNN Kustom dan skenario MobileNetV2 lainnya. Selain itu, MobileNetV2 memiliki jumlah parameter dan ukuran model yang lebih kecil dibandingkan CNN Kustom.

## 11. Catatan Tambahan

1. Jalankan perintah dari folder `source_code_final`.
2. Folder ini hanya memuat source code yang digunakan pada skripsi final tujuh kelas.
3. Dataset dan model hasil pelatihan perlu diletakkan secara terpisah jika kode ingin dijalankan ulang.
4. File `.ps1` pada folder `scripts` merupakan script bantu PowerShell untuk Windows.
5. Pelatihan model dapat memerlukan waktu cukup lama, terutama pada percobaan ablasi MobileNetV2.
