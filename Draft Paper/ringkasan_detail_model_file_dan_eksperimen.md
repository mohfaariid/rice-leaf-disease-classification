# Ringkasan Detail Model, File, dan Eksperimen

Dokumen ini merangkum posisi file, model, dataset, konfigurasi, dan hasil eksperimen pada project klasifikasi penyakit daun padi.

## 1. Fokus Penelitian

Project ini memiliki dua fokus besar:

1. **Paper**
   - Fokus pada **CNN Kustom**.
   - Menggunakan **6 kelas tanpa Tungro**.
   - Tujuannya agar pembahasan lebih sederhana dan dataset lebih seimbang.

2. **Skripsi**
   - Membandingkan **CNN Kustom** dan **MobileNetV2**.
   - Menggunakan dataset **7 kelas original** dan dataset **7 kelas balanced Tungro**.
   - Tujuannya untuk menganalisis pengaruh imbalance dataset, terutama karena kelas Tungro memiliki jumlah data lebih banyak.

## 2. Dataset

### 2.1 Dataset Original 7 Kelas

Dataset original terdiri dari tujuh kelas:

| No | Kelas | Jumlah Citra |
|---:|---|---:|
| 1 | Bacterial Leaf Blight | 636 |
| 2 | Brown Spot | 646 |
| 3 | Healthy Rice Leaf | 653 |
| 4 | Leaf Blast | 634 |
| 5 | Leaf Scald | 628 |
| 6 | Sheath Blight | 632 |
| 7 | Tungro | 1308 |
| | **Total** | **5137** |

File distribusi:

`rice_leaf_disease_project/reports/distribusi_dataset_original.csv`

### 2.2 Dataset Balanced 7 Kelas

Dataset balanced tetap terdiri dari tujuh kelas, tetapi kelas Tungro dikurangi menjadi sekitar 600-an gambar.

| No | Kelas | Jumlah Citra |
|---:|---|---:|
| 1 | Bacterial Leaf Blight | 636 |
| 2 | Brown Spot | 646 |
| 3 | Healthy Rice Leaf | 653 |
| 4 | Leaf Blast | 634 |
| 5 | Leaf Scald | 628 |
| 6 | Sheath Blight | 632 |
| 7 | Tungro | 636 |
| | **Total** | **4465** |

File distribusi:

`rice_leaf_disease_project/reports/distribusi_dataset_balanced_tungro_636.csv`

### 2.3 Dataset 6 Kelas Tanpa Tungro

Dataset ini digunakan untuk paper CNN Kustom.

| No | Kelas | Jumlah Citra |
|---:|---|---:|
| 1 | Bacterial Leaf Blight | 636 |
| 2 | Brown Spot | 646 |
| 3 | Healthy Rice Leaf | 653 |
| 4 | Leaf Blast | 634 |
| 5 | Leaf Scald | 628 |
| 6 | Sheath Blight | 632 |
| | **Total** | **3829** |

Path dataset:

`rice_leaf_disease_project/data/rice_leaf_aug_6_no_tungro`

Kelas **Tungro** tidak digunakan karena jumlah datanya jauh lebih banyak dibandingkan kelas lain pada dataset original.

## 3. Perbedaan CNN Kustom dan CNN Kustom Stable

### 3.1 CNN Kustom Biasa

CNN Kustom biasa adalah model baseline yang lebih sederhana.

Arsitektur:

```text
Input
Rescaling
Augmentasi
Conv2D 16
MaxPooling
Conv2D 32
MaxPooling
Conv2D 64
MaxPooling
Flatten
Dense 128
Output
```

File utama:

`rice_leaf_disease_project/models/custom_cnn.py`

File pipeline 6 kelas:

`rice_leaf_disease_project/main_custom_cnn_6_classes.py`

`rice_leaf_disease_project/training/train_custom_cnn_6_classes.py`

### 3.2 CNN Kustom Stable

CNN Kustom Stable adalah versi eksperimen yang pernah dibuat untuk mencoba membuat training lebih stabil dan mengurangi overfitting.

Arsitektur awal stable:

```text
Input
Rescaling
Augmentasi
Conv2D 32
MaxPooling
Dropout 0.10
Conv2D 64
MaxPooling
Dropout 0.15
Conv2D 128
MaxPooling
Dropout 0.20
Conv2D 128
MaxPooling
Flatten
Dense 256
Dropout 0.50
Output
```

Perbedaan utama:

| Aspek | CNN Kustom Biasa | CNN Kustom Stable |
|---|---|---|
| Tujuan | Baseline/model utama | Eksperimen tuning |
| Jumlah Conv2D | 3 layer | 4 layer |
| Filter | 16, 32, 64 | 32, 64, 128, 128 |
| Dense | 128 neuron | 256 neuron |
| Dropout | Tidak ada | Ada |
| Augmentasi | Flip dan rotation | Flip, rotation, zoom |
| Kompleksitas | Lebih sederhana | Lebih kompleks |

## 4. Notebook yang Sedang Dipakai untuk Paper

Notebook yang sekarang menjadi perhatian:

`rice_leaf_disease_project/notebooks/custom_cnn_6kelas_stable_experiment.ipynb`

Catatan penting:

- Nama filenya masih mengandung kata `stable`.
- Namun isi notebook sudah disesuaikan agar mengikuti **CNN Kustom biasa/revised**.
- Jadi untuk paper, model tetap sebaiknya disebut **CNN Kustom**, bukan CNN Kustom Stable.

Backup sebelum perubahan tuning:

`rice_leaf_disease_project/notebooks/custom_cnn_6kelas_stable_experiment.backup_before_revised_config.ipynb`

`rice_leaf_disease_project/notebooks/custom_cnn_6kelas_stable_experiment.backup_before_val_loss_tuning.ipynb`

## 5. Konfigurasi Notebook CNN Kustom 6 Kelas Saat Ini

Notebook:

`rice_leaf_disease_project/notebooks/custom_cnn_6kelas_stable_experiment.ipynb`

Konfigurasi saat ini:

| Komponen | Nilai |
|---|---|
| Eksperimen | `custom_cnn_revised_6kelas_tanpa_tungro_val_loss_tuned` |
| Dataset | 6 kelas tanpa Tungro |
| Input citra | 256 x 256 x 3 |
| Batch size | 32 |
| Epoch maksimum | 30 |
| Optimizer | Adam |
| Learning rate | 0.0003 |
| Loss function | SparseCategoricalCrossentropy(from_logits=True) |
| Output | Logits |
| Jumlah kelas | 6 kelas |
| Monitor model terbaik | Validation loss |

Arsitektur model:

```text
Input 256 x 256 x 3
        |
Rescaling 1/255
        |
RandomFlip(horizontal_and_vertical)
        |
RandomRotation(0.2)
        |
Conv2D 16, kernel 3x3, ReLU
        |
MaxPooling2D
        |
Conv2D 32, kernel 3x3, ReLU
        |
MaxPooling2D
        |
Conv2D 64, kernel 3x3, ReLU
        |
MaxPooling2D
        |
Flatten
        |
Dense 128, ReLU
        |
Dense 6 Output
```

Output layer tidak memakai softmax eksplisit. Karena itu, loss menggunakan:

```text
SparseCategoricalCrossentropy(from_logits=True)
```

## 6. Penyesuaian agar Validation Loss Lebih Rendah

Notebook sudah disesuaikan agar training lebih fokus menurunkan `val_loss`.

Perubahan yang dilakukan:

| Bagian | Sebelumnya | Sekarang |
|---|---|---|
| Learning rate | 0.001 | 0.0003 |
| ModelCheckpoint | Monitor `val_accuracy` | Monitor `val_loss` |
| Mode checkpoint | Max | Min |
| EarlyStopping | Tidak aktif | Aktif, monitor `val_loss` |
| ReduceLROnPlateau | Tidak aktif | Aktif, monitor `val_loss` |
| Evaluasi akhir | Bisa memakai model epoch terakhir | Memakai model terbaik berdasarkan `val_loss` |

Tujuan perubahan ini:

- Membuat training lebih stabil.
- Mengurangi lonjakan validation loss.
- Menyimpan model terbaik berdasarkan validation loss paling rendah.
- Menghindari penggunaan model epoch terakhir jika epoch terakhir tidak memiliki performa validasi terbaik.

Model terbaik akan disimpan ke:

`rice_leaf_disease_project/outputs/custom_cnn_revised_6kelas_tanpa_tungro_val_loss_tuned_best_val_loss.keras`

Ringkasan hasil akan disimpan ke:

`rice_leaf_disease_project/reports/ringkasan_custom_cnn_revised_6kelas_tanpa_tungro_val_loss_tuned.csv`

Classification report akan disimpan ke:

`rice_leaf_disease_project/reports/classification_report_custom_cnn_revised_6kelas_tanpa_tungro_val_loss_tuned.txt`

Confusion matrix akan disimpan ke:

`rice_leaf_disease_project/reports/confusion_matrix_custom_cnn_revised_6kelas_tanpa_tungro_val_loss_tuned.png`

## 7. File Notebook Revised 6 Kelas

File lain yang juga membahas CNN Kustom 6 kelas:

`rice_leaf_disease_project/notebooks/rice_disease_cnn_revised_6kelas_tanpa_tungro.ipynb`

Konfigurasi notebook tersebut:

| Komponen | Nilai |
|---|---|
| Dataset | 6 kelas tanpa Tungro |
| Input citra | 256 x 256 x 3 |
| Batch size | 32 |
| Epoch | 20 |
| Optimizer | Adam |
| Learning rate | Default Adam, 0.001 |
| Augmentasi | RandomFlip dan RandomRotation(0.2) |
| Arsitektur | Conv2D 16, Conv2D 32, Conv2D 64, Dense 128 |
| Output | 6 neuron logits |
| Loss | SparseCategoricalCrossentropy(from_logits=True) |

Draft metode untuk notebook ini:

`rice_leaf_disease_project/Draft Paper/metode_rice_disease_cnn_revised_6kelas_tanpa_tungro.md`

## 8. Perbedaan Notebook Revised dan Notebook Tuning Saat Ini

| Aspek | `rice_disease_cnn_revised_6kelas_tanpa_tungro.ipynb` | `custom_cnn_6kelas_stable_experiment.ipynb` saat ini |
|---|---|---|
| Nama model | CNN Kustom revised | CNN Kustom revised val-loss tuned |
| Dataset | 6 kelas tanpa Tungro | 6 kelas tanpa Tungro |
| Input | 256 x 256 x 3 | 256 x 256 x 3 |
| Conv2D | 16, 32, 64 | 16, 32, 64 |
| Dense | 128 | 128 |
| Dropout | Tidak ada | Tidak ada |
| Epoch | 20 | 30 maksimum |
| Learning rate | 0.001 default Adam | 0.0003 |
| Checkpoint | Tidak fokus val_loss | Model terbaik berdasarkan val_loss |
| EarlyStopping | Tidak ada | Ada |
| ReduceLROnPlateau | Tidak ada | Ada |
| Tujuan | Eksperimen dasar | Menurunkan dan menstabilkan validation loss |

## 9. Hasil CNN Kustom 6 Kelas dari Notebook Revised

Berdasarkan output notebook `rice_disease_cnn_revised_6kelas_tanpa_tungro.ipynb`, hasil evaluasi yang pernah tercetak adalah:

```text
Test Loss     : 0.8881
Test Accuracy : 67.71%
```

Classification report yang tercetak menunjukkan:

| Metrik | Nilai |
|---|---:|
| Accuracy | 71% |
| Precision macro | 72% |
| Recall macro | 70% |
| F1-score macro | 71% |
| Precision weighted | 71% |
| Recall weighted | 71% |
| F1-score weighted | 71% |

Catatan:

Ada perbedaan antara nilai accuracy dari `model.evaluate()` dan `classification_report`. Sebelum hasil final dimasukkan ke paper, sebaiknya notebook dijalankan ulang dari kernel bersih agar angka evaluasi konsisten.

## 10. Hasil Eksperimen Skripsi 7 Kelas

### 10.1 Dataset Original

File hasil:

`rice_leaf_disease_project/reports/tabel_perbandingan_model.csv`

Hasil terbaru:

| Model | Accuracy | F1-score macro |
|---|---:|---:|
| MobileNetV2 | 81.77% | 78.49% |
| CNN Kustom | 79.69% | sekitar 79% |

Kesimpulan pada dataset original:

MobileNetV2 menjadi model terbaik.

### 10.2 Dataset Balanced Tungro

File hasil:

`rice_leaf_disease_project/reports/tabel_perbandingan_model_balanced_tungro_636.csv`

Hasil terbaru:

| Model | Accuracy | F1-score macro |
|---|---:|---:|
| MobileNetV2 balanced | 81.25% | 80.76% |
| CNN Kustom balanced | 74.85% | sekitar 74-78% |

Kesimpulan pada dataset balanced:

MobileNetV2 tetap menjadi model terbaik, tetapi penggunaan dataset balanced membuat evaluasi antar kelas lebih adil karena kelas Tungro tidak lagi mendominasi jumlah data.

## 11. Rekomendasi Penamaan untuk Paper

Untuk paper, gunakan istilah:

```text
CNN Kustom
```

Jangan gunakan istilah:

```text
CNN Kustom Stable
```

Alasannya:

- Arsitektur yang ingin dibahas adalah CNN Kustom sederhana/revised.
- Kata stable dapat membingungkan pembaca karena pernah ada versi stable yang arsitekturnya berbeda.
- Paper hanya fokus pada CNN Kustom 6 kelas tanpa Tungro.

Judul model yang disarankan di paper:

```text
CNN Kustom untuk Klasifikasi Enam Kelas Penyakit Daun Padi
```

atau:

```text
Model CNN Kustom Berbasis Citra untuk Klasifikasi Penyakit Daun Padi Tanpa Kelas Tungro
```

## 12. Rekomendasi Alur Kerja Selanjutnya

1. Jalankan ulang notebook:

   `rice_leaf_disease_project/notebooks/custom_cnn_6kelas_stable_experiment.ipynb`

2. Pilih kernel:

   `Rice Leaf Disease (.venv)`

3. Jalankan dari awal:

   `Restart Kernel and Run All`

4. Setelah selesai, gunakan file hasil berikut untuk paper:

   - `ringkasan_custom_cnn_revised_6kelas_tanpa_tungro_val_loss_tuned.csv`
   - `classification_report_custom_cnn_revised_6kelas_tanpa_tungro_val_loss_tuned.txt`
   - `confusion_matrix_custom_cnn_revised_6kelas_tanpa_tungro_val_loss_tuned.png`
   - `grafik_pelatihan_custom_cnn_revised_6kelas_tanpa_tungro_val_loss_tuned.png`

5. Masukkan hasil final ke draft paper setelah angka dari evaluation dan classification report konsisten.

