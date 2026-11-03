# Ringkasan Isi Paper CNN Kustom 6 Kelas Tanpa Tungro

## Judul

**Penerapan CNN Kustom dalam Klasifikasi Penyakit Daun Padi Menggunakan Citra Digital**

Alternatif judul bahasa Inggris:

**Application of a Custom CNN in Rice Leaf Disease Classification Using Digital Images**

---

## Abstract

Abstract ditulis dalam bahasa Inggris. Isinya menjelaskan tujuan penelitian, dataset yang digunakan, metode CNN Kustom, proses preprocessing dan augmentasi, serta hasil utama model. Karena paper ini berfokus pada CNN Kustom, abstract tidak perlu membahas MobileNetV2 atau imbalance Tungro secara luas.

Poin penting abstract:

- Rice leaf disease classification is important to support early disease identification.
- This study applies a custom Convolutional Neural Network (CNN) to classify rice leaf disease images.
- The dataset consists of 6 classes after excluding the Tungro class.
- Images are resized to 256 x 256 pixels and augmented using random flip and random rotation.
- The model is trained using Adam optimizer, sparse categorical crossentropy loss, early stopping, dropout, and L2 regularization.
- The best model is selected based on the lowest validation loss.
- The experimental result shows that the custom CNN is able to classify rice leaf disease images, although the performance still needs improvement due to visual similarity between classes and limited dataset variation.

---

## 1. Pendahuluan

Bagian pendahuluan menjelaskan latar belakang penelitian. Penyakit daun padi dapat memengaruhi kualitas dan hasil produksi padi. Identifikasi penyakit secara manual masih bergantung pada pengamatan manusia, sehingga dapat bersifat subjektif dan membutuhkan pengalaman. Oleh karena itu, pendekatan berbasis citra digital dan deep learning dapat digunakan untuk membantu proses klasifikasi penyakit daun padi.

Poin yang perlu ditulis:

- Padi merupakan tanaman pangan penting.
- Penyakit daun padi dapat menurunkan produktivitas.
- Identifikasi manual memiliki keterbatasan.
- Citra digital dapat digunakan sebagai data masukan untuk klasifikasi otomatis.
- CNN cocok untuk mengenali pola visual seperti warna, bercak, tekstur, dan bentuk kerusakan daun.
- Penelitian ini berfokus pada penerapan CNN Kustom untuk klasifikasi 6 kelas penyakit/daun padi tanpa kelas Tungro.

Rumusan masalah yang dapat digunakan:

- Bagaimana menerapkan CNN Kustom untuk klasifikasi penyakit daun padi berdasarkan citra digital?
- Bagaimana performa CNN Kustom dalam mengklasifikasikan 6 kelas citra daun padi?
- Faktor apa saja yang memengaruhi hasil klasifikasi model?

Tujuan penelitian:

- Membangun model CNN Kustom untuk klasifikasi citra daun padi.
- Melakukan preprocessing, augmentasi, pelatihan, dan evaluasi model.
- Menganalisis hasil performa model berdasarkan accuracy, precision, recall, F1-score, dan confusion matrix.

---

## 2. Metode Penelitian

Bagian metode menjelaskan tahapan eksperimen dari dataset sampai evaluasi model. Alur metode dapat dibuat sebagai diagram:

```text
Pengumpulan Dataset
        ↓
Preprocessing dan Augmentasi Data
        ↓
Pembagian Dataset
        ↓
Perancangan Arsitektur CNN Kustom
        ↓
Pelatihan Model
        ↓
Evaluasi Model
```

### 2.1 Pengumpulan Dataset

Dataset yang digunakan adalah citra daun padi. Pada eksperimen paper, kelas Tungro tidak digunakan sehingga dataset terdiri dari 6 kelas:

1. Bacterial Leaf Blight
2. Brown Spot
3. Healthy Rice Leaf
4. Leaf Blast
5. Leaf scald
6. Sheath Blight

Jumlah total data yang digunakan adalah **3.829 citra**.

Alasan kelas Tungro tidak digunakan:

- Jumlah data Tungro jauh lebih banyak dibandingkan kelas lain pada dataset original.
- Paper difokuskan pada eksperimen CNN Kustom 6 kelas agar distribusi data lebih seimbang.
- Pembahasan imbalance Tungro akan dibahas lebih luas pada laporan skripsi, bukan paper ini.

### 2.2 Preprocessing dan Augmentasi Data

Preprocessing dilakukan agar seluruh citra memiliki ukuran dan format yang seragam sebelum masuk ke model CNN.

Preprocessing yang digunakan:

- Resize citra menjadi **256 x 256 piksel**.
- Citra memiliki 3 channel warna RGB.
- Normalisasi piksel menggunakan `Rescaling(1./255)`.

Augmentasi data digunakan untuk membantu model mengenali variasi citra.

Augmentasi yang digunakan:

- `RandomFlip(horizontal_and_vertical)`
- `RandomRotation(0.2)`

Penjelasan:

- `RandomFlip` membantu model agar tidak terlalu bergantung pada arah daun.
- `RandomRotation(0.2)` memutar citra secara acak sehingga model lebih tahan terhadap variasi posisi daun.

### 2.3 Pembagian Dataset

Dataset dibagi menjadi data training, validation, dan testing.

Pembagian data pada notebook:

| Jenis Data | Jumlah Citra |
|---|---:|
| Training | 2.681 |
| Validation | 572 |
| Testing | 576 |
| Total | 3.829 |

Fungsi masing-masing data:

- Training digunakan untuk melatih model.
- Validation digunakan untuk memantau performa model selama training dan memilih model terbaik.
- Testing digunakan untuk evaluasi akhir setelah training selesai.

### 2.4 Perancangan Arsitektur CNN Kustom

Model yang digunakan adalah CNN Kustom. CNN dirancang untuk mengekstraksi fitur visual dari citra daun padi, seperti bercak, warna daun, pola tekstur, dan area kerusakan.

Arsitektur utama:

| Urutan | Layer | Fungsi |
|---:|---|---|
| 1 | Input 256 x 256 x 3 | Menerima citra RGB |
| 2 | Rescaling 1./255 | Normalisasi nilai piksel |
| 3 | RandomFlip + RandomRotation | Augmentasi data |
| 4 | Conv2D 16 + MaxPooling | Ekstraksi fitur awal |
| 5 | Dropout 0.10 | Mengurangi overfitting |
| 6 | Conv2D 32 + MaxPooling | Ekstraksi fitur menengah |
| 7 | Dropout 0.15 | Mengurangi overfitting |
| 8 | Conv2D 64 + MaxPooling | Ekstraksi fitur lebih kompleks |
| 9 | Dropout 0.20 | Mengurangi overfitting |
| 10 | Flatten | Mengubah feature map menjadi vektor |
| 11 | Dense 128 | Menggabungkan fitur untuk klasifikasi |
| 12 | Dropout 0.50 | Mengurangi overfitting pada fully connected layer |
| 13 | Dense 6 neuron | Output untuk 6 kelas |

Catatan penting:

- Output model berupa **logits**, bukan softmax eksplisit.
- Karena output berupa logits, loss yang digunakan adalah `SparseCategoricalCrossentropy(from_logits=True)`.
- Model menggunakan regularisasi L2 sebesar `1e-4` untuk menekan overfitting.

### 2.5 Pelatihan Model

Pelatihan model dilakukan dengan konfigurasi:

| Hyperparameter | Nilai |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.0003 |
| Batch size | 32 |
| Epoch maksimum | 100 |
| Loss function | SparseCategoricalCrossentropy(from_logits=True) |
| Input size | 256 x 256 x 3 |
| Jumlah kelas | 6 kelas |
| Output | Logits |
| Regularisasi | L2 regularization 1e-4 |
| Dropout | 0.10, 0.15, 0.20, 0.50 |
| Model terbaik | Berdasarkan validation loss terendah |
| Callback | ModelCheckpoint dan EarlyStopping |

Model dilatih sampai maksimum 100 epoch, tetapi proses training dapat berhenti lebih cepat jika validation loss tidak lagi membaik. Model terbaik dipilih berdasarkan nilai validation loss paling rendah.

### 2.6 Evaluasi Model

Evaluasi dilakukan menggunakan data testing yang tidak digunakan saat training.

Metrik evaluasi:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Confusion matrix digunakan untuk melihat kelas mana yang sering benar diklasifikasikan dan kelas mana yang sering tertukar.

---

## 3. Hasil dan Pembahasan

### 3.1 Hasil Pelatihan Model

Berdasarkan hasil training pada notebook:

| Komponen | Hasil |
|---|---:|
| Epoch maksimum | 100 |
| Epoch yang berjalan | 94 |
| Epoch terbaik berdasarkan validation loss | 84 |
| Best validation loss | 0.9184 |
| Best validation accuracy | 0.6871 |

Hasil ini menunjukkan bahwa model masih mengalami keterbatasan dalam mempelajari pola visual antar kelas. Validation loss berhasil turun selama training, tetapi nilainya masih relatif tinggi jika dibandingkan dengan eksperimen CNN Kustom pada dataset 7 kelas skripsi.

### 3.2 Hasil Evaluasi Model

Hasil evaluasi pada data testing:

| Metrik | Nilai |
|---|---:|
| Test accuracy | 0.6684 |
| Precision macro | Sesuaikan dari classification report |
| Recall macro | Sesuaikan dari classification report |
| F1-score macro | Sesuaikan dari classification report |

Interpretasi:

- Model mampu melakukan klasifikasi, tetapi performanya belum sangat tinggi.
- Nilai accuracy sekitar 66% menunjukkan bahwa model masih sering keliru membedakan beberapa kelas.
- Hasil ini masih dapat diterima sebagai eksperimen awal CNN Kustom, tetapi perlu dianalisis penyebab keterbatasannya.

### 3.3 Analisis Confusion Matrix

Confusion matrix digunakan untuk melihat kesalahan klasifikasi model. Pada bagian ini, bahas kelas-kelas yang sering tertukar.

Poin pembahasan:

- Beberapa penyakit daun padi memiliki pola visual yang mirip.
- Warna bercak dan tekstur daun pada beberapa kelas dapat saling menyerupai.
- Model CNN Kustom sederhana mungkin belum cukup kuat untuk membedakan fitur halus antar penyakit.
- Kelas dengan variasi citra yang tinggi cenderung lebih sulit diklasifikasikan.

Contoh kalimat:

> Berdasarkan confusion matrix, kesalahan klasifikasi terjadi karena beberapa kelas penyakit memiliki kemiripan visual, terutama pada pola bercak dan perubahan warna daun. Kondisi ini menyebabkan model sulit membedakan batas karakteristik antar kelas secara konsisten.

### 3.4 Pembahasan Penyebab Rendahnya Accuracy

Beberapa faktor yang dapat menyebabkan accuracy tidak terlalu tinggi:

1. Kemiripan visual antar penyakit

   Beberapa penyakit daun padi memiliki gejala visual yang mirip, seperti bercak cokelat, area menguning, atau perubahan tekstur daun.

2. Dataset relatif terbatas

   Total dataset 3.829 citra terbagi ke 6 kelas. Untuk deep learning, jumlah ini masih tergolong terbatas, terutama jika variasi citra tinggi.

3. Variasi kualitas citra

   Perbedaan pencahayaan, sudut pengambilan gambar, latar belakang, dan kondisi daun dapat memengaruhi model.

4. Arsitektur CNN masih sederhana

   CNN Kustom yang digunakan hanya terdiri dari 3 blok konvolusi. Model ini cukup ringan, tetapi mungkin belum mampu menangkap fitur visual yang sangat kompleks.

5. Regularisasi membuat model lebih hati-hati

   Dropout dan L2 regularization membantu mengurangi overfitting, tetapi jika terlalu kuat dapat membuat model belajar lebih lambat dan menghasilkan accuracy yang lebih rendah.

6. Tidak menggunakan transfer learning

   Paper ini hanya membahas CNN Kustom, sehingga model belajar dari awal tanpa memanfaatkan bobot awal dari model besar seperti MobileNetV2.

Kalimat pembahasan:

> Rendahnya accuracy pada eksperimen ini tidak hanya disebabkan oleh jumlah data, tetapi juga oleh kemiripan visual antar kelas, variasi citra, serta keterbatasan arsitektur CNN Kustom yang digunakan. Selain itu, penggunaan dropout dan regularisasi L2 membuat model lebih stabil terhadap overfitting, tetapi dapat memperlambat proses pembelajaran fitur.

### 3.5 Perbandingan dengan Penelitian Terdahulu

Bagian ini membandingkan hasil penelitian dengan penelitian sebelumnya.

Poin yang bisa dibahas:

- Penelitian terdahulu sering memperoleh accuracy lebih tinggi karena menggunakan transfer learning atau arsitektur CNN yang lebih dalam.
- Beberapa penelitian menggunakan dataset yang lebih besar atau lebih bersih.
- Beberapa penelitian hanya menggunakan kelas penyakit tertentu sehingga tingkat kesulitan klasifikasi berbeda.
- Penelitian ini berkontribusi dengan menunjukkan penerapan CNN Kustom sederhana pada dataset 6 kelas tanpa Tungro.

Contoh isi tabel:

| Peneliti | Dataset | Metode | Jumlah Kelas | Hasil |
|---|---|---|---:|---|
| Penelitian A | Rice leaf disease | CNN | 3 kelas | Accuracy tinggi |
| Penelitian B | Rice disease image | MobileNetV2 | 4 kelas | Accuracy tinggi |
| Penelitian ini | Rice Leaf AUG tanpa Tungro | CNN Kustom | 6 kelas | Accuracy 0.6684 |

Kalimat pembahasan:

> Dibandingkan dengan penelitian terdahulu yang menggunakan transfer learning atau arsitektur lebih kompleks, hasil penelitian ini lebih rendah. Namun, penelitian ini tetap menunjukkan bahwa CNN Kustom dapat digunakan sebagai pendekatan dasar untuk klasifikasi penyakit daun padi, meskipun masih membutuhkan pengembangan lebih lanjut.

---

## 4. Kesimpulan

Kesimpulan berisi ringkasan hasil penelitian.

Poin kesimpulan:

- CNN Kustom berhasil diterapkan untuk klasifikasi 6 kelas citra daun padi tanpa kelas Tungro.
- Model menggunakan preprocessing resize, normalisasi piksel, augmentasi, dropout, L2 regularization, EarlyStopping, dan checkpoint.
- Model terbaik diperoleh pada epoch 84 berdasarkan validation loss terendah.
- Hasil test accuracy sebesar 0.6684 menunjukkan bahwa model mampu mengenali pola penyakit, tetapi performanya masih terbatas.
- Faktor yang memengaruhi hasil antara lain kemiripan visual antar kelas, keterbatasan jumlah data, variasi kualitas citra, dan arsitektur CNN yang masih sederhana.
- Penelitian selanjutnya dapat meningkatkan performa dengan menambah data, memperbaiki kualitas dataset, menggunakan arsitektur CNN yang lebih kuat, atau membandingkan dengan transfer learning.

Contoh paragraf kesimpulan:

> Berdasarkan hasil penelitian, CNN Kustom dapat diterapkan untuk klasifikasi penyakit daun padi pada dataset 6 kelas tanpa Tungro. Model terbaik diperoleh pada epoch 84 dengan validation loss terendah sebesar 0.9184 dan test accuracy sebesar 0.6684. Hasil tersebut menunjukkan bahwa model mampu mempelajari pola visual penyakit daun padi, tetapi performanya masih belum optimal. Keterbatasan ini dipengaruhi oleh kemiripan visual antar kelas, jumlah data yang terbatas, variasi kualitas citra, serta arsitektur CNN yang sederhana. Oleh karena itu, pengembangan lebih lanjut dapat dilakukan melalui peningkatan kualitas dataset, penambahan jumlah data, serta penggunaan metode pembanding seperti transfer learning.

---

## 5. Daftar Pustaka

Daftar pustaka berisi jurnal atau paper yang mendukung:

- Penyakit daun padi
- Klasifikasi citra digital
- Convolutional Neural Network
- Deep learning untuk klasifikasi penyakit tanaman
- Augmentasi data
- Evaluasi model klasifikasi

Referensi yang dapat digunakan:

- Paper tentang CNN untuk klasifikasi penyakit daun padi.
- Paper tentang deep learning untuk plant disease detection.
- Paper tentang penggunaan citra digital pada bidang pertanian.
- Paper tentang augmentasi data dan regularisasi pada CNN.

