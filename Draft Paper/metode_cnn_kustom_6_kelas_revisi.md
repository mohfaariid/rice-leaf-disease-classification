# Metode CNN Kustom 6 Kelas Tanpa Tungro

## 2.2 Preprocessing Citra

Preprocessing citra dilakukan sebelum data dimasukkan ke dalam model CNN Kustom. Tahap ini mencakup penyesuaian ukuran gambar agar seluruh citra memiliki dimensi input yang seragam. Pada penelitian ini, seluruh citra diubah menjadi ukuran **128 x 128 piksel** dengan tiga kanal warna RGB, sehingga ukuran input model menjadi **128 x 128 x 3**.

Selain penyesuaian ukuran citra, normalisasi piksel dilakukan dengan mengubah nilai piksel dari rentang **0-255** menjadi **0-1** menggunakan layer `Rescaling(1.0 / 255)`. Normalisasi ini bertujuan untuk menyederhanakan skala data dan membantu proses pelatihan model menjadi lebih stabil.

Dataset yang digunakan pada penelitian ini terdiri dari **enam kelas citra daun padi**, yaitu Bacterial Leaf Blight, Brown Spot, Healthy Rice Leaf, Leaf Blast, Leaf Scald, dan Sheath Blight. Kelas Tungro tidak digunakan pada eksperimen ini karena jumlah datanya jauh lebih banyak dibandingkan kelas lain, sehingga penelitian difokuskan pada enam kelas yang memiliki distribusi data lebih seimbang.

## 2.3 Augmentasi Data

Augmentasi data diterapkan pada data training dengan teknik **RandomFlip**, **RandomRotation**, dan **RandomZoom**. Teknik ini digunakan untuk menambah variasi tampilan citra latih, sehingga model dapat mempelajari pola penyakit dari orientasi, posisi, dan skala daun yang berbeda.

Augmentasi pada penelitian ini diterapkan secara **real-time saat training** melalui layer augmentasi di dalam model. Dengan demikian, jumlah file citra asli pada folder dataset tetap sama, yaitu **3.829 gambar**, tetapi variasi citra yang dipelajari model bertambah selama proses pelatihan berlangsung.

Augmentasi digunakan sebagai strategi untuk membantu model mengenali variasi bentuk dan orientasi daun. Namun, performa akhir tetap dipengaruhi oleh kualitas citra asli, jumlah data, serta tingkat kemiripan ciri visual antar kategori penyakit. Oleh karena itu, augmentasi pada penelitian ini berperan sebagai upaya peningkatan variasi data latih, sedangkan keterbatasannya dibahas sebagai salah satu faktor yang memengaruhi hasil evaluasi model.

## 2.4 Pembagian Dataset

Setelah tahap preprocessing, dataset dibagi menjadi tiga bagian, yaitu data training, validation, dan testing. Data training digunakan untuk melatih model, data validation digunakan untuk memantau perkembangan model selama proses training, sedangkan data testing digunakan sebagai data uji akhir untuk menilai performa model pada citra yang tidak dilibatkan dalam pelatihan.

**Tabel 1. Pembagian Dataset**

| Jenis Data | Jumlah Citra |
|---|---:|
| Training | 2.680 |
| Validation | 574 |
| Testing | 575 |
| **Total** | **3.829** |

Pembagian dataset dilakukan agar proses pembelajaran, pemantauan, dan pengujian model menggunakan data yang berbeda. Dengan demikian, hasil pengujian akhir dapat memberikan gambaran yang lebih objektif terhadap kemampuan model dalam mengklasifikasikan citra baru.

## 2.5 Perancangan Arsitektur CNN Kustom

Arsitektur CNN Kustom dalam penelitian ini dirancang sebagai model klasifikasi citra yang dilatih dari awal tanpa memanfaatkan bobot pretrained. Model terdiri atas beberapa komponen utama, yaitu layer input, rescaling, augmentasi data, convolutional layer, pooling layer, dropout layer, flatten layer, dense layer, dan output layer.

Convolutional layer berfungsi untuk memperoleh ciri visual dari citra daun padi, seperti warna, tekstur, tepi daun, dan bentuk bercak. Pooling layer digunakan untuk mengurangi ukuran fitur agar proses komputasi lebih ringan. Dropout digunakan untuk mengurangi risiko overfitting. Flatten layer mengubah hasil ekstraksi fitur menjadi vektor satu dimensi, dense layer mempelajari hubungan antar fitur, sedangkan output layer menghasilkan skor prediksi untuk enam kategori yang digunakan.

Secara umum, model menerima input citra berukuran **128 x 128 x 3**. Setelah itu, citra dinormalisasi menggunakan rescaling dan diberi augmentasi secara real-time. Citra kemudian diproses melalui empat layer Conv2D untuk mengekstraksi fitur visual. Setelah fitur diekstraksi, hasilnya diubah menjadi vektor satu dimensi melalui Flatten, lalu diproses oleh Dense layer untuk menghasilkan prediksi akhir.

**Tabel 2. Rancangan Arsitektur CNN Kustom 6 Kelas**

| Lapisan | Konfigurasi |
|---|---|
| Input | 128 x 128 x 3 |
| Rescaling | 1/255 |
| Augmentasi | RandomFlip horizontal dan vertikal, RandomRotation 0.10, RandomZoom 0.10 |
| Conv2D 1 | 32 filter, kernel 3 x 3, padding same, aktivasi ReLU |
| MaxPooling2D 1 | Pooling default 2 x 2 |
| Dropout 1 | 0.10 |
| Conv2D 2 | 64 filter, kernel 3 x 3, padding same, aktivasi ReLU |
| MaxPooling2D 2 | Pooling default 2 x 2 |
| Dropout 2 | 0.15 |
| Conv2D 3 | 128 filter, kernel 3 x 3, padding same, aktivasi ReLU |
| MaxPooling2D 3 | Pooling default 2 x 2 |
| Dropout 3 | 0.20 |
| Conv2D 4 | 128 filter, kernel 3 x 3, padding same, aktivasi ReLU |
| MaxPooling2D 4 | Pooling default 2 x 2 |
| Flatten | Mengubah fitur menjadi vektor satu dimensi |
| Dense | 256 neuron, aktivasi ReLU |
| Dropout 4 | 0.50 |
| Output | 6 neuron tanpa aktivasi softmax eksplisit |

Output layer pada model ini tidak menggunakan aktivasi softmax secara eksplisit. Oleh karena itu, output model masih berupa **logits**, yaitu skor mentah untuk setiap kelas. Perhitungan probabilitas dan kesalahan prediksi ditangani melalui fungsi loss `SparseCategoricalCrossentropy(from_logits=True)`.

Model CNN Kustom berbeda dengan pendekatan transfer learning yang biasanya memanfaatkan model pretrained dari dataset berskala besar. Pada penelitian ini, seluruh proses pembelajaran fitur visual hanya bergantung pada dataset citra daun padi yang tersedia. Dengan demikian, kemampuan model dalam mengenali pola seperti tepi daun, warna, tekstur, bentuk bercak, dan pola kerusakan sangat dipengaruhi oleh jumlah, kualitas, serta variasi citra pada dataset penelitian.

## 2.6 Pelatihan Model

Pelatihan model dilakukan menggunakan data training, sedangkan data validation digunakan untuk memantau kinerja model selama training berlangsung. Pada penelitian ini, model CNN Kustom dilatih menggunakan optimizer **Adam** dengan learning rate **0.0001**, batch size **32**, dan jumlah epoch maksimum **30**.

Fungsi loss yang digunakan adalah **SparseCategoricalCrossentropy(from_logits=True)** karena penelitian ini merupakan klasifikasi multi-kategori dengan label kelas numerik dan output model berupa logits. Selama proses pelatihan, perubahan accuracy dan loss pada data training serta validation diamati untuk mengetahui perkembangan pembelajaran model.

Selain itu, proses pelatihan menggunakan mekanisme **early stopping** dengan memantau nilai validation loss. Early stopping digunakan agar proses pelatihan dapat dihentikan ketika performa validasi tidak lagi membaik, sehingga dapat membantu mengurangi risiko overfitting. Pada hasil eksperimen, model dikonfigurasi untuk berjalan hingga 30 epoch, tetapi proses training berhenti pada epoch ke-27 karena mekanisme early stopping.

**Tabel 3. Konfigurasi Hyperparameter Pelatihan**

| Hyperparameter | Nilai |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Batch size | 32 |
| Epoch maksimum | 30 |
| Epoch berjalan | 27 |
| Loss function | SparseCategoricalCrossentropy(from_logits=True) |
| Input size | 128 x 128 x 3 |
| Jumlah kelas | 6 kelas |
| Metrik training | Accuracy |
| Early stopping | Monitor validation loss, patience 8, restore best weights |

## Hasil Evaluasi Singkat

Berdasarkan hasil pengujian pada data testing, model CNN Kustom 6 kelas memperoleh accuracy sebesar **61,57%** dengan macro F1-score sebesar **61,33%**. Hasil ini menunjukkan bahwa model mampu mengenali sebagian pola penyakit daun padi, tetapi performanya masih dipengaruhi oleh keterbatasan jumlah data, kemiripan visual antar kelas penyakit, serta kualitas citra yang digunakan.

**Tabel 4. Ringkasan Hasil Evaluasi CNN Kustom 6 Kelas**

| Metrik | Nilai |
|---|---:|
| Accuracy | 61,57% |
| Precision macro | 64,45% |
| Recall macro | 61,43% |
| F1-score macro | 61,33% |
| Precision weighted | 64,63% |
| Recall weighted | 61,57% |
| F1-score weighted | 61,50% |
