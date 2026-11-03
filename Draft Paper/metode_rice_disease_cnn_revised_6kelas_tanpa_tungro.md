# Metode CNN Kustom 6 Kelas Berdasarkan Notebook Revised

Dokumen ini disesuaikan dengan file notebook:

`rice_leaf_disease_project/notebooks/rice_disease_cnn_revised_6kelas_tanpa_tungro.ipynb`

Notebook tersebut menggunakan dataset **6 kelas tanpa Tungro** dan model **CNN Kustom** yang dibangun dari awal.

## 2.2 Preprocessing Citra

Preprocessing citra dilakukan sebelum data dimasukkan ke dalam model CNN Kustom. Tahap ini mencakup penyesuaian ukuran gambar agar seluruh citra memiliki dimensi input yang seragam. Pada penelitian ini, seluruh citra diubah menjadi ukuran **256 x 256 piksel** sesuai dengan kebutuhan input model. Karena citra yang digunakan merupakan citra berwarna RGB, maka bentuk input model adalah **256 x 256 x 3**.

Selain penyesuaian ukuran citra, normalisasi piksel dilakukan dengan mengubah nilai piksel dari rentang **0-255** menjadi **0-1** menggunakan layer `Rescaling(1./255)`. Normalisasi ini dilakukan agar skala data lebih sederhana dan proses pelatihan model menjadi lebih stabil.

Dataset yang digunakan pada eksperimen ini terdiri dari **enam kelas citra daun padi**, yaitu:

| No | Kelas |
|---:|---|
| 1 | Bacterial Leaf Blight |
| 2 | Brown Spot |
| 3 | Healthy Rice Leaf |
| 4 | Leaf Blast |
| 5 | Leaf Scald |
| 6 | Sheath Blight |

Kelas **Tungro** tidak digunakan pada eksperimen ini karena jumlah datanya jauh lebih banyak dibandingkan kelas lainnya. Dengan mengeluarkan kelas Tungro, eksperimen difokuskan pada enam kelas yang memiliki jumlah data relatif lebih seimbang.

## 2.3 Augmentasi Data

Augmentasi data diterapkan pada data training dengan teknik **RandomFlip** dan **RandomRotation**. Teknik ini digunakan untuk menambah variasi tampilan citra latih, sehingga model dapat mempelajari pola penyakit dari orientasi dan posisi daun yang berbeda.

Pada notebook yang digunakan, augmentasi dilakukan melalui layer berikut:

```text
RandomFlip("horizontal_and_vertical")
RandomRotation(0.2)
```

Nilai `RandomRotation(0.2)` berarti citra dapat diputar secara acak hingga sekitar 20% dari satu putaran penuh. Augmentasi ini diterapkan secara **real-time saat training**, sehingga tidak menghasilkan file gambar baru di folder dataset. Jumlah file citra asli tetap **3.829 gambar**, tetapi variasi citra yang dipelajari model bertambah selama proses pelatihan berlangsung.

Augmentasi digunakan sebagai strategi untuk membantu model mengenali variasi bentuk dan orientasi daun. Namun, performa akhir tetap dipengaruhi oleh kualitas citra asli, jumlah data, serta tingkat kemiripan ciri visual antar kategori penyakit.

## 2.4 Pembagian Dataset

Dataset dibagi menjadi tiga bagian, yaitu data training, validation, dan testing. Data training digunakan untuk melatih model, data validation digunakan untuk memantau perkembangan model selama proses training, sedangkan data testing digunakan sebagai data uji akhir untuk menilai performa model pada citra yang tidak dilibatkan dalam pelatihan.

Pembagian data dilakukan dengan skema sekitar **70% training**, **15% validation**, dan **15% testing**. Berdasarkan output notebook, jumlah data yang digunakan adalah sebagai berikut:

**Tabel 1. Pembagian Dataset**

| Jenis Data | Jumlah Citra |
|---|---:|
| Training | 2.681 |
| Validation | 572 |
| Testing | 576 |
| **Total** | **3.829** |

Pada notebook, data training dimuat menggunakan `validation_split=0.3` dengan subset `training`, sehingga 70% data digunakan untuk training. Sisa 30% data dimuat sebagai `val_test_ds`, kemudian dibagi lagi menjadi validation dan testing menggunakan operasi `take()` dan `skip()` pada batch dataset.

Pembagian dataset dilakukan agar proses pembelajaran, pemantauan, dan pengujian model menggunakan data yang berbeda. Dengan demikian, hasil pengujian akhir dapat memberikan gambaran yang lebih objektif terhadap kemampuan model dalam mengklasifikasikan citra baru.

## 2.5 Perancangan Arsitektur CNN Kustom

Arsitektur CNN Kustom dalam penelitian ini dirancang sebagai model baseline yang dilatih dari awal tanpa memanfaatkan bobot pretrained. Model terdiri atas beberapa komponen utama, yaitu input layer, rescaling layer, data augmentation layer, convolutional layer, pooling layer, flatten layer, dense layer, dan output layer.

Convolutional layer berfungsi untuk memperoleh ciri visual dari citra daun padi, seperti warna, tekstur, tepi daun, dan bentuk bercak. Pooling layer digunakan untuk mengurangi ukuran fitur agar proses komputasi lebih ringan. Flatten layer mengubah hasil ekstraksi fitur menjadi vektor satu dimensi. Dense layer digunakan untuk mempelajari hubungan antar fitur, sedangkan output layer menghasilkan skor prediksi akhir berdasarkan enam kategori yang digunakan.

Secara umum, rancangan arsitektur CNN Kustom pada notebook ini adalah sebagai berikut:

```text
Input 256 x 256 x 3
        |
Rescaling 1/255
        |
Data Augmentation
        |
Conv2D 16 + ReLU
        |
MaxPooling2D
        |
Conv2D 32 + ReLU
        |
MaxPooling2D
        |
Conv2D 64 + ReLU
        |
MaxPooling2D
        |
Flatten
        |
Dense 128 + ReLU
        |
Dense 6 Output
```

**Tabel 2. Rancangan Arsitektur CNN Kustom**

| Lapisan | Konfigurasi |
|---|---|
| Input | 256 x 256 x 3 |
| Rescaling | 1/255 |
| Augmentasi | RandomFlip horizontal dan vertikal, RandomRotation 0.2 |
| Conv2D 1 | 16 filter, kernel 3 x 3, padding same, aktivasi ReLU |
| MaxPooling2D 1 | Pooling default 2 x 2 |
| Conv2D 2 | 32 filter, kernel 3 x 3, padding same, aktivasi ReLU |
| MaxPooling2D 2 | Pooling default 2 x 2 |
| Conv2D 3 | 64 filter, kernel 3 x 3, padding same, aktivasi ReLU |
| MaxPooling2D 3 | Pooling default 2 x 2 |
| Flatten | Mengubah feature map menjadi vektor satu dimensi |
| Dense | 128 neuron, aktivasi ReLU |
| Output | 6 neuron tanpa aktivasi softmax eksplisit |

Output layer pada model ini tidak menggunakan aktivasi softmax secara eksplisit. Oleh karena itu, output model masih berupa **logits**, yaitu skor mentah untuk setiap kelas. Perhitungan probabilitas dan kesalahan prediksi ditangani oleh fungsi loss `SparseCategoricalCrossentropy(from_logits=True)`.

Model CNN Kustom berbeda dengan pendekatan transfer learning yang memanfaatkan model pretrained dari dataset berskala besar. Pada penelitian ini, seluruh proses pembelajaran fitur visual hanya bergantung pada dataset citra daun padi yang tersedia. Dengan demikian, kemampuan model dalam mengenali pola seperti tepi daun, warna, tekstur, bentuk bercak, dan pola kerusakan sangat dipengaruhi oleh jumlah, kualitas, serta variasi citra pada dataset penelitian.

## 2.6 Pelatihan Model

Pelatihan model dilakukan menggunakan data training, sedangkan data validation digunakan untuk memantau kinerja model selama training berlangsung. Pada penelitian ini, model dilatih menggunakan optimizer **Adam**, batch size **32**, dan jumlah epoch **20**. Learning rate yang digunakan adalah learning rate default dari Adam, yaitu **0.001**.

Fungsi loss yang digunakan adalah **SparseCategoricalCrossentropy(from_logits=True)** karena penelitian ini merupakan klasifikasi multi-kategori dengan label numerik dan output model berupa logits. Selama proses pelatihan, perubahan accuracy dan loss pada data training serta validation diamati untuk mengetahui perkembangan pembelajaran model.

Selain accuracy dan loss, notebook juga menggunakan callback tambahan untuk menampilkan nilai **precision macro**, **recall macro**, dan **F1-score macro** pada data validation di setiap akhir epoch. Metrik macro digunakan karena setiap kelas dinilai dengan bobot yang sama, sehingga cocok untuk melihat performa model pada seluruh kelas secara lebih seimbang.

**Tabel 3. Konfigurasi Hyperparameter Pelatihan**

| Hyperparameter | Nilai |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 atau default Adam |
| Batch size | 32 |
| Epoch | 20 |
| Loss function | SparseCategoricalCrossentropy(from_logits=True) |
| Input size | 256 x 256 x 3 |
| Jumlah kelas | 6 kelas |
| Data augmentation | RandomFlip dan RandomRotation |
| Metrik training | Accuracy |
| Metrik validation tambahan | Precision macro, recall macro, F1-score macro |

## Hasil Evaluasi Notebook

Berdasarkan output notebook, evaluasi akhir pada test set menghasilkan:

```text
Test Loss     : 0.8881
Test Accuracy : 67.71%
```

Classification report yang tercetak pada notebook menunjukkan ringkasan performa sebagai berikut:

**Tabel 4. Ringkasan Classification Report**

| Metrik | Nilai |
|---|---:|
| Accuracy | 71% |
| Precision macro | 72% |
| Recall macro | 70% |
| F1-score macro | 71% |
| Precision weighted | 71% |
| Recall weighted | 71% |
| F1-score weighted | 71% |

Detail performa setiap kelas berdasarkan classification report adalah sebagai berikut:

| Kelas | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Bacterial Leaf Blight | 0.63 | 0.66 | 0.64 | 88 |
| Brown Spot | 0.75 | 0.50 | 0.60 | 88 |
| Healthy Rice Leaf | 0.91 | 0.94 | 0.92 | 94 |
| Leaf Blast | 0.64 | 0.68 | 0.66 | 115 |
| Leaf Scald | 0.56 | 0.66 | 0.61 | 96 |
| Sheath Blight | 0.80 | 0.80 | 0.80 | 95 |

Catatan: terdapat perbedaan antara nilai accuracy dari `model.evaluate()` sebesar **67,71%** dan nilai accuracy pada classification report sebesar sekitar **71%**. Sebelum angka final dimasukkan ke paper, disarankan menjalankan ulang cell evaluasi dan classification report secara berurutan dari kernel yang bersih agar hasil akhir yang digunakan konsisten.
