# Draft Paper MALCOM

Template acuan: `C:\Users\farid\Downloads\New Malcom Template.docx`

Catatan:

- Draft ini mengikuti struktur template MALCOM.
- Seluruh isi menggunakan bahasa Indonesia, kecuali bagian `Abstract` yang menggunakan bahasa Inggris.
- Sitasi angka `[1]`, `[2]`, dan seterusnya masih perlu disesuaikan dengan metadata referensi final.
- Format akhir seperti font, margin, kolom, dan style perlu dipindahkan ke file `.docx` template.

---

# Comparative Analysis of Custom CNN and MobileNetV2 for Rice Leaf Disease Classification

# Analisis Perbandingan CNN Kustom dan MobileNetV2 untuk Klasifikasi Penyakit Daun Padi

Author1*, Author2

1Program Studi Informatika, Fakultas ..., Universitas ..., Indonesia

E-Mail: 1emailanda@example.com, 2emaildosen@example.com

Corresponding Author: Nama Penulis

## Abstract

Rice leaf disease can reduce crop productivity and affect food security. Manual identification of rice leaf disease is often subjective because several diseases have similar visual symptoms. This study compares a Custom Convolutional Neural Network (CNN) and MobileNetV2 transfer learning for classifying seven rice leaf conditions: Bacterial Leaf Blight, Brown Spot, Healthy Rice Leaf, Leaf Blast, Leaf Scald, Sheath Blight, and Tungro. The dataset was divided into training, validation, and testing sets with a 70:15:15 ratio. The Custom CNN was built from convolution, pooling, flatten, and dense layers, while MobileNetV2 used pretrained ImageNet weights with a modified classification head. Model performance was evaluated using accuracy, precision, recall, F1-score, classification report, and confusion matrix. Experimental results show that MobileNetV2 achieved the best performance with 81.64% accuracy and 79.49% macro F1-score, while the Custom CNN achieved 78.13% accuracy and 74.62% macro F1-score. These results indicate that MobileNetV2 is more effective for rice leaf disease classification due to its pretrained feature extraction capability.

Keyword: CNN, deep learning, MobileNetV2, rice leaf disease, transfer learning

## Abstrak

Penyakit daun padi dapat menurunkan produktivitas tanaman dan berdampak pada ketahanan pangan. Identifikasi penyakit secara manual masih memiliki keterbatasan karena beberapa penyakit memiliki gejala visual yang mirip. Penelitian ini membandingkan Convolutional Neural Network (CNN) Kustom dan MobileNetV2 berbasis transfer learning untuk klasifikasi tujuh kondisi daun padi, yaitu Bacterial Leaf Blight, Brown Spot, Healthy Rice Leaf, Leaf Blast, Leaf Scald, Sheath Blight, dan Tungro. Dataset dibagi menjadi training set, validation set, dan test set dengan rasio 70:15:15. CNN Kustom dibangun menggunakan layer konvolusi, pooling, flatten, dan dense, sedangkan MobileNetV2 menggunakan bobot pretrained ImageNet dengan classification head yang disesuaikan. Evaluasi dilakukan menggunakan accuracy, precision, recall, F1-score, classification report, dan confusion matrix. Hasil eksperimen menunjukkan bahwa MobileNetV2 memperoleh performa terbaik dengan accuracy sebesar 81,64% dan F1-score macro sebesar 79,49%, sedangkan CNN Kustom memperoleh accuracy sebesar 78,13% dan F1-score macro sebesar 74,62%. Hasil ini menunjukkan bahwa MobileNetV2 lebih efektif untuk klasifikasi penyakit daun padi karena memiliki kemampuan ekstraksi fitur dari model pretrained.

Kata Kunci: CNN, deep learning, MobileNetV2, penyakit daun padi, transfer learning

## INTRODUCTION

Padi merupakan salah satu komoditas pangan utama yang memiliki peran penting dalam menunjang ketahanan pangan, khususnya di Indonesia. Produktivitas tanaman padi perlu dijaga karena padi menjadi sumber makanan pokok bagi sebagian besar masyarakat. Salah satu faktor yang dapat menurunkan produktivitas tanaman padi adalah serangan penyakit pada daun. Penyakit daun padi dapat mengganggu proses fotosintesis, menghambat pertumbuhan tanaman, dan menurunkan hasil panen.

Beberapa penyakit daun padi memiliki gejala visual yang mirip, seperti bercak, perubahan warna, garis pada daun, dan kerusakan jaringan. Proses identifikasi penyakit secara manual masih banyak bergantung pada pengalaman pengamat. Kondisi tersebut dapat menyebabkan hasil identifikasi menjadi subjektif, terutama jika citra daun memiliki pencahayaan yang berbeda atau gejala penyakit saling menyerupai. Oleh karena itu, dibutuhkan pendekatan otomatis yang mampu membantu proses klasifikasi penyakit daun padi berdasarkan citra digital.

Perkembangan computer vision dan deep learning memungkinkan proses klasifikasi citra dilakukan secara otomatis. Salah satu metode yang banyak digunakan untuk klasifikasi citra adalah Convolutional Neural Network (CNN). CNN mampu mempelajari fitur visual seperti warna, tekstur, tepi, bercak, dan bentuk lesi secara otomatis melalui proses training [1], [2]. Dalam konteks penyakit tanaman, CNN dapat digunakan untuk mengenali pola visual pada daun dan mengelompokkannya ke dalam kelas penyakit tertentu.

Pendekatan CNN dapat dilakukan dengan membangun model dari awal atau menggunakan transfer learning. CNN Kustom memiliki kelebihan karena arsitekturnya sederhana dan dapat disesuaikan dengan kebutuhan penelitian. Namun, model yang dilatih dari awal harus mempelajari seluruh fitur visual dari dataset yang tersedia. Hal ini membuat performa CNN Kustom sangat dipengaruhi oleh jumlah data, variasi citra, dan kompleksitas arsitektur. Sebaliknya, transfer learning memanfaatkan model yang telah dilatih pada dataset besar sehingga model sudah memiliki kemampuan awal dalam mengenali fitur visual umum [3], [4].

MobileNetV2 merupakan salah satu arsitektur deep learning yang dirancang ringan dan efisien. Arsitektur ini menggunakan depthwise separable convolution dan inverted residual block untuk mengurangi kebutuhan komputasi. MobileNetV2 relevan digunakan untuk klasifikasi citra penyakit tanaman karena mampu memberikan keseimbangan antara performa dan efisiensi [5], [6].

Penelitian ini membandingkan CNN Kustom dan MobileNetV2 untuk klasifikasi tujuh kondisi daun padi. Kelas yang digunakan meliputi Bacterial Leaf Blight, Brown Spot, Healthy Rice Leaf, Leaf Blast, Leaf Scald, Sheath Blight, dan Tungro. Evaluasi dilakukan menggunakan accuracy, precision, recall, F1-score, classification report, dan confusion matrix. Tujuan penelitian ini adalah mengetahui model yang memberikan performa terbaik pada klasifikasi penyakit daun padi berdasarkan citra digital.

Kontribusi penelitian ini adalah:

1. Membangun model CNN Kustom untuk klasifikasi tujuh kondisi daun padi.
2. Menerapkan MobileNetV2 berbasis transfer learning sebagai model pembanding.
3. Membandingkan performa kedua model menggunakan dataset dan metrik evaluasi yang sama.
4. Menentukan model terbaik berdasarkan accuracy dan F1-score macro.

## MATERIALS AND METHOD

## Alur Penelitian

Alur penelitian terdiri dari tahap persiapan dataset, preprocessing, pembagian data, augmentasi, pembangunan model, training, evaluasi, dan perbandingan hasil. Tahapan penelitian yang dilakukan adalah sebagai berikut:

1. Load dataset citra daun padi.
2. Membagi dataset menjadi training set, validation set, dan test set.
3. Melakukan preprocessing dan augmentasi data.
4. Membangun model CNN Kustom.
5. Melakukan training CNN Kustom.
6. Mengevaluasi CNN Kustom.
7. Membangun model MobileNetV2 berbasis transfer learning.
8. Melakukan training MobileNetV2.
9. Mengevaluasi MobileNetV2.
10. Membandingkan performa kedua model.
11. Menganalisis confusion matrix.
12. Menentukan model terbaik.

## Dataset

Dataset yang digunakan terdiri dari tujuh kelas kondisi daun padi, yaitu:

1. Bacterial Leaf Blight
2. Brown Spot
3. Healthy Rice Leaf
4. Leaf Blast
5. Leaf Scald
6. Sheath Blight
7. Tungro

Dataset dibagi menjadi training set, validation set, dan test set dengan rasio 70:15:15. Training set digunakan untuk melatih model, validation set digunakan untuk memantau performa model selama proses training, sedangkan test set digunakan untuk evaluasi akhir. Test set tidak digunakan selama proses training agar hasil evaluasi dapat menggambarkan kemampuan generalisasi model terhadap data baru.

## Preprocessing dan Augmentasi

Seluruh citra diubah ukurannya menjadi 256 x 256 piksel. Pada CNN Kustom, nilai piksel dinormalisasi menggunakan rescaling 1/255. Pada MobileNetV2, citra diproses menggunakan fungsi preprocessing bawaan MobileNetV2.

Augmentasi data diterapkan pada data training untuk meningkatkan variasi citra. Teknik augmentasi yang digunakan adalah RandomFlip dan RandomRotation. Augmentasi tidak diterapkan pada validation set dan test set.

## Arsitektur CNN Kustom

CNN Kustom dibangun dari awal menggunakan beberapa layer konvolusi dan pooling. Arsitektur CNN Kustom terdiri dari:

1. Input layer
2. Rescaling layer
3. Data augmentation layer
4. Conv2D dengan 16 filter
5. MaxPooling2D
6. Conv2D dengan 32 filter
7. MaxPooling2D
8. Conv2D dengan 64 filter
9. MaxPooling2D
10. Flatten
11. Dense dengan 128 neuron
12. Output layer tujuh kelas

CNN Kustom digunakan sebagai baseline karena model ini dilatih dari awal tanpa memanfaatkan bobot pretrained.

## Arsitektur MobileNetV2

MobileNetV2 digunakan sebagai model transfer learning. Model ini menggunakan bobot pretrained ImageNet dengan `include_top=False`. Classification head bawaan diganti dengan layer baru yang disesuaikan dengan jumlah kelas pada dataset, yaitu tujuh kelas.

Struktur MobileNetV2 yang digunakan dalam penelitian ini terdiri dari:

1. Input layer
2. Data augmentation
3. Preprocessing MobileNetV2
4. MobileNetV2 base model
5. GlobalAveragePooling2D
6. Dropout
7. Dense output tujuh kelas

MobileNetV2 dipilih karena memiliki arsitektur ringan dan efisien, serta mampu memanfaatkan fitur visual dari proses pretraining.

## Parameter Training

Parameter training yang digunakan ditunjukkan pada Tabel 1.

Table 1. Parameter Training

| Parameter | Nilai |
|---|---:|
| Image size | 256 x 256 |
| Batch size | 32 |
| Epoch | 20 |
| Optimizer | Adam |
| Loss function | Sparse Categorical Crossentropy |
| Metrik evaluasi | Accuracy, Precision, Recall, F1-score |

## Metrik Evaluasi

Evaluasi dilakukan menggunakan accuracy, precision, recall, F1-score, classification report, dan confusion matrix. Accuracy digunakan untuk melihat proporsi prediksi benar terhadap seluruh data uji. Precision digunakan untuk mengukur ketepatan prediksi pada suatu kelas. Recall digunakan untuk mengukur kemampuan model dalam mengenali data pada suatu kelas. F1-score digunakan sebagai rata-rata harmonis antara precision dan recall.

F1-score macro digunakan sebagai salah satu metrik utama karena menghitung rata-rata performa setiap kelas tanpa terlalu dipengaruhi oleh jumlah data pada kelas yang lebih dominan.

## RESULTS AND DISCUSSION

## Hasil CNN Kustom

CNN Kustom memperoleh accuracy sebesar 78,13% pada test set. Nilai precision macro, recall macro, dan F1-score macro masing-masing sebesar 76,42%, 74,77%, dan 74,62%. Hasil ini menunjukkan bahwa CNN Kustom mampu melakukan klasifikasi tujuh kondisi daun padi dengan performa cukup baik meskipun dilatih dari awal.

Table 2. Classification Report CNN Kustom

| Kelas | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Bacterial Leaf Blight | 0,65 | 0,58 | 0,61 | 104 |
| Brown Spot | 0,56 | 0,79 | 0,66 | 101 |
| Healthy Rice Leaf | 0,92 | 0,94 | 0,93 | 102 |
| Leaf Blast | 0,63 | 0,57 | 0,59 | 92 |
| Leaf scald | 0,63 | 0,44 | 0,52 | 88 |
| Sheath Blight | 0,77 | 0,80 | 0,78 | 94 |
| Tungro | 1,00 | 0,99 | 1,00 | 187 |

Berdasarkan hasil tersebut, CNN Kustom memiliki performa sangat baik pada kelas Tungro dan Healthy Rice Leaf. Namun, performa pada kelas Leaf scald (F1-score 0,52) dan Leaf Blast (F1-score 0,59) masih lebih rendah dibandingkan kelas lainnya. Hal ini dapat disebabkan oleh kemiripan gejala visual antarpenyakit.

## Hasil MobileNetV2

MobileNetV2 memperoleh accuracy sebesar 81,64% pada test set. Nilai precision macro, recall macro, dan F1-score macro masing-masing sebesar 80,94%, 79,20%, dan 79,49%. Hasil ini menunjukkan bahwa MobileNetV2 memiliki performa lebih baik dibandingkan CNN Kustom.

Table 3. Classification Report MobileNetV2

| Kelas | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Bacterial Leaf Blight | 0,74 | 0,62 | 0,67 | 104 |
| Brown Spot | 0,79 | 0,68 | 0,73 | 101 |
| Healthy Rice Leaf | 0,96 | 0,96 | 0,96 | 102 |
| Leaf Blast | 0,51 | 0,77 | 0,61 | 92 |
| Leaf scald | 0,78 | 0,70 | 0,74 | 88 |
| Sheath Blight | 0,89 | 0,81 | 0,85 | 94 |
| Tungro | 1,00 | 1,00 | 1,00 | 187 |

MobileNetV2 menunjukkan peningkatan performa pada beberapa kelas seperti Leaf scald (0,52 → 0,74), Brown Spot (0,66 → 0,73), dan Sheath Blight (0,78 → 0,85). Pada kelas Leaf Blast, kedua model memperoleh F1-score yang hampir sama (0,59 vs 0,61).

## Perbandingan Model

Perbandingan performa CNN Kustom dan MobileNetV2 ditunjukkan pada Tabel 4.

Table 4. Perbandingan Performa Model

| Model | Accuracy | Precision Macro | Recall Macro | F1 Macro | F1 Weighted |
|---|---:|---:|---:|---:|---:|
| MobileNetV2 | 81,64% | 80,94% | 79,20% | 79,49% | 82,02% |
| CNN Kustom | 78,13% | 76,42% | 74,77% | 74,62% | 77,81% |

MobileNetV2 unggul pada seluruh metrik utama. Selisih accuracy antara MobileNetV2 dan CNN Kustom adalah 3,51 persen poin. Selisih F1-score macro adalah 4,87 persen poin. Hal ini menunjukkan bahwa transfer learning menggunakan MobileNetV2 mampu meningkatkan performa klasifikasi dibandingkan CNN Kustom yang dilatih dari awal.

Keunggulan MobileNetV2 dapat dijelaskan oleh penggunaan bobot pretrained ImageNet. Dengan bobot pretrained, MobileNetV2 telah memiliki kemampuan awal dalam mengenali fitur visual umum seperti bentuk, tepi, tekstur, dan pola warna. Fitur tersebut membantu model dalam mengenali pola penyakit pada daun padi.

CNN Kustom tetap relevan sebagai model baseline. Meskipun performanya lebih rendah dibandingkan MobileNetV2, CNN Kustom mampu menunjukkan kemampuan klasifikasi yang cukup baik tanpa menggunakan model pretrained. Hal ini menunjukkan bahwa arsitektur CNN sederhana masih dapat mempelajari pola visual penyakit daun padi, tetapi transfer learning memberikan performa yang lebih baik pada dataset ini.

## CONCLUSION

Penelitian ini membandingkan CNN Kustom dan MobileNetV2 untuk klasifikasi tujuh kondisi daun padi. Berdasarkan hasil eksperimen, MobileNetV2 memperoleh performa lebih baik dibandingkan CNN Kustom. MobileNetV2 menghasilkan accuracy sebesar 81,64% dan F1-score macro sebesar 79,49%, sedangkan CNN Kustom menghasilkan accuracy sebesar 78,13% dan F1-score macro sebesar 74,62%.

Berdasarkan hasil tersebut, MobileNetV2 menjadi model terbaik dalam penelitian ini. Penggunaan transfer learning membantu MobileNetV2 memperoleh performa lebih baik karena model dapat memanfaatkan fitur visual dari proses pretraining. Sementara itu, CNN Kustom tetap dapat digunakan sebagai baseline karena mampu memberikan performa cukup baik meskipun dilatih dari awal.

Penelitian selanjutnya dapat mengembangkan eksperimen dengan menambahkan jumlah data, melakukan fine-tuning pada MobileNetV2, membandingkan model pretrained lain seperti EfficientNetB0 atau MobileNetV3, serta menggunakan metode explainable AI seperti Grad-CAM untuk melihat area citra yang diperhatikan model saat melakukan prediksi.

## ACKNOWLEDGMENTS

Penulis mengucapkan terima kasih kepada dosen pembimbing, institusi akademik, dan seluruh pihak yang telah memberikan dukungan dalam pelaksanaan penelitian ini. Bagian ini dapat disesuaikan kembali sesuai kebutuhan publikasi.

## REFERENCES

[1] Frontiers article, "Automatic Diagnosis of Rice Diseases Using Deep Learning," 2021.

[2] Frontiers article, "Image Dataset Configuration for CNN-Based Rice Disease Classification," 2022.

[3] Article on MobileNetV2 and EfficientNetB0 for Rice Leaf Disease Classification.

[4] Article on MobileNetV2 and MobileNetV3 for Rice Leaf Disease Classification.

[5] Article on Lightweight MobileNetV2 for Paddy Disease Detection.

[6] Article on MobileNetV2 and TensorFlow Lite for Rice Disease Detection.

[7] Article on Enhanced CAE-CNN Architecture for Rice Leaf Disease Classification.

[8] Article on Lightweight CNNs for Rice Leaf Disease Identification.

[9] Article on CNN and Transfer Learning for Rice Leaf Disease Classification.

[10] Article on Deep Learning for Rice Leaf Disease Classification.

Catatan: daftar referensi sementara di atas perlu diganti dengan format IEEE lengkap berdasarkan metadata jurnal yang sudah diunduh pada folder `references/journals`.
