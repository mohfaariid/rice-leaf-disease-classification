# Poin Pembahasan Paper CNN Kustom

## Fokus Paper

Paper ini difokuskan pada penerapan **Convolutional Neural Network (CNN) Kustom** untuk klasifikasi penyakit daun padi berdasarkan citra digital.

Paper ini tidak membahas:

- Perbandingan dengan MobileNetV2.
- Analisis imbalance dataset.
- Eksperimen pengurangan data kelas tertentu.

Topik tersebut dapat dibahas nanti pada laporan skripsi.

## Usulan Judul

Pilihan judul yang disarankan:

1. Klasifikasi Penyakit Daun Padi Menggunakan Convolutional Neural Network Kustom
2. Penerapan CNN Kustom untuk Klasifikasi Penyakit Daun Padi Berdasarkan Citra Digital
3. Implementasi CNN Kustom pada Citra Daun Padi untuk Identifikasi Penyakit Tanaman

Judul yang paling aman:

**Klasifikasi Penyakit Daun Padi Menggunakan Convolutional Neural Network Kustom**

## 1. Latar Belakang

Poin yang dapat dibahas:

- Padi merupakan salah satu tanaman pangan penting.
- Penyakit daun padi dapat menurunkan kualitas dan hasil panen.
- Identifikasi penyakit secara manual membutuhkan pengalaman dan dapat bersifat subjektif.
- Beberapa penyakit daun padi memiliki gejala visual yang mirip, seperti bercak, perubahan warna, dan bagian daun yang mengering.
- Citra digital dapat dimanfaatkan sebagai dasar untuk membantu identifikasi penyakit daun padi.
- CNN cocok digunakan untuk data citra karena mampu mengenali pola visual seperti warna, tekstur, tepi, dan bentuk bercak.

Contoh narasi:

> Penyakit daun padi dapat berdampak pada produktivitas tanaman, sehingga proses identifikasi penyakit perlu dilakukan secara cepat dan tepat. Identifikasi secara manual masih memiliki keterbatasan karena bergantung pada pengalaman pengamat dan kondisi visual daun. Oleh karena itu, pendekatan berbasis citra digital dan deep learning dapat digunakan sebagai alternatif untuk membantu proses klasifikasi penyakit daun padi.

## 2. Tujuan Penelitian

Poin tujuan:

- Membangun model CNN Kustom untuk klasifikasi citra daun padi.
- Menguji kemampuan CNN Kustom dalam mengenali beberapa kondisi daun padi.
- Mengevaluasi performa model menggunakan accuracy, precision, recall, F1-score, dan confusion matrix.

Contoh narasi:

> Penelitian ini bertujuan untuk menerapkan model CNN Kustom dalam klasifikasi citra daun padi serta mengevaluasi performanya berdasarkan metrik accuracy, precision, recall, F1-score, dan confusion matrix.

## 3. Dataset

Dataset yang digunakan berupa citra daun padi dengan enam kelas:

1. Bacterial Leaf Blight
2. Brown Spot
3. Healthy Rice Leaf
4. Leaf Blast
5. Leaf scald
6. Sheath Blight

Poin pembahasan:

- Dataset berbentuk gambar daun padi.
- Setiap kelas disimpan dalam folder berbeda.
- Nama folder digunakan sebagai label kelas.
- Dataset dibagi menjadi data training, validation, dan testing.
- Data training digunakan untuk melatih model.
- Data validation digunakan untuk memantau performa selama training.
- Data testing digunakan untuk evaluasi akhir.

Contoh narasi:

> Dataset yang digunakan terdiri dari citra daun padi yang dikelompokkan ke dalam enam kelas. Setiap kelas direpresentasikan oleh folder tersendiri, sehingga label data diperoleh dari nama folder. Dataset kemudian dibagi menjadi training set, validation set, dan test set agar proses pelatihan dan evaluasi dapat dilakukan secara terpisah.

## 4. Preprocessing Data

Poin preprocessing:

- Gambar dibaca dari folder dataset.
- Semua gambar diubah ke ukuran seragam.
- Nilai pixel dinormalisasi dari rentang 0-255 menjadi 0-1.
- Label kelas dihasilkan secara otomatis dari nama folder.

Contoh narasi:

> Setiap citra diubah ke ukuran seragam agar dapat diproses oleh model CNN. Nilai pixel kemudian dinormalisasi dari rentang 0-255 menjadi 0-1 untuk membantu proses pelatihan menjadi lebih stabil.

## 5. Data Augmentation

Augmentasi yang dapat dibahas:

- RandomFlip
- RandomRotation
- RandomZoom, jika digunakan pada eksperimen final

Tujuan augmentasi:

- Menambah variasi gambar selama training.
- Mengurangi risiko overfitting.
- Membuat model tidak terlalu bergantung pada orientasi daun.
- Membantu model mengenali pola penyakit dalam kondisi gambar yang bervariasi.

Contoh narasi:

> Data augmentation digunakan untuk menambah variasi citra pada data training. Transformasi seperti flip dan rotasi membantu model agar tidak hanya menghafal orientasi gambar tertentu, tetapi mampu mengenali pola visual penyakit pada berbagai posisi daun.

## 6. Arsitektur CNN Kustom

Layer utama yang dapat dibahas:

1. Input Layer
2. Rescaling
3. Data Augmentation
4. Conv2D
5. MaxPooling2D
6. Flatten
7. Dense
8. Output Layer

Penjelasan fungsi layer:

- **Input Layer** menerima citra dengan ukuran tertentu.
- **Rescaling** menormalisasi nilai pixel.
- **Data Augmentation** memberi variasi gambar saat training.
- **Conv2D** mengekstraksi fitur visual seperti tepi, warna, tekstur, dan pola bercak.
- **MaxPooling2D** mengurangi ukuran fitur dan membantu menekan komputasi.
- **Flatten** mengubah fitur hasil konvolusi menjadi vektor satu dimensi.
- **Dense Layer** memproses fitur untuk menentukan kelas.
- **Output Layer** menghasilkan skor prediksi untuk setiap kelas.

Contoh narasi:

> Model CNN Kustom dibangun menggunakan beberapa layer konvolusi dan pooling. Layer konvolusi berfungsi untuk mengekstraksi fitur visual dari citra daun, sedangkan pooling digunakan untuk mereduksi ukuran fitur. Setelah fitur diekstraksi, hasilnya diubah menjadi vektor melalui Flatten dan diproses oleh Dense layer untuk menghasilkan prediksi kelas.

## 7. Proses Training

Poin yang dapat dibahas:

- Optimizer yang digunakan adalah Adam.
- Loss function yang digunakan adalah Sparse Categorical Crossentropy.
- Batch size menentukan jumlah gambar yang diproses dalam satu iterasi.
- Epoch menunjukkan jumlah pengulangan proses training.
- Training set digunakan untuk pembelajaran model.
- Validation set digunakan untuk memantau performa selama training.

Contoh narasi:

> Model dilatih menggunakan optimizer Adam dan fungsi loss Sparse Categorical Crossentropy. Optimizer Adam digunakan karena mampu menyesuaikan proses pembaruan bobot secara adaptif, sedangkan Sparse Categorical Crossentropy sesuai untuk kasus klasifikasi multi-kelas dengan label berbentuk bilangan integer.

## 8. Metrik Evaluasi

Metrik yang digunakan:

1. Accuracy
2. Precision
3. Recall
4. F1-score
5. Confusion Matrix

Penjelasan metrik:

- **Accuracy** menunjukkan proporsi prediksi yang benar dari seluruh data.
- **Precision** menunjukkan seberapa tepat prediksi model pada suatu kelas.
- **Recall** menunjukkan kemampuan model menemukan data asli dari suatu kelas.
- **F1-score** merupakan rata-rata harmonis antara precision dan recall.
- **Confusion Matrix** menunjukkan distribusi prediksi benar dan salah antar kelas.

Contoh narasi:

> Evaluasi model tidak hanya dilakukan menggunakan accuracy, tetapi juga precision, recall, dan F1-score. Hal ini dilakukan karena accuracy hanya menunjukkan performa umum, sedangkan precision, recall, dan F1-score memberikan informasi lebih detail mengenai performa model pada setiap kelas.

## 9. Hasil Eksperimen

Jika menggunakan hasil CNN Kustom Stable terakhir, hasil utama adalah:

| Metrik | Nilai |
|---|---:|
| Accuracy | 61,56% |
| Precision Macro | 64,45% |
| Recall Macro | 61,43% |
| F1-score Macro | 61,33% |
| F1-score Weighted | 61,50% |

Hasil per kelas:

| Kelas | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Bacterial Leaf Blight | 0,65 | 0,53 | 0,59 |
| Brown Spot | 0,64 | 0,66 | 0,65 |
| Healthy Rice Leaf | 0,91 | 0,88 | 0,89 |
| Leaf Blast | 0,61 | 0,66 | 0,63 |
| Leaf scald | 0,39 | 0,63 | 0,48 |
| Sheath Blight | 0,67 | 0,33 | 0,44 |

Poin pembahasan:

- Model memperoleh accuracy sebesar 61,56%.
- F1-score macro sebesar 61,33% menunjukkan performa rata-rata antar kelas.
- Kelas Healthy Rice Leaf memiliki performa terbaik.
- Kelas Leaf scald dan Sheath Blight masih sulit dikenali oleh model.
- Perbedaan performa antar kelas dapat disebabkan oleh kemiripan gejala visual antar penyakit.

Contoh narasi:

> Berdasarkan hasil evaluasi, model CNN Kustom memperoleh accuracy sebesar 61,56% dan F1-score macro sebesar 61,33%. Hasil ini menunjukkan bahwa model mampu mempelajari sebagian pola visual pada citra daun padi, meskipun performanya masih bervariasi antar kelas. Kelas Healthy Rice Leaf memperoleh performa terbaik, sedangkan kelas Leaf scald dan Sheath Blight masih memiliki F1-score yang lebih rendah.

## 10. Analisis Kelas yang Mudah dan Sulit Dikenali

Kelas yang relatif mudah dikenali:

- Healthy Rice Leaf

Kemungkinan alasan:

- Daun sehat memiliki pola visual yang lebih berbeda dari daun sakit.
- Warna dan tekstur daun sehat lebih konsisten.

Kelas yang relatif sulit dikenali:

- Leaf scald
- Sheath Blight

Kemungkinan alasan:

- Gejala visual dapat mirip dengan penyakit lain.
- Bercak atau kerusakan daun tidak selalu memiliki pola yang konsisten.
- Variasi pencahayaan, posisi daun, dan kualitas gambar dapat memengaruhi pembelajaran model.

Contoh narasi:

> Performa tertinggi diperoleh pada kelas Healthy Rice Leaf. Hal ini dapat terjadi karena citra daun sehat memiliki pola visual yang relatif berbeda dibandingkan daun yang terinfeksi penyakit. Sebaliknya, kelas Leaf scald dan Sheath Blight memiliki performa lebih rendah, yang mengindikasikan bahwa model masih kesulitan membedakan gejala visual yang mirip antar penyakit.

## 11. Pembahasan Confusion Matrix

Poin pembahasan:

- Diagonal utama menunjukkan prediksi benar.
- Sel di luar diagonal menunjukkan kesalahan klasifikasi.
- Confusion matrix membantu melihat kelas mana yang sering tertukar.
- Kesalahan antar kelas dapat terjadi karena kemiripan warna, bercak, atau tekstur daun.

Contoh narasi:

> Confusion matrix digunakan untuk melihat distribusi kesalahan model. Prediksi benar ditunjukkan pada diagonal utama, sedangkan nilai di luar diagonal menunjukkan kesalahan klasifikasi. Jika suatu kelas sering diprediksi sebagai kelas lain, hal tersebut dapat mengindikasikan adanya kemiripan pola visual antar penyakit.

## 12. Keterbatasan Penelitian

Poin keterbatasan:

- Model hanya menggunakan CNN Kustom yang dilatih dari awal.
- Arsitektur model masih relatif sederhana.
- Model masih kesulitan membedakan beberapa kelas penyakit dengan gejala visual yang mirip.
- Performa model dipengaruhi oleh kualitas citra, pencahayaan, posisi daun, dan variasi data.
- Hasil penelitian berlaku pada dataset dan skenario eksperimen yang digunakan.

Contoh narasi:

> Penelitian ini memiliki keterbatasan karena model yang digunakan merupakan CNN Kustom yang dilatih dari awal. Model belum memanfaatkan pengetahuan dari model pretrained, sehingga kemampuan ekstraksi fitur sangat bergantung pada data training yang tersedia. Selain itu, beberapa penyakit daun padi memiliki gejala visual yang mirip, sehingga model masih mengalami kesulitan dalam membedakan kelas tertentu.

## 13. Kesimpulan

Poin kesimpulan:

- CNN Kustom dapat digunakan untuk klasifikasi citra daun padi.
- Model mampu mengenali pola visual dasar pada beberapa kelas.
- Performa terbaik diperoleh pada kelas Healthy Rice Leaf.
- Beberapa kelas penyakit masih sulit dibedakan.
- CNN Kustom dapat dijadikan baseline untuk pengembangan penelitian selanjutnya.

Contoh kesimpulan:

> Penelitian ini menunjukkan bahwa CNN Kustom dapat digunakan untuk klasifikasi penyakit daun padi berdasarkan citra digital. Model memperoleh accuracy sebesar 61,56% dan F1-score macro sebesar 61,33% pada test set. Hasil terbaik diperoleh pada kelas Healthy Rice Leaf, sedangkan kelas Leaf scald dan Sheath Blight masih sulit dikenali. Dengan demikian, CNN Kustom dapat digunakan sebagai baseline untuk pengembangan sistem klasifikasi penyakit daun padi yang lebih baik pada penelitian selanjutnya.

## 14. Struktur Paper yang Disarankan

Struktur paper:

1. Title
2. Abstract
3. Keywords
4. Introduction
5. Materials and Method
6. Results and Discussion
7. Conclusion
8. References

Isi tiap bagian:

### Introduction

- Latar belakang penyakit daun padi.
- Keterbatasan identifikasi manual.
- Alasan penggunaan CNN.
- Tujuan penelitian.

### Materials and Method

- Dataset.
- Preprocessing.
- Data augmentation.
- Arsitektur CNN Kustom.
- Training model.
- Metrik evaluasi.

### Results and Discussion

- Hasil accuracy, precision, recall, dan F1-score.
- Hasil per kelas.
- Confusion matrix.
- Analisis kelas yang mudah dan sulit dikenali.
- Keterbatasan model.

### Conclusion

- Ringkasan hasil.
- Jawaban terhadap tujuan penelitian.
- Saran pengembangan berikutnya.

