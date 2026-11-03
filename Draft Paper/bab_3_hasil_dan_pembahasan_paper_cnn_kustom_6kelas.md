# 3. Hasil dan Pembahasan

Bagian ini menjelaskan hasil eksperimen model CNN Kustom dalam melakukan klasifikasi citra daun padi pada dataset 6 kelas tanpa kelas Tungro. Pembahasan difokuskan pada hasil pelatihan model, hasil evaluasi pada data testing, analisis confusion matrix, penyebab performa model yang belum optimal, serta perbandingan dengan penelitian terdahulu.

Eksperimen dilakukan menggunakan dataset sebanyak 3.829 citra yang terdiri dari enam kelas, yaitu Bacterial Leaf Blight, Brown Spot, Healthy Rice Leaf, Leaf Blast, Leaf scald, dan Sheath Blight. Dataset dibagi menjadi data training, validation, dan testing. Data training digunakan untuk melatih model, data validation digunakan untuk memantau performa model selama pelatihan, sedangkan data testing digunakan untuk mengevaluasi performa akhir model.

## 3.1 Hasil Pelatihan Model

Model CNN Kustom dilatih menggunakan citra berukuran 256 x 256 piksel dengan tiga channel warna RGB. Arsitektur model terdiri dari tiga lapisan konvolusi utama dengan jumlah filter 16, 32, dan 64, kemudian dilanjutkan dengan lapisan Flatten, Dense 128 neuron, dan output layer sebanyak 6 neuron sesuai jumlah kelas yang diklasifikasikan.

Pada proses pelatihan, model menggunakan optimizer Adam dengan learning rate 0.0003. Fungsi loss yang digunakan adalah `SparseCategoricalCrossentropy(from_logits=True)` karena output model berupa logits dan tidak menggunakan softmax eksplisit pada layer terakhir. Selain itu, model menggunakan dropout dan regularisasi L2 untuk mengurangi risiko overfitting.

Konfigurasi pelatihan model ditunjukkan pada Tabel 3.1.

**Tabel 3.1 Konfigurasi Pelatihan Model CNN Kustom**

| Parameter | Nilai |
|---|---|
| Model | CNN Kustom |
| Input citra | 256 x 256 x 3 |
| Jumlah kelas | 6 kelas |
| Optimizer | Adam |
| Learning rate | 0.0003 |
| Batch size | 32 |
| Epoch maksimum | 100 |
| Loss function | SparseCategoricalCrossentropy(from_logits=True) |
| Output model | Logits |
| Regularisasi | L2 regularization 1e-4 |
| Dropout | 0.10, 0.15, 0.20, dan 0.50 |
| Callback | ModelCheckpoint dan EarlyStopping |
| Pemilihan model terbaik | Berdasarkan validation loss terendah |

Berdasarkan hasil pelatihan, model tidak menggunakan epoch terakhir sebagai model terbaik, melainkan memilih model berdasarkan nilai validation loss terendah. Hal ini dilakukan agar model yang digunakan untuk evaluasi adalah model yang memiliki kemampuan generalisasi terbaik terhadap data validation.

Ringkasan hasil pelatihan ditunjukkan pada Tabel 3.2.

**Tabel 3.2 Ringkasan Hasil Pelatihan Model**

| Komponen | Hasil |
|---|---:|
| Epoch maksimum | 100 |
| Epoch yang berjalan | 94 |
| Epoch terbaik berdasarkan validation loss | 84 |
| Validation loss terbaik | 0.9184 |
| Validation accuracy terbaik | 0.6871 |

Berdasarkan Tabel 3.2, nilai validation loss terendah diperoleh pada epoch ke-84, yaitu sebesar 0.9184. Nilai validation accuracy terbaik yang diperoleh adalah 0.6871. Hasil ini menunjukkan bahwa model masih mengalami kesulitan dalam mempelajari pola visual antar kelas secara optimal. Meskipun demikian, penerapan EarlyStopping dan ModelCheckpoint membantu memilih model terbaik selama proses pelatihan, sehingga model yang digunakan untuk evaluasi bukan model pada epoch terakhir, melainkan model dengan validation loss paling rendah.

Nilai validation loss yang masih cukup tinggi menunjukkan bahwa model belum sepenuhnya mampu melakukan generalisasi terhadap data validation. Hal ini dapat disebabkan oleh kemiripan visual antar kelas penyakit daun padi, jumlah data yang terbatas, serta variasi kondisi citra seperti pencahayaan, sudut pengambilan gambar, dan bentuk daun.

## 3.2 Hasil Evaluasi Model

Setelah proses pelatihan selesai, model terbaik berdasarkan validation loss digunakan untuk evaluasi pada data testing. Data testing merupakan data yang tidak digunakan pada proses training maupun validation, sehingga hasil evaluasi pada data ini dapat digunakan untuk melihat kemampuan model dalam mengklasifikasikan citra baru.

Hasil evaluasi model ditunjukkan pada Tabel 3.3.

**Tabel 3.3 Hasil Evaluasi Model CNN Kustom pada Data Testing**

| Metrik | Nilai |
|---|---:|
| Test accuracy | 0.6684 |
| Precision macro | Sesuaikan dengan classification report |
| Recall macro | Sesuaikan dengan classification report |
| F1-score macro | Sesuaikan dengan classification report |

Berdasarkan hasil evaluasi, model memperoleh test accuracy sebesar 0.6684. Nilai ini menunjukkan bahwa model mampu mengklasifikasikan sebagian besar citra dengan benar, tetapi performanya belum optimal. Dengan accuracy sekitar 66,84%, model masih memiliki kesalahan klasifikasi yang cukup besar pada beberapa kelas.

Nilai accuracy tersebut menunjukkan bahwa CNN Kustom dapat digunakan sebagai pendekatan dasar untuk klasifikasi penyakit daun padi, tetapi arsitektur dan dataset yang digunakan masih perlu ditingkatkan agar hasil klasifikasi menjadi lebih baik. Untuk memperoleh analisis yang lebih mendalam, hasil evaluasi tidak cukup hanya dilihat dari accuracy, tetapi juga perlu dianalisis menggunakan precision, recall, F1-score, dan confusion matrix.

Precision menunjukkan seberapa tepat prediksi model pada suatu kelas, recall menunjukkan seberapa banyak data pada suatu kelas yang berhasil dikenali, sedangkan F1-score merupakan rata-rata harmonik antara precision dan recall. Metrik macro average digunakan untuk melihat performa rata-rata antar kelas tanpa mempertimbangkan jumlah data pada masing-masing kelas.

## 3.3 Analisis Confusion Matrix

Confusion matrix digunakan untuk melihat pola kesalahan klasifikasi model pada masing-masing kelas. Melalui confusion matrix, dapat diketahui kelas mana yang berhasil diklasifikasikan dengan baik dan kelas mana yang sering tertukar dengan kelas lain.

Pada eksperimen ini, kesalahan klasifikasi dapat terjadi karena beberapa penyakit daun padi memiliki karakteristik visual yang mirip. Beberapa kelas memiliki gejala berupa bercak pada daun, perubahan warna, atau area kerusakan yang secara visual sulit dibedakan. Akibatnya, model dapat salah mengenali satu kelas sebagai kelas lain, terutama jika pola bercak atau tekstur daun terlihat serupa.

Beberapa hal yang perlu dibahas saat memasukkan gambar confusion matrix:

- Kelas yang memiliki nilai diagonal tinggi berarti lebih sering diklasifikasikan dengan benar.
- Kelas yang memiliki banyak nilai di luar diagonal berarti sering tertukar dengan kelas lain.
- Kesalahan antar kelas dapat menunjukkan adanya kemiripan visual pada gejala penyakit.
- Confusion matrix membantu menjelaskan mengapa accuracy model belum terlalu tinggi.

Contoh pembahasan yang dapat digunakan:

> Berdasarkan confusion matrix, terlihat bahwa model masih mengalami kesalahan klasifikasi pada beberapa kelas penyakit. Kesalahan ini dapat disebabkan oleh kemiripan pola visual antar penyakit, seperti bercak daun, perubahan warna, dan tekstur kerusakan. Kelas-kelas dengan gejala visual yang mirip cenderung lebih sering tertukar dibandingkan kelas yang memiliki karakteristik visual lebih berbeda.

Dengan demikian, confusion matrix menunjukkan bahwa tantangan utama pada eksperimen ini bukan hanya jumlah data, tetapi juga kemiripan visual antar kelas penyakit daun padi.

## 3.4 Pembahasan Penyebab Rendahnya Accuracy

Hasil accuracy sebesar 0.6684 menunjukkan bahwa performa model masih belum optimal. Terdapat beberapa faktor yang dapat memengaruhi rendahnya accuracy model pada eksperimen ini.

Pertama, beberapa kelas penyakit daun padi memiliki kemiripan visual. Misalnya, beberapa penyakit menunjukkan gejala berupa bercak, perubahan warna, atau bagian daun yang mengering. Kemiripan ini membuat model sulit membedakan fitur antar kelas secara konsisten.

Kedua, jumlah data pada setiap kelas masih terbatas untuk kebutuhan deep learning. Walaupun total dataset berjumlah 3.829 citra, jumlah tersebut terbagi ke dalam enam kelas. Model CNN membutuhkan variasi data yang cukup besar agar dapat mempelajari pola visual dengan baik, terutama jika citra memiliki kondisi yang beragam.

Ketiga, variasi kualitas citra dapat memengaruhi proses pembelajaran model. Perbedaan pencahayaan, latar belakang, posisi daun, sudut pengambilan gambar, dan tingkat kejelasan gejala penyakit dapat membuat model lebih sulit mengenali pola yang stabil.

Keempat, arsitektur CNN Kustom yang digunakan masih relatif sederhana. Model hanya menggunakan tiga blok konvolusi utama, sehingga kemampuan ekstraksi fitur masih terbatas jika dibandingkan dengan arsitektur yang lebih dalam atau model transfer learning.

Kelima, penggunaan regularisasi seperti Dropout dan L2 regularization dapat mengurangi overfitting, tetapi juga dapat membuat model belajar lebih lambat. Pada eksperimen ini, regularisasi membantu menjaga model agar tidak terlalu menghafal data training, tetapi performa validation dan testing masih belum terlalu tinggi.

Berdasarkan faktor-faktor tersebut, rendahnya accuracy tidak dapat disebabkan oleh satu hal saja. Performa model dipengaruhi oleh kombinasi antara karakteristik dataset, kemiripan visual antar kelas, variasi citra, serta kapasitas arsitektur model.

Paragraf pembahasan yang dapat digunakan:

> Accuracy model yang belum optimal dipengaruhi oleh beberapa faktor, antara lain kemiripan visual antar kelas penyakit, keterbatasan jumlah data, variasi kualitas citra, serta arsitektur CNN Kustom yang masih sederhana. Meskipun model telah menggunakan augmentasi data, dropout, dan regularisasi L2, model masih mengalami kesulitan dalam membedakan beberapa kelas penyakit yang memiliki pola visual serupa. Hal ini menunjukkan bahwa klasifikasi penyakit daun padi berbasis citra merupakan permasalahan yang cukup kompleks dan membutuhkan pengembangan lebih lanjut.

## 3.5 Perbandingan dengan Penelitian Terdahulu

Hasil penelitian ini dapat dibandingkan dengan penelitian terdahulu yang membahas klasifikasi penyakit tanaman menggunakan CNN atau deep learning. Pada beberapa penelitian sebelumnya, accuracy yang diperoleh cenderung lebih tinggi karena menggunakan arsitektur yang lebih kompleks, transfer learning, atau dataset yang lebih besar dan lebih terkontrol.

Penelitian ini menggunakan CNN Kustom yang dilatih dari awal tanpa bobot awal dari model pretrained. Hal ini berbeda dengan pendekatan transfer learning seperti MobileNetV2, ResNet, EfficientNet, atau VGG yang telah memiliki bobot awal dari dataset besar seperti ImageNet. Model transfer learning biasanya lebih cepat memperoleh performa tinggi karena sudah memiliki kemampuan ekstraksi fitur visual dasar.

Namun, kontribusi penelitian ini adalah menunjukkan penerapan CNN Kustom sederhana untuk klasifikasi penyakit daun padi pada dataset 6 kelas tanpa Tungro. Hasil yang diperoleh dapat menjadi dasar untuk pengembangan model lebih lanjut, baik dengan meningkatkan arsitektur CNN, memperbaiki kualitas dataset, menambah jumlah data, maupun membandingkan dengan metode transfer learning.

Tabel perbandingan dengan penelitian terdahulu dapat dibuat seperti Tabel 3.4.

**Tabel 3.4 Perbandingan dengan Penelitian Terdahulu**

| Peneliti | Dataset | Metode | Jumlah Kelas | Hasil |
|---|---|---|---:|---|
| Penelitian 1 | Penyakit daun padi | CNN / Transfer Learning | Sesuaikan | Sesuaikan |
| Penelitian 2 | Penyakit tanaman | MobileNet / ResNet / EfficientNet | Sesuaikan | Sesuaikan |
| Penelitian ini | Rice Leaf AUG tanpa Tungro | CNN Kustom | 6 kelas | Accuracy 0.6684 |

Contoh pembahasan:

> Dibandingkan dengan penelitian terdahulu yang menggunakan arsitektur transfer learning atau model CNN yang lebih dalam, hasil penelitian ini masih lebih rendah. Hal tersebut dapat terjadi karena model yang digunakan pada penelitian ini merupakan CNN Kustom sederhana yang dilatih dari awal. Meskipun demikian, penelitian ini tetap memberikan gambaran bahwa CNN Kustom dapat digunakan sebagai pendekatan dasar untuk klasifikasi penyakit daun padi, serta menunjukkan beberapa tantangan yang perlu diperhatikan dalam pengembangan model klasifikasi citra penyakit tanaman.

Secara keseluruhan, hasil eksperimen menunjukkan bahwa CNN Kustom mampu mempelajari sebagian pola visual penyakit daun padi, tetapi performanya masih perlu ditingkatkan. Pengembangan selanjutnya dapat dilakukan dengan menambah data, memperbaiki kualitas citra, menerapkan preprocessing yang lebih baik, memperdalam arsitektur CNN, atau membandingkan hasil dengan model transfer learning.

