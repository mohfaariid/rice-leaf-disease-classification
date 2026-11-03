# Catatan Paper: Advancing Green AI - Efficient and Accurate Lightweight CNNs untuk Klasifikasi Penyakit Daun Padi

Dokumen ini berisi catatan bacaan dari paper yang relevan dengan penelitian perbandingan CNN Kustom dan MobileNetV2 untuk klasifikasi penyakit daun padi.

---

## 1. Identitas Paper

**Judul:**

```
Advancing Green AI: Efficient and Accurate Lightweight CNNs for Rice Leaf Disease Identification
```

**Penulis:**

```
Khairun Saddami, Yudha Nurdin, Mutia Zahramita, Muhammad Shahreeza Safiruz
```

**Institusi:**

```
a) Department of Electrical and Computer Engineering, Faculty of Engineering, Universitas Syiah Kuala
b) Telematic Research Centers, Universitas Syiah Kuala
c) Department of Artificial Intelligence, Faculty of Computer Science and Information Technology, 
   University of Malaya
```

**Jurnal/Konferensi:**

```
arXiv Preprint, 2024
```

**Topik Utama:**

```
Green AI, lightweight CNNs, mobile-compatible models, rice leaf disease classification, EfficientNet, 
MobileNetV2, ShuffleNet, efficient neural networks, sustainability.
```

---

## 2. Ringkasan Singkat Paper

Paper ini menginvestigasi penerapan tiga arsitektur CNN berbasis mobile, yaitu **ShuffleNet**, **MobileNetV2**, dan **EfficientNet-B0**, untuk klasifikasi penyakit daun padi.

Penelitian ini menggunakan dataset dari Kaggle yang terdiri dari empat kelas: Brown Spot, Healthy, Hispa, dan Leaf Blast. Dataset original berisi 3.355 gambar, kemudian didownsample menjadi 2.092 gambar dengan 523 gambar per kelas untuk memastikan dataset yang seimbang.

Setiap model ditingkatkan performanya dengan menambahkan dua fully connected layers yang dipisahkan oleh dropout layer. Teknik early stopping diterapkan untuk mencegah overfitting.

**Hasil utama:**
- EfficientNet-B0: Accuracy 99.52%, Precision 99.58%, Recall 99.59%, F1-score 99.58%
- MobileNetV2: Accuracy 84.21%, Precision 84.35%, Recall 84.23%, F1-score 84.13%
- ShuffleNet: Accuracy 66.03%, Precision 65.78%, Recall 65.64%, F1-score 65.56%

Paper menekankan pentingnya "Green AI" - mengembangkan model AI dengan akurasi tinggi namun dengan computational requirements yang rendah untuk mengurangi dampak lingkungan dan menjaga keberlanjutan.

---

## 3. Masalah yang Dibahas Paper

**Masalah Utama:**

1. **Deteksi Penyakit Padi yang Akurat dan Cepat:**
   - Penyakit daun padi (Brown Spot, Hispa, Leaf Blast) dapat mengurangi yield dan kualitas panen secara signifikan
   - Deteksi dini penting untuk mencegah penyebaran dan meminimalkan kerugian
   - Sistem deteksi otomatis berbasis machine learning dapat membantu petani mengambil tindakan cepat

2. **Model Deep Learning yang Berat dan Tidak Efisien:**
   - Penelitian terdahulu fokus pada akurasi tinggi tanpa mempertimbangkan efisiensi komputasi
   - Contoh: VGG16 (144 juta parameter, 535 MB), ResNet50 (25 juta parameters), InceptionV3 (24 juta parameters)
   - Model besar tidak cocok untuk deployment di mobile devices atau IoT systems

3. **Kebutuhan Green AI untuk Sustainability:**
   - Model besar memerlukan banyak sumber daya komputasi (processing power, memory, energy)
   - Data centers yang menjalankan model besar mengonsumsi listrik besar dan menghasilkan emisi gas rumah kaca
   - Perlu mengembangkan model yang akurat namun lightweight untuk mengurangi dampak lingkungan

4. **Gap dalam Penelitian:**
   - Fokus penelitian terdahulu masih pada mencapai akurasi tertinggi, bukan pada efisiensi model
   - Transformer/ViT models mencapai akurasi tinggi tapi sangat besar dan tidak praktis untuk deployment
   - Perlu komparasi sistematis antara lightweight models yang cocok untuk mobile deployment

---

## 4. Metode yang Digunakan

### 4.1 Dataset

**Sumber:** Kaggle (Rice Leaf Disease Dataset)

**Kelas:** 4 kelas
- Brown Spot
- Healthy
- Hispa
- Leaf Blast

**Ukuran Dataset:**
- Original: 3.355 gambar
- Setelah downsampling: 2.092 gambar
- Per kelas: 523 gambar (balanced dataset)

**Preprocessing & Augmentasi:**
- Resize ke ukuran standar
- Augmentasi: rotation (30°), zoom (0.15), width/height shift (0.2), shear (0.15), horizontal flip

### 4.2 Arsitektur Model

**Tiga model lightweight yang dievaluasi:**

1. **EfficientNet-B0**
   - Menggunakan "compound scaling" untuk mengoptimalkan width, depth, dan resolution secara simultaneous
   - Depthwise separable convolution untuk mengurangi parameters
   - Scaling factor terkecil dalam family EfficientNet

2. **MobileNetV2**
   - Depthwise separable convolutions
   - Bottlenecks dan inverted residuals
   - Linear bottlenecks dengan channel attention mechanism
   - Dirancang untuk balance akurasi dan efisiensi

3. **ShuffleNet**
   - Pointwise group convolutions untuk mengurangi parameters
   - Channel shuffling untuk memfasilitasi information flow antar group
   - Fokus pada efisiensi ekstrem dengan parameter minimal

**Custom Layer Addition:**
Untuk meningkatkan performance, setiap model ditambahkan:
- Fully Connected layer (128 features) 
- Dropout layer (0.3)
- Fully Connected layer lagi (untuk 4 kelas)
- L2 regularization (lambda = 10^-4)

### 4.3 Training Configuration

**Hyperparameters:**
- Image size: Standard size (disesuaikan dengan model)
- Batch size: 32
- Epochs: 200 dan 300 (dengan early stopping di epoch 15)
- Optimizer: Adam (adaptive moment estimation)
- Loss function: Categorical cross entropy
- Learning rate: Variated dari 10^-1 hingga 10^-5

**Teknik Regularisasi:**
- Early stopping: Menghentikan training ketika validation loss tidak menurun selama 15 epoch berturut-turut
- Dropout: 0.3
- L2 regularization: 10^-4

### 4.4 Evaluation Metrics

**Metrik Akurasi:**
- Accuracy
- Precision
- Recall
- F-measure (F1-score)

**Metrik Efisiensi:**
- Learning parameters (jumlah parameters model)
- Energy usage (normalized relative energy consumption)

---

## 5. Hasil Utama Paper

### 5.1 Training Results Analysis

Paper menunjukkan learning curves untuk ketiga model dengan berbagai learning rates (10^-1 hingga 10^-5):

**EfficientNet-B0:**
- Learning rate 10^-4 dengan early stopping: konvergensi di epoch 285
- Menunjukkan hasil stabil dan mencapai akurasi tertinggi

**MobileNetV2:**
- Learning rate 10^-4 dengan early stopping: konvergensi di epoch 112
- Menunjukkan sensitivity terhadap durasi training

**ShuffleNet:**
- Learning rate 10^-3 dengan early stopping
- Paling cepat converge namun dengan accuracy terendah

### 5.2 Testing Results

**Tabel Performa (dengan 300 epochs dan early stopping):**

| Model | Accuracy | Recall | Precision | F-Measure | Learning Parameters |
|-------|----------|--------|-----------|-----------|---------------------|
| EfficientNet-B0 | 99.52% | 99.59% | 99.58% | 99.58% | 5.4 M |
| MobileNetV2 | 84.21% | 84.23% | 84.35% | 84.13% | 3.5 M |
| ShuffleNet | 66.03% | 65.64% | 65.78% | 65.56% | 1.4 M |

**EfficientNet-B0:**
- Exceptional performance dengan semua metrics di atas 99.5%
- Perbedaan margin kecil sebelum dan sesudah early stopping menunjukkan robustness
- Akurasi tertinggi namun dengan parameter count moderate (5.4M)

**MobileNetV2:**
- Moderate performance dengan akurasi 84.21%
- Balance antara recall dan precision yang konsisten
- Signifikan impact dari early stopping (turun dari 84.21% menjadi 79.90%)
- Menunjukkan MobileNetV2 memerlukan full training epochs untuk optimal results

**ShuffleNet:**
- Akurasi terendah pada 66.03%
- Lightweight dengan parameters paling minimal (1.4M)
- Early stopping sedikit meningkatkan performa (66.03% → 66.51%)
- Paling cocok untuk resource-constrained environments

### 5.3 Comparison with Related Work

**Komparasi dengan model lain:**

| Model | Accuracy | Learning Parameters | Energy Used |
|-------|----------|---------------------|-------------|
| VGG16 | 91.66% | 138 M | 1.00 |
| Xception | 71.95% | 27.3 M | 0.20 |
| ResNet50 | 91.90% | 25 M | 0.18 |
| GhostNet | 89.30% | 5.2 M | 0.04 |
| **EfficientNet-B0** | **99.52%** | **5.4 M** | **0.04** |
| MobileNetV2 | 84.21% | 3.5 M | 0.03 |
| ModLeafNet | 87.76% | 1.3 M | 0.01 |
| ShuffleNet | 66.03% | 1.4 M | 0.01 |

**Key Findings:**
- EfficientNet-B0 mencapai akurasi tertinggi (99.52%) dengan parameter count moderate
- VGG16 dan ResNet50 memiliki akurasi tinggi namun dengan parameter yang sangat besar
- ShuffleNet dan ModLeafNet memiliki akurasi terendah tapi parameter minimal
- EfficientNet-B0 menunjukkan optimal balance antara performance dan efficiency

---

## 6. Keterbatasan Paper

1. **Dataset Terbatas:**
   - Hanya 4 kelas penyakit (Brown Spot, Healthy, Hispa, Leaf Blast)
   - Dataset final hanya 2.092 gambar setelah downsampling
   - Tidak mencakup semua jenis penyakit padi yang ada di lapangan

2. **Hyperparameter Tuning:**
   - Tuning parameters dilakukan pada Kaggle dataset saja
   - Optimal learning rate dan epoch count mungkin berbeda untuk dataset lain
   - Dropout value (0.3) dan L2 regularization (10^-4) tidak dijelaskan bagaimana cara menentukannya

3. **Evaluasi Time Series:**
   - Paper tidak menjelaskan detail mengenai train-test split
   - Tidak ada discussion tentang cross-validation
   - Unclear apakah ada temporal consideration dalam pembagian data

4. **Early Stopping Implementation:**
   - Early stopping menggunakan patience 15 epoch - tidak dijelaskan alasan pemilihan angka ini
   - Impact signifikan pada MobileNetV2 menunjukkan perlu lebih careful tuning

5. **Missing Ablation Study:**
   - Tidak ada ablation study untuk custom FC+Dropout+FC layer yang ditambahkan
   - Kontribusi masing-masing component tidak terlihat jelas

6. **Limited Real-World Testing:**
   - Dataset dari Kaggle adalah dataset controlled
   - Tidak ada testing pada real farmer-captured images dengan variasi lighting, angle, disease severity

7. **Energy Usage Definition:**
   - "Energy usage" adalah relative normalized value, bukan actual energy consumption
   - Actual power consumption di production environment tidak diukur

8. **Green AI Claims:**
   - Paper klaim Green AI tapi tidak ada actual carbon footprint atau energy measurement
   - Training dilakukan pada NVIDIA RTX 3090 dengan 64GB RAM - setup yang expensive

---

## 7. Relevansi dengan Penelitian Anda

Paper ini **sangat relevan** dengan penelitian Anda karena:

### 7.1 Kesamaan Topik
- ✅ Sama-sama fokus pada klasifikasi penyakit daun padi menggunakan deep learning
- ✅ Menggunakan CNN sebagai base model (seperti CNN Kustom Anda)
- ✅ Mengevaluasi model lightweight untuk mobile deployment (seperti interest Anda pada MobileNetV2)
- ✅ Dataset dari Kaggle dengan multiple disease classes

### 7.2 Kesamaan Metodologi
- ✅ Menggunakan pre-trained models (transfer learning approach untuk EfficientNet-B0 dan MobileNetV2)
- ✅ Mengevaluasi dengan multiple metrics: accuracy, precision, recall, F1-score
- ✅ Data augmentation untuk improve training
- ✅ Early stopping untuk prevent overfitting

### 7.3 Perbedaan dan Kontribusi Unik Penelitian Anda
- 📍 Penelitian Anda fokus pada **perbandingan CNN Kustom vs MobileNetV2**, bukan 3-way comparison
- 📍 Dataset Anda: **7 kelas penyakit** (lebih comprehensive) vs paper ini 4 kelas
- 📍 Anda fokus pada **transfer learning mechanism** dan bagaimana pre-trained knowledge membantu
- 📍 Anda menggunakan **dataset augmented (Rice_Leaf_AUG)** dengan 5.137 citra vs paper 2.092
- 📍 Anda fokus pada **baseline vs state-of-the-art** comparison, paper ini fokus pada Green AI sustainability

---

## 8. Perbedaan Detail dengan Penelitian Anda

| Aspek | Paper: Green AI Lightweight CNNs | Penelitian Anda |
|-------|----------------------------------|-----------------|
| **Model Utama** | EfficientNet-B0, MobileNetV2, ShuffleNet | CNN Kustom, MobileNetV2 |
| **Jumlah Kelas** | 4 kelas penyakit | 7 kelas (6 penyakit + 1 healthy) |
| **Dataset Size** | 2.092 gambar (downsampled) | 5.137 gambar (Rice_Leaf_AUG) |
| **Model Comparison Focus** | Lightweight/Green AI perspective | Custom vs Transfer Learning |
| **Pre-training Approach** | ImageNet weights untuk all models | ImageNet weights hanya untuk MobileNetV2 |
| **Evaluation Metrics** | Accuracy, Precision, Recall, F1, Energy | Accuracy, Precision, Recall, F1, Classification Report, Confusion Matrix |
| **Primary Metric** | Accuracy | F1-score macro (due to imbalanced dataset) |
| **Main Contribution** | Green AI, sustainability aspect | Comparative performance analysis |

---

## 9. Kalimat yang Bisa Ditulis di BAB 1 (Latar Belakang)

```
Perkembangan teknologi deep learning telah membuka peluang bagi pengembangan sistem 
deteksi penyakit padi yang akurat dan efisien. Meskipun model-model besar seperti VGG16 
dan ResNet50 dapat mencapai akurasi tinggi dalam klasifikasi penyakit daun padi, mereka 
memerlukan sumber daya komputasi yang signifikan, membuatnya kurang cocok untuk deployment 
pada perangkat mobile atau IoT di tingkat petani (Saddami et al., 2024).
```

```
Pendekatan lightweight CNN architecture, seperti MobileNetV2 dan EfficientNet-B0, 
menawarkan alternatif yang seimbang antara akurasi tinggi dan efisiensi komputasi. 
Arsitektur-arsitektur ini dirancang khusus untuk mengurangi jumlah parameter dan 
computational requirements sambil mempertahankan performa yang kompetitif (Saddami et al., 2024).
```

```
Penelitian yang membandingkan secara sistematis performa model CNN Kustom dengan model 
transfer learning seperti MobileNetV2 masih diperlukan untuk memahami trade-off antara 
kompleksitas model, jumlah data training, dan akurasi klasifikasi pada kasus deteksi 
penyakit daun padi.
```

---

## 10. Kalimat yang Bisa Ditulis di BAB 2 (Tinjauan Pustaka)

### Subtopic: Lightweight CNN Architectures

```
Dalam penelitian terbaru, Saddami et al. (2024) mengevaluasi tiga lightweight CNN 
architecture - EfficientNet-B0, MobileNetV2, dan ShuffleNet - untuk klasifikasi penyakit 
daun padi. Penelitian tersebut menggunakan dataset Kaggle dengan 4 kelas penyakit dan 
menunjukkan bahwa EfficientNet-B0 mencapai akurasi 99.52% dengan 5.4 juta parameters, 
sementara MobileNetV2 mencapai 84.21% dengan 3.5 juta parameters, dan ShuffleNet 
mencapai 66.03% dengan 1.4 juta parameters.
```

### Subtopic: Green AI and Sustainability

```
Konsep "Green AI" menekankan pengembangan model artificial intelligence yang memiliki 
akurasi tinggi namun dengan computational requirements yang rendah (Saddami et al., 2024). 
Pendekatan ini relevan untuk agriculture technology, di mana deployment pada mobile devices 
dan IoT systems memerlukan model yang lightweight. Dengan mengurangi computational resources 
yang diperlukan, model yang lebih efficient dapat berkontribusi pada pengurangan energy 
consumption dan greenhouse gas emissions.
```

### Subtopic: Model Architecture Selection

```
Pemilihan architecture model harus mempertimbangkan balance antara akurasi dan efisiensi. 
Saddami et al. (2024) menunjukkan bahwa EfficientNet-B0 memberikan optimal trade-off 
dengan akurasi 99.52% dan parameter count yang moderate (5.4M), sementara model yang lebih 
lightweight seperti ShuffleNet (1.4M parameters) memberikan trade-off ekstrem dengan akurasi 
terendah (66.03%) namun paling cocok untuk resource-constrained environments.
```

---

## 11. Kalimat untuk Gap Penelitian

```
Meskipun penelitian terdahulu telah menunjukkan bahwa model lightweight seperti MobileNetV2 
dapat digunakan untuk klasifikasi penyakit padi dengan akurasi yang cukup baik, kajian yang 
secara detail membandingkan performa CNN Kustom (dilatih dari awal tanpa pre-trained weights) 
dengan MobileNetV2 berbasis transfer learning pada dataset yang sama, dengan kelas penyakit 
yang lebih comprehensive (7 kelas), dan dengan metrik evaluasi yang lebih detailed masih perlu 
dikembangkan.
```

```
Selain itu, understanding tentang bagaimana transfer learning mechanism bekerja dalam konteks 
klasifikasi penyakit padi, dan bagaimana feature extraction dari ImageNet pre-trained weights 
dapat diadaptasi untuk domain penyakit padi yang spesifik, masih memerlukan eksplorasi lebih 
lanjut.
```

---

## 12. Informasi Penting yang Perlu Dicek Lebih Lanjut

Sebelum menggunakan paper ini sebagai rujukan utama, hal berikut perlu diverifikasi:

- [ ] Detail tentang bagaimana test set dipilih dan ukurannya
- [ ] Apakah ada cross-validation atau hanya single train-test split
- [ ] Bagaimana early stopping patience (15 epochs) ditentukan - ada justification?
- [ ] Apakah custom FC+Dropout+FC layer addition memberikan significant improvement (ada ablation study?)
- [ ] Actual energy consumption measurement atau hanya normalized calculation?
- [ ] Robustness terhadap different image qualities, angles, lighting conditions?

---

## 13. Kontribusi dan Insight dari Paper

### Insights yang Valuable untuk Penelitian Anda:

1. **EfficientNet-B0 sebagai Strong Baseline:**
   - Paper menunjukkan EfficientNet-B0 mencapai 99.52% accuracy
   - Namun accuracy yang sangat tinggi ini perlu diinterpretasi dengan hati-hati
   - Anda bisa bandingkan MobileNetV2 di research Anda dengan EfficientNet-B0 sebagai reference

2. **MobileNetV2 Performance Profile:**
   - Paper: MobileNetV2 84.21% dengan 3.5M params
   - Anda mungkin akan dapatkan hasil berbeda karena 7 classes (vs paper 4 classes)
   - Impact early stopping pada MobileNetV2 (turun 4.31%) adalah insight penting

3. **Lightweight Model Trade-offs:**
   - ShuffleNet dengan 1.4M params hanya 66.03% accuracy - too low untuk practical use
   - Balance point ada di MobileNetV2 dan EfficientNet-B0
   - Penelitian Anda fokus pada useful balance point

4. **Dataset Quality Matters:**
   - Paper downsampling dari 3.355 ke 2.092 untuk balanced dataset
   - Anda punya 5.137 images dalam Rice_Leaf_AUG
   - Larger dataset Anda mungkin memberikan lebih better generalization

5. **Custom Layer Addition:**
   - FC(128) + Dropout(0.3) + FC(4) ditambahkan ke all models
   - Ini adalah common practice untuk adapt pre-trained model untuk specific task
   - Anda juga bisa apply similar approach untuk MobileNetV2 Anda

---

## 14. Kesimpulan untuk Penelitian Anda

Paper ini dapat digunakan sebagai:

✅ **Referensi Strong untuk:**
- State-of-the-art lightweight models untuk rice disease classification
- Trade-off analysis antara accuracy dan model size/computational efficiency
- Metodologi perbandingan model (metrics, evaluation approach)
- Diskusi tentang importance dari model efficiency dan Green AI

⚠️ **HATI-HATI UNTUK:**
- Direct benchmark comparison karena dataset berbeda (paper 4 classes, Anda 7 classes)
- Hyperparameter tuning values mungkin tidak cocok untuk dataset Anda
- Accuracy values yang sangat tinggi (99.52%) perlu critical review

🎯 **Posisi Best dalam Penelitian Anda:**
- Sebagai **state-of-the-art reference** untuk lightweight architectures
- Sebagai **comparison point** untuk memahami bagaimana MobileNetV2 perform pada rice disease dataset
- Sebagai **methodological reference** untuk evaluation metrics dan model comparison approach
- Sebagai **context** untuk diskusi tentang importance dari computational efficiency dalam agricultural AI

---

## 15. Catatan Tambahan & Rekomendasi

### Untuk Penulisan Skripsi Anda:

1. **Dalam BAB 1 (Latar Belakang):**
   - Gunakan paper ini untuk context tentang importance dari efficient models
   - Mention bahwa lightweight models seperti MobileNetV2 sudah proven effective

2. **Dalam BAB 2 (Related Works / State of the Art):**
   - Gunakan sebagai main reference untuk lightweight CNN section
   - Bandingkan dengan older works yang menggunakan larger models
   - Highlight trade-off analysis antara accuracy dan efficiency

3. **Dalam BAB 3 (Metodologi):**
   - Justifikasi pemilihan MobileNetV2 dapat merujuk pada paper ini
   - Custom layer addition juga ada precedent di paper ini

4. **Dalam BAB 4 (Hasil & Pembahasan):**
   - Gunakan untuk comparison point
   - Highlight perbedaan hasil Anda dengan paper (7 classes vs 4 classes)

5. **Dalam BAB 5 (Kesimpulan):**
   - Diskusikan bagaimana Anda contribute beyond this paper
   - Highlight unique aspects dari penelitian Anda

### Critical Notes:

🔴 **Accuracy 99.52% seems too high** - likely ada:
- Dataset that's relatively easy (4 healthy vs diseased)
- Potential data leakage
- Biased evaluation
- Very different augmentation/preprocessing
→ Anda perlu critical assessment jika result Anda jauh lebih rendah

🟡 **Green AI aspect** tidak prioritas untuk skripsi Anda
- Paper emphasize sustainability
- Anda fokus pada comparative performance
- Bisa mention tapi jangan jadi main focus

---

**Status Screening:** ✅ COMPLETE  
**Relevance Score:** 8/10 (High relevance, but different scope)  
**Recommendation:** ✅ USE as reference for lightweight models and state-of-the-art comparison
