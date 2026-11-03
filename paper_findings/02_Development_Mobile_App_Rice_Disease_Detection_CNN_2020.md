# Catatan Paper: Development of a Prototype Application untuk Rice Disease Detection Menggunakan CNN

Dokumen ini berisi catatan bacaan dari paper yang relevan dengan penelitian pengembangan sistem deteksi penyakit daun padi menggunakan deep learning.

---

## 1. Identitas Paper

**Judul:**

```
Development of a Prototype Application for Rice Disease Detection Using Convolutional Neural Networks
```

**Penulis:**

```
Harold Costales, Arpee Callejo-Arruejo, Noel Rafanan
```

**Institusi:**

```
University of Northern Philippines, Philippines
```

**Jurnal/Konferensi:**

```
International Journal of Emerging Trends in Engineering Research (IJETER)
Vol. 8, No. 10, October 2020, Pages 7076-7081
DOI: https://doi.org/10.30534/ijeter/2020/708102020
```

**Topik Utama:**

```
Rice disease detection, convolutional neural networks, transfer learning, mobile application, 
MobileNetV2, image analysis, agricultural technology, Feature-Driven Development, 
real-time detection.
```

---

## 2. Ringkasan Singkat Paper

Paper ini mendeskripsikan pengembangan aplikasi mobile prototype untuk deteksi penyakit daun padi menggunakan CNN dan transfer learning. Penelitian ini difokuskan pada solusi praktis untuk petani Filipino yang memerlukan cara cepat dan akurat untuk mendeteksi penyakit pada padi mereka.

**Dataset:**
- Sumber: Kaggle (Rice Disease Image Dataset by Huy Minh Do)
- Total: 1.260 labeled rice leaf images
- Kelas: 3 jenis penyakit
  - Leaf Blast
  - Brown Spot
  - Hispa

**Metodologi:**
- Preprocessing: Downscale ke 500x500 pixels
- Data split: 80% training, 20% validation
- Augmentation: Flipping, shearing, rescaling
- Base model: MobileNetV2 dengan ImageNet pre-trained weights
- Development approach: Feature-Driven Development (FDD) methodology

**Hasil:**
- Iterasi 1 (baseline): Top layer frozen, result tidak optimal
- Iterasi 2 (ImageNet weights): 97% training accuracy, 94% validation (overfitting)
- Iterasi 3 (fine-tuning + augmentation): **98.9% training accuracy, 98% validation** ✅ BEST

Paper menekankan pentingnya transfer learning dan fine-tuning untuk meningkatkan performa model pada domain-specific problem (rice disease classification).

---

## 3. Masalah yang Dibahas Paper

### 3.1 Masalah Praktis di Pertanian
1. **Ketidaktahuan Petani:** Banyak petani Filipino tidak familiar dengan berbagai tipe penyakit daun padi
2. **Kerugian Ekonomi:** Bacterial Leaf Blight dapat mengurangi yield hingga 75% atau bahkan crop loss total
3. **Kurangnya Expertise:** Ahli patologi tanaman terbatas, terutama di daerah rural
4. **Keterlambatan Deteksi:** Deteksi manual lambat dan tidak reliable, menyebabkan penyebaran penyakit
5. **Biaya Tinggi:** Konsultasi dengan ahli mahal dan tidak terjangkau untuk petani kecil

### 3.2 Solusi yang Diperlukan
Paper mengidentifikasi bahwa teknologi ICT dan mobile applications dapat menjadi solusi:
- Bawa expertise ke lapangan melalui mobile app
- Deteksi real-time tanpa perlu konsultasi expert
- Affordable dan accessible untuk petani kecil
- Informative dan educational untuk petani

### 3.3 Gap dalam Penelitian Terdahulu
- Penelitian existing fokus pada accuracy tapi tidak pada practical deployment
- Tidak ada banyak aplikasi yang spesifik untuk Philippine agricultural context
- Need untuk mobile-based solution yang affordable

---

## 4. Metode yang Digunakan

### 4.1 Dataset Preparation

**Source:** Kaggle (Rice Disease Image Dataset)

**Kelas penyakit:** 3 kelas
- Leaf Blast
- Brown Spot  
- Hispa

**Ukuran:** 1.260 labeled images

**Preprocessing:**
- Downscale semua gambar ke 500×500 pixels (untuk mengurangi computational complexity)
- Data split: 80% training, 20% validation
- Augmentation techniques:
  - Flipping (horizontal flip)
  - Shearing
  - Rescaling
  - Purpose: Increase dataset size dan improve model generalization

### 4.2 Deep Learning Algorithm: Convolutional Neural Networks (CNN)

**Arsitektur CNN Typical (sesuai Tajbakhsh 2016):**
1. Input layer (images)
2. Convolutional layer (automatic feature extraction)
   - Matrix operation (convolution)
   - Produces feature maps
3. Pooling layer (downsampling untuk reduce computational complexity)
   - Max pooling atau average pooling
   - Max pooling is preferred
4. Fully Connected layer (vectorize features)
   - Transform multi-dimensional feature map ke row vector
5. Classifier (output probabilities)
   - Activation function: Softmax (untuk multi-class classification)
6. Output layer (4 classes + disease probabilities)

### 4.3 Transfer Learning dengan MobileNetV2

**Base Model:** MobileNetV2
- Pre-trained pada ImageNet dataset
- Winner of ImageNet Large Scale Visual Recognition Challenge (ILSVRC)
- State-of-the-art CNN model

**Rationale untuk MobileNetV2:**
- Target dataset kecil (1.260 images)
- Domain spesifik (rice disease) berbeda dari ImageNet
- Need untuk frozen top layer pada initial training

### 4.4 Training Iterations

**Iterasi 1 - Baseline:**
- Base model: MobileNetV2
- Top layer: False (frozen)
- Initial weights: Random
- Result: Poor performance (not specified)

**Iterasi 2 - With ImageNet Weights:**
```
Hyperparameters:
- Base model: MobileNetV2
- Top layer: False (frozen)
- Initial weights: ImageNet
- Loss: Categorical cross entropy
- Optimizer: ADAM
- Metrics: Accuracy
- Epochs: 10
- Trainable: FALSE (all layers frozen)

Results:
- Training accuracy: 97%
- Validation accuracy: 94%
- Issue: OVERFITTING detected
```

**Iterasi 3 - Fine-tuning + Augmentation:**
```
Hyperparameters:
- Base model: MobileNetV2
- Image augmentation: YES (applied)
- Top layer: False
- Initial weights: ImageNet
- Loss: Categorical cross entropy
- Optimizer: ADAM
- Metrics: Accuracy
- Epochs: 20
- Trainable: TRUE (allow fine-tuning) ← KEY CHANGE

Results:
- Training accuracy: 98.9%
- Validation accuracy: 98%
- Issue: Overfitting SOLVED ✅
```

**Key Improvement:** 
- Menaikkan trainable dari FALSE ke TRUE (fine-tuning)
- Meningkatkan epochs dari 10 ke 20
- Menambahkan image augmentation
- Hasil: validation accuracy 94% → 98% (+4%)

### 4.5 Software Development Methodology

**Approach:** Feature-Driven Development (FDD)

FDD dipilih karena:
- Customer-centric process
- Iterative development approach
- Suitable untuk agile methodology

**5 Phases:**
1. Develop Overall Model - Create fundamental foundation
2. Build Feature List - Identify features for mobile app
3. Plan by Feature - Concrete planning for each feature
4. Design by Feature - Design individual features
5. Build by Feature - Implement the planned and designed features

---

## 5. Hasil Utama Paper

### 5.1 Model Performance

**Best Result (Iterasi 3):**
- Training accuracy: **98.9%**
- Validation accuracy: **98%**
- Model: MobileNetV2 dengan fine-tuning
- Dataset: 1.260 rice leaf images (3 classes)

**Comparison across iterations:**
- Iterasi 2: 94% validation (overfitting, poor generalization)
- Iterasi 3: 98% validation (balanced, good generalization)

### 5.2 Key Insights dari Training Process

1. **Importance of Fine-tuning:**
   - Iterasi 2 (all layers frozen): 94% validation
   - Iterasi 3 (trainable=true): 98% validation
   - Fine-tuning improves domain adaptation

2. **Overfitting Problem & Solution:**
   - Iterasi 2: Training 97% vs Validation 94% → 3% gap (overfitting)
   - Iterasi 3: Training 98.9% vs Validation 98% → 0.9% gap (almost no overfitting)
   - Solution: Image augmentation + more epochs

3. **Image Augmentation Effect:**
   - Critical untuk meningkatkan model robustness
   - Membantu prevent overfitting
   - Meningkatkan generalization capability

### 5.3 Mobile Application Features

Paper mengembangkan prototype aplikasi dengan 3 fitur utama:

**1. Real-time Detection**
- Menggunakan embedded camera dari smartphone
- Input langsung dari user: foto daun padi yang dicurigai sakit
- Output: Probabilitas penyakit dalam real-time
- Benefit: Farmers bisa deteksi tanpa pathologist

**2. Artificial Intelligence**
- Model CNN yang terus belajar seiring bertambahnya data
- Semakin banyak data → semakin akurat hasil
- Mereplikasi knowledge dari agriculture experts

**3. Image Analysis**
- Menggunakan CNN algorithm untuk analyze leaf images
- Feature extraction dari gambar
- Classification berdasarkan extracted features

### 5.4 Application Context

**Target users:** Farmers di Northern Philippines
**Primary goal:** Early detection dan prevention measures
**Value proposition:**
- Virtual expert di lapangan
- Reduce crop losses dari 75% to manageable level
- Educational untuk farmers
- Cost-effective alternative to hiring experts

---

## 6. Keterbatasan Paper

### 6.1 Dataset Limitations

1. **Ukuran Dataset Kecil:**
   - Hanya 1.260 images (dibanding research lain dengan 3.000+)
   - Untuk deep learning, 1.260 adalah relatively small

2. **Limited Classes:**
   - Hanya 3 disease classes (Leaf Blast, Brown Spot, Hispa)
   - Tidak termasuk healthy rice leaf class untuk reference
   - Tidak cover semua rice diseases di lapangan

3. **Single Source:**
   - Dataset hanya dari Kaggle (Huy Minh Do)
   - Tidak ada mention tentang bagaimana images dikumpulkan
   - Unclear tentang image quality, diversity, dan consistency

### 6.2 Methodological Issues

1. **Limited Hyperparameter Tuning:**
   - Hanya 3 iterations dengan parameter variations terbatas
   - Learning rate tidak di-specify atau di-tune
   - Batch size tidak di-mention

2. **No Cross-Validation:**
   - Simple train-val split (80-20), no k-fold cross-validation
   - Single split bisa biased

3. **Missing Ablation Study:**
   - Tidak clear contribution dari masing-masing component:
     - Berapa improvement dari fine-tuning saja?
     - Berapa improvement dari augmentation saja?
     - Berapa improvement dari additional epochs?

4. **No Test Set Evaluation:**
   - Paper hanya report training dan validation accuracy
   - Tidak ada dedicated test set untuk final evaluation
   - Cannot assess true generalization performance

### 6.3 Transfer Learning Details

1. **Shallow Fine-tuning:**
   - Paper tidak specify yang layers di-unfreeze
   - Mungkin hanya top layers yang trainable, tidak seluruh network

2. **ImageNet Domain Gap:**
   - ImageNet adalah natural objects, rice disease sangat different
   - Paper tidak discuss berapa effectivenya transfer learning untuk ini

### 6.4 Model Architecture Details

1. **Top Classifier Architecture:**
   - Paper tidak specify custom layers ditambahkan di atas MobileNetV2
   - Tidak ada FC layers specification
   - Tidak ada regularization (dropout, L2) mentioned

2. **Loss Function & Metrics:**
   - Hanya mention categorical cross entropy
   - No other metrics beyond accuracy

### 6.5 Real-world Deployment Issues

1. **No Real-world Testing:**
   - Hanya tested pada Kaggle dataset
   - No testing dengan farmer-taken photos
   - No testing dengan different lighting, angles, disease severity

2. **Model Size & Latency:**
   - MobileNetV2 adalah lightweight, tapi exact model size not mentioned
   - No inference latency reported
   - Important untuk mobile deployment!

3. **Validation Plan:**
   - Paper recommend validation oleh pathologists tapi tidak describe process
   - No detail tentang how farmers akan use app after deployment

### 6.6 Research Design Issues

1. **Limited Context:**
   - Paper written dalam 2020 (early days of mobile ML)
   - Technology sudah berkembang significantly sejak itu

2. **No Comparison:**
   - Tidak ada comparison dengan other methods or baseline models
   - Tidak ada comparison dengan existing apps

3. **Short Paper:**
   - Conference paper dengan 6 pages saja
   - Limited depth dan detail dibanding journal paper

---

## 7. Relevansi dengan Penelitian Anda

Paper ini **relevan tapi dengan scope berbeda** dari penelitian Anda:

### 7.1 Kesamaan Topik ✅
- ✅ Sama-sama fokus pada rice disease classification menggunakan CNN
- ✅ Menggunakan MobileNetV2 sebagai salah satu model
- ✅ Transfer learning approach
- ✅ Dataset dari Kaggle dengan multiple disease classes
- ✅ Practical application focus

### 7.2 Kesamaan Metodologi ✅
- ✅ Preprocessing: Downscale, augmentation
- ✅ Data split: Train-val (atau train-val-test)
- ✅ Using pre-trained models (ImageNet weights)
- ✅ Fine-tuning approach
- ✅ Evaluation: Accuracy, training vs validation comparison

### 7.3 Perbedaan Signifikan 🔴
1. **Focus Area:**
   - Paper ini: Development of mobile application prototype
   - Anda: Comparative analysis of CNN Kustom vs MobileNetV2

2. **Model Comparison:**
   - Paper: Single model (MobileNetV2) dengan iterations
   - Anda: Two models (Custom CNN vs MobileNetV2) dengan direct comparison

3. **Dataset:**
   - Paper: 1.260 images, 3 classes
   - Anda: 5.137 images, 7 classes (more comprehensive)

4. **Evaluation Depth:**
   - Paper: Only training & validation accuracy
   - Anda: Comprehensive evaluation (accuracy, precision, recall, F1, confusion matrix, classification report)

5. **Methodological Rigor:**
   - Paper: Conference paper dengan limited detail
   - Anda: Thesis dengan more rigorous evaluation

6. **Research Question:**
   - Paper: "Can CNN detect rice diseases on mobile?"
   - Anda: "Which model is better: Custom CNN or MobileNetV2? And why?"

---

## 8. Perbedaan Detail dengan Penelitian Anda

| Aspek | Paper: Mobile App Rice Disease | Penelitian Anda |
|-------|-------------------------------|-----------------|
| **Primary Focus** | Mobile app development | Model comparison & analysis |
| **Development Methodology** | FDD (software dev) | Experimental research (ML) |
| **Models Evaluated** | 1 model (MobileNetV2 only) | 2 models (Custom CNN, MobileNetV2) |
| **Dataset Size** | 1.260 images | 5.137 images |
| **Number of Classes** | 3 disease classes | 7 classes (6 disease + 1 healthy) |
| **Evaluation Metrics** | Accuracy only | Accuracy, Precision, Recall, F1, Classification Report, Confusion Matrix |
| **Fine-tuning Approach** | Simple trainable=true/false toggle | More detailed if applicable |
| **Main Contribution** | Working prototype app | Insights on model effectiveness trade-offs |
| **Test Set** | Not explicitly mentioned | Formal test set (768 images) |
| **Ablation Study** | None | Implicit through model comparison |

---

## 9. Kalimat yang Bisa Ditulis di BAB 1 (Latar Belakang)

```
Penyakit daun padi seperti Leaf Blast, Brown Spot, dan Hispa merupakan ancaman serius 
dalam produksi padi. Bacterial Leaf Blight saja dapat mengurangi hasil panen hingga 75% 
atau bahkan menyebabkan kegagalan panen total. Deteksi dini sangat penting untuk 
pencegahan dan pengendalian penyakit (Costales et al., 2020).
```

```
Tantangan utama dalam deteksi penyakit padi di tingkat petani adalah keterbatasan akses 
ke ahli patologi tanaman, biaya konsultasi yang tinggi, dan ketidaktahuan petani tentang 
berbagai jenis penyakit. Teknologi deep learning dan aplikasi mobile dapat menjadi solusi 
yang accessible dan affordable untuk petani (Costales et al., 2020).
```

```
Penelitian terdahulu telah menunjukkan bahwa CNN dengan transfer learning dapat mencapai 
akurasi tinggi dalam klasifikasi penyakit daun padi. Costales et al. (2020) mengembangkan 
aplikasi mobile menggunakan MobileNetV2 dan berhasil mencapai akurasi validasi 98% 
pada dataset 1.260 gambar dengan 3 kelas penyakit.
```

---

## 10. Kalimat yang Bisa Ditulis di BAB 2 (Tinjauan Pustaka)

### Subtopic: Transfer Learning untuk Rice Disease

```
Transfer learning merupakan pendekatan yang memanfaatkan pengetahuan dari model yang 
telah dilatih sebelumnya pada dataset besar (seperti ImageNet) dan mengadaptasinya untuk 
problem domain yang berbeda. Costales et al. (2020) menggunakan MobileNetV2 dengan 
pre-trained ImageNet weights sebagai base model untuk klasifikasi penyakit daun padi. 
Melalui fine-tuning (mengubah trainable=false menjadi trainable=true) dan image 
augmentation, model mencapai akurasi validasi 98%, meningkat dari 94% pada versi yang 
fully frozen.
```

### Subtopic: Fine-tuning Strategy

```
Fine-tuning merupakan teknik penting dalam transfer learning di mana layer-layer tertentu 
dari pre-trained model dibuka untuk training. Costales et al. (2020) menunjukkan bahwa 
approach yang fully frozen (trainable=false) dengan 10 epoch menghasilkan 94% akurasi 
validasi dengan overfitting yang signifikan. Ketika diubah menjadi fine-tuning (trainable=true) 
dengan augmentation dan 20 epoch, akurasi meningkat menjadi 98% dengan gap training-validation 
kurang dari 1%.
```

### Subtopic: Image Augmentation untuk Disease Detection

```
Image augmentation adalah teknik preprocessing yang meningkatkan variasi dataset dengan 
transformasi geometrik seperti flipping, shearing, dan rescaling. Costales et al. (2020) 
menerapkan augmentation pada training set untuk mengatasi masalah overfitting. Kombinasi 
augmentation dengan fine-tuning terbukti efektif dalam meningkatkan generalization capability 
model dari 94% menjadi 98% akurasi validasi.
```

### Subtopic: Practical Application di Agriculture

```
Pengembangan aplikasi praktis untuk agriculture technology memerlukan pendekatan yang 
tidak hanya akurat secara model tapi juga feasible untuk deployment. Costales et al. (2020) 
mengembangkan prototype aplikasi mobile untuk deteksi penyakit padi real-time menggunakan 
CNN. Fitur-fitur aplikasi mencakup real-time detection melalui camera, artificial intelligence 
yang belajar dari data baru, dan image analysis capabilities yang dapat diakses oleh petani 
tanpa expertise khusus.
```

---

## 11. Kalimat untuk Gap Penelitian

```
Meskipun penelitian sebelumnya telah menunjukkan bahwa transfer learning dengan MobileNetV2 
dapat mencapai akurasi tinggi untuk rice disease classification, kajian yang lebih mendalam 
tentang perbandingan performa CNN Kustom (trained from scratch) dengan MobileNetV2 pada 
dataset yang lebih komprehensif dan dengan class distribution yang lebih balanced masih 
diperlukan.
```

```
Selain itu, kebanyakan penelitian existing fokus pada pencapaian accuracy metrics tertinggi 
tanpa mempertimbangkan trade-off antara model complexity, jumlah parameters, training time, 
dan practical feasibility untuk deployment. Investigasi lebih detail tentang mengapa transfer 
learning bekerja lebih baik dan dalam konteks apa CNN Kustom masih relevan perlu dikembangkan.
```

```
Paper terdahulu seperti Costales et al. (2020) menggunakan dataset relatif kecil (1.260 images) 
dengan limited classes (3 disease). Penelitian dengan dataset lebih besar, class distribution 
yang lebih balanced (termasuk healthy class), dan evaluation yang lebih rigorous (including 
dedicated test set, confusion matrix, per-class metrics) akan memberikan insights yang lebih 
comprehensive.
```

---

## 12. Informasi Penting yang Perlu Dicek Lebih Lanjut

- [ ] Bagaimana cara Kaggle dataset dikurasi? Quality dan authenticity images?
- [ ] Spesifik layers mana yang di-unfreeze ketika trainable=true?
- [ ] Exact architecture dari custom FC layers di atas MobileNetV2?
- [ ] Berapa model size? Inference latency untuk mobile deployment?
- [ ] Apakah ada actual deployment? Feedback dari farmers?
- [ ] Hasil validasi oleh pathologists? Accuracy vs ground truth?
- [ ] Bagaimana performance pada images dengan lighting, angle, severity variations?
- [ ] Comparison dengan existing disease detection methods atau apps?

---

## 13. Kontribusi dan Insight dari Paper

### Insights Valuable untuk Penelitian Anda:

1. **Fine-tuning Effectiveness:**
   - Paper menunjukkan freezing vs fine-tuning punya impact besar (94% → 98%)
   - Anda bisa apply similar insight untuk MobileNetV2 tuning

2. **Augmentation Importance:**
   - Critical untuk overcome overfitting, especially pada small-medium datasets
   - Anda juga apply augmentation, so validation dari paper ini penting

3. **MobileNetV2 as Solid Baseline:**
   - Paper: 98% pada 3 classes dengan 1.260 images
   - Anda: akan dapatkan hasil berbeda pada 7 classes dengan 5.137 images
   - Important untuk understand expected ballpark performance

4. **Practical Consideration:**
   - Paper remind bahwa mobile deployment memerlukan lightweight model
   - MobileNetV2 dipilih tepat karena balance accuracy-efficiency
   - Relevant untuk justifikasi pemilihan model Anda

5. **Training Dynamics:**
   - Early overfitting (Iter 2: gap 3% antara train-val)
   - Augmentation + fine-tuning solve ini (Iter 3: gap <1%)
   - Anda bisa expect similar dynamics

---

## 14. Catatan Kritis

### 🔴 RED FLAGS:

1. **98% Accuracy Seems High:**
   - Paper tidak provide detail tentang baseline atau comparison
   - Only 3 classes bisa easier to classify
   - Missing test set evaluation
   - Result pada real-world data mungkin berbeda

2. **Limited Statistical Rigor:**
   - Tidak ada cross-validation
   - Tidak ada confidence intervals
   - Single train-val split bisa biased
   - Tidak ada statistical significance testing

3. **Incomplete Reporting:**
   - Conference paper dengan limited space
   - Missing important implementation details
   - Tidak clear tentang exact methodology

### 🟡 CAUTIONS:

1. Paper dari 2020 - standards sudah berubah
2. Limited dataset - generalization bisa questionable
3. Single model comparison - tidak ada baseline
4. No real-world validation - gap antara lab dan field mungkin signifikan

---

## 15. Kesimpulan untuk Penelitian Anda

Paper ini dapat digunakan sebagai:

✅ **USE FOR:**
- Reference untuk transfer learning dengan MobileNetV2 (practical approach)
- Reference untuk fine-tuning strategy (trainable=true vs false)
- Reference untuk importance of image augmentation
- Reference untuk practical application context (mobile deployment)
- Background reading untuk understanding MobileNetV2 effectiveness

⚠️ **CAUTION:**
- Tidak gunakan 98% accuracy sebagai target benchmark (kondisi berbeda)
- Tidak gunakan sebagai methodological gold standard (limited rigor)
- Understand limitations (dataset size, class balance, no test set)

🎯 **BEST POSITION dalam Penelitian Anda:**
- Sebagai **supporting reference** untuk transfer learning approach
- Sebagai **comparison point** untuk understanding MobileNetV2 performance range
- Sebagai **practical example** dari successful deployment
- Sebagai **motivation** untuk more rigorous evaluation (dibanding paper ini)

---

## 16. Rekomendasi untuk Implementasi dalam Skripsi

### Untuk BAB 1 (Latar Belakang):
- Mention practical importance dari rice disease detection
- Cite paper ini untuk relevance transfer learning dalam agriculture
- Use sebagai motivation untuk penelitian Anda (improve upon their work)

### Untuk BAB 2 (Tinjauan Pustaka):
- Pakai untuk section tentang transfer learning
- Pakai untuk section tentang fine-tuning strategies
- Pakai untuk discussion tentang practical application

### Untuk BAB 3 (Metodologi):
- Reference untuk justify augmentation approach Anda
- Reference untuk explain fine-tuning strategy if applicable
- Highlight how Anda improve upon their methodology:
  - Larger, more balanced dataset
  - More comprehensive evaluation
  - Direct model comparison (vs their single model)
  - Dedicated test set
  - More detailed metrics

### Untuk BAB 4 (Hasil & Pembahasan):
- Gunakan sebagai comparison/context
- Highlight similarities dan differences dalam results
- Explain mengapa Anda expect different results (7 classes vs 3, larger dataset, etc)

### Untuk BAB 5 (Kesimpulan):
- Acknowledge kontribusi paper ini
- Explain bagaimana Anda build upon atau improve their work
- Positioning: lebih rigorous, more comprehensive

---

**Status Screening:** ✅ COMPLETE  
**Relevance Score:** 7/10 (Relevant but different scope and rigor)  
**Recommendation:** ✅ USE as supporting reference for practical application and transfer learning insights
