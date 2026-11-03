# Hasil Prediksi Foto Lapangan

Pengujian ini menggunakan foto lapangan tanpa label ground truth. Hasil berikut digunakan sebagai pengujian tambahan secara kualitatif.

| No | File | Prediksi CNN Kustom + Dropout | Confidence CNN | Prediksi MobileNetV2 | Confidence MobileNetV2 | Prediksi Sama |
|---:|---|---|---:|---|---:|---|
| 1 | `foto_001.jpeg` | Healthy Rice Leaf | 100.00% | Healthy Rice Leaf | 84.25% | Ya |
| 2 | `foto_002.jpeg` | Healthy Rice Leaf | 99.41% | Healthy Rice Leaf | 92.67% | Ya |
| 3 | `foto_003.jpeg` | Bacterial Leaf Blight | 45.65% | Healthy Rice Leaf | 91.62% | Tidak |
| 4 | `foto_004.jpeg` | Healthy Rice Leaf | 98.90% | Healthy Rice Leaf | 99.45% | Ya |
| 5 | `foto_005.jpeg` | Healthy Rice Leaf | 97.47% | Healthy Rice Leaf | 99.71% | Ya |
| 6 | `foto_006.jpeg` | Healthy Rice Leaf | 97.42% | Healthy Rice Leaf | 99.33% | Ya |
| 7 | `foto_007.jpeg` | Healthy Rice Leaf | 100.00% | Healthy Rice Leaf | 99.96% | Ya |
| 8 | `foto_008.jpeg` | Healthy Rice Leaf | 97.66% | Tungro | 55.15% | Tidak |
| 9 | `foto_009.jpeg` | Healthy Rice Leaf | 99.83% | Sheath Blight | 84.37% | Tidak |
| 10 | `foto_010.jpeg` | Healthy Rice Leaf | 99.97% | Healthy Rice Leaf | 56.56% | Ya |
| 11 | `foto_011.jpeg` | Healthy Rice Leaf | 99.58% | Healthy Rice Leaf | 78.65% | Ya |

## Top-3 Prediction

| No | File | Top-3 CNN | Top-3 MobileNetV2 |
|---:|---|---|---|
| 1 | `foto_001.jpeg` | Healthy Rice Leaf (100.00%); Brown Spot (0.00%); Leaf Blast (0.00%) | Healthy Rice Leaf (84.25%); Tungro (13.27%); Leaf scald (2.12%) |
| 2 | `foto_002.jpeg` | Healthy Rice Leaf (99.41%); Brown Spot (0.57%); Leaf Blast (0.03%) | Healthy Rice Leaf (92.67%); Leaf scald (6.82%); Tungro (0.37%) |
| 3 | `foto_003.jpeg` | Bacterial Leaf Blight (45.65%); Healthy Rice Leaf (35.42%); Sheath Blight (6.59%) | Healthy Rice Leaf (91.62%); Tungro (7.54%); Leaf scald (0.61%) |
| 4 | `foto_004.jpeg` | Healthy Rice Leaf (98.90%); Bacterial Leaf Blight (0.43%); Leaf Blast (0.26%) | Healthy Rice Leaf (99.45%); Tungro (0.37%); Leaf scald (0.18%) |
| 5 | `foto_005.jpeg` | Healthy Rice Leaf (97.47%); Leaf Blast (1.10%); Brown Spot (1.07%) | Healthy Rice Leaf (99.71%); Tungro (0.16%); Leaf scald (0.07%) |
| 6 | `foto_006.jpeg` | Healthy Rice Leaf (97.42%); Brown Spot (1.58%); Bacterial Leaf Blight (0.91%) | Healthy Rice Leaf (99.33%); Leaf scald (0.44%); Tungro (0.17%) |
| 7 | `foto_007.jpeg` | Healthy Rice Leaf (100.00%); Brown Spot (0.00%); Sheath Blight (0.00%) | Healthy Rice Leaf (99.96%); Brown Spot (0.02%); Sheath Blight (0.02%) |
| 8 | `foto_008.jpeg` | Healthy Rice Leaf (97.66%); Brown Spot (1.95%); Leaf Blast (0.36%) | Tungro (55.15%); Healthy Rice Leaf (36.64%); Leaf scald (5.81%) |
| 9 | `foto_009.jpeg` | Healthy Rice Leaf (99.83%); Leaf Blast (0.16%); Sheath Blight (0.01%) | Sheath Blight (84.37%); Tungro (7.28%); Leaf scald (5.54%) |
| 10 | `foto_010.jpeg` | Healthy Rice Leaf (99.97%); Brown Spot (0.02%); Leaf Blast (0.00%) | Healthy Rice Leaf (56.56%); Sheath Blight (24.53%); Leaf scald (16.97%) |
| 11 | `foto_011.jpeg` | Healthy Rice Leaf (99.58%); Leaf Blast (0.20%); Sheath Blight (0.18%) | Healthy Rice Leaf (78.65%); Tungro (20.40%); Sheath Blight (0.52%) |
