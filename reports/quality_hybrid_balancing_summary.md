# Ringkasan Dataset Balanced Quality Hybrid

Folder output: `C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\data\rice_leaf_aug_7_balanced_quality_hybrid_653`
Target jumlah per kelas: 653
Report detail: `C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project\reports\quality_hybrid_balancing_report.csv`

| Kelas | Data awal | Asli dipakai | Dikeluarkan | Augmentasi | Total akhir |
|---|---:|---:|---:|---:|---:|
| Bacterial Leaf Blight | 636 | 636 | 0 | 17 | 653 |
| Brown Spot | 646 | 646 | 0 | 7 | 653 |
| Healthy Rice Leaf | 653 | 653 | 0 | 0 | 653 |
| Leaf Blast | 634 | 634 | 0 | 19 | 653 |
| Leaf scald | 628 | 628 | 0 | 25 | 653 |
| Sheath Blight | 632 | 632 | 0 | 21 | 653 |
| Tungro | 1308 | 653 | 655 | 0 | 653 |

Metode: quality-based filtering digunakan saat memilih citra asli yang dipertahankan. Kelas dengan jumlah data lebih banyak dari target dikurangi berdasarkan skor kualitas citra. Kelas dengan jumlah data kurang dari target ditambah menggunakan augmentasi citra.
