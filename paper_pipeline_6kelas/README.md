# Pipeline Paper CNN Kustom 6 Kelas Tanpa Tungro

Folder ini berisi pipeline eksperimen khusus paper. Pipeline ini dibuat terpisah dari pipeline skripsi agar hasil eksperimen 6 kelas tanpa Tungro tidak bercampur dengan hasil eksperimen 7 kelas original/balanced.

## Tujuan

Menguji apakah CNN Kustom pada dataset 6 kelas tanpa Tungro menghasilkan `validation loss` yang tinggi seperti notebook sebelumnya, atau dapat turun lebih rendah ketika dilatih dengan maksimum 100 epoch, EarlyStopping, dan checkpoint model terbaik.

## Dataset

Dataset yang digunakan:

```text
rice_leaf_disease_project/data/rice_leaf_aug_6_no_tungro
```

Kelas yang digunakan:

1. Bacterial Leaf Blight
2. Brown Spot
3. Healthy Rice Leaf
4. Leaf Blast
5. Leaf scald
6. Sheath Blight

Kelas Tungro tidak digunakan.

## Model

Arsitektur mengikuti CNN Kustom dasar:

```text
Input 256x256x3
Rescaling 1/255
RandomFlip(horizontal_and_vertical)
RandomRotation(0.2)
Conv2D 16 + MaxPooling
Conv2D 32 + MaxPooling
Conv2D 64 + MaxPooling
Flatten
Dense 128
Dense output 6 neuron
```

Output berupa logits, sehingga loss yang digunakan adalah:

```text
SparseCategoricalCrossentropy(from_logits=True)
```

## Cara Menjalankan

Dari root project:

```powershell
cd "C:\Users\farid\Downloads\archive (5)\rice_leaf_disease_project"

$env:PAPER_EPOCHS="100"
$env:PAPER_IMAGE_SIZE="256"
$env:PAPER_BATCH_SIZE="32"
$env:PAPER_LEARNING_RATE="0.001"

.\.venv\Scripts\python.exe -u .\paper_pipeline_6kelas\run_paper_pipeline_6kelas.py
```

## Output

Hasil eksperimen disimpan di:

```text
paper_pipeline_6kelas/reports
paper_pipeline_6kelas/outputs
```

File penting:

- `training_history.csv`
- `summary_metrics.csv`
- `classification_report.txt`
- `confusion_matrix.png`
- `training_curves.png`
- `best_model_val_loss.keras`
- `final_model.keras`
