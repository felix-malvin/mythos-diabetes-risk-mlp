# Prediksi Risiko Diabetes Menggunakan Multilayer Perceptron (MLP)

Proyek Tugas 1 dan UTS mata kuliah Pembelajaran Mendalam (Deep Learning), Semester Ganjil TA 2026-2027.

> **Catatan:** hasil model ini adalah estimasi risiko berbasis data survei, **bukan diagnosis medis**.

## Anggota Kelompok

| NIM | Nama |
|---|---|
| 231111970 | Felix Malvin |
| 231110702 | Afriyoga Haganta Tarigan |
| 231110595 | Wilbert Edric Tanjiro |
| 231110730 | Johannes Hartanto |
| 231110980 | Erik Hamli |

## Dataset

- **Nama:** Diabetes Health Indicators Dataset (Alex Teboul, Kaggle)
- **Link:** https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
- **File training:** `data/diabetes_binary_health_indicators_BRFSS2015.csv`
- **File pembanding (tidak untuk training):** `data/diabetes_012_health_indicators_BRFSS2015.csv`
- **Sumber asli:** Behavioral Risk Factor Surveillance System (BRFSS) 2015, CDC
- **Lisensi:** CC0 Public Domain
- **Ukuran:** 253.680 baris, 21 fitur, target biner `Diabetes_binary` (0 = tidak diabetes, 1 = diabetes). Catatan: berbeda dengan deskripsi Kaggle, responden prediabetes masuk ke label 0 (diverifikasi di notebook 01 bagian 6.1)

## Struktur Repository

```
mythos_diabetes_risk_mlp/
├── data/          # dataset mentah
├── notebooks/     # notebook pengerjaan, dijalankan berurutan 01 -> 07
├── src/           # kode yang dipakai bersama oleh notebook dan aplikasi
├── models/        # scaler.joblib, preprocessing_config.json, model final
├── reports/       # figures/ (grafik) dan results/ (metrik dalam csv)
├── app/           # aplikasi deployment (UTS)
└── docs/          # laporan PDF dan screenshot deployment
```

## Cara Menjalankan

Disarankan memakai **Python 3.12**.

```bash
git clone https://github.com/felix-malvin/mythos_diabetes_risk_mlp.git
cd mythos_diabetes_risk_mlp

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Buka folder repository di VSCode, pilih kernel `.venv`, lalu jalankan notebook di folder `notebooks/` secara berurutan.

## Ringkasan Preprocessing

| Tahap | Jumlah baris |
|---|---|
| Data mentah | 253.680 |
| Setelah hapus duplikat | 229.474 |
| Setelah hapus BMI >= 70 | 228.875 |
| Train / Validation / Test | 160.211 / 34.332 / 34.332 |

Notebook model cukup memanggil `prepare_datasets()` dari `src/preprocessing.py`.

## Model Final

| File | Isi |
|---|---|
| `models/mlp_final.keras` | MLP [128-64], ReLU, dropout 0,2, Adam 0,001, batch 512, loss BCE, seed 42 |
| `models/model_config.json` | Urutan fitur dan threshold keputusan **0,135** (F2 maksimum di data validation) |
| `models/scaler.joblib` | StandardScaler hasil fit di data train |

Probabilitas >= 0,135 dikategorikan berisiko diabetes. Hasil ini adalah estimasi risiko, bukan diagnosis medis.

## Susunan Notebook

| Notebook | Isi | Penanggung jawab |
|---|---|---|
| 01_eda | Exploratory data analysis | Afriyoga Haganta Tarigan |
| 02_preprocessing | Cleaning, split, scaling | Felix Malvin |
| 03_model_mlp | Arsitektur MLP, dropout, L2, BatchNorm | Wilbert Edric Tanjiro |
| 04_eksperimen_optimasi | Learning rate, batch size, model final | Afriyoga Haganta Tarigan |
| 05_fitur_unik_imbalance | Focal Loss dan threshold | Johannes Hartanto |
| 06_fitur_unik_explainability | Integrated Gradients | Erik Hamli |
| 07_evaluasi_final | Evaluasi di data test | Felix Malvin |

Notebook 03 melatih 30 model dan notebook 04 melatih 25 model. Di kedua notebook, set `RUN_EXPERIMENTS = False` untuk memuat hasil yang tersimpan di `reports/results/` tanpa training ulang.

## Status Pengerjaan

- [x] Fase 2: setup repository, environment, dataset, EDA
- [x] Fase 3a: preprocessing (dedup, outlier BMI, split 70/15/15, scaler)
- [x] Fase 3b: arsitektur MLP dan regularisasi (notebook 03)
- [x] Fase 3c: learning rate, batch size, model tanpa fitur unik (notebook 04)
- [x] Fase 3d: fitur unik 1, threshold F2 (notebook 05). Model final dibekukan
- [ ] Fase 3e: fitur unik 2 (Integrated Gradients) dan evaluasi final di data test
- [ ] Fase 4: penyelesaian Tugas 1
- [ ] Fase 5: aplikasi dan deployment (UTS)
