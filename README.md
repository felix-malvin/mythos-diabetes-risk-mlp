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
├── app/           # aplikasi Streamlit (UTS): app.py, contoh_kasus.json, requirements.txt
├── tests/         # pengujian otomatis aplikasi dan inferensi
├── .streamlit/    # tema dan konfigurasi Streamlit
└── docs/          # laporan, panduan deployment, dan screenshot
```

## Cara Menjalankan

Disarankan memakai **Python 3.12**.

```bash
git clone https://github.com/felix-malvin/mythos-diabetes-risk-mlp.git
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
| `models/mlp_final.keras` | MLP [128-64-32-16], ReLU, tanpa dropout, BatchNorm, L2 1e-6, Adam 0,0044, batch 256, loss BCE, seed 42 (hasil uji konfirmasi Random Search, notebook 04b) |
| `models/model_config.json` | Urutan fitur dan threshold keputusan **0,13** (F2 maksimum di data validation) |
| `models/scaler.joblib` | StandardScaler hasil fit di data train |
| `models/explain_config.json` | Baseline dan jumlah langkah Integrated Gradients (128) untuk penjelasan prediksi |

Probabilitas >= 0,13 dikategorikan berisiko diabetes. Prediksi dan penjelasan untuk satu responden dibuat dengan `src/inference.py` (`load_artifacts()` dan `predict_and_explain()`). Hasil ini adalah estimasi risiko, bukan diagnosis medis.

## Hasil Evaluasi Final (Data Test)

Data test (34.332 responden, 5.253 diabetes) hanya dipakai sekali di `07_evaluasi_final.ipynb`. Angka di bawah bersumber dari `reports/results/07_metrik_final_test.json`.

| Metrik | Threshold 0,5 (tanpa fitur unik) | Threshold 0,13 (model final) | IK 95% model final |
|---|---|---|---|
| Recall | 11,14% | **80,94%** | 79,84% - 81,95% |
| Precision | 62,77% | 30,43% | 29,66% - 31,24% |
| F2-score | 0,1333 | **0,6077** | 0,5987 - 0,6166 |
| Accuracy | 85,39% | 68,77% | 68,28% - 69,23% |
| PR-AUC | 0,4318 | 0,4318 | 0,4175 - 0,4464 |
| ROC-AUC | 0,8128 | 0,8128 | 0,8072 - 0,8183 |

Keterbatasan utama: 40,70% responden ditandai berisiko, recall pada usia 18-44 tahun hanya 45,70%, dan label berasal dari pengakuan responden survei. Model ini adalah alat skrining awal, bukan diagnosis medis.

## Aplikasi (UTS)

Aplikasi web Streamlit di `app/app.py` memakai model final yang sama persis dengan laporan (tidak ada training ulang).

| Halaman | Isi |
|---|---|
| Cek Risiko | Form 21 pertanyaan (BMI bisa dihitung dari tinggi dan berat badan), probabilitas dan kategori risiko, perbandingan threshold 0,13 dengan threshold bawaan 0,5 (fitur unik 1), grafik kontribusi Integrated Gradients (fitur unik 2) |
| Contoh Kasus | Lima responden asli dari data test (TP, TN, FP, FP prediabetes, FN), dipilih dengan aturan tetap: probabilitas paling dekat ke median kategorinya |
| Tentang Model | Metrik data test dari `reports/results/07_metrik_final_test.json`, confusion matrix, arsitektur, keterbatasan |

Menjalankan aplikasi dan pengujian dari root repository:

```bash
streamlit run app/app.py
python -m pytest tests -q
```

- **Link aplikasi:** https://mythos-diabetes-risk-mlp.streamlit.app/
- **Dependensi server:** `app/requirements.txt` 

## Susunan Notebook

| Notebook | Isi | Penanggung jawab |
|---|---|---|
| 01_eda | Exploratory data analysis | Afriyoga Haganta Tarigan |
| 02_preprocessing | Cleaning, split, scaling | Felix Malvin |
| 03_model_mlp | Arsitektur MLP, dropout, L2, BatchNorm | Wilbert Edric Tanjiro |
| 04_eksperimen_optimasi | Learning rate, batch size, model tanpa fitur unik (tahap bertahap) | Afriyoga Haganta Tarigan |
| 04b_random_search_kerastuner | Uji konfirmasi Random Search KerasTuner, konfigurasi final | Wilbert Edric Tanjiro |
| 05_fitur_unik_imbalance | Focal Loss dan threshold | Johannes Hartanto |
| 06_fitur_unik_explainability | Integrated Gradients | Erik Hamli |
| 07_evaluasi_final | Evaluasi di data test | Felix Malvin |

Notebook 03, 04, 04b, dan 05 melatih banyak model (masing-masing sekitar 30 sampai 60 menit di CPU). Di keempat notebook, set `RUN_EXPERIMENTS = False` untuk memuat hasil yang tersimpan di `reports/results/` tanpa training ulang.

## Status Pengerjaan

- [x] Fase 2: setup repository, environment, dataset, EDA
- [x] Fase 3a: preprocessing (dedup, outlier BMI, split 70/15/15, scaler)
- [x] Fase 3b: arsitektur MLP dan regularisasi (notebook 03)
- [x] Fase 3c: learning rate, batch size, model tanpa fitur unik (notebook 04)
- [x] Fase 3c-2: uji konfirmasi Random Search KerasTuner, konfigurasi diganti (notebook 04b)
- [x] Fase 3d: fitur unik 1, threshold F2 (notebook 05). Model final dibekukan
- [x] Fase 3e: fitur unik 2, Integrated Gradients (notebook 06)
- [x] Fase 3f: evaluasi final di data test (notebook 07)
- [x] Fase 4: laporan Tugas 1
- [x] Fase 5a: aplikasi Streamlit dan pengujian otomatis
- [x] Fase 5b: screenshot aplikasi
- [x] Fase 6: laporan UTS
