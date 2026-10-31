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
- **File:** `data/diabetes_binary_health_indicators_BRFSS2015.csv`
- **Sumber asli:** Behavioral Risk Factor Surveillance System (BRFSS) 2015, CDC
- **Lisensi:** CC0 Public Domain
- **Ukuran:** 253.680 baris, 21 fitur, target biner `Diabetes_binary` (0 = tidak diabetes, 1 = prediabetes atau diabetes)

## Struktur Repository

```
mythos_diabetes_risk_mlp/
├── data/          # dataset mentah
├── notebooks/     # notebook pengerjaan, dijalankan berurutan 01 -> 06
├── src/           # kode yang dipakai bersama oleh notebook dan aplikasi
├── models/        # model final dan artefak preprocessing
├── reports/       # figures/ (grafik) dan results/ (metrik dalam csv)
├── app/           # aplikasi deployment (UTS)
└── docs/          # laporan PDF dan screenshot deployment
```

## Cara Menjalankan

Disarankan memakai **Python 3.12**.

```bash
git clone https://github.com/<username>/mythos_diabetes_risk_mlp.git
cd mythos_diabetes_risk_mlp

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Buka folder repository di VSCode, pilih kernel `.venv`, lalu jalankan notebook di folder `notebooks/` secara berurutan.

## Status Pengerjaan

- [x] Fase 2: setup repository, environment, dataset, EDA
- [ ] Fase 3: preprocessing, model MLP, fitur unik, evaluasi
- [ ] Fase 4: penyelesaian Tugas 1
- [ ] Fase 5: aplikasi dan deployment (UTS)
