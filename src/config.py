"""Konfigurasi global proyek: path, seed, dan daftar fitur.

Semua notebook dan aplikasi mengambil nilai dari file ini supaya
path, urutan fitur, dan seed tidak ditulis ulang di banyak tempat.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------
# Root repository dihitung dari lokasi file ini (src/config.py -> root),
# jadi tetap benar walaupun notebook dijalankan dari folder notebooks/.
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "diabetes_binary_health_indicators_BRFSS2015.csv"
# File pembanding (3 kelas) dari dataset Kaggle yang sama. TIDAK dipakai untuk training,
# hanya untuk memverifikasi makna label dan analisis kesalahan (responden prediabetes).
REFERENCE_012_PATH = DATA_DIR / "diabetes_012_health_indicators_BRFSS2015.csv"

MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# Proporsi pembagian data (train sisa dari keduanya = 0.70)
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# ---------------------------------------------------------------------------
# Keputusan preprocessing (hasil notebooks/02_preprocessing.ipynb)
# ---------------------------------------------------------------------------
# Baris dengan BMI >= 70 dihapus karena terindikasi salah input (lihat notebook 02).
# Aplikasi juga hanya menerima input BMI sampai nilai ini.
BMI_MAX_VALID = 69

# Artefak preprocessing yang dipakai ulang oleh notebook lain dan aplikasi
SCALER_PATH = MODELS_DIR / "scaler.joblib"
PREPROCESSING_CONFIG_PATH = MODELS_DIR / "preprocessing_config.json"

# ---------------------------------------------------------------------------
# Kolom dataset
# ---------------------------------------------------------------------------
TARGET_COL = "Diabetes_binary"

# Urutan fitur dikunci di sini. Model dan aplikasi WAJIB memakai urutan ini.
FEATURE_COLS = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]

# Fitur biner (0/1)
BINARY_COLS = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex",
]

# Fitur ordinal (kategori berurutan, dikodekan angka)
ORDINAL_COLS = ["GenHlth", "Age", "Education", "Income"]

# Fitur numerik kontinu / hitungan hari
CONTINUOUS_COLS = ["BMI", "MentHlth", "PhysHlth"]

# Fitur yang nantinya di-scaling (ordinal + kontinu)
SCALED_COLS = CONTINUOUS_COLS + ORDINAL_COLS

# ---------------------------------------------------------------------------
# Keterangan kode kategori (BRFSS 2015 codebook, variabel turunan CDC)
# Dipakai untuk label grafik EDA dan form input aplikasi.
# ---------------------------------------------------------------------------
AGE_LABELS = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
    6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
    11: "70-74", 12: "75-79", 13: "80+",
}

GENHLTH_LABELS = {
    1: "Istimewa", 2: "Sangat baik", 3: "Baik",
    4: "Cukup", 5: "Buruk",
}

EDUCATION_LABELS = {
    1: "Tidak sekolah / hanya TK",
    2: "Kelas 1-8 (SD-SMP)",
    3: "Kelas 9-11 (SMA tidak tamat)",
    4: "Kelas 12 / GED (lulus SMA)",
    5: "Kuliah/sekolah teknik 1-3 thn",
    6: "Kuliah 4 tahun atau lebih",
}

INCOME_LABELS = {
    1: "< $10.000", 2: "$10.000-15.000", 3: "$15.000-20.000",
    4: "$20.000-25.000", 5: "$25.000-35.000", 6: "$35.000-50.000",
    7: "$50.000-75.000", 8: ">= $75.000",
}

SEX_LABELS = {0: "Perempuan", 1: "Laki-laki"}
YES_NO_LABELS = {0: "Tidak", 1: "Ya"}

# Makna label target (sudah diverifikasi dengan file Diabetes_012, lihat notebook 01).
# Label 1 HANYA berisi responden diabetes. Responden prediabetes dan diabetes saat
# hamil saja masuk ke label 0, berbeda dengan deskripsi di halaman Kaggle.
CLASS_LABELS = {0: "Tidak diabetes", 1: "Diabetes"}

# ---------------------------------------------------------------------------
# Kamus fitur: acuan untuk laporan (bab 3) dan form input aplikasi.
# Sumber: deskripsi variabel dataset Kaggle (Teboul, 2021) yang diturunkan dari
# codebook BRFSS 2015 (CDC). Rentang nilai sudah dicek langsung pada data.
# ---------------------------------------------------------------------------
FEATURE_INFO = {
    "HighBP": {"nama": "Tekanan darah tinggi", "jenis": "biner",
               "keterangan": "Pernah dinyatakan memiliki tekanan darah tinggi", "nilai": YES_NO_LABELS},
    "HighChol": {"nama": "Kolesterol tinggi", "jenis": "biner",
                 "keterangan": "Pernah dinyatakan memiliki kolesterol tinggi", "nilai": YES_NO_LABELS},
    "CholCheck": {"nama": "Cek kolesterol", "jenis": "biner",
                  "keterangan": "Melakukan cek kolesterol dalam 5 tahun terakhir", "nilai": YES_NO_LABELS},
    "BMI": {"nama": "Indeks massa tubuh", "jenis": "numerik",
            "keterangan": "Body Mass Index (berat kg / tinggi m kuadrat), bilangan bulat",
            "nilai": "12-98 di data mentah, 12-69 setelah cleaning"},
    "Smoker": {"nama": "Pernah merokok", "jenis": "biner",
               "keterangan": "Pernah merokok minimal 100 batang (5 bungkus) seumur hidup", "nilai": YES_NO_LABELS},
    "Stroke": {"nama": "Riwayat stroke", "jenis": "biner",
               "keterangan": "Pernah dinyatakan mengalami stroke", "nilai": YES_NO_LABELS},
    "HeartDiseaseorAttack": {"nama": "Penyakit jantung", "jenis": "biner",
                             "keterangan": "Riwayat penyakit jantung koroner (CHD) atau serangan jantung (MI)",
                             "nilai": YES_NO_LABELS},
    "PhysActivity": {"nama": "Aktivitas fisik", "jenis": "biner",
                     "keterangan": "Melakukan aktivitas fisik di luar pekerjaan dalam 30 hari terakhir",
                     "nilai": YES_NO_LABELS},
    "Fruits": {"nama": "Konsumsi buah", "jenis": "biner",
               "keterangan": "Makan buah 1 kali atau lebih per hari", "nilai": YES_NO_LABELS},
    "Veggies": {"nama": "Konsumsi sayur", "jenis": "biner",
                "keterangan": "Makan sayur 1 kali atau lebih per hari", "nilai": YES_NO_LABELS},
    "HvyAlcoholConsump": {"nama": "Konsumsi alkohol berat", "jenis": "biner",
                          "keterangan": "Laki-laki lebih dari 14 minuman per minggu, perempuan lebih dari 7 minuman per minggu",
                          "nilai": YES_NO_LABELS},
    "AnyHealthcare": {"nama": "Jaminan kesehatan", "jenis": "biner",
                      "keterangan": "Memiliki jaminan kesehatan apa pun (asuransi, HMO, dan sejenisnya)",
                      "nilai": YES_NO_LABELS},
    "NoDocbcCost": {"nama": "Tidak ke dokter karena biaya", "jenis": "biner",
                    "keterangan": "Dalam 12 bulan terakhir perlu ke dokter tetapi tidak bisa karena biaya",
                    "nilai": YES_NO_LABELS},
    "GenHlth": {"nama": "Kesehatan umum", "jenis": "ordinal",
                "keterangan": "Penilaian responden terhadap kesehatannya secara umum", "nilai": GENHLTH_LABELS},
    "MentHlth": {"nama": "Hari kesehatan mental buruk", "jenis": "numerik",
                 "keterangan": "Jumlah hari dalam 30 hari terakhir dengan kondisi mental tidak baik (stres, depresi, masalah emosi)",
                 "nilai": "0-30 hari (0 = tidak pernah)"},
    "PhysHlth": {"nama": "Hari kesehatan fisik buruk", "jenis": "numerik",
                 "keterangan": "Jumlah hari dalam 30 hari terakhir dengan kondisi fisik tidak baik (sakit atau cedera)",
                 "nilai": "0-30 hari (0 = tidak pernah)"},
    "DiffWalk": {"nama": "Kesulitan berjalan", "jenis": "biner",
                 "keterangan": "Kesulitan serius saat berjalan atau naik tangga", "nilai": YES_NO_LABELS},
    "Sex": {"nama": "Jenis kelamin", "jenis": "biner", "keterangan": "Jenis kelamin responden",
            "nilai": SEX_LABELS},
    "Age": {"nama": "Kelompok usia", "jenis": "ordinal", "keterangan": "Kategori usia dalam rentang 5 tahun",
            "nilai": AGE_LABELS},
    "Education": {"nama": "Pendidikan", "jenis": "ordinal", "keterangan": "Jenjang pendidikan tertinggi",
                  "nilai": EDUCATION_LABELS},
    "Income": {"nama": "Pendapatan", "jenis": "ordinal",
               "keterangan": "Kategori pendapatan tahunan rumah tangga (USD)", "nilai": INCOME_LABELS},
}
