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
    2: "SD-SMP (kelas 1-8)",
    3: "SMA tidak tamat (kelas 9-11)",
    4: "SMA tamat",
    5: "Kuliah 1-3 tahun",
    6: "Lulus kuliah (4 tahun+)",
}

INCOME_LABELS = {
    1: "< $10.000", 2: "$10.000-15.000", 3: "$15.000-20.000",
    4: "$20.000-25.000", 5: "$25.000-35.000", 6: "$35.000-50.000",
    7: "$50.000-75.000", 8: ">= $75.000",
}

CLASS_LABELS = {0: "Tidak diabetes", 1: "Prediabetes/Diabetes"}
