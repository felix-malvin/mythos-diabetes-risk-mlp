"""Pipeline data: load, cleaning, split, scaling, dan penyimpanan artefak.

Keputusan yang dipakai di file ini berasal dari notebooks/02_preprocessing.ipynb:
1. Baris duplikat (identik di semua kolom) dihapus.
2. Baris dengan BMI di atas config.BMI_MAX_VALID (69) dihapus.
3. Split stratified 70/15/15 dengan seed tetap.
4. StandardScaler di-fit hanya pada data train, khusus kolom ordinal dan numerik.

Cara pakai di notebook lain:
    from src.preprocessing import prepare_datasets
    data = prepare_datasets()
    data["X_train"], data["y_train"], ...   # numpy array siap masuk model
"""

from __future__ import annotations

import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import config


# ---------------------------------------------------------------------------
# Load dan cleaning
# ---------------------------------------------------------------------------
def load_raw_data(path=config.RAW_DATA_PATH) -> pd.DataFrame:
    """Membaca CSV mentah dan mengubah semua kolom ke integer.

    Seluruh kolom di file asli bertipe float, padahal isinya bilangan
    bulat (kode kategori, biner, BMI dibulatkan, jumlah hari).
    """
    df = pd.read_csv(path)
    expected = [config.TARGET_COL] + config.FEATURE_COLS
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"Kolom tidak ditemukan di dataset: {missing}")
    return df[expected].astype("int64")


def clean_data(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    bmi_max: int | None = config.BMI_MAX_VALID,
) -> tuple[pd.DataFrame, dict]:
    """Membersihkan data dan mengembalikan ringkasan jumlah baris tiap langkah.

    drop_duplicates : hapus baris yang identik di semua kolom (fitur + label).
    bmi_max         : hapus baris dengan BMI > bmi_max. None berarti tidak difilter.
    """
    summary = {"rows_raw": len(df)}

    if drop_duplicates:
        df = df.drop_duplicates()
    summary["duplicates_removed"] = summary["rows_raw"] - len(df)

    if bmi_max is not None:
        before = len(df)
        df = df[df["BMI"] <= bmi_max]
        summary["bmi_outliers_removed"] = before - len(df)
    else:
        summary["bmi_outliers_removed"] = 0

    df = df.reset_index(drop=True)
    summary["rows_clean"] = len(df)
    return df, summary


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def split_data(df: pd.DataFrame, seed: int = config.SEED) -> dict:
    """Stratified split menjadi train / validation / test (70/15/15)."""
    X = df[config.FEATURE_COLS]
    y = df[config.TARGET_COL]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=seed
    )
    # Ukuran val relatif terhadap sisa data setelah test dipisahkan
    val_relative = config.VAL_SIZE / (1 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, stratify=y_temp, random_state=seed
    )
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


def get_splits(
    drop_duplicates: bool = True,
    bmi_max: int | None = config.BMI_MAX_VALID,
) -> dict:
    """Shortcut: load -> clean -> split. Hasilnya masih berupa DataFrame (belum di-scaling)."""
    df = load_raw_data()
    df, summary = clean_data(df, drop_duplicates=drop_duplicates, bmi_max=bmi_max)
    splits = split_data(df)
    splits["clean_summary"] = summary
    return splits


def split_fingerprint(index: pd.Index) -> str:
    """Hash pendek dari index baris. Dipakai untuk memastikan semua notebook memakai split yang sama."""
    raw = ",".join(map(str, sorted(index.tolist())))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------
def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit StandardScaler hanya pada kolom ordinal dan numerik milik data train."""
    scaler = StandardScaler()
    scaler.fit(X_train[config.SCALED_COLS])
    return scaler


def transform_features(X: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """Menerapkan scaler lalu mengembalikan array float32 dengan urutan config.FEATURE_COLS.

    Kolom biner dibiarkan 0/1. Fungsi ini juga yang dipakai aplikasi saat inferensi,
    jadi transformasi saat training dan saat prediksi dijamin sama.
    """
    X = X[config.FEATURE_COLS].astype("float64").copy()
    X[config.SCALED_COLS] = scaler.transform(X[config.SCALED_COLS])
    return X.to_numpy(dtype=np.float32)


# ---------------------------------------------------------------------------
# Data siap pakai untuk model
# ---------------------------------------------------------------------------
def prepare_datasets() -> dict:
    """Load -> clean -> split -> fit scaler (train) -> transform semua split.

    Mengembalikan numpy array siap masuk model, beserta DataFrame asli
    (belum di-scaling) untuk keperluan analisis kesalahan dan explainability.
    """
    splits = get_splits()
    scaler = fit_scaler(splits["X_train"])
    data = {"scaler": scaler, "clean_summary": splits["clean_summary"]}
    for part in ["train", "val", "test"]:
        X_df = splits[f"X_{part}"]
        data[f"X_{part}"] = transform_features(X_df, scaler)
        data[f"y_{part}"] = splits[f"y_{part}"].to_numpy(dtype=np.float32)
        data[f"X_{part}_raw"] = X_df
    data["test_fingerprint"] = split_fingerprint(splits["X_test"].index)
    return data


# ---------------------------------------------------------------------------
# Simpan dan muat artefak
# ---------------------------------------------------------------------------
def save_preprocessing_artifacts(scaler: StandardScaler, info: dict) -> None:
    """Menyimpan scaler dan konfigurasi preprocessing ke folder models/."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, config.SCALER_PATH)
    with open(config.PREPROCESSING_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def load_scaler() -> StandardScaler:
    return joblib.load(config.SCALER_PATH)


def load_preprocessing_config() -> dict:
    with open(config.PREPROCESSING_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)
