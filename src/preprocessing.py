"""Pipeline data: load, cleaning, split, dan scaling.

Versi awal (Fase 2). Keputusan final soal outlier BMI ditetapkan di
notebooks/02_preprocessing.ipynb berdasarkan temuan EDA, lalu nilai
defaultnya diperbarui di file ini.

Prinsip utama:
- Split dilakukan SETELAH cleaning, dengan stratifikasi dan seed tetap,
  sehingga setiap notebook yang memanggil get_splits() mendapat
  pembagian data yang identik tanpa perlu menyimpan file hasil split.
- Scaler hanya di-fit pada data training untuk mencegah kebocoran data.
"""

from __future__ import annotations

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
    bmi_upper: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Membersihkan data dan mengembalikan ringkasan perubahan.

    drop_duplicates : hapus baris yang identik di semua kolom (fitur + label).
                      Tujuannya mencegah baris yang sama muncul di train dan test.
    bmi_upper       : batas atas BMI (clipping). None berarti belum diterapkan.
    """
    summary = {"rows_before": len(df)}

    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    summary["rows_after_dedup"] = len(df)
    summary["duplicates_removed"] = summary["rows_before"] - len(df)

    if bmi_upper is not None:
        summary["bmi_clipped"] = int((df["BMI"] > bmi_upper).sum())
        df = df.assign(BMI=df["BMI"].clip(upper=bmi_upper))
    else:
        summary["bmi_clipped"] = 0

    summary["rows_after"] = len(df)
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


def get_splits(drop_duplicates: bool = True, bmi_upper: int | None = None) -> dict:
    """Shortcut: load -> clean -> split. Dipakai oleh semua notebook."""
    df = load_raw_data()
    df, summary = clean_data(df, drop_duplicates=drop_duplicates, bmi_upper=bmi_upper)
    splits = split_data(df)
    splits["clean_summary"] = summary
    return splits


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------
def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit StandardScaler hanya pada kolom ordinal dan kontinu milik data train."""
    scaler = StandardScaler()
    scaler.fit(X_train[config.SCALED_COLS])
    return scaler


def transform_features(X: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """Menerapkan scaler lalu mengembalikan array float32 sesuai FEATURE_COLS.

    Kolom biner dibiarkan 0/1. Urutan kolom output selalu mengikuti
    config.FEATURE_COLS supaya sama persis antara training dan aplikasi.
    """
    X = X[config.FEATURE_COLS].astype("float64").copy()
    X[config.SCALED_COLS] = scaler.transform(X[config.SCALED_COLS])
    return X.to_numpy(dtype=np.float32)
