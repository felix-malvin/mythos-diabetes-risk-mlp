"""Inferensi dan penjelasan prediksi (Integrated Gradients).

Modul ini dipakai oleh notebook 06, notebook 07, dan aplikasi deployment, sehingga
alur prediksi di aplikasi sama persis dengan notebook:

    input dictionary (nilai asli) -> validasi -> scaler -> model -> probabilitas
    -> kategori risiko (threshold dari model_config.json) -> Integrated Gradients

Integrated Gradients (Sundararajan dkk., 2017):
    IG_i(x) = (x_i - x'_i) * integral_0^1 dF(x' + a (x - x')) / dx_i  da
dengan F = probabilitas diabetes dan x' = baseline. Sifat completeness:
    sum_i IG_i(x) = F(x) - F(x')
sehingga kontribusi setiap fitur bisa dibaca dalam satuan poin probabilitas.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from src import config
from src import preprocessing as prep

EXPLAIN_CONFIG_PATH = config.MODELS_DIR / "explain_config.json"
MODEL_CONFIG_PATH = config.MODELS_DIR / "model_config.json"
IG_STEPS = 64


# ---------------------------------------------------------------------------
# Integrated Gradients
# ---------------------------------------------------------------------------
def _gradients(model: keras.Model, x: np.ndarray) -> np.ndarray:
    """Gradien probabilitas output terhadap input (dropout nonaktif)."""
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        prob = model(x, training=False)
    return tape.gradient(prob, x).numpy()


def integrated_gradients(model: keras.Model, X: np.ndarray, baseline: np.ndarray,
                         steps: int = IG_STEPS, chunk: int = 512) -> np.ndarray:
    """IG untuk banyak baris sekaligus, integral dihitung dengan aturan trapesium.

    X        : (n, d) input yang sudah di-scaling
    baseline : (d,) satu baseline yang sama untuk semua baris
    return   : (n, d) atribusi dalam satuan probabilitas
    """
    X = np.asarray(X, dtype=np.float32)
    baseline = np.asarray(baseline, dtype=np.float32).reshape(1, -1)
    alphas = np.linspace(0.0, 1.0, steps + 1, dtype=np.float32)
    weights = np.full(steps + 1, 1.0 / steps, dtype=np.float32)
    weights[[0, -1]] = 0.5 / steps
    out = np.zeros_like(X)
    for start in range(0, len(X), chunk):
        xb = X[start:start + chunk]
        diff = xb - baseline
        path = baseline[None, :, :] + alphas[None, :, None] * diff[:, None, :]
        grads = _gradients(model, path.reshape(-1, X.shape[1])).reshape(len(xb), steps + 1, -1)
        out[start:start + chunk] = diff * np.tensordot(weights, grads, axes=([0], [1]))
    return out


def expected_gradients(model: keras.Model, X: np.ndarray, baselines: np.ndarray, steps: int = 16) -> np.ndarray:
    """Rata-rata IG terhadap banyak baseline. Dipakai untuk uji sensitivitas baseline."""
    hasil = [integrated_gradients(model, X, b, steps=steps) for b in np.asarray(baselines, dtype=np.float32)]
    return np.mean(hasil, axis=0)


def predict_proba(model: keras.Model, X: np.ndarray) -> np.ndarray:
    return model.predict(np.asarray(X, dtype=np.float32), batch_size=4096, verbose=0).ravel()


# ---------------------------------------------------------------------------
# Artefak untuk aplikasi
# ---------------------------------------------------------------------------
def load_artifacts(root: Path = config.ROOT_DIR) -> dict:
    """Memuat model, scaler, konfigurasi model, dan konfigurasi penjelasan dari folder models/."""
    model_cfg = json.loads((root / "models" / "model_config.json").read_text(encoding="utf-8"))
    explain_cfg = json.loads((root / "models" / "explain_config.json").read_text(encoding="utf-8"))
    prep_cfg = json.loads((root / "models" / "preprocessing_config.json").read_text(encoding="utf-8"))
    return {
        "model": keras.models.load_model(root / model_cfg["model_path"], compile=False),
        "scaler": prep.load_scaler(),
        "model_config": model_cfg,
        "explain_config": explain_cfg,
        "input_ranges": prep_cfg["input_ranges"],
        "baseline": np.asarray(explain_cfg["baseline_scaled"], dtype=np.float32),
    }


def validate_input(values: dict, input_ranges: dict) -> list[str]:
    """Mengembalikan daftar pesan kesalahan. List kosong berarti input valid."""
    errors = []
    for col in config.FEATURE_COLS:
        if col not in values:
            errors.append(f"{col} belum diisi")
            continue
        v = values[col]
        if not float(v).is_integer():
            errors.append(f"{col} harus bilangan bulat")
            continue
        lo, hi = input_ranges[col]["min"], input_ranges[col]["max"]
        if not lo <= v <= hi:
            errors.append(f"{col} harus di antara {lo} dan {hi}")
    return errors


def risk_category(prob: float, threshold: float) -> str:
    return "Berisiko diabetes" if prob >= threshold else "Risiko rendah"


def format_value(col: str, value) -> str:
    """Nilai fitur dalam bentuk yang mudah dibaca, memakai label dari config.FEATURE_INFO."""
    nilai = config.FEATURE_INFO[col]["nilai"]
    if isinstance(nilai, dict):
        return nilai.get(int(value), str(value))
    if col in ("MentHlth", "PhysHlth"):
        return f"{int(value)} hari"
    return str(int(value))


def predict_and_explain(values: dict, art: dict, steps: int | None = None) -> dict:
    """Prediksi satu responden beserta kontribusi setiap fitur.

    Jumlah langkah IG diambil dari explain_config.json kalau tidak diisi.
    """
    steps = steps or int(art["explain_config"]["steps"])
    errors = validate_input(values, art["input_ranges"])
    if errors:
        raise ValueError("; ".join(errors))
    row = pd.DataFrame([{c: int(values[c]) for c in config.FEATURE_COLS}])
    x = prep.transform_features(row, art["scaler"])
    prob = float(predict_proba(art["model"], x)[0])
    attr = integrated_gradients(art["model"], x, art["baseline"], steps=steps)[0]
    kontribusi = pd.DataFrame({
        "fitur": config.FEATURE_COLS,
        "nama": [config.FEATURE_INFO[c]["nama"] for c in config.FEATURE_COLS],
        "nilai": [format_value(c, values[c]) for c in config.FEATURE_COLS],
        "kontribusi": attr,
    }).sort_values("kontribusi", key=np.abs, ascending=False).reset_index(drop=True)
    threshold = art["model_config"]["threshold"]
    return {
        "probabilitas": prob,
        "threshold": threshold,
        "kategori": risk_category(prob, threshold),
        "probabilitas_baseline": art["explain_config"]["baseline_prob"],
        "kontribusi": kontribusi,
        "selisih_completeness": float(prob - art["explain_config"]["baseline_prob"] - attr.sum()),
    }
