"""Model MLP, pipeline tf.data, callback, dan helper eksperimen.

Dipakai oleh notebook 03 (arsitektur dan regularisasi), 04 (learning rate dan
batch size, training model final), serta notebook fitur unik. Keputusan yang
dikunci kelompok:
- Aktivasi hidden layer: ReLU
- Optimizer: Adam
- Input pipeline: tf.data
- Setiap konfigurasi dijalankan dengan 3 seed
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config

SEEDS = [42, 123, 2026]
MAX_EPOCHS = 100
PATIENCE = 8
MONITOR = "val_pr_auc"
EVAL_BATCH = 4096

# Konfigurasi awal sebelum eksperimen. Nilai ini yang diuji dan diubah bertahap.
DEFAULT_CONFIG = {
    "hidden_units": [128, 64],
    "dropout": 0.2,
    "l2": 0.0,
    "batch_norm": False,
    "learning_rate": 1e-3,
    "batch_size": 512,
}

CHECKPOINT_DIR = config.MODELS_DIR / "tmp"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Mengunci seed Python, NumPy, dan TensorFlow, lalu mengaktifkan operasi deterministik."""
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


# ---------------------------------------------------------------------------
# Input pipeline
# ---------------------------------------------------------------------------
def make_dataset(X, y, batch_size: int, shuffle: bool = False, seed: int = config.SEED) -> tf.data.Dataset:
    """Membuat tf.data.Dataset: (shuffle) -> batch -> prefetch.

    Shuffle memakai buffer seukuran data dan seed tetap, lalu diacak ulang setiap epoch.
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(
    hidden_units=(128, 64),
    dropout: float = 0.2,
    l2: float = 0.0,
    batch_norm: bool = False,
    learning_rate: float = 1e-3,
    loss="binary_crossentropy",
    n_features: int = len(config.FEATURE_COLS),
    name: str = "mlp_diabetes",
    **_,
) -> keras.Model:
    """MLP untuk klasifikasi biner.

    Susunan setiap hidden layer: Dense (He normal, L2 opsional) -> BatchNorm (opsional)
    -> ReLU -> Dropout (opsional). Output: 1 neuron sigmoid (probabilitas diabetes).
    Parameter `loss` bisa diganti, misalnya dengan Focal Loss di notebook fitur unik 1.
    """
    reg = keras.regularizers.L2(l2) if l2 > 0 else None
    inputs = keras.Input(shape=(n_features,), name="fitur_input")
    x = inputs
    for i, units in enumerate(hidden_units, start=1):
        x = keras.layers.Dense(units, kernel_initializer="he_normal", kernel_regularizer=reg,
                               use_bias=not batch_norm, name=f"dense_{i}")(x)
        if batch_norm:
            x = keras.layers.BatchNormalization(name=f"batchnorm_{i}")(x)
        x = keras.layers.Activation("relu", name=f"relu_{i}")(x)
        if dropout > 0:
            x = keras.layers.Dropout(dropout, name=f"dropout_{i}")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs, outputs, name=name)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc"),
                 keras.metrics.AUC(curve="ROC", name="roc_auc")],
    )
    return model


def get_callbacks(checkpoint_path: Path, patience: int = PATIENCE, monitor: str = MONITOR) -> list:
    """EarlyStopping dan ModelCheckpoint berdasarkan PR-AUC validation."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(monitor=monitor, mode="max", patience=patience,
                                      restore_best_weights=True, verbose=0),
        keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor=monitor, mode="max",
                                        save_best_only=True, verbose=0),
    ]


# ---------------------------------------------------------------------------
# Training dan evaluasi
# ---------------------------------------------------------------------------
def score(model: keras.Model, X, y) -> dict:
    """PR-AUC (average precision) dan ROC-AUC dihitung dengan scikit-learn dari probabilitas."""
    prob = model.predict(X, batch_size=EVAL_BATCH, verbose=0).ravel()
    return {"pr_auc": float(average_precision_score(y, prob)), "roc_auc": float(roc_auc_score(y, prob))}


def config_label(cfg: dict) -> str:
    units = "-".join(map(str, cfg["hidden_units"]))
    label = f"[{units}] do={cfg['dropout']}"
    if cfg.get("l2", 0) > 0:
        label += f" l2={cfg['l2']:g}"
    if cfg.get("batch_norm"):
        label += " BN"
    if cfg.get("learning_rate", 1e-3) != 1e-3 or cfg.get("batch_size", 512) != 512:
        label += f" lr={cfg['learning_rate']:g} bs={cfg['batch_size']}"
    return label


def train_one(cfg: dict, data: dict, seed: int, run_name: str, max_epochs: int = MAX_EPOCHS,
              loss="binary_crossentropy") -> tuple[keras.Model, dict, dict]:
    """Melatih satu konfigurasi dengan satu seed. Mengembalikan model (bobot terbaik), ringkasan, dan history."""
    set_seed(seed)
    model = build_model(**cfg, loss=loss)
    train_ds = make_dataset(data["X_train"], data["y_train"], cfg["batch_size"], shuffle=True, seed=seed)
    val_ds = make_dataset(data["X_val"], data["y_val"], EVAL_BATCH)
    ckpt = CHECKPOINT_DIR / f"{run_name}_seed{seed}.keras"

    start = time.time()
    # shuffle=False karena pengacakan sudah dilakukan oleh tf.data (dengan seed)
    history = model.fit(train_ds, validation_data=val_ds, epochs=max_epochs,
                        callbacks=get_callbacks(ckpt), shuffle=False, verbose=0)
    duration = time.time() - start

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    best_epoch = int(np.argmax(hist[MONITOR])) + 1
    val = score(model, data["X_val"], data["y_val"])
    train = score(model, data["X_train"], data["y_train"])

    summary = {
        "config": config_label(cfg),
        "seed": seed,
        "params": int(model.count_params()),
        "epochs_run": len(hist["loss"]),
        "best_epoch": best_epoch,
        "train_loss_best": hist["loss"][best_epoch - 1],
        "val_loss_best": hist["val_loss"][best_epoch - 1],
        "train_pr_auc": train["pr_auc"],
        "val_pr_auc": val["pr_auc"],
        "val_roc_auc": val["roc_auc"],
        "gap_pr_auc": train["pr_auc"] - val["pr_auc"],
        "seconds": round(duration, 1),
        "seconds_per_epoch": round(duration / len(hist["loss"]), 2),
    }
    return model, summary, hist


def run_configs(configs: dict, data: dict, seeds=SEEDS, prefix: str = "run", cache: dict | None = None,
                verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Menjalankan beberapa konfigurasi, masing-masing dengan semua seed.

    configs : {nama_skenario: cfg}
    cache   : dictionary hasil run sebelumnya {(config_label, seed): (summary, history)}.
              Konfigurasi yang identik tidak dilatih ulang.
    """
    cache = {} if cache is None else cache
    rows, histories = [], {}
    for scenario, cfg in configs.items():
        for seed in seeds:
            key = (config_label(cfg), seed)
            if key in cache:
                summary, hist = cache[key]
                status = "dipakai ulang"
            else:
                _, summary, hist = train_one(cfg, data, seed, run_name=f"{prefix}_{scenario}")
                cache[key] = (summary, hist)
                status = f"{summary['seconds']:.0f} detik"
            rows.append({"skenario": scenario, **summary, "cfg_json": json.dumps(cfg)})
            histories[f"{scenario}|{seed}"] = hist
            if verbose:
                print(f"{scenario:<14} seed {seed:<5} val PR-AUC {summary['val_pr_auc']:.4f}  "
                      f"epoch terbaik {summary['best_epoch']:>3}/{summary['epochs_run']:<3} ({status})")
            keras.backend.clear_session()
    return pd.DataFrame(rows), histories


# ---------------------------------------------------------------------------
# Ringkasan dan aturan pemilihan
# ---------------------------------------------------------------------------
def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """Rata-rata dan standar deviasi metrik per skenario (di semua seed)."""
    g = runs.groupby("skenario", sort=False)
    out = pd.DataFrame({
        "config": g["config"].first(),
        "params": g["params"].first(),
        "val_pr_auc_mean": g["val_pr_auc"].mean(),
        "val_pr_auc_std": g["val_pr_auc"].std(ddof=1),
        "val_roc_auc_mean": g["val_roc_auc"].mean(),
        "train_pr_auc_mean": g["train_pr_auc"].mean(),
        "gap_pr_auc_mean": g["gap_pr_auc"].mean(),
        "best_epoch_mean": g["best_epoch"].mean(),
        "seconds_mean": g["seconds"].mean(),
        "cfg_json": g["cfg_json"].first(),
    })
    return out


def _extras(cfg: dict) -> int:
    """Jumlah komponen tambahan di luar dropout (L2 dan BatchNorm)."""
    return int(cfg.get("l2", 0) > 0) + int(bool(cfg.get("batch_norm")))


def select_best(summary: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Aturan pemilihan konfigurasi.

    1. Cari skenario dengan rata-rata val PR-AUC tertinggi.
    2. Skenario lain dianggap SETARA jika rata-ratanya >= (terbaik - 1 std milik skenario terbaik).
    3. Dari yang setara, pilih yang paling sederhana: komponen tambahan paling sedikit,
       lalu parameter paling sedikit, lalu PR-AUC tertinggi.
    """
    s = summary.copy()
    best = s["val_pr_auc_mean"].idxmax()
    batas = s.loc[best, "val_pr_auc_mean"] - s.loc[best, "val_pr_auc_std"]
    s["setara_dengan_terbaik"] = s["val_pr_auc_mean"] >= batas
    s["komponen_tambahan"] = [_extras(json.loads(c)) for c in s["cfg_json"]]
    kandidat = s[s["setara_dengan_terbaik"]].sort_values(
        ["komponen_tambahan", "params", "val_pr_auc_mean"], ascending=[True, True, False])
    return kandidat.index[0], s


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
