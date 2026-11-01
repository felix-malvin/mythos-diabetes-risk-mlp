"""Helper tabel dan grafik yang dipakai bersama oleh notebook 04 dan seterusnya.

Warna dibuat konsisten di semua notebook:
- train      : oranye, garis putus-putus
- validation : biru, garis utuh
- terpilih   : biru, kandidat lain abu-abu
"""

from __future__ import annotations

import json

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

WARNA_TRAIN = "#DD8452"
WARNA_VAL = "#4C72B0"
WARNA_ABU = "#9A9A9A"
WARNA_PILIH = "#4C72B0"


def set_style() -> None:
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.3, "font.size": 10})


def save_fig(fig, name: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")


def tabel_ringkas(summary: pd.DataFrame, terpilih: str) -> pd.DataFrame:
    """Tabel ringkasan skenario dengan nama kolom berbahasa Indonesia."""
    t = summary[["config", "params", "val_pr_auc_mean", "val_pr_auc_std", "val_roc_auc_mean",
                 "train_pr_auc_mean", "gap_pr_auc_mean", "best_epoch_mean", "seconds_mean",
                 "setara_dengan_terbaik"]].copy()
    t.columns = ["konfigurasi", "parameter", "val PR-AUC", "std", "val ROC-AUC", "train PR-AUC",
                 "gap train-val", "epoch terbaik", "detik/run", "setara terbaik"]
    t["dipilih"] = ["ya" if i == terpilih else "" for i in t.index]
    return t.round({"val PR-AUC": 4, "std": 4, "val ROC-AUC": 4, "train PR-AUC": 4,
                    "gap train-val": 4, "epoch terbaik": 1, "detik/run": 0})


def plot_tahap(runs: pd.DataFrame, summary: pd.DataFrame, terpilih: str, judul: str, nama_file: str,
               label_x: dict | None = None) -> None:
    """Dot plot: titik kecil = tiap seed, titik besar + garis = rata-rata dan +-1 std."""
    urutan = list(summary.index)
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(urutan)), 4))
    for i, sk in enumerate(urutan):
        warna = WARNA_PILIH if sk == terpilih else WARNA_ABU
        nilai = runs.loc[runs["skenario"] == sk, "val_pr_auc"].to_numpy()
        ax.scatter(i + np.linspace(-0.08, 0.08, len(nilai)), nilai, s=18, color=warna, alpha=0.5, zorder=2)
        m, s = summary.loc[sk, "val_pr_auc_mean"], summary.loc[sk, "val_pr_auc_std"]
        ax.errorbar(i + 0.25, m, yerr=s, fmt="o", ms=8, color=warna, capsize=4, lw=2, zorder=3)
        ax.annotate(f"{m:.4f}", (i + 0.33, m), va="center", fontsize=9, color="#333333")
    best = summary["val_pr_auc_mean"].idxmax()
    batas = summary.loc[best, "val_pr_auc_mean"] - summary.loc[best, "val_pr_auc_std"]
    ax.axhline(batas, ls=":", color="#555555", lw=1)
    ax.text(-0.45, batas, "batas setara", va="bottom", ha="left", fontsize=8, color="#555555")
    label_x = label_x or {sk: sk for sk in urutan}
    ax.set_xticks(range(len(urutan)), [label_x[sk] for sk in urutan])
    ax.set_xlim(-0.5, len(urutan) - 0.2)
    ax.set_ylabel("PR-AUC validation")
    ax.set_title(judul, fontweight="bold")
    ax.legend(handles=[mpatches.Patch(color=WARNA_PILIH, label="dipilih"),
                       mpatches.Patch(color=WARNA_ABU, label="kandidat lain")], loc="lower right", fontsize=8)
    fig.text(0.01, -0.02, "Titik kecil = hasil tiap seed. Titik besar dan garis = rata-rata +- 1 std. "
             "Sumbu y diperbesar dan tidak dimulai dari 0.", fontsize=8, color="#555555")
    fig.tight_layout()
    save_fig(fig, nama_file)
    plt.show()


def plot_kurva(histories: dict, judul: str, nama_file: str, loss_label: str = "Loss") -> None:
    """Kurva loss dan PR-AUC train vs validation. histories = {label_seed: history_dict}."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for metrik, ax, sub in [("loss", axes[0], loss_label), ("pr_auc", axes[1], "PR-AUC")]:
        for j, (lbl, h) in enumerate(histories.items()):
            ep = np.arange(1, len(h[metrik]) + 1)
            alpha = 1.0 if j == 0 else 0.45
            ax.plot(ep, h[metrik], ls="--", lw=1.5, color=WARNA_TRAIN, alpha=alpha, label="train" if j == 0 else None)
            ax.plot(ep, h[f"val_{metrik}"], ls="-", lw=2, color=WARNA_VAL, alpha=alpha,
                    label="validation" if j == 0 else None)
            ax.axvline(int(np.argmax(h["val_pr_auc"])) + 1, color=WARNA_VAL, lw=0.8, alpha=0.35)
        ax.set_xlabel("Epoch")
        ax.set_title(sub)
        ax.legend()
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("PR-AUC")
    fig.suptitle(judul, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, nama_file)
    plt.show()


def cfg_dari_json(s: str) -> dict:
    return json.loads(s)
