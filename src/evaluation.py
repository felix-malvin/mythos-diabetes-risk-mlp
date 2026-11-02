"""Metrik klasifikasi pada threshold tertentu dan pemilihan threshold.

Dipakai oleh notebook 05 (fitur unik 1), notebook 07 (evaluasi final), dan aplikasi.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

THRESHOLD_GRID = np.round(np.arange(0.01, 0.99, 0.005), 3)


def fbeta(precision, recall, beta: float):
    precision, recall = np.asarray(precision, float), np.asarray(recall, float)
    b2 = beta ** 2
    denom = b2 * precision + recall
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, (1 + b2) * precision * recall / denom, 0.0)


def metrics_at_threshold(y_true, prob, threshold: float) -> dict:
    """Confusion matrix dan metrik turunannya pada satu threshold."""
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(prob) >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": float(fbeta(precision, recall, 1)),
        "f2": float(fbeta(precision, recall, 2)),
        "accuracy": (tp + tn) / len(y_true),
        "persen_diprediksi_positif": (tp + fp) / len(y_true),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def threshold_curve(y_true, prob, grid=THRESHOLD_GRID) -> pd.DataFrame:
    """Metrik untuk setiap threshold di grid (dihitung vektor, cepat untuk ribuan threshold)."""
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob)
    pred = prob[None, :] >= np.asarray(grid)[:, None]
    pos = y_true == 1
    tp = (pred & pos).sum(axis=1)
    fp = (pred & ~pos).sum(axis=1)
    fn = pos.sum() - tp
    tn = (~pos).sum() - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
    return pd.DataFrame({
        "threshold": grid, "precision": precision, "recall": recall, "specificity": specificity,
        "f1": fbeta(precision, recall, 1), "f2": fbeta(precision, recall, 2),
        "youden_j": recall + specificity - 1,
        "persen_diprediksi_positif": (tp + fp) / len(y_true),
    })


def pick_threshold(curve: pd.DataFrame, criterion: str, min_recall: float = 0.80) -> float:
    """Memilih threshold dari kurva berdasarkan kriteria.

    'f1' / 'f2' / 'youden_j' : nilai maksimum
    'recall_min'             : precision tertinggi di antara threshold dengan recall >= min_recall
    """
    if criterion in ("f1", "f2", "youden_j"):
        return float(curve.loc[curve[criterion].idxmax(), "threshold"])
    if criterion == "recall_min":
        ok = curve[curve["recall"] >= min_recall]
        return float(ok.loc[ok["precision"].idxmax(), "threshold"])
    raise ValueError(criterion)


def ranking_metrics(y_true, prob) -> dict:
    return {"pr_auc": float(average_precision_score(y_true, prob)),
            "roc_auc": float(roc_auc_score(y_true, prob))}
