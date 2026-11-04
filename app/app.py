from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  

st.set_page_config(page_title="Cek Risiko Diabetes", page_icon="🩺", layout="wide")

THRESHOLD_BAWAAN = 0.5
LABEL_012 = {0: "Tidak diabetes", 1: "Prediabetes", 2: "Diabetes"}
GITHUB_URL = "https://github.com/felix-malvin/mythos_diabetes_risk_mlp" 


# ---------------------------------------------------------------------------
# Artefak (dimuat sekali per server)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model...")
def muat_artefak() -> dict:
    from src import inference
    art = inference.load_artifacts()
    # Probabilitas baseline tanpa pembulatan, untuk cek completeness yang tepat di tampilan.
    art["baseline_prob_tepat"] = float(inference.predict_proba(art["model"], art["baseline"][None, :])[0])
    return art


@st.cache_data
def muat_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def prediksi(values: dict) -> dict:
    from src import inference
    art = muat_artefak()
    h = inference.predict_and_explain(values, art)
    h["baseline_tepat"] = art["baseline_prob_tepat"]
    return h


def persen(x: float, d: int = 2) -> str:
    return f"{x * 100:.{d}f}%".replace(".", ",")


def angka(x: float, d: int = 4) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def ribuan(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def poin(x: float, d: int = 2) -> str:
    """Selisih probabilitas dalam poin persen dengan tanda, format Indonesia."""
    return f"{x * 100:+.{d}f}".replace(".", ",")