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


# ---------------------------------------------------------------------------
# Nilai awal form
# ---------------------------------------------------------------------------
NILAI_AWAL = {
    "HighBP": 0, "HighChol": 0, "CholCheck": 1, "BMI": 27, "Smoker": 0, "Stroke": 0,
    "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1, "Veggies": 1,
    "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 3,
    "MentHlth": 0, "PhysHlth": 0, "DiffWalk": 0, "Sex": 0, "Age": 8, "Education": 4, "Income": 6,
}


MODE_HITUNG, MODE_LANGSUNG = "Hitung dari tinggi dan berat badan", "Isi BMI langsung"


def isi_form(values: dict, sumber: str | None = None, mode_bmi: str = MODE_LANGSUNG) -> None:
    """Mengisi semua widget form lewat session_state. Dipakai juga oleh halaman contoh kasus.

    Contoh kasus dari data test hanya punya nilai BMI, jadi mode BMI-nya "Isi BMI langsung".
    Form kosong memakai mode tinggi dan berat badan karena lebih mudah untuk pengguna awam.
    """
    for c in config.FEATURE_COLS:
        st.session_state[f"f_{c}"] = int(values[c])
    st.session_state["mode_bmi"] = mode_bmi
    st.session_state["bmi_langsung"] = int(values["BMI"])
    st.session_state["tinggi"] = 165
    st.session_state["berat"] = 70
    st.session_state["sumber_input"] = sumber
    st.session_state.pop("hasil", None)


def pastikan_state() -> None:
    if "f_HighBP" not in st.session_state:
        isi_form(NILAI_AWAL, mode_bmi=MODE_HITUNG)


# ---------------------------------------------------------------------------
# Komponen form
# ---------------------------------------------------------------------------
def ya_tidak(col: str, pertanyaan: str, bantuan: str | None = None) -> None:
    st.radio(pertanyaan, options=[1, 0], format_func=lambda v: config.YES_NO_LABELS[v],
             horizontal=True, key=f"f_{col}", help=bantuan)


def pilihan(col: str, pertanyaan: str, label: dict, bantuan: str | None = None) -> None:
    st.selectbox(pertanyaan, options=list(label.keys()), format_func=lambda v: label[v],
                 key=f"f_{col}", help=bantuan)


def bagian_bmi() -> tuple[int | None, str | None]:
    """Mengembalikan (BMI bulat, pesan kesalahan)."""
    lo, hi = 12, config.BMI_MAX_VALID
    mode = st.radio("Cara mengisi BMI", [MODE_HITUNG, MODE_LANGSUNG], horizontal=True, key="mode_bmi")
    if mode == MODE_LANGSUNG:
        bmi = st.number_input(f"BMI (bilangan bulat, {lo} sampai {hi})", min_value=lo, max_value=hi,
                              step=1, key="bmi_langsung")
        return int(bmi), None
    c1, c2 = st.columns(2)
    tinggi = c1.number_input("Tinggi badan (cm)", min_value=100, max_value=230, step=1, key="tinggi")
    berat = c2.number_input("Berat badan (kg)", min_value=25, max_value=250, step=1, key="berat")
    bmi_float = berat / (tinggi / 100) ** 2
    bmi = int(round(bmi_float))
    st.caption(f"BMI = {berat} / ({tinggi / 100:.2f} x {tinggi / 100:.2f}) = {bmi_float:.1f}, "
               f"dibulatkan menjadi **{bmi}** karena data training memakai BMI bilangan bulat.")
    if not lo <= bmi <= hi:
        return None, (f"BMI {bmi} di luar rentang data training ({lo} sampai {hi}). "
                      "Model tidak bisa memberi estimasi yang dapat dipercaya untuk nilai ini.")
    return bmi, None


def form_input() -> tuple[dict | None, str | None]:
    pastikan_state()
    k1, k2 = st.columns(2, gap="large")
    with k1:
        st.markdown("##### Data diri")
        st.radio("Jenis kelamin", options=[0, 1], format_func=lambda v: config.SEX_LABELS[v],
                 horizontal=True, key="f_Sex")
        pilihan("Age", "Kelompok usia", {k: f"{v} tahun" for k, v in config.AGE_LABELS.items()})
        pilihan("Education", "Pendidikan terakhir", config.EDUCATION_LABELS)
        pilihan("Income", "Pendapatan tahunan rumah tangga (USD)", config.INCOME_LABELS,
                "Kategori mengikuti survei BRFSS 2015 di Amerika Serikat, dalam dolar AS.")
        st.markdown("##### Ukuran tubuh")
        bmi, galat = bagian_bmi()
        st.markdown("##### Gaya hidup")
        ya_tidak("Smoker", "Pernah merokok minimal 100 batang (5 bungkus) seumur hidup",
                 "Bukan status merokok saat ini, sesuai definisi pertanyaan survei.")
        ya_tidak("PhysActivity", "Berolahraga atau beraktivitas fisik di luar pekerjaan dalam 30 hari terakhir")
        ya_tidak("Fruits", "Makan buah minimal 1 kali sehari")
        ya_tidak("Veggies", "Makan sayur minimal 1 kali sehari")
        ya_tidak("HvyAlcoholConsump", "Minum alkohol berat",
                 "Laki-laki lebih dari 14 minuman per minggu, perempuan lebih dari 7 minuman per minggu.")
    with k2:
        st.markdown("##### Riwayat kesehatan")
        ya_tidak("HighBP", "Pernah dinyatakan dokter memiliki tekanan darah tinggi")
        ya_tidak("HighChol", "Pernah dinyatakan dokter memiliki kolesterol tinggi")
        ya_tidak("CholCheck", "Cek kolesterol dalam 5 tahun terakhir")
        ya_tidak("Stroke", "Pernah dinyatakan mengalami stroke")
        ya_tidak("HeartDiseaseorAttack", "Pernah mengalami penyakit jantung koroner atau serangan jantung")
        ya_tidak("DiffWalk", "Kesulitan serius saat berjalan atau naik tangga")
        st.markdown("##### Kondisi 30 hari terakhir")
        pilihan("GenHlth", "Secara umum, kesehatan Anda", config.GENHLTH_LABELS)
        st.number_input("Jumlah hari kondisi mental terasa tidak baik (stres, depresi, masalah emosi)",
                        min_value=0, max_value=30, step=1, key="f_MentHlth")
        st.number_input("Jumlah hari kondisi fisik terasa tidak baik (sakit atau cedera)",
                        min_value=0, max_value=30, step=1, key="f_PhysHlth")
        st.markdown("##### Akses layanan kesehatan")
        ya_tidak("AnyHealthcare", "Memiliki jaminan atau asuransi kesehatan apa pun")
        ya_tidak("NoDocbcCost", "Dalam 12 bulan terakhir pernah batal ke dokter karena biaya")
    if galat:
        return None, galat
    values = {c: int(st.session_state[f"f_{c}"]) for c in config.FEATURE_COLS if c != "BMI"}
    values["BMI"] = bmi
    return values, None