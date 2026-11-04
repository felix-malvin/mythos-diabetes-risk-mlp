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
GITHUB_URL = "https://github.com/felix-malvin/mythos-diabetes-risk-mlp.git" 


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


# ---------------------------------------------------------------------------
# Tampilan hasil
# ---------------------------------------------------------------------------
def grafik_threshold(prob: float, thr: float) -> alt.Chart:
    """Pita probabilitas 0 sampai 1, zona berisiko (merah muda), dua garis threshold, dan posisi responden."""
    X = alt.X("x:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title="Probabilitas diabetes"))
    Y = alt.Y("y:Q", scale=alt.Scale(domain=[0, 1]), axis=None)
    pita = pd.DataFrame({"x": [0.0, thr], "x2": [1.0, 1.0], "y": [0.38, 0.38], "y2": [0.62, 0.62],
                         "warna": ["#e9eef3", "#f6d5cf"]})
    dasar = alt.Chart(pita).mark_rect().encode(x=X, x2="x2:Q", y=Y, y2="y2:Q", color=alt.Color("warna:N", scale=None))
    garis = pd.DataFrame({"x": [thr, THRESHOLD_BAWAAN], "y": [0.2, 0.2], "y2": [0.8, 0.8],
                          "teks": [f"threshold model {angka(thr, 2)}", f"threshold bawaan {angka(THRESHOLD_BAWAAN, 1)}"],
                          "warna": ["#c0392b", "#555555"], "yt": [0.92, 0.92]})
    rule = alt.Chart(garis).mark_rule(strokeWidth=2.5).encode(x="x:Q", y="y:Q", y2="y2:Q",
                                                              color=alt.Color("warna:N", scale=None))
    label = alt.Chart(garis).mark_text(align="left", dx=5, fontSize=12).encode(
        x="x:Q", y="yt:Q", text="teks:N", color=alt.Color("warna:N", scale=None))
    titik = alt.Chart(pd.DataFrame({"x": [prob], "y": [0.5], "yt": [0.06], "teks": [f"responden {persen(prob, 1)}"]}))
    poin_ = titik.mark_point(shape="diamond", size=260, filled=True, color="#1f3b73", opacity=1).encode(x="x:Q", y="y:Q")
    tp = titik.mark_text(fontSize=12, fontWeight="bold", color="#1f3b73").encode(x="x:Q", y="yt:Q", text="teks:N")
    return (dasar + rule + label + poin_ + tp).properties(height=130)


def grafik_kontribusi(kontribusi: pd.DataFrame, n: int = 10) -> alt.Chart:
    top = kontribusi.head(n).copy()
    top["poin"] = top["kontribusi"] * 100
    top["arah"] = top["poin"].apply(lambda v: "Menaikkan risiko" if v > 0 else "Menurunkan risiko")
    top["label"] = top["nama"] + ": " + top["nilai"].astype(str)
    top["teks"] = top["poin"].map(lambda v: f"{v:+.1f}".replace(".", ","))
    urutan = top["label"].tolist()
    batang = alt.Chart(top).mark_bar().encode(
        x=alt.X("poin:Q", title="Kontribusi (poin persen probabilitas)"),
        y=alt.Y("label:N", sort=urutan, title=None, axis=alt.Axis(labelLimit=320)),
        color=alt.Color("arah:N", scale=alt.Scale(domain=["Menaikkan risiko", "Menurunkan risiko"],
                                                   range=["#c0392b", "#2e6db4"]),
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=[alt.Tooltip("label:N", title="Faktor"), alt.Tooltip("teks:N", title="Poin persen")])
    teks_neg = alt.Chart(top[top["poin"] < 0]).mark_text(align="right", dx=-4, color="#333").encode(
        x="poin:Q", y=alt.Y("label:N", sort=urutan), text="teks:N")
    nol = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#555").encode(x="x:Q")
    teks_pos = alt.Chart(top[top["poin"] > 0]).mark_text(align="left", dx=4, color="#333").encode(
        x="poin:Q", y=alt.Y("label:N", sort=urutan), text="teks:N")
    return (batang + nol + teks_pos + teks_neg).properties(height=34 * len(top) + 40)


def tampilkan_hasil(h: dict, values: dict) -> None:
    prob, thr = h["probabilitas"], h["threshold"]
    berisiko = prob >= thr
    st.divider()
    st.subheader("Hasil")
    c1, c2, c3 = st.columns([1.1, 1, 1])
    c1.metric("Probabilitas diabetes (estimasi model)", persen(prob, 1))
    if berisiko:
        c2.error("**Berisiko diabetes**\n\nDisarankan memeriksakan gula darah di fasilitas kesehatan.")
    else:
        c2.success("**Risiko rendah**\n\nTetap jaga pola hidup sehat dan lakukan pemeriksaan rutin.")
    c3.info("Hasil ini estimasi skrining dari data survei, **bukan diagnosis medis**.")

    # ---------------- Fitur unik 1 ----------------
    st.markdown("#### Fitur unik 1: Threshold untuk data tidak seimbang")
    st.altair_chart(grafik_threshold(prob, thr), width="stretch")
    b1, b2 = st.columns(2)
    kat_model = "Berisiko diabetes" if berisiko else "Risiko rendah"
    kat_bawaan = "Berisiko diabetes" if prob >= THRESHOLD_BAWAAN else "Risiko rendah"
    b1.markdown(f"**Threshold model ({angka(thr, 2)})**  \nKeputusan: **{kat_model}**")
    b2.markdown(f"**Threshold bawaan ({angka(THRESHOLD_BAWAAN, 1)})**  \nKeputusan: **{kat_bawaan}**")
    if thr <= prob < THRESHOLD_BAWAAN:
        st.warning(f"Responden ini **hanya tertangkap karena threshold {angka(thr, 2)}**. Dengan threshold bawaan "
                   f"{angka(THRESHOLD_BAWAAN, 1)}, probabilitas {persen(prob, 1)} akan dianggap risiko rendah.")
    st.caption("Threshold 0,13 dipilih dari data validation dengan kriteria F2 maksimum (recall dianggap dua kali "
               "lebih penting dari precision). Di data test, recall naik dari 11,14% pada threshold 0,5 menjadi "
               "80,94% pada threshold 0,13. Harganya, precision turun menjadi 30,43%, sehingga sebagian orang "
               "yang ditandai berisiko sebenarnya tidak diabetes.")

    # ---------------- Fitur unik 2 ----------------
    st.markdown("#### Fitur unik 2: Faktor yang memengaruhi perhitungan model (Integrated Gradients)")
    base = h["baseline_tepat"]
    total = float(h["kontribusi"]["kontribusi"].sum())
    selisih = prob - base - total
    st.markdown(f"Pembanding: **profil responden rata-rata** dengan risiko {persen(base, 1)}. "
                f"Merah menaikkan risiko, biru menurunkan risiko, dalam poin persen.")
    st.altair_chart(grafik_kontribusi(h["kontribusi"]), width="stretch")
    st.caption(f"Cek completeness: {persen(base, 2)} (baseline) {poin(total)} poin (jumlah 21 kontribusi) = "
               f"{persen(base + total, 2)}, sedangkan probabilitas model {persen(prob, 2)}. Selisihnya "
               f"{angka(abs(selisih) * 100, 3)} poin, dihitung dengan 128 langkah integrasi.")
    with st.expander("Lihat kontribusi semua 21 faktor"):
        tabel = h["kontribusi"][["nama", "nilai", "kontribusi"]].copy()
        tabel["kontribusi"] = tabel["kontribusi"].map(poin)
        tabel.columns = ["Faktor", "Jawaban", "Kontribusi (poin persen)"]
        st.dataframe(tabel, hide_index=True, width="stretch")
    st.caption("Penjelasan ini menunjukkan pola yang dipakai model, bukan hubungan sebab akibat dan bukan saran medis. "
               "Contohnya pendapatan atau kesehatan mental bisa menggeser perhitungan model, tetapi bukan berarti "
               "faktor tersebut menyebabkan diabetes.")
    

# ---------------------------------------------------------------------------
# Halaman
# ---------------------------------------------------------------------------
def halaman_cek() -> None:
    st.title("🩺 Cek Risiko Diabetes")
    st.markdown("Isi 21 pertanyaan berikut sesuai kondisi Anda. Model MLP akan memperkirakan risiko diabetes "
                "dan menjelaskan faktor yang paling memengaruhi hasilnya.")
    if st.session_state.get("sumber_input"):
        st.info(f"Form diisi dari contoh kasus: **{st.session_state['sumber_input']}**. Ubah jawaban mana pun "
                "lalu tekan tombol di bawah untuk menghitung ulang.")
    values, galat = form_input()
    a, b, _ = st.columns([1, 1, 3])
    hitung = a.button("Hitung risiko", type="primary", width="stretch")
    b.button("Kosongkan form", width="stretch", on_click=isi_form, args=(NILAI_AWAL,), kwargs={"mode_bmi": MODE_HITUNG})
    if hitung:
        if galat:
            st.error(galat)
            st.session_state.pop("hasil", None)
        else:
            try:
                st.session_state["hasil"] = (prediksi(values), values)
            except ValueError as e:
                st.error(f"Input tidak valid: {e}")
    if galat and "hasil" in st.session_state:
        st.session_state.pop("hasil", None)
    if "hasil" in st.session_state:
        hasil, nilai = st.session_state["hasil"]
        if nilai == values:
            tampilkan_hasil(hasil, nilai)
        else:
            st.caption("Jawaban berubah setelah perhitungan terakhir. Tekan **Hitung risiko** untuk memperbarui hasil.")


def halaman_contoh(halaman_cek_obj) -> None:
    st.title("Contoh Kasus dari Data Testing")
    data = muat_json("app/contoh_kasus.json")
    thr = muat_artefak()["model_config"]["threshold"]
    st.markdown("Responden di bawah adalah data asli dari data test yang tidak pernah dipakai saat training. "
                "Untuk setiap kategori dipilih responden dengan probabilitas **paling dekat ke median** kategorinya, "
                "jadi contoh ini mewakili kasus yang tipikal, bukan kasus yang dipilih agar model terlihat bagus. "
                "Label prediabetes diambil dari file pembanding `Diabetes_012` pada dataset yang sama.")
    for k in data["kasus"]:
        v = k["nilai"]
        benar = (k["prob"] >= thr) == (k["label"] == 1)
        with st.container(border=True):
            c1, c2 = st.columns([3, 1.2])
            with c1:
                st.markdown(f"**{'✅' if benar else '❌'} {k['judul']}**")
                st.markdown(
                    f"Label asli: **{config.CLASS_LABELS[k['label']]}** (Diabetes_012: {LABEL_012[k['label_012']]})  \n"
                    f"Probabilitas model: **{persen(k['prob'], 1)}**, keputusan: "
                    f"**{'Berisiko diabetes' if k['prob'] >= thr else 'Risiko rendah'}**")
                ringkas = (f"{config.SEX_LABELS[v['Sex']]}, {config.AGE_LABELS[v['Age']]} tahun, BMI {v['BMI']}, "
                           f"kesehatan umum {config.GENHLTH_LABELS[v['GenHlth']].lower()}, "
                           f"tekanan darah tinggi {config.YES_NO_LABELS[v['HighBP']].lower()}, "
                           f"kolesterol tinggi {config.YES_NO_LABELS[v['HighChol']].lower()}")
                st.caption(ringkas)
                st.caption(f"Kategori ini berisi {ribuan(k['jumlah_di_kategori'])} responden di data test, median "
                           f"probabilitasnya {persen(k['median_prob_kategori'], 1)}.")
            with c2:
                if st.button("Muat ke form", key=f"muat_{k['kode']}", width="stretch",
                             on_click=isi_form, args=(v,), kwargs={"sumber": k["judul"]}):
                    st.switch_page(halaman_cek_obj)


def halaman_tentang() -> None:
    st.title("Tentang Model")
    m = muat_json("reports/results/07_metrik_final_test.json")
    cfg = muat_artefak()["model_config"]
    st.markdown(
        f"Model adalah **Multilayer Perceptron** [128-64-32-16] dengan BatchNorm, L2 1e-6, ReLU, dan output sigmoid "
        f"(14.417 parameter), dilatih dengan Adam dan binary cross-entropy pada **Diabetes Health Indicators Dataset** "
        f"(BRFSS 2015, CDC). Keputusan memakai threshold **{angka(cfg['threshold'], 2)}**. Semua angka di halaman ini "
        f"berasal dari evaluasi sekali di data test ({ribuan(m['jumlah_test'])} responden, "
        f"{ribuan(m['jumlah_diabetes_test'])} diabetes).")
    nama = {"Recall (sensitivitas)": "Recall (sensitivitas)", "Precision": "Precision", "Specificity": "Specificity",
            "F1-score": "F1-score", "F2-score": "F2-score", "Accuracy": "Accuracy", "PR-AUC": "PR-AUC", "ROC-AUC": "ROC-AUC"}
    baris = []
    for k in nama:
        val = m["metrik"][k]
        lo, hi = m["ci95_bootstrap"][k]
        pakai_persen = k in ("Recall (sensitivitas)", "Precision", "Specificity", "Accuracy")
        f = (lambda x: persen(x, 2)) if pakai_persen else (lambda x: angka(x, 4))
        baris.append({"Metrik": k, "Nilai di data test": f(val), "IK 95% (bootstrap)": f"{f(lo)} sampai {f(hi)}"})
    k1, k2 = st.columns([1.2, 1])
    with k1:
        st.markdown("##### Performa di data test")
        st.dataframe(pd.DataFrame(baris), hide_index=True, width="stretch")
        t = m["tanpa_fitur_unik_threshold_05"]
        st.caption(f"Tanpa fitur unik (threshold 0,5): recall {persen(t['recall'])}, precision {persen(t['precision'])}, "
                   f"F2 {angka(t['f2'])}.")
    with k2:
        st.markdown("##### Confusion matrix (threshold 0,13)")
        st.image(str(ROOT / "reports/figures/eval_01_confusion_matrix_test.png"), width="stretch")
    st.markdown("##### Arsitektur")
    st.image(str(ROOT / "reports/figures/lap_01_arsitektur_final.png"), width="stretch")
    st.markdown("##### Keterbatasan")
    st.markdown(
        "- Ini alat **skrining awal**, bukan diagnosis. Dari 10 orang yang ditandai berisiko, sekitar 3 yang benar diabetes.\n"
        "- Recall pada usia 18 sampai 44 tahun hanya 45,70%, jauh di bawah usia 65 tahun ke atas (86,85%). Hasil "
        "\"risiko rendah\" pada pengguna muda kurang bisa diandalkan.\n"
        "- Label berasal dari pengakuan responden survei telepon di Amerika Serikat tahun 2015, sehingga belum tentu "
        "berlaku sama untuk populasi lain.\n"
        "- Probabilitas cenderung sedikit lebih rendah dari kenyataan (rata-rata 13,92% berbanding 15,30% di data test).\n"
        "- Penderita yang profil jawabannya terlihat sehat sulit dikenali dari kuesioner saja.")
    st.markdown(f"Source code: [{GITHUB_URL}]({GITHUB_URL})")


def main() -> None:
    pastikan_state()
    p_cek = st.Page(halaman_cek, title="Cek Risiko", icon="🩺", default=True)
    p_contoh = st.Page(lambda: halaman_contoh(p_cek), title="Contoh Kasus", icon="🧪", url_path="contoh-kasus")
    p_tentang = st.Page(halaman_tentang, title="Tentang Model", icon="📊", url_path="tentang-model")
    nav = st.navigation([p_cek, p_contoh, p_tentang])
    with st.sidebar:
        st.markdown("**Prediksi Risiko Diabetes dengan MLP**")
        st.caption("Proyek UTS Pembelajaran Mendalam, Semester Ganjil TA 2026-2027.")
        st.caption("Hasil aplikasi adalah estimasi risiko berbasis data survei, bukan diagnosis medis.")
    nav.run()


main()
