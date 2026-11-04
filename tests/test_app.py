"""Uji otomatis aplikasi: semua halaman terbuka tanpa error, dan angka aplikasi sama dengan notebook.

Jalankan dari root repository:  python -m pytest tests -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
APP = str(ROOT / "app" / "app.py")
KASUS = json.loads((ROOT / "app" / "contoh_kasus.json").read_text(encoding="utf-8"))["kasus"]


@pytest.fixture(scope="module")
def art():
    from src import inference
    return inference.load_artifacts()


def test_probabilitas_contoh_kasus_sama_dengan_notebook(art):
    """Probabilitas lewat predict_and_explain harus sama dengan probabilitas yang dihitung dari data test."""
    from src import inference
    for k in KASUS:
        h = inference.predict_and_explain(k["nilai"], art)
        assert abs(h["probabilitas"] - k["prob"]) < 1e-5, k["kode"]
        assert abs(h["selisih_completeness"]) < 0.01, k["kode"]


def test_threshold_dan_kategori(art):
    assert art["model_config"]["threshold"] == 0.13
    from src import inference
    assert inference.risk_category(0.13, 0.13) == "Berisiko diabetes"
    assert inference.risk_category(0.1299, 0.13) == "Risiko rendah"


def test_validasi_menolak_bmi_di_luar_rentang(art):
    from src import inference
    v = dict(KASUS[0]["nilai"], BMI=80)
    with pytest.raises(ValueError):
        inference.predict_and_explain(v, art)


def test_halaman_cek_risiko_menghitung():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    at.button[0].click().run()          # tombol "Hitung risiko"
    assert not at.exception
    teks = " ".join(m.value for m in at.markdown)
    assert "Fitur unik 1" in teks and "Fitur unik 2" in teks
    assert at.metric[0].value.endswith("%")


def test_bmi_dari_tinggi_berat_di_luar_rentang():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.radio(key="mode_bmi").set_value("Hitung dari tinggi dan berat badan").run()
    at.number_input(key="tinggi").set_value(150).run()
    at.number_input(key="berat").set_value(200).run()   # BMI 88,9 -> ditolak
    at.button[0].click().run()
    assert not at.exception
    assert any("di luar rentang" in e.value for e in at.error)


@pytest.mark.parametrize("kode", [k["kode"] for k in KASUS])
def test_muat_contoh_kasus(kode):
    at = AppTest.from_file(APP, default_timeout=120).run()
    k = next(x for x in KASUS if x["kode"] == kode)
    # isi state seperti tombol "Muat ke form", lalu hitung
    for c, v in k["nilai"].items():
        at.session_state[f"f_{c}"] = v
    at.session_state["mode_bmi"] = "Isi BMI langsung"
    at.session_state["bmi_langsung"] = k["nilai"]["BMI"]
    at.run()
    at.button[0].click().run()
    assert not at.exception
    tampil = at.metric[0].value
    harap = f"{k['prob'] * 100:.1f}%".replace(".", ",")
    assert tampil == harap, (tampil, harap)
