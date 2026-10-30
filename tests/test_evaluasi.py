"""Tes evaluasi kesepakatan Bloom per CPMK (CLAUDE.md §7): weighted kappa, akurasi ±1, afektif terpisah."""
from __future__ import annotations

import pandas as pd
import pytest

from src import db
from src.export import ekspor_evaluasi, kappa_berbobot, kesepakatan, template_pakar
from src.llm.beban import impor_pakar, jalankan_penilaian_bloom
from tests.conftest import buat_cfg


def test_kappa_berbobot_hitungan_manual():
    """a = [1,2,3], b = [1,3,3]. O = 1/3 di (1,1), (2,3), (3,3); marginal a = 1/3 ×3, b: 1 → 1/3, 3 → 2/3.
    Linear |i−j|: ΣwO = 1/3; ΣwE = (2/9·2 + 1/9·1 + 2/9·1 + 1/9·2) = 9/9 → κ = 1 − (1/3)/1 = 0,6667.
    Kuadratik (i−j)²: ΣwO = 1/3; ΣwE = (2/9·4 + 1/9·1 + 2/9·1 + 1/9·4) = 15/9 → κ = 1 − 0,2 = 0,8."""
    assert kappa_berbobot([1, 2, 3], [1, 3, 3], "linear") == pytest.approx(0.6667, abs=1e-4)
    assert kappa_berbobot([1, 2, 3], [1, 3, 3], "kuadratik") == pytest.approx(0.8, abs=1e-4)
    assert kappa_berbobot([1, 4, 6, 2], [1, 4, 6, 2]) == 1.0
    assert kappa_berbobot([3, 3], [3, 3]) is None        # tidak terdefinisi (tanpa variasi)
    assert kappa_berbobot([], []) is None


def test_kappa_berbobot_simetris_dan_lebih_buruk_bila_jauh():
    dekat = kappa_berbobot([1, 2, 3, 4, 5, 6], [2, 2, 3, 4, 5, 5])
    jauh = kappa_berbobot([1, 2, 3, 4, 5, 6], [6, 2, 3, 4, 5, 1])
    assert dekat > jauh
    assert kappa_berbobot([1, 2, 3, 4], [2, 2, 4, 4]) == kappa_berbobot([2, 2, 4, 4], [1, 2, 3, 4])


def test_kesepakatan_afektif_dipisah_dan_akurasi_pm1():
    r1 = {"a": "C2", "b": "C4", "c": "C6", "d": "A", "e": "C3"}
    r2 = {"a": "C2", "b": "C5", "c": "C3", "d": "A", "e": "A", "x": "C1"}
    k = kesepakatan(r1, r2)
    assert (k["n_cpmk"], k["n_kognitif"]) == (5, 3)          # e tidak masuk kognitif (salah satu A)
    assert k["akurasi_eksak"] == pytest.approx(1 / 3, abs=1e-4) and k["akurasi_pm1"] == pytest.approx(2 / 3, abs=1e-4)
    assert (k["afektif_cocok"], k["afektif_tidak_cocok"]) == (4, 1)
    sub = kesepakatan(r1, r2, subset={"a", "b"})
    assert sub["n_cpmk"] == 2 and sub["akurasi_eksak"] == 0.5


def test_ekspor_evaluasi_tanpa_dan_dengan_pakar(tmp_path):
    cfg = buat_cfg(tmp_path, llm={"n_pengulangan": 1})
    r = ekspor_evaluasi(cfg)
    assert r["sumber_tersedia"] == ["heuristik"] and not r["kesepakatan"] and "belum diimpor" in r["catatan"] and "banding_llm_pakar" in r["catatan"]
    jalankan_penilaian_bloom(cfg, "mock", kode_list=["COM60014", "CIE61206"])
    template = pd.read_csv(template_pakar(cfg), dtype=str, keep_default_na=False)
    isi = template[template["kode_mk"].isin(["COM60014", "CIE61206"])].copy()
    isi["kode_pakar"] = "P1"
    isi["bloom"] = ["C3", "C3", "C6", "C3", "C3", "C2", "C2", "C6", "C4", "C2"]
    assert impor_pakar(cfg, isi)["n_dinilai"] == 10
    r = ekspor_evaluasi(cfg)
    assert {"llm_mock~heuristik", "heuristik~pakar"} <= set(r["kesepakatan"])
    hp = r["kesepakatan"]["heuristik~pakar"]["semua"]
    assert hp["n_cpmk"] == 10 and 0 <= hp["akurasi_eksak"] <= 1
    tabel = pd.read_csv(cfg.ekspor_dir / "bloom_perbandingan.csv")
    assert {"bloom_heuristik", "bloom_llm_mock", "bloom_pakar"} <= set(tabel.columns) and len(tabel) == 348
    assert db.baca_ringkasan(cfg, "evaluasi")["kesepakatan"]
