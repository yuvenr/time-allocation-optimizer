"""Tes statistik RM1 (src/statistik.py) dan evaluasi tingkat MK (src/export.evaluasi_rm1).
Nilai acuan: contoh baku Shrout & Fleiss (1979) dan hitungan manual di docstring."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import db
from src.export import ekspor_evaluasi, evaluasi_rm1, template_pakar
from src.llm.beban import impor_pakar, jalankan_penilaian_bloom
from src.statistik import bland_altman, icc, kategori_koo_li, mae, uji_t_berpasangan
from tests.conftest import buat_cfg

SHROUT_FLEISS = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], float)


def test_icc_contoh_shrout_fleiss():
    """Shrout & Fleiss (1979): ICC(2,1) = ICC(A,1) = 0,29 [0,019; 0,761]; ICC(3,1) = ICC(C,1) = 0,71 [0,342; 0,946]."""
    r = icc(SHROUT_FLEISS)
    assert (r["n_objek"], r["n_penilai"]) == (6, 4)
    assert r["ICC_A1"] == pytest.approx(0.2898, abs=1e-4) and r["ICC_C1"] == pytest.approx(0.7148, abs=1e-4)
    assert r["ICC_A1_ci95"] == pytest.approx((0.0188, 0.7611), abs=1e-3)
    assert r["ICC_C1_ci95"] == pytest.approx((0.3425, 0.9459), abs=1e-3)
    # komponen varians: σ²objek = (MSR−MSE)/k, σ²penilai = (MSC−MSE)/n, σ²galat = MSE
    assert r["var_objek"] == pytest.approx((r["MSR"] - r["MSE"]) / 4)
    assert r["var_penilai"] == pytest.approx((r["MSC"] - r["MSE"]) / 6)
    assert r["ICC_A1"] <= r["ICC_C1"]                     # ICC(A,1) ≤ ICC(C,1) (Bab 2.9.1)
    assert r["kategori_A1"] == "buruk"


def test_icc_hitungan_manual_dua_penilai():
    """Matriks [[2,3],[4,3],[6,5]]: MSR 4,6667; MSC 0,1667; MSE 0,6667 →
    ICC(C,1) = 4/5,3333 = 0,75; ICC(A,1) = 4/(5,3333 + 2(0,1667−0,6667)/3) = 4/5 = 0,80."""
    r = icc(pd.DataFrame({"P1": [2, 4, 6], "P2": [3, 3, 5]}))
    assert (r["MSR"], r["MSC"], r["MSE"]) == pytest.approx((14 / 3, 1 / 6, 2 / 3))
    assert r["ICC_C1"] == pytest.approx(0.75) and r["ICC_A1"] == pytest.approx(0.80)
    assert "catatan" in icc(pd.DataFrame({"a": [3, 3, 3], "b": [3, 3, 3]}))     # varians nol
    assert "catatan" in icc(pd.DataFrame({"a": [1.0]}))
    assert icc(pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [1, 3, 5, 4]}))["n_objek"] == 3   # baris kosong dibuang


@pytest.mark.parametrize("nilai,harap", [(0.49, "buruk"), (0.5, "sedang"), (0.75, "baik"), (0.9, "baik"),
                                         (0.91, "sangat baik"), (None, None), (float("nan"), None)])
def test_kategori_koo_li(nilai, harap):
    assert kategori_koo_li(nilai) == harap


def test_mae_bland_altman_uji_t_manual():
    """x = [3, 4, 5, 6], y = [2, 4, 4, 5] → d = [1, 0, 1, 1]: MAE 0,75; bias 0,75; SD(d) = 0,5;
    LoA = 0,75 ± 0,98 = [−0,23; 1,73]; t = 0,75 / (0,5/√4) = 3,0 (df 3)."""
    x, y = [3, 4, 5, 6], [2, 4, 4, 5]
    assert mae(x, y) == pytest.approx(0.75)
    ba = bland_altman(x, y)
    assert (ba["bias"], ba["sd_selisih"]) == pytest.approx((0.75, 0.5))
    assert (ba["loa_bawah"], ba["loa_atas"]) == pytest.approx((0.75 - 0.98, 0.75 + 0.98))
    t = uji_t_berpasangan(x, y)
    assert t["t"] == pytest.approx(3.0) and t["df"] == 3 and 0 < t["p"] < 0.1 and "shapiro_p" in t
    assert t["cohen_dz"] == pytest.approx(1.5)
    assert "catatan" in uji_t_berpasangan([1, 2], [1, 2]) and "catatan" in bland_altman([1], [2])
    assert "catatan" in uji_t_berpasangan([2, 3, 4], [1, 2, 3])                     # selisih konstan


def test_evaluasi_rm1_dengan_pakar(tmp_path):
    """Dua pakar menilai 3 MK; setiap pakar memberi label sama untuk semua CPMK satu MK sehingga skor MK
    (median level) = level itu: P1 = [2, 4, 6], P2 = [3, 3, 5] → rata pakar [2,5; 3,5; 5,5];
    ICC(A,1) antarpakar = 0,80 (lihat hitungan manual di atas)."""
    cfg = buat_cfg(tmp_path, llm={"n_pengulangan": 2})
    mk = ["COM60014", "CIE61206", "CIE62303"]
    jalankan_penilaian_bloom(cfg, "mock", kode_list=mk)
    form = pd.read_csv(template_pakar(cfg), dtype=str, keep_default_na=False)
    isi = form[form["kode_mk"].isin(mk)]
    p1 = isi.assign(kode_pakar="P1", bloom=isi["kode_mk"].map(dict(zip(mk, ["C2", "C4", "C6"]))))
    p2 = isi.assign(kode_pakar="P2", bloom=isi["kode_mk"].map(dict(zip(mk, ["C3", "C3", "C5"]))))
    impor_pakar(cfg, pd.concat([p1, p2]))
    tabel, r = evaluasi_rm1(cfg)
    t = tabel.set_index("kode_mk").loc[mk]
    assert list(t["skor_pakar_P1"]) == [2, 4, 6] and list(t["skor_pakar_P2"]) == [3, 3, 5]
    assert list(t["rata_pakar"]) == [2.5, 3.5, 5.5]
    assert list(t["skor_llm"]) == list(t["skor_heuristik"])                 # mock = heuristik
    assert {"skor_llm_run1", "skor_llm_run2"} <= set(tabel.columns)
    assert r["sumber_llm"] == "llm_mock" and "BUKAN hasil penelitian" in r["peringatan"]
    assert r["n_pakar"] == 2 and r["antarpakar"]["ICC_A1"] == pytest.approx(0.8)
    utama = r["utama_llm_vs_rata_pakar"]
    assert utama["n_objek"] == 3 and utama["MAE"] == pytest.approx(float((t["skor_llm"] - t["rata_pakar"]).abs().mean()),
                                                                   abs=1e-4)
    assert r["antarpengulangan_llm"]["ICC_A1"] == pytest.approx(1.0)        # mock deterministik
    ringkasan = ekspor_evaluasi(cfg)
    assert ringkasan["rm1"]["antarpakar"]["ICC_A1"] == pytest.approx(0.8)
    ekspor = pd.read_csv(cfg.ekspor_dir / "rm1_skor_mk.csv")
    assert len(ekspor) == 72 and ekspor["rata_pakar"].notna().sum() == 3
    assert db.baca_hasil(cfg, "rm1_skor_mk") is not None


def test_evaluasi_rm1_tanpa_pakar(tmp_path):
    cfg = buat_cfg(tmp_path)
    _, r = evaluasi_rm1(cfg)
    assert "belum diimpor" in r["catatan"] and "banding_llm_pakar" in r["catatan"] and "utama_llm_vs_rata_pakar" not in r
