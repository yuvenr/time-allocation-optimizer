"""Tes metrik dan simulasi komparasi mode arm vs hybrid (UC-12). Simulasi dijalankan pada fixture
mini buatan tangan (KRS fiktif berkode Kurikulum 2020 → dinormalisasi ke 2024), bukan data sintetis."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src import db
from src.config import load_config
from src.logutil import baca_log
from src.metrics import METRIK_BOOLEAN, METRIK_NUMERIK, jaccard, metrik_rencana
from src.schemas import ItemRencana, Plan, Profile, SemesterRencana, Transaksi, ValidationResult
from src.simulate import bentuk_profil, jalankan_simulasi

CFG = load_config(use_override_file=False, use_env=False)


def _plan(sems, theta=100.0, per_aturan=None):
    s = [SemesterRencana(semester=n, total_beban=b, total_sks=k,
                         items=[ItemRencana(kode=x, nama=x, sks=3, sifat="wajib", semester_anjuran=1) for x in mk])
         for n, b, k, mk in sems]
    per = per_aturan or {f"AD-0{i}": True for i in range(1, 10)}
    return Plan(id="t", mode="hybrid", dibuat=datetime.now(), profil=Profile(semester_saat_ini=5, ip_terakhir=3),
                theta=theta, semester=s, validasi=ValidationResult(lolos=all(per.values()), per_aturan=per))


def test_jaccard():
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert jaccard({"a"}, set()) == 0.0


def test_metrik_rencana_nilai_benar():
    plan = _plan([(5, 90.0, 18, ["A", "B"]), (6, 110.0, 21, ["C"]), (7, 0.0, 0, []), (8, 60.0, 6, ["D"])])
    aktual = {5: {"A", "B"}, 6: {"C", "X"}, 7: set(), 8: {"D"}}
    m = metrik_rencana(plan, CFG, aktual)
    b = pd.Series([90.0, 110.0, 60.0])
    assert m["sd_beban"] == pytest.approx(b.std(ddof=0), abs=1e-4)  # semester kosong diabaikan
    assert m["beban_puncak"] == 110.0
    assert m["sd_sks"] == pytest.approx(pd.Series([18, 21, 6]).std(ddof=0), abs=1e-4)
    assert m["jaccard_rata"] == pytest.approx((1 + 0.5 + 1) / 3, abs=1e-4)  # S7 kosong di keduanya dilewati
    assert m["lolos_theta"] is False  # 110 > θ 100
    assert m["lolos_semua"] and m["n_semester_terisi"] == 3


def test_jaccard_pers_2_12_hanya_semester_direkomendasikan():
    """Pers. 2.12: T = semester yang DIREKOMENDASIKAN. S5: J({A,B},{A}) = 0,5; S6 rencana kosong tetapi alumni
    mengambil X → tidak dihitung; S9 (di luar data alumni) direkomendasikan C → J = 0. Rata (0,5 + 0)/2 = 0,25."""
    plan = _plan([(5, 50.0, 6, ["A", "B"]), (6, 0.0, 0, []), (9, 20.0, 3, ["C"])])
    m = metrik_rencana(plan, CFG, {5: {"A"}, 6: {"X"}})
    assert m["jaccard_rata"] == pytest.approx(0.25) and m["semester_selesai"] == 9


def test_lolos_145_harus_dalam_horizon():
    """145 SKS tercapai dalam delapan semester. Awal 120 SKS lulus; S5–S8 menambah 24 SKS → 144 < 145
    (gagal walau AD-01 lolos karena rencana diperpanjang ke S9); +1 SKS di S8 → 145 → lolos."""
    per = {f"AD-0{i}": True for i in range(1, 10)}
    sems = [(5, 10.0, 6, ["A"]), (6, 10.0, 6, ["B"]), (7, 10.0, 6, ["C"]), (8, 10.0, 6, ["D"]), (9, 10.0, 3, ["E"])]
    plan = _plan(sems, per_aturan=per).model_copy(update={"sks_lulus_awal": 120})
    assert not metrik_rencana(plan, CFG)["lolos_145"]
    sems[3] = (8, 10.0, 7, ["D"])
    plan = _plan(sems, per_aturan=per).model_copy(update={"sks_lulus_awal": 120})
    assert metrik_rencana(plan, CFG)["lolos_145"]


def test_metrik_kepatuhan_dari_validator():
    per = {f"AD-0{i}": True for i in range(1, 10)} | {"AD-03": False, "AD-01": False}
    m = metrik_rencana(_plan([(5, 50.0, 12, ["A"])], per_aturan=per), CFG)
    assert not m["lolos_prasyarat"] and not m["lolos_145"] and m["lolos_sks"] and not m["lolos_semua"]


def _tr(s, lulus, gagal=(), ip=3.0):
    """Transaksi berisi MK yang diambil (lulus + tidak lulus); ``gagal`` = yang tidak lulus."""
    return Transaksi(id_anon="x", angkatan=2017, semester_ke=s, ip_semester=ip,
                     items=[f"{k}@S{s}" for k in list(lulus) + list(gagal)], gagal=list(gagal), sumber="sintetis")


def test_bentuk_profil_potong_di_k():
    riwayat = [_tr(1, ["A", "B"], ["C"], 3.1), _tr(2, ["D"], [], 2.7), _tr(3, ["C", "E"], [], 3.4), _tr(4, ["F"])]
    prof, aktual = bentuk_profil(riwayat, 2)
    assert prof.semester_saat_ini == 3 and prof.ip_terakhir == 2.7
    assert prof.mk_lulus == {"A", "B", "D"} and prof.mk_pernah_gagal == {"C"}
    assert aktual == {3: {"C", "E"}, 4: {"F"}}
    assert bentuk_profil(riwayat, 4) is None  # tidak ada semester setelah k


def test_bentuk_profil_menyaring_kode_tak_dikenal():
    """Item yang tidak dapat direncanakan (OLD:<kode> = MK 2020 dihapus, kode asing) dibuang dari profil
    dan dari MK aktual bila ``kode_dikenal`` diberikan."""
    riwayat = [_tr(1, ["A", "OLD:X"], ["Z"]), _tr(2, ["B", "OLD:Y"])]
    prof, aktual = bentuk_profil(riwayat, 1, kode_dikenal={"A", "B"})
    assert prof.mk_lulus == {"A"} and prof.mk_pernah_gagal == set()
    assert aktual == {2: {"B"}}


def test_simulasi_menolak_transaksi_uji_basi(simulasi_mini):
    """Transaksi uji yang dibentuk dari data KRS lain (sidik berbeda) tidak boleh disimulasikan."""
    cfg, _ = simulasi_mini
    with open(cfg.raw_dir / "histori_krs.csv", "a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="basi"):
        jalankan_simulasi(cfg)


@pytest.fixture
def simulasi_mini(cfg_mini):
    from src.arm import jalankan_arm
    from src.preprocessing import jalankan_preprocessing

    jalankan_preprocessing(cfg_mini)
    jalankan_arm(cfg_mini)
    return cfg_mini, jalankan_simulasi(cfg_mini)


def test_simulasi_format_berpasangan(simulasi_mini):
    """Fixture mini: 1 alumni uji dengan 4 semester → profil k = 2 dan 3 (k ≥ 4 tidak punya semester sesudahnya)."""
    cfg, hasil = simulasi_mini
    df = pd.read_csv(cfg.ekspor_dir / "simulasi_metrik.csv")
    # asal riwayat (sumber_data) hanya di basis data, tidak ikut diekspor
    assert "sumber_data" not in df.columns
    pd.testing.assert_frame_equal(df, db.baca_hasil(cfg, "simulasi_metrik").drop(columns="sumber_data"),
                                  check_dtype=False)
    assert len(df) == hasil["n_profil"] == 2 and sorted(df["k"]) == [2, 3]
    assert not df.duplicated(["id_anon", "k"]).any()  # satu baris per profil
    for m in METRIK_NUMERIK + METRIK_BOOLEAN:
        assert f"{m}_arm" in df.columns and f"{m}_hybrid" in df.columns
    for m in METRIK_BOOLEAN:
        assert set(df[f"{m}_arm"]) <= {0, 1}
    ringkas = db.baca_hasil(cfg, "simulasi_ringkasan")
    assert set(ringkas["metrik"]) == set(METRIK_NUMERIK + METRIK_BOOLEAN)


def test_simulasi_definition_of_done(simulasi_mini):
    """Hybrid tidak pernah melampaui θ; kedua mode lolos AD-02/03/04 (atau sebab tercatat)."""
    cfg, hasil = simulasi_mini
    df = db.baca_hasil(cfg, "simulasi_metrik")
    assert (df["lolos_theta_hybrid"] == 1).all()
    for mode in ("arm", "hybrid"):
        keras = df[[f"lolos_prasyarat_{mode}", f"lolos_sks_{mode}", f"lolos_paritas_{mode}"]].all(axis=1)
        assert keras.all(), df[f"pelanggaran_{mode}"].tolist()
        assert hasil["profil_gagal_aturan_keras"][mode] == []
    assert len(baca_log(cfg, "simulasi")) == 1
