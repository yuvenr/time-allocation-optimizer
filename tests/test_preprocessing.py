"""Tes preprocessing (UC-07) + normalisasi kode Kurikulum 2020 → 2024 (CLAUDE.md §6, AT21)."""
from __future__ import annotations

import pandas as pd
import pytest

from src import db
from src.config import load_config
from src.data_loader import Kurikulum, load_kurikulum, load_transaksi
from src.preprocessing import (anonimisasi, bentuk_transaksi, filter_masa_studi, jalankan_preprocessing,
                               label_item, normalisasi_kode, normalisasi_krs, normalisasi_riwayat, pisah_item,
                               profil_dari_riwayat, split_alumni, tandai_lulus)
from src.rules import Keadaan, cek_mk
from src.schemas import MataKuliah, Penyetaraan

CFG = load_config(use_override_file=False, use_env=False)
KUR = load_kurikulum(CFG)


def _kur():
    """Kurikulum buatan: B1..B4; penyetaraan L1 → B1 & B2 (1:n), L2/L3 → B3 (n:1), LH dihapus."""
    mk = {
        k: MataKuliah.model_validate({"kode_mk": k, "nama_mk": k, "sks": s, "status": "Wajib",
                                      "kategori": "Wajib Program Studi", "semester_rekomendasi_jalur_A": 1})
        for k, s in [("B1", 3), ("B2", 2), ("B3", 3), ("B4", 4)]
    }
    pen = [Penyetaraan.model_validate(r) for r in [
        {"kode_lama": "L1", "kode_baru": "B1", "sks_lama": 5, "sks_baru": 3, "jenis_perubahan": "berubah_kode_sks"},
        {"kode_lama": "L1", "kode_baru": "B2", "sks_lama": 5, "sks_baru": 2, "jenis_perubahan": "berubah_kode_sks"},
        {"kode_lama": "L2", "kode_baru": "B3", "sks_lama": 2, "sks_baru": 3, "jenis_perubahan": "berubah_kode_sks"},
        {"kode_lama": "L3", "kode_baru": "B3", "sks_lama": 2, "sks_baru": 3, "jenis_perubahan": "berubah_kode_sks"},
        {"kode_lama": "LH", "jenis_perubahan": "dihapus"},
    ]]
    return Kurikulum(mk=mk, penyetaraan=pen)


def _krs(rows):
    kol = ["nim", "nama", "angkatan", "semester_ke", "kode_mk", "sks", "nilai_huruf", "ip_semester",
           "kurikulum", "masa_studi_semester"]
    return pd.DataFrame(rows, columns=kol)


def test_anonimisasi_hash_dan_buang_identitas():
    df = _krs([["123", "Budi", 2017, 1, "B1", 3, "A", 3.5, "K", 8],
               ["123", "Budi", 2017, 1, "B2", 2, "B", 3.5, "K", 8],
               ["456", "Ani", 2017, 1, "B1", 3, "A", 3.0, "K", 8]])
    out = anonimisasi(df, CFG, salt="s")
    assert "nim" not in out.columns and "nama" not in out.columns
    assert out["id_anon"].nunique() == 2
    assert not out["id_anon"].isin(["123", "456"]).any()
    # deterministik untuk salt yang sama, berbeda untuk salt lain
    assert list(anonimisasi(df, CFG, salt="s")["id_anon"]) == list(out["id_anon"])
    assert list(anonimisasi(df, CFG, salt="t")["id_anon"]) != list(out["id_anon"])


def test_filter_masa_studi():
    df = _krs([["1", "x", 2017, 1, "B1", 3, "A", 3, "K", 8], ["2", "y", 2017, 1, "B1", 3, "A", 3, "K", 9]])
    assert list(filter_masa_studi(df, CFG)["nim"]) == ["1"]


def test_tandai_lulus_mengikuti_nilai_lulus_minimum():
    df = pd.DataFrame({"nilai_huruf": ["A", "D+", "D", "E", " e ", "T"]})
    out = tandai_lulus(df, CFG)
    assert list(out["lulus"]) == [True, True, True, False, False, False]    # NILAI_LULUS_MINIMUM = D (AD-07)
    assert list(out["nilai_dikenal"]) == [True, True, True, True, True, False]
    ketat = load_config({"aturan": {"NILAI_LULUS_MINIMUM": "C"}}, use_override_file=False, use_env=False)
    assert list(tandai_lulus(df, ketat)["lulus"]) == [True, False, False, False, False, False]


def _df(kode, nilai, semester=None):
    n = len(kode)
    return pd.DataFrame({"id_anon": ["a"] * n, "angkatan": [2020] * n, "semester_ke": semester or [1] * n,
                         "kode_mk": kode, "sks": [5, 2, 2, 3, 4, 3][:n], "nilai_huruf": nilai,
                         "ip_semester": [3] * n, "kurikulum": ["K2020"] * n, "masa_studi_semester": [8] * n})


def test_normalisasi_1n_n1_dihapus_dan_tidak_dikenal():
    df = tandai_lulus(_df(["L1", "L2", "L3", "LX", "B4", "LH"], ["A", "E", "B", "A", "A", "A"]), CFG)
    out, lap = normalisasi_krs(df, _kur(), CFG)
    assert sorted(out["kode_mk"]) == ["B1", "B2", "B3", "B4"]   # 1:n → B1+B2, n:1 → B3, kode 2024 tetap
    b = out.set_index("kode_mk")
    assert b.loc["B3", "lulus"]                                  # salah satu komponen n:1 lulus
    assert b.loc["B1", "sks"] == 3 and b.loc["B1", "sks_asal"] == 5   # SKS mengikuti Kurikulum 2024
    assert lap["mk_tidak_dikenal"] == {"LX": 1} and lap["mk_dihapus"] == {"LH": 1}
    assert lap["proporsi_baris_terpetakan"] == pytest.approx(round(4 / 6, 4))
    assert lap["baris_per_jenis"] == {"berubah_kode_sks": 3, "dihapus": 1, "kurikulum_2024": 1, "tidak_dikenal": 1}


def test_normalisasi_n1_lintas_semester():
    """L2 gagal di S1, L3 lulus di S2 → B3 lulus di S2; komponen yang diulang setelah lulus diabaikan."""
    df = tandai_lulus(_df(["L2", "L3", "L2"], ["E", "B", "A"], semester=[1, 2, 3]), CFG)
    out, _ = normalisasi_krs(df, _kur(), CFG)
    assert list(zip(out["semester_ke"], out["kode_mk"], out["lulus"])) == [(1, "B3", False), (2, "B3", True)]


def test_kebijakan_mk_dihapus_prefiks_old():
    cfg = load_config({"preprocessing": {"mk_dihapus": "prefiks_old"}}, use_override_file=False, use_env=False)
    out, lap = normalisasi_krs(tandai_lulus(_df(["LH", "B4"], ["A", "A"]), cfg), _kur(), cfg)
    assert sorted(out["kode_mk"]) == ["B4", "OLD:LH"] and lap["kebijakan_mk_dihapus"] == "prefiks_old"
    assert out.set_index("kode_mk").loc["OLD:LH", "sks"] == 5   # SKS asal dipertahankan


@pytest.mark.parametrize("kode,harap", [
    ("UBU60004", (["UBU60004"], "identik")),
    ("COM60014", (["COM60014"], "berubah_sks")),
    ("CIE62005", (["CIE62104"], "berubah_kode_nama")),
    ("CSD60004", (["CSD60709"], "berubah_kode_nama_sks_status")),
    ("CIE60044", ([], "dihapus")),
    ("CIE61403", (["CIE61403"], "kurikulum_2024")),   # mk_baru: tidak punya kode lama
    ("XYZ99999", ([], "tidak_dikenal")),
])
def test_normalisasi_kode_dataset_asli(kode, harap):
    assert normalisasi_kode(kode, KUR) == harap


def test_at21_prasyarat_dipenuhi_mk_lama_yang_disetarakan():
    """AT21: CIE62005 (lama) yang lulus memenuhi prasyarat CIE61108 (← CIE62104). Bernilai E: tidak memenuhi
    menurut AD-03 (harus lulus), tetapi memenuhi menurut tafsiran literal "≥ E" (flag config)."""
    profil, lap = profil_dari_riwayat({"CIE62005": "D", "CIE61010": "B", "CIE60044": "A"}, KUR, CFG,
                                      semester_saat_ini=5, ip_terakhir=3.0)
    assert profil.riwayat_nilai == {"CIE62104": "D", "CIE61402": "B"}
    assert profil.mk_lulus == {"CIE62104", "CIE61402"} and not profil.mk_pernah_gagal
    assert lap["dihapus"] == ["CIE60044"]
    kd = Keadaan.dari_profil(profil, KUR, CFG)
    assert kd.ipk == round((1.0 * 3 + 3.0 * 3) / 6, 2)                     # IPK dari riwayat ter-normalisasi
    assert {h.status for h in cek_mk("CIE61108", 5, kd, KUR, CFG) if h.kode == "AD-03"} == {"lolos"}
    gagal, _ = profil_dari_riwayat({"CIE62005": "E"}, KUR, CFG, semester_saat_ini=5, ip_terakhir=3.0)
    kd_e = Keadaan.dari_profil(gagal, KUR, CFG)
    assert {h.status for h in cek_mk("CIE61108", 5, kd_e, KUR, CFG) if h.kode == "AD-03"} == {"gagal"}
    literal = load_config({"aturan": {"PRASYARAT_E_BERARTI_PERNAH_MENEMPUH": True}}, use_override_file=False,
                          use_env=False)
    assert {h.status for h in cek_mk("CIE61108", 5, kd_e, KUR, literal) if h.kode == "AD-03"} == {"lolos"}


def test_normalisasi_riwayat_n1_memakai_nilai_terbaik():
    baru, _ = normalisasi_riwayat({"L2": "D", "L3": "B+", "B4": "C"}, _kur(), CFG)
    assert baru == {"B3": "B+", "B4": "C"}


def test_transaksi_berlabel_semester_mk_diambil_dan_gagal():
    """Transaksi = MK yang diambil (lulus maupun tidak); MK tidak lulus dicatat di ``gagal``."""
    df = pd.DataFrame({
        "id_anon": ["a", "a", "a", "b"], "angkatan": [2017] * 4, "semester_ke": [1, 1, 2, 1],
        "kode_mk": ["B1", "B2", "B2", "B1"], "lulus": [True, False, True, True], "ip_semester": [3, 3, 3.5, 2],
    })
    tr = bentuk_transaksi(df, "sintetis")
    assert len(tr) == 3 and {t.sumber for t in tr} == {"sintetis"}
    a1 = next(t for t in tr if t.id_anon == "a" and t.semester_ke == 1)
    assert a1.items == ["B1@S1", "B2@S1"] and a1.gagal == ["B2"] and a1.kode == {"B1", "B2"}
    a2 = next(t for t in tr if t.id_anon == "a" and t.semester_ke == 2)
    assert a2.items == ["B2@S2"] and a2.gagal == [] and a2.ip_semester == 3.5
    assert pisah_item(label_item("PTI101", 3)) == ("PTI101", 3)


def test_split_per_alumni_deterministik():
    ids = [f"id{i}" for i in range(100)]
    latih, uji = split_alumni(ids, CFG)
    assert len(uji) == 20 and len(latih) == 80
    assert not set(latih) & set(uji)
    assert split_alumni(list(reversed(ids)), CFG) == (latih, uji)


# ---- fixture mini buatan tangan (tests/fixtures/mini) -------------------------


def test_pipeline_mini_hitungan_manual(cfg_mini):
    """Hitungan manual atas tests/fixtures/mini/raw/histori_krs.csv + 11_penyetaraan asli.

    - 54 baris mentah, 6 mahasiswa; M06 masa studi 10 → dibuang (7 baris) → 47 baris.
    - CIE60044 berjenis dihapus di 11 → 2 baris dibuang (M04 & M05 S3) → 45/47 = 0.9574 terpetakan.
    - Kode unik setelah filter = 11, yang terpetakan 10 → 10/11 = 0.9091.
    - Baris per jenis (47): berubah_kode_nama 25 (CIE61014, CIE61013, CIE62016, CIE61011, CIE62018 × 5),
      berubah_kode_nama_status 6 (CIE61010 ×3, CIE62020 ×3), berubah_kode_nama_sks_status 4 (CSD60004),
      berubah_sks 5 (COM60014), berubah_nama 5 (COM60015), dihapus 2 (CIE60044).
    - SKS berubah: COM60014 4→5 (5 baris) + CSD60004 3→2 (4 baris) = 9; status berubah: CSD60004 (4) +
      CIE61010 (3) + CIE62020 (3) = 10.
    - Nilai tidak lulus (< D) setelah filter: hanya M02 CIE61010 (E) → 1 baris.
    - Transaksi (MK diambil) = 5 mahasiswa × 4 semester = 20; split 80:20 → 4 latih (16 transaksi), 1 uji (4).
    """
    r = jalankan_preprocessing(cfg_mini)
    assert r["n_baris_mentah"] == 54
    assert (r["n_alumni_awal"], r["n_alumni_masa_studi_ok"], r["n_alumni_dibuang_masa_studi"]) == (6, 5, 1)
    assert (r["n_baris"], r["n_baris_terpetakan"]) == (47, 45)
    assert r["proporsi_baris_terpetakan"] == pytest.approx(0.9574)
    assert (r["n_kode_unik"], r["n_kode_terpetakan"]) == (11, 10)
    assert r["proporsi_kode_terpetakan"] == pytest.approx(0.9091)
    assert r["mk_dihapus"] == {"CIE60044": 2} and r["mk_tidak_dikenal"] == {}
    assert r["baris_per_jenis"] == {"berubah_kode_nama": 25, "berubah_kode_nama_sks_status": 4,
                                    "berubah_kode_nama_status": 6, "berubah_nama": 5, "berubah_sks": 5,
                                    "dihapus": 2}
    assert (r["n_baris_berubah_sks"], r["n_baris_berubah_status"]) == (9, 10)
    assert r["mk_baru_cold_start"] == ["CIE61114", "CIE61120", "CIE61403", "CIE62117", "CIE62305", "CIE62306",
                                       "COM60061"]
    assert r["n_nilai_di_luar_skala"] == 0
    assert r["n_baris_gagal"] == 1
    assert (r["n_transaksi"], r["n_alumni_latih"], r["n_alumni_uji"]) == (20, 4, 1)
    assert (r["n_transaksi_asli"], r["sumber_riwayat"]) == (0, "fixture")
    assert (r["n_transaksi_latih"], r["n_transaksi_uji"]) == (16, 4)

    semua = load_transaksi(cfg_mini, "latih") + load_transaksi(cfg_mini, "uji")
    assert not {t.id_anon for t in load_transaksi(cfg_mini, "latih")} & {t.id_anon for t in load_transaksi(cfg_mini, "uji")}
    per_s = {s: sorted(tuple(sorted(pisah_item(i)[0] for i in t.items)) for t in semua if t.semester_ke == s)
             for s in (1, 2, 3, 4)}
    assert per_s[1] == [("COM60014", "COM60015")] * 5
    assert per_s[2] == [("CIE62201", "CIE62204")] * 5
    assert per_s[3] == sorted([("CIE61206", "CIE61208", "CIE61402"),   # M01
                               ("CIE61206", "CIE61208", "CIE61402"),   # M02 (CIE61402 diambil, gagal)
                               ("CIE61206", "CIE61208", "CIE61402"),   # M03
                               ("CIE61206", "CIE61208"),               # M04 (CIE60044 dibuang)
                               ("CIE61206",)])                         # M05 (CIE60044 dibuang)
    assert per_s[4] == sorted([("CIE62209", "CSD60709", "CSD60711"),   # M01
                               ("CIE62209", "CSD60709"),               # M02
                               ("CIE62209", "CSD60709", "CSD60711"),   # M03
                               ("CIE62209", "CSD60711"),               # M04
                               ("CIE61208", "CIE62209", "CSD60709")])  # M05 (CIE61011 diambil di S4)
    assert sorted(g for t in semua for g in t.gagal) == ["CIE61402"]

    bersih = db.baca(cfg_mini, "SELECT * FROM histori")
    # COM60014: 4 SKS di Kurikulum 2020 → 5 SKS di Kurikulum 2024 (berubah_sks)
    assert set(bersih.loc[bersih.kode_mk == "COM60014", "sks"]) == {5}
    assert set(bersih.loc[bersih.kode_mk == "COM60014", "kode_asal"]) == {"COM60014"}
    assert set(bersih.loc[bersih.kode_mk == "CIE62204", "kode_asal"]) == {"CIE61014"}
    assert set(bersih.loc[bersih.kode_mk == "CSD60709", "jenis_perubahan"]) == {"berubah_kode_nama_sks_status"}
    tanpa = db.baca_hasil(cfg_mini, "mk_tanpa_padanan")
    assert tanpa.to_dict("records") == [{"kode_mk": "CIE60044", "jenis": "dihapus", "jumlah_baris": 2}]
    assert len(db.baca_hasil(cfg_mini, "perubahan_penyetaraan")) == 4
    assert set(bersih["split"]) == {"latih", "uji"} and set(bersih["lulus"]) <= {0, 1}
    assert "nim" not in bersih.columns and "nama" not in bersih.columns
