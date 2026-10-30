"""Tes data akademik nyata (``src/akademik.py``), integrasinya ke preprocessing/ARM/θ, dan validasi KRS.

Data di sini FIKTIF buatan tangan dengan format ekspor akademik (KAMUS_DATA): KRS satu semester tanpa
nilai + IP/IPK semester sebelumnya. NIM fiktif hanya ada di memori tes xlsx dan harus hilang setelah impor.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import db
from src.akademik import baca_folder, baca_xlsx, id_anon, impor, ip_sebelumnya, muat_krs, periode_dari_judul
from src.arm import pilih_transaksi
from src.data_loader import label_sumber_data, load_kurikulum, load_transaksi, sumber_data
from src.preprocessing import jalankan_preprocessing
from src.schemas import Transaksi
from tests.conftest import buat_cfg, buat_cfg_mini

S1 = ["CIE61301", "COM60014", "COM60015", "COM60016", "MPK60006", "MPK60007", "UBU60003", "UBU60004"]
S3 = ["CIE61101", "CIE61102", "CIE61103", "CIE61205", "CIE61206", "CIE61207", "CIE61208"]
S2_SKS = 20   # paket semester 2 Jalur A (termasuk 2 SKS agama)


def _krs(n_s1: int = 30, n_s3: int = 30, periode: str = "Ganjil 2026/2027") -> pd.DataFrame:
    baris = []
    for i in range(n_s1):
        baris += [{"id_anon": f"MA{i:07d}", "angkatan": 2026, "semester_ke": 1, "periode": periode, "kode_mk": k,
                   "nama_mk": k, "sks": 1, "kurikulum_kode": "2024"} for k in S1]
    for i in range(n_s3):
        # separuh mahasiswa S3 juga mengambil MK pilihan Tema A (CIE61402) lebih awal
        kode = S3 + (["CIE61402"] if i % 2 == 0 else [])
        baris += [{"id_anon": f"MB{i:07d}", "angkatan": 2025, "semester_ke": 3, "periode": periode, "kode_mk": k,
                   "nama_mk": k, "sks": 1, "kurikulum_kode": "2024"} for k in kode]
    return pd.DataFrame(baris)     # SKS dinormalisasi ke Kurikulum 2024 saat preprocessing


def _ip(n_s3: int = 30) -> pd.DataFrame:
    """IP semester 2 mahasiswa S3: SKS lulus = 40 (paket S1 + S2) agar konsisten dengan struktur Jalur A."""
    return pd.DataFrame([{"id_anon": f"MB{i:07d}", "angkatan": 2025, "semester_ke": 2, "periode": "Genap 2025/2026",
                          "ip_beban": 3.6, "sks_beban": 20, "ipk_beban": 3.55, "sksk_beban": 40, "ip_lulus": 3.6,
                          "sks_lulus": 20, "ipk_lulus": 3.55, "sksk_lulus": 40, "sks_tidak_lulus": 0}
                         for i in range(n_s3)])


def test_id_anon_dan_periode():
    assert id_anon("12345", "garam") == id_anon(" 12345 ", "garam") != id_anon("12345", "lain")
    assert id_anon("12345", "garam").startswith("M") and len(id_anon("1", "g")) == 9
    assert periode_dari_judul("KRS Ganjil 2026") == "Ganjil 2026/2027"
    assert periode_dari_judul("IP dan IPK Genap 2025") == "Genap 2025/2026"
    with pytest.raises(ValueError):
        periode_dari_judul("Sheet1")


def test_xlsx_dianonimkan_tanpa_nim(tmp_path):
    xlsx = tmp_path / "akademik.xlsx"
    with pd.ExcelWriter(xlsx) as w:
        pd.DataFrame({"NIM": ["245150700111001", "245150700111002"], "PROGRAM STUDI": "PTI", "ANGKATAN": 2025,
                      "SEMESTER": 3, "KODE MK": ["CIE61101", "CIE61102"], "KURIKULUM": 2024,
                      "NAMA MK": ["a", "b"], "SKS": [3, 4], "KELAS": "A"}).to_excel(w, sheet_name="KRS Ganjil 2026",
                                                                                    index=False)
        pd.DataFrame({"NIM": ["245150700111001"], "PROGRAM STUDI": "PTI", "ANGKATAN": 2025, "SEMESTER": 2,
                      "IP BEBAN": 3.5, "SKS BEBAN": 20, "IPK BEBAN": 3.4, "SKSK BEBAN": 40, "IP LULUS": 3.5,
                      "SKS LULUS": 20, "IPK LULUS": 3.4, "SKSK LULUS": 40}).to_excel(
            w, sheet_name="IP dan IPK Genap 2025", index=False)
    with pytest.raises(ValueError, match="ANON_SALT"):
        baca_xlsx(xlsx, salt="")
    krs, ip = baca_xlsx(xlsx.read_bytes(), salt="garam-uji")
    assert "NIM" not in krs.columns and "KELAS" not in krs.columns and "PROGRAM STUDI" not in ip.columns
    assert not krs.astype(str).apply(lambda c: c.str.contains("2451507001")).any().any()
    assert krs["id_anon"].iloc[0] == ip["id_anon"].iloc[0] == id_anon("245150700111001", "garam-uji")
    assert set(krs["periode"]) == {"Ganjil 2026/2027"} and set(ip["periode"]) == {"Genap 2025/2026"}
    assert ip["sks_tidak_lulus"].iloc[0] == 0


def test_impor_validasi_dan_periode_diganti(tmp_path):
    cfg = buat_cfg(tmp_path)
    with pytest.raises(ValueError, match="NIM"):
        impor(cfg, _krs(2, 0).assign(nim="1"), _ip(0), asal="tes")
    lap = impor(cfg, _krs(3, 2), _ip(2), asal="tes")
    assert (lap["n_mahasiswa_krs"], lap["n_mahasiswa_ip"], lap["n_mahasiswa_krs_dengan_ip"]) == (5, 2, 2)
    assert lap["kode_mk_tidak_dikenal"] == [] and lap["mahasiswa_per_semester"] == {1: 3, 3: 2}
    impor(cfg, _krs(1, 0), _ip(0), asal="tes ulang")          # periode yang sama diganti, bukan ditambah
    assert muat_krs(cfg)["id_anon"].nunique() == 1
    assert sumber_data(cfg) == {"krs_akademik": "asli", "riwayat": None} and label_sumber_data(cfg) == "asli"
    ganjil = _krs(1, 0).assign(kode_mk="XYZ123")
    assert impor(cfg, ganjil, _ip(0), asal="tes")["kode_mk_tidak_dikenal"] == ["XYZ123"]


def test_folder_praproses(tmp_path):
    _krs(2, 1).to_csv(tmp_path / "histori_krs.csv", index=False)
    _ip(1).to_csv(tmp_path / "ip_semester.csv", index=False)
    krs, ip = baca_folder(tmp_path)
    assert len(krs) == 2 * len(S1) + len(S3) + 1 and len(ip) == 1
    assert ip_sebelumnya(ip) == {"MB0000000": (3.6, 3.55, 40, 2)}


def test_pilih_transaksi_per_semester():
    def tr(i, s, sumber):
        return Transaksi(id_anon=f"{sumber}{i}", angkatan=2024, semester_ke=s, items=[f"X@S{s}"], sumber=sumber)

    data = ([tr(i, 1, "asli") for i in range(25)] + [tr(i, 1, "sintetis") for i in range(40)]
            + [tr(i, 2, "sintetis") for i in range(40)] + [tr(i, 3, "asli") for i in range(5)]
            + [tr(i, 3, "sintetis") for i in range(40)] + [tr(i, 5, "asli") for i in range(3)])
    pilih, sumber = pilih_transaksi(data, "utamakan_asli", min_asli=20)
    assert sumber == {1: "asli", 2: "sintetis", 3: "sintetis", 5: "asli"}   # S5: tanpa sumber lain → asli
    assert len(pilih) == 25 + 40 + 40 + 3
    assert pilih_transaksi(data, "asli")[1] == {1: "asli", 3: "asli", 5: "asli"}
    assert pilih_transaksi(data, "sintetis")[1] == {1: "sintetis", 2: "sintetis", 3: "sintetis"}
    assert pilih_transaksi(data, "gabungan")[1][1] == "asli+sintetis"


def test_preprocessing_arm_theta_dan_validasi_krs(tmp_path):
    """KRS akademik + riwayat fixture → transaksi berlabel sumber; ARM per semester memakai KRS akademik bila
    cukup (S1, S3) dan riwayat untuk semester lain; θ dikalibrasi data; validasi KRS nyata pada S3."""
    from src.arm import jalankan_arm
    from src.llm.beban import jalankan_cl
    from src.pipeline import hitung_dan_simpan_theta
    from src.simulate import jalankan_validasi_krs

    cfg = buat_cfg_mini(tmp_path, arm={"min_transaksi_asli": 10})
    impor(cfg, _krs(30, 30), _ip(30), asal="tes")
    r = jalankan_preprocessing(cfg)
    assert r["n_transaksi_asli"] == 60 and r["krs_asli"]["transaksi_per_semester"] == {1: 30, 3: 30}
    assert r["n_mahasiswa_asli_uji"] == 12 and r["sumber_riwayat"] == "fixture"
    latih = load_transaksi(cfg, "latih")
    asli = [t for t in latih if t.sumber == "asli"]
    assert len(asli) == 48 and all(t.ip_semester is None and t.gagal == [] for t in asli)

    a = jalankan_arm(cfg)
    assert a["sumber_per_semester"] == {"1": "asli", "2": "fixture", "3": "asli", "4": "fixture"}
    sup = db.baca(cfg, "SELECT * FROM support_item").set_index("item")
    s3_latih = sum(t.semester_ke == 3 for t in asli)
    assert sup.loc["CIE61402@S3", "sumber"] == "asli" and sup.loc["CIE62201@S2", "sumber"] == "fixture"
    assert sup.loc["CIE61402@S3", "n_transaksi"] == s3_latih

    jalankan_cl(cfg)
    info = hitung_dan_simpan_theta(cfg)
    assert info["dikalibrasi_data"] and info["sumber_per_semester"]["3"] == "asli"

    v = jalankan_validasi_krs(cfg)
    s3_uji = 30 - s3_latih                                                  # semua profil S3 konsisten (40 SKS)
    assert v["n_kandidat"] == v["n_profil"] == s3_uji > 0 and v["per_semester"] == {3: s3_uji}
    df = db.baca_hasil(cfg, "validasi_krs")
    assert {"jaccard_arm", "jaccard_hybrid", "presisi_arm", "recall_hybrid", "lolos_sks_arm"} <= set(df.columns)
    assert df["lolos_prasyarat_arm"].eq(1).all() and df["lolos_paritas_hybrid"].eq(1).all()
    assert (tmp_path / "ekspor" / "validasi_krs.csv").exists()


def test_validasi_krs_mengecualikan_profil_tidak_konsisten(tmp_path):
    from src.simulate import profil_krs_asli

    cfg = buat_cfg(tmp_path)
    kur = load_kurikulum(cfg)
    sks_s1 = kur.total_sks(set(S1))
    assert sks_s1 + S2_SKS == 40
    prof, _ = profil_krs_asli(3, set(S3), (3.6, 3.55, 40, 2), kur, cfg)
    assert prof.semester_saat_ini == 3 and prof.ipk == 3.55 and set(S1) <= prof.mk_lulus
    assert profil_krs_asli(3, set(S3), (3.6, 3.55, 46, 2), kur, cfg) == (None, "sks_lebih_dari_struktur")
    assert profil_krs_asli(3, set(S3), (3.6, 3.55, 40, 4), kur, cfg) == (None, "semester_ip_tidak_berurutan")
    assert profil_krs_asli(3, set(S3), None, kur, cfg) == (None, "tanpa_ip_semester_sebelumnya")
    # MK semester lalu yang ada di KRS berjalan = diulang/tertunda → tidak dihitung lulus
    prof, _ = profil_krs_asli(3, set(S3) | {"COM60015"}, (3.6, 3.55, 38, 2), kur, cfg)
    assert "COM60015" not in prof.mk_lulus
