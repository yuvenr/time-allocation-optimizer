"""Tes iterasi 3: templat data (``src/templat.py``), daftar periksa dari nol & atur ulang (``src/kesiapan.py``),
cadangan/pemulihan rating LLM, dan sinkronisasi templat dengan panduan administrator."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from src import db
from src.akademik import baca_xlsx, validasi
from src.kesiapan import BELUM, OPSIONAL, PERHATIAN, SELESAI, atur_ulang, periksa_kesiapan
from src.llm.beban import ekspor_rating_llm, impor_pakar, impor_rating_llm
from src.pipeline import jalankan_semua
from src.templat import SHEET_XLSX, TEMPLAT, baca_csv_unggahan, csv_kosong, kamus, xlsx_kosong

PANDUAN_ADMIN = (Path(__file__).resolve().parent.parent / "docs" / "PANDUAN_ADMINISTRATOR.md").read_text("utf-8")


def _status(cfg) -> dict[str, str]:
    return {x.judul: x.status for x in periksa_kesiapan(cfg)}


# ---- templat ---------------------------------------------------------------------------------


@pytest.mark.parametrize("kunci", list(TEMPLAT))
def test_templat_csv_hanya_nama_kolom(kunci):
    df = baca_csv_unggahan(io.BytesIO(csv_kosong(kunci)))
    assert list(df.columns) == TEMPLAT[kunci].nama_kolom and df.empty
    assert set(TEMPLAT[kunci].wajib) <= set(TEMPLAT[kunci].nama_kolom)
    assert kamus(kunci)["Keterangan"].str.len().min() > 0


def test_templat_dan_kolom_tercantum_di_panduan_administrator():
    for t in TEMPLAT.values():
        assert t.berkas in PANDUAN_ADMIN, t.berkas
        for k in t.nama_kolom:
            assert f"`{k}`" in PANDUAN_ADMIN, (t.berkas, k)
    for judul, kolom in SHEET_XLSX.items():
        assert judul in PANDUAN_ADMIN and all(f"`{k}`" in PANDUAN_ADMIN for k in kolom)


def test_templat_xlsx_terisi_diterima_impor(cfg_kosong):
    """Templat xlsx yang diisi sesuai kolomnya terbaca baca_xlsx (NIM dianonimkan) dan lolos validasi."""
    buku = pd.read_excel(io.BytesIO(xlsx_kosong()), sheet_name=None, dtype=str)
    assert {j: list(df.columns) for j, df in buku.items()} == SHEET_XLSX
    krs = pd.DataFrame([{"NIM": "245150700111001", "ANGKATAN": 2024, "SEMESTER": 3, "KODE MK": "CIE61206",
                         "NAMA MK": "Jaringan Komputer", "SKS": 3, "KURIKULUM": "2024"}])
    ip = pd.DataFrame([{"NIM": "245150700111001", "ANGKATAN": 2024, "SEMESTER": 2, "IP BEBAN": 3.5, "SKS BEBAN": 20,
                        "IPK BEBAN": 3.4, "SKSK BEBAN": 40, "IP LULUS": 3.5, "SKS LULUS": 20, "IPK LULUS": 3.4,
                        "SKSK LULUS": 40}])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        krs.to_excel(w, sheet_name="KRS Ganjil 2026", index=False)
        ip.to_excel(w, sheet_name="IP dan IPK Genap 2025", index=False)
    k, i = baca_xlsx(buf.getvalue(), salt="salt-uji-templat")
    assert "NIM" not in k.columns and k["id_anon"].iloc[0].startswith("M")
    _, _, lap = validasi(k, i, cfg_kosong)
    assert lap["periode_krs"] == ["Ganjil 2026/2027"] and lap["n_mahasiswa_krs_dengan_ip"] == 1


def test_baca_csv_excel_indonesia():
    """Excel berlocale Indonesia: pemisah titik koma, koma desimal, BOM UTF-8."""
    teks = "﻿id_anon;angkatan;ip_beban;kode_mk\nM0001;2024;3,25;COM60014\nM0002;2024;2,5;CIE61206\n"
    df = baca_csv_unggahan(io.BytesIO(teks.encode("utf-8")), dtype={"id_anon": str, "kode_mk": str})
    assert list(df.columns) == ["id_anon", "angkatan", "ip_beban", "kode_mk"]
    assert df["ip_beban"].tolist() == [3.25, 2.5] and df["kode_mk"].tolist() == ["COM60014", "CIE61206"]
    df = baca_csv_unggahan(io.BytesIO(b"kode_mk,nilai_huruf\nCOM60014,A\n"), dtype=str)
    assert df.to_dict("records") == [{"kode_mk": "COM60014", "nilai_huruf": "A"}]


# ---- dari nol -------------------------------------------------------------------------------


def test_sistem_kosong(cfg_kosong):
    st = _status(cfg_kosong)
    assert st["Dataset Kurikulum 2024 terbaca"] == SELESAI
    for judul in ("Impor data akademik (KRS + IP/IPK)", "Riwayat KRS multisemester", "Preprocessing",
                  "Penambangan aturan asosiasi (ARM)", "Penilaian Bloom per CPMK oleh LLM", "Skor CL per MK dan ambang θ",
                  "Simulasi komparasi & validasi KRS", "Uji alur Mahasiswa → DPA"):
        assert st[judul] == BELUM, judul
    assert st["Impor penilaian pakar (RM1)"] == OPSIONAL


def test_dari_nol_sampai_siap_lalu_atur_ulang(cfg_mini):
    """Riwayat fixture → pipeline (Bloom mock) → langkah selesai; atur ulang → kembali kosong, rating LLM tetap."""
    assert _status(cfg_mini)["Riwayat KRS multisemester"] == SELESAI
    jalankan_semua(cfg_mini, provider="mock", nilai_bloom=True)
    st = _status(cfg_mini)
    for judul in ("Preprocessing", "Penambangan aturan asosiasi (ARM)", "Skor CL per MK dan ambang θ",
                  "Simulasi komparasi & validasi KRS"):
        assert st[judul] == SELESAI, (judul, st[judul])
    assert st["Penilaian Bloom per CPMK oleh LLM"] == PERHATIAN        # mock ≠ hasil penelitian
    n_rating = db.jumlah(cfg_mini, "rating_bloom")
    assert n_rating > 0

    r = atur_ulang(cfg_mini, pertahankan_rating_llm=True)
    assert "histori_krs.csv" in r["berkas_dihapus"] and r["baris_dihapus"]["transaksi"] > 0
    assert db.jumlah(cfg_mini, "rating_bloom") == n_rating and db.jumlah(cfg_mini, "transaksi") == 0
    assert db.baca_ringkasan(cfg_mini, "preprocessing") is None and db.baca_hasil(cfg_mini, "simulasi_metrik") is None
    st = _status(cfg_mini)
    assert st["Riwayat KRS multisemester"] == BELUM and st["Preprocessing"] == BELUM
    assert st["Penilaian Bloom per CPMK oleh LLM"] == PERHATIAN
    assert db.jumlah(cfg_mini, "log_proses", "proses = 'atur_ulang'") == 1      # jejak audit tidak dihapus


def test_cadangan_rating_llm_pulih_utuh(cfg_kosong):
    jalankan_semua(cfg_kosong, provider="mock", nilai_bloom=True)
    impor_pakar(cfg_kosong, pd.DataFrame([{"id_cpmk": "CIE62303-CPMK01", "kode_mk": "CIE62303", "kode_pakar": "P1",
                                           "bloom": "C3"}]))
    cadangan = ekspor_rating_llm(cfg_kosong)
    assert len(cadangan) and set(cadangan["sumber"]) == {"llm_mock"}
    berkas = io.BytesIO(cadangan.to_csv(index=False).encode("utf-8-sig"))

    atur_ulang(cfg_kosong, pertahankan_rating_llm=False)
    assert db.jumlah(cfg_kosong, "rating_bloom") == 0
    r = impor_rating_llm(cfg_kosong, baca_csv_unggahan(berkas, dtype={"penilai": str}))
    assert r["n_baris"] == len(cadangan) and r["n_pengulangan"] == 3
    pulih = ekspor_rating_llm(cfg_kosong)
    kolom = ["sumber", "id_cpmk", "kode_mk", "penilai", "model", "versi_prompt", "bloom", "status", "timestamp"]
    urut = ["model", "penilai", "id_cpmk"]
    assert pulih[kolom].sort_values(urut).astype(str).values.tolist() == \
        cadangan[kolom].sort_values(urut).astype(str).values.tolist()
    assert db.jumlah(cfg_kosong, "rating_bloom", "sumber = 'pakar'") == 0          # pakar tidak ikut cadangan LLM


@pytest.mark.parametrize("ubah, pesan", [
    (lambda d: d.drop(columns="model"), "kolom wajib"),
    (lambda d: d.assign(sumber="pakar"), "sumber harus"),
    (lambda d: d.assign(id_cpmk="XX-CPMK99"), "id_cpmk tidak dikenal"),
    (lambda d: d.assign(bloom="C9"), "label bloom"),
    (lambda d: pd.concat([d, d.head(1)]), "baris ganda"),
])
def test_pulihkan_rating_ditolak_bila_tidak_sah(cfg_kosong, ubah, pesan):
    jalankan_semua(cfg_kosong, provider="mock", nilai_bloom=True)
    cadangan = ekspor_rating_llm(cfg_kosong)
    with pytest.raises(ValueError, match=pesan):
        impor_rating_llm(cfg_kosong, ubah(cadangan.copy()))
