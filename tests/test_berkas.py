"""Tes salinan berkas di basis data (``src/berkas.py``) untuk hosting tanpa disk persisten."""
from __future__ import annotations

import src.config as konfigurasi
from src import db
from src.berkas import daftar, pulihkan, simpan
from src.kesiapan import atur_ulang
from tests.conftest import buat_cfg


def _cfg(tmp_path, monkeypatch):
    (tmp_path / "raw").mkdir()
    monkeypatch.setattr(konfigurasi, "OVERRIDE_PATH", tmp_path / "config.override.yaml")
    return buat_cfg(tmp_path)


def test_simpan_lalu_pulihkan_byte_identik(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch)
    isi = "id_anon,semester_ke\r\nA,1\n".encode("utf-8") + bytes(range(256))     # byte apa pun dipertahankan
    (cfg.raw_dir / "histori_krs.csv").write_bytes(isi)
    (cfg.raw_dir / "SUMBER_DATA.txt").write_text("sintetis\n", encoding="utf-8")
    konfigurasi.OVERRIDE_PATH.write_text("beban:\n  theta: 70\n", encoding="utf-8")
    assert set(simpan(cfg)) == {"raw/histori_krs.csv", "raw/SUMBER_DATA.txt", "config.override.yaml"}

    for p in (cfg.raw_dir / "histori_krs.csv", cfg.raw_dir / "SUMBER_DATA.txt", konfigurasi.OVERRIDE_PATH):
        p.unlink()                                                   # kontainer dimulai ulang: disk kosong
    assert sorted(pulihkan(cfg)) == ["config.override.yaml", "raw/SUMBER_DATA.txt", "raw/histori_krs.csv"]
    assert (cfg.raw_dir / "histori_krs.csv").read_bytes() == isi
    assert konfigurasi.OVERRIDE_PATH.read_text(encoding="utf-8") == "beban:\n  theta: 70\n"


def test_berkas_lokal_tidak_ditimpa_dan_hapus_ikut_tersalin(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch)
    (cfg.raw_dir / "histori_krs.csv").write_text("lama", encoding="utf-8")
    simpan(cfg)
    (cfg.raw_dir / "histori_krs.csv").write_text("baru di disk", encoding="utf-8")
    assert pulihkan(cfg) == []
    assert (cfg.raw_dir / "histori_krs.csv").read_text(encoding="utf-8") == "baru di disk"
    (cfg.raw_dir / "histori_krs.csv").unlink()
    simpan(cfg)                                                      # berkas dihapus → salinan juga dihapus
    assert daftar(cfg).empty and pulihkan(cfg) == []


def test_atur_ulang_mengosongkan_salinan(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch)
    (cfg.raw_dir / "histori_krs.csv").write_text("x", encoding="utf-8")
    simpan(cfg)
    atur_ulang(cfg, pertahankan_rating_llm=False)
    assert db.jumlah(cfg, "berkas") == 0 and pulihkan(cfg) == []


def test_sidik_data_tidak_bergantung_akhir_baris(tmp_path):
    """Dataset kurikulum di Windows (git autocrlf → CRLF) dan di server Linux (LF) harus bersidik sama, agar hasil
    proses yang dipindah ke cloud tidak dianggap basi. Isi yang benar-benar berubah tetap terdeteksi."""
    import shutil

    from src.data_loader import sidik_raw

    sidik = {}
    for nama, ubah in (("lf", lambda b: b.replace(b"\r\n", b"\n")), ("crlf", lambda b: b.replace(b"\r\n", b"\n")
                                                                       .replace(b"\n", b"\r\n"))):
        kur = tmp_path / nama / "kurikulum"
        shutil.copytree(buat_cfg(tmp_path / "asal").kurikulum_dir, kur)
        for f in kur.glob("*.csv"):
            f.write_bytes(ubah(f.read_bytes()))
        (tmp_path / nama / "raw").mkdir()
        (tmp_path / nama / "raw" / "histori_krs.csv").write_bytes(ubah(b"id_anon,semester_ke\r\nA,1\r\n"))
        sidik[nama] = sidik_raw(buat_cfg(tmp_path / nama, paths={"kurikulum": str(kur)}))
    assert sidik["lf"] == sidik["crlf"]
    (tmp_path / "lf" / "raw" / "histori_krs.csv").write_bytes(b"id_anon,semester_ke\nA,2\n")
    assert sidik_raw(buat_cfg(tmp_path / "lf", paths={"kurikulum": str(tmp_path / "lf" / "kurikulum")})) != sidik["lf"]
