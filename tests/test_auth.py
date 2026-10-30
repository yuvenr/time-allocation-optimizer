"""Tes autentikasi peran (KF-22, ``src/auth.py``)."""
from __future__ import annotations

import pytest

from src import auth, db
from tests.conftest import buat_cfg


def test_hash_pbkdf2_dan_pencocokan():
    h = auth.buat_hash("rahasia-dpa", iterasi=1000)
    assert h.startswith("pbkdf2_sha256$1000$") and "rahasia" not in h
    assert auth.cocok("rahasia-dpa", h) and not auth.cocok("rahasia-dpX", h)
    assert auth.buat_hash("rahasia-dpa", iterasi=1000) != h          # garam acak per hash
    assert not auth.cocok("x", "bukan-hash") and not auth.cocok("x", "md5$1$a$b")
    with pytest.raises(ValueError, match="minimal"):
        auth.buat_hash("pendek")


def test_masuk_dicatat_dan_aman_bawaan(tmp_path, monkeypatch):
    cfg = buat_cfg(tmp_path)
    monkeypatch.delenv("CRS_HASH_ADMIN", raising=False)
    assert auth.perlu_masuk("Administrator", cfg) and not auth.perlu_masuk("Mahasiswa", cfg)
    assert auth.hash_peran("Administrator") is None and not auth.masuk("Administrator", "apa saja", cfg)
    monkeypatch.setenv("CRS_HASH_ADMIN", auth.buat_hash("sandi-admin-1", iterasi=1000))
    assert not auth.masuk("Administrator", "salah", cfg) and auth.masuk("Administrator", "sandi-admin-1", cfg)
    log = db.baca(cfg, "SELECT * FROM log_akses ORDER BY rowid")
    assert list(log["berhasil"]) == [0, 0, 1] and log["keterangan"].iloc[0] == "hash peran belum dikonfigurasi"
    assert "sandi" not in log.to_string()                              # kata sandi tidak pernah dicatat
    tanpa = buat_cfg(tmp_path / "x", auth={"aktif": False})
    assert not auth.perlu_masuk("Administrator", tanpa)


def test_hash_di_env_tidak_rusak_oleh_dotenv(tmp_path, monkeypatch):
    """Format hash memuat '$'; python-dotenv tidak boleh menginterpolasinya bila ditulis dengan kutip tunggal."""
    from dotenv import dotenv_values

    h = auth.buat_hash("sandi-dpa-123", iterasi=1000)
    (tmp_path / ".env").write_text(f"CRS_HASH_DPA='{h}'\n", encoding="utf-8")
    assert dotenv_values(tmp_path / ".env")["CRS_HASH_DPA"] == h
