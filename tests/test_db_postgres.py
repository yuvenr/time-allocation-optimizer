"""Tes backend PostgreSQL (src/db.py) — dijalankan bila CRS_TEST_DATABASE_URL berisi URL PostgreSQL uji, mis.

    CRS_TEST_DATABASE_URL=postgresql://postgres@localhost/crs_tes pytest

Dengan variabel itu SELURUH tes lain juga berjalan di PostgreSQL (setiap basis data tes = satu skema), sehingga
berkas ini hanya menguji hal yang khas PostgreSQL: pemetaan skema, upsert ``INSERT OR REPLACE`` → ``ON CONFLICT``,
urutan penyisipan log (``rowid``), tabel hasil berkolom dinamis, dan pemindahan data dari SQLite.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from src import db
from tests.conftest import PG_TES, buat_cfg

pytestmark = pytest.mark.skipif(not PG_TES, reason="CRS_TEST_DATABASE_URL tidak diisi")


@pytest.fixture
def cfg(tmp_path):
    (tmp_path / "raw").mkdir()
    c = buat_cfg(tmp_path)
    c.paths.db_url, c.paths.db_schema = PG_TES, db.id_skema_tes(str(c.db_path))
    return c


def test_backend_dan_skema_dibuat_otomatis(cfg):
    import psycopg
    assert db.backend(cfg) == "postgres" and db.ada_db(cfg)
    assert "PostgreSQL" in db.keterangan(cfg) and "@" not in db.keterangan(cfg)   # kata sandi tidak bocor
    with db.koneksi(cfg) as c:
        tabel = {r[0] for r in c.execute("SELECT table_name FROM information_schema.tables "
                                         "WHERE table_schema = current_schema()")}
        tipe = dict(c.execute("SELECT column_name, data_type FROM information_schema.columns "
                              "WHERE table_schema = current_schema() AND table_name = 'aturan'").fetchall())
        assert c.execute("SELECT nilai FROM meta WHERE kunci = ?", ("versi_skema",)).fetchone()["nilai"] \
            == str(db.VERSI_SKEMA)
    assert {"ringkasan", "log_proses", "histori", "transaksi", "aturan", "support_item", "rating_bloom", "cl_mk",
            "rencana", "override_cpmk", "override_prasyarat", "krs_asli", "ip_asli", "batch_llm", "log_akses"} <= tabel
    assert tipe["lift"] == "double precision"                 # REAL SQLite → presisi ganda
    with pytest.raises(psycopg.IntegrityError):               # CHECK constraint tetap ditegakkan
        with db.koneksi(cfg) as c:
            db.tulis(c, "rencana", [{"id": "r1", "dibuat": "2026-01-01", "mode": "arm", "isi": "{}"}])
            db.tulis(c, "rencana", [{"id": "r2", "dibuat": "2026-01-01", "mode": "salah", "isi": "{}"}])
    assert db.jumlah(cfg, "rencana") == 0                     # seluruh transaksi dibatalkan


def test_upsert_setara_insert_or_replace(cfg):
    with db.koneksi(cfg) as c:
        db.tulis(c, "override_prasyarat", [{"kode_mk": "A", "kode_mk_syarat": "B", "aksi": "tambah",
                                            "alasan": "x", "diperbarui": "1"}], ganti=True)
        db.tulis(c, "override_prasyarat", [{"kode_mk": "A", "kode_mk_syarat": "B", "aksi": "hapus",
                                            "alasan": "y", "diperbarui": "2"}], ganti=True)
        db.tulis(c, "rencana", [{"id": "r1", "dibuat": "1", "mode": "arm", "isi": "{}", "catatan_dpa": "lama"}])
        db.tulis(c, "rencana", [{"id": "r1", "dibuat": "2", "mode": "hybrid", "isi": "{}"}], ganti=True)
    df = db.baca(cfg, "SELECT * FROM override_prasyarat")
    assert df.to_dict("records") == [{"kode_mk": "A", "kode_mk_syarat": "B", "aksi": "hapus", "alasan": "y",
                                      "diperbarui": "2"}]
    r = db.baca(cfg, "SELECT * FROM rencana").iloc[0]
    assert r["mode"] == "hybrid" and r["catatan_dpa"] == ""     # kolom yang tidak diisi kembali ke DEFAULT


def test_log_urut_penyisipan_dan_rowid_tersembunyi(cfg):
    with db.koneksi(cfg) as c:
        db.tulis(c, "log_akses", [{"waktu": "t", "peran": p, "berhasil": True} for p in ("DPA", "Administrator")])
        assert "rowid" not in db.kolom_tabel(c, "log_akses")
    df = db.baca(cfg, "SELECT * FROM log_akses ORDER BY rowid")
    assert list(df["peran"]) == ["DPA", "Administrator"] and list(df["berhasil"]) == [1, 1]
    assert "rowid" not in df.columns


def test_tabel_hasil_dan_ringkasan(cfg):
    df = pd.DataFrame({"min_support": [0.1, 0.2], "n_aturan": [5, 2], "label": ["a", None], "Kolom Spasi": [1.5, 2.0]})
    db.simpan_hasil(cfg, "sensitivitas_arm", df)
    db.simpan_hasil(cfg, "sensitivitas_arm", df)              # replace, bukan menambah
    pd.testing.assert_frame_equal(db.baca_hasil(cfg, "sensitivitas_arm"), df)
    assert db.baca_hasil(cfg, "tidak_ada") is None
    db.simpan_ringkasan(cfg, "arm", {"n": 3, "persen": "50%"})   # tanda % aman di SQL berparameter
    assert db.baca_ringkasan(cfg, "arm") == {"n": 3, "persen": "50%"}


def test_skema_lebih_baru_ditolak(cfg):
    with db.koneksi(cfg) as c:
        c.execute("UPDATE meta SET nilai = ? WHERE kunci = 'versi_skema'", (str(db.VERSI_SKEMA + 1),))
    db._SKEMA_SIAP.clear()                                    # proses baru (mis. aplikasi versi lama)
    with pytest.raises(RuntimeError, match="lebih baru"):
        with db.koneksi(cfg):
            pass


def test_pindah_sqlite_ke_postgres(tmp_path, cfg):
    from scripts.migrasi_sqlite_ke_postgres import pindahkan
    lokal = buat_cfg(tmp_path / "lokal")
    lokal.paths.db_url = ""
    with db.koneksi(lokal) as c:
        db.tulis(c, "rencana", [{"id": "r1", "dibuat": "1", "mode": "arm", "isi": '{"a": 1}'}])
        db.tulis(c, "log_akses", [{"waktu": "t", "peran": "DPA", "berhasil": 1}])
    db.simpan_ringkasan(lokal, "arm", {"n": 1})
    db.simpan_hasil(lokal, "simulasi_metrik", pd.DataFrame({"x": [1, 2]}))
    laporan = pindahkan(lokal, cfg)
    assert laporan.set_index("tabel").loc["rencana", "cocok"] and laporan["cocok"].all()
    assert db.baca(cfg, "SELECT isi FROM rencana")["isi"][0] == '{"a": 1}'
    assert db.baca_ringkasan(cfg, "arm") == {"n": 1}
    pd.testing.assert_frame_equal(db.baca_hasil(cfg, "simulasi_metrik"), pd.DataFrame({"x": [1, 2]}))
    with pytest.raises(RuntimeError, match="tidak kosong"):     # tidak menimpa data cloud tanpa --timpa
        pindahkan(lokal, cfg)
    assert isinstance(sqlite3.connect(lokal.db_path), sqlite3.Connection)
