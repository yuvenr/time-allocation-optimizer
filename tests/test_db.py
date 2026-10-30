"""Tes lapisan penyimpanan SQLite (src/db.py): skema, transaksi, validasi kolom, ringkasan, tabel hasil."""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from src import db
from src.data_loader import ada_transaksi, load_transaksi, simpan_transaksi, transaksi_segar
from src.schemas import Transaksi
from tests.conftest import PG_TES, buat_cfg

# Tes yang memeriksa berkas/katalog SQLite secara langsung; padanan PostgreSQL ada di test_db_postgres.py.
khusus_sqlite = pytest.mark.skipif(bool(PG_TES), reason="khusus backend SQLite (lihat test_db_postgres.py)")


@pytest.fixture
def cfg(tmp_path):
    (tmp_path / "raw").mkdir()
    return buat_cfg(tmp_path)


@khusus_sqlite
def test_skema_dibuat_otomatis_dan_berversi(cfg):
    with db.koneksi(cfg) as c:
        tabel = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        versi = c.execute("SELECT nilai FROM meta WHERE kunci = 'versi_skema'").fetchone()[0]
    assert {"ringkasan", "log_proses", "histori", "transaksi", "aturan", "support_item", "rating_bloom", "cl_mk",
            "rencana", "override_cpmk", "override_prasyarat"} <= tabel
    assert int(versi) == db.VERSI_SKEMA
    assert cfg.db_path.exists()


@khusus_sqlite
def test_skema_lebih_baru_ditolak(cfg):
    with db.koneksi(cfg) as c:
        c.execute("UPDATE meta SET nilai = ? WHERE kunci = 'versi_skema'", (str(db.VERSI_SKEMA + 1),))
    with pytest.raises(RuntimeError, match="lebih baru"):
        with db.koneksi(cfg):
            pass


@khusus_sqlite
def test_tulis_menolak_kolom_asing_dan_rollback(cfg):
    with pytest.raises(ValueError, match="kolom tidak dikenal"):
        with db.koneksi(cfg) as c:
            db.tulis(c, "cl_mk", [{"kode_mk": "X", "cl": 2.0, "n_cpmk": 1, "n_cpmk_kognitif": 1, "n_cpmk_afektif": 0,
                                   "metode": "median", "sumber": "uji", "kolom_salah": 1}])
    # transaksi gagal di tengah → baris sebelumnya ikut dibatalkan
    with pytest.raises(sqlite3.IntegrityError):
        with db.koneksi(cfg) as c:
            db.tulis(c, "rencana", [{"id": "r1", "dibuat": "2026-01-01", "mode": "arm", "isi": "{}"}])
            db.tulis(c, "rencana", [{"id": "r2", "dibuat": "2026-01-01", "mode": "salah", "isi": "{}"}])
    assert db.jumlah(cfg, "rencana") == 0


def test_ringkasan_dan_tabel_hasil(cfg):
    assert db.baca_ringkasan(cfg, "arm") is None and db.baca_hasil(cfg, "sensitivitas_arm") is None
    db.simpan_ringkasan(cfg, "arm", {"n": 3, "himpunan": {"b", "a"}})
    db.simpan_ringkasan(cfg, "arm", {"n": 4})
    assert db.baca_ringkasan(cfg, "arm") == {"n": 4}
    df = pd.DataFrame({"min_support": [0.1, 0.2], "n_aturan": [5, 2]})
    db.simpan_hasil(cfg, "sensitivitas_arm", df)
    pd.testing.assert_frame_equal(db.baca_hasil(cfg, "sensitivitas_arm"), df)
    with pytest.raises(ValueError):
        db.simpan_hasil(cfg, "x; DROP TABLE rencana", df)


def test_transaksi_roundtrip_dan_kesegaran(cfg):
    tr = [Transaksi(id_anon="a", angkatan=2020, semester_ke=1, ip_semester=3.5, items=["COM60014@S1"], gagal=["X"],
                    sumber="sintetis"),
          Transaksi(id_anon="a", angkatan=2020, semester_ke=2, ip_semester=3.1, items=[], sumber="sintetis")]
    simpan_transaksi(cfg, "latih", tr)
    assert not ada_transaksi(cfg)                       # tanpa ringkasan preprocessing = belum sah
    with pytest.raises(LookupError):
        load_transaksi(cfg, "latih")
    db.simpan_ringkasan(cfg, "preprocessing", {"sidik_data": "lama"})
    assert load_transaksi(cfg, "latih") == tr and load_transaksi(cfg, "uji") == []
    assert transaksi_segar(cfg, "latih")[1] and "basi" in transaksi_segar(cfg, "latih")[1]
    simpan_transaksi(cfg, "latih", tr[:1])              # ganti isi split, bukan menambah
    assert len(load_transaksi(cfg, "latih")) == 1


def test_sidik_konteks_tidak_berubah_karena_rencana_atau_log(cfg):
    awal = db.sidik_konteks(cfg)
    with db.koneksi(cfg) as c:
        db.tulis(c, "rencana", [{"id": "r1", "dibuat": "2026-01-01", "mode": "arm", "isi": "{}"}])
    from src.logutil import catat_proses
    with catat_proses(cfg, "uji", {}):
        pass
    assert db.sidik_konteks(cfg) == awal
    db.simpan_ringkasan(cfg, "cl", {"n": 1})
    assert db.sidik_konteks(cfg) != awal


def test_log_proses_mencatat_galat(cfg):
    from src.logutil import baca_log, catat_proses
    with pytest.raises(ZeroDivisionError):
        with catat_proses(cfg, "uji_galat", {"p": 1}) as log:
            log.ringkasan["x"] = 1
            1 / 0
    entri = baca_log(cfg, "uji_galat")
    assert len(entri) == 1 and entri[0]["status"] == "error" and "ZeroDivisionError" in entri[0]["galat"]
    assert entri[0]["parameter"] == {"p": 1} and entri[0]["ringkasan"] == {"x": 1}


@khusus_sqlite
def test_migrasi_skema_v1_ke_v2(cfg):
    """Basis data v1 (rating_bloom tanpa kolom ringkasan_thinking) dimigrasikan otomatis tanpa kehilangan data."""
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(cfg.db_path)
    conn.executescript("""
        CREATE TABLE meta (kunci TEXT PRIMARY KEY, nilai TEXT NOT NULL);
        INSERT INTO meta VALUES ('versi_skema', '1');
        CREATE TABLE rating_bloom (sumber TEXT NOT NULL, id_cpmk TEXT NOT NULL, kode_mk TEXT NOT NULL,
            penilai TEXT NOT NULL, model TEXT NOT NULL DEFAULT '', versi_prompt TEXT NOT NULL DEFAULT '',
            bloom TEXT NOT NULL DEFAULT '', ranah TEXT, penalaran TEXT, alasan TEXT, keyakinan REAL, provider TEXT,
            model_pelayan TEXT, temperature REAL, thinking INTEGER, status TEXT NOT NULL, percobaan INTEGER,
            catatan TEXT, timestamp TEXT, PRIMARY KEY (sumber, model, versi_prompt, penilai, id_cpmk));
        INSERT INTO rating_bloom (sumber, id_cpmk, kode_mk, penilai, bloom, status)
            VALUES ('pakar', 'COM60014-CPMK01', 'COM60014', 'P1', 'C3', 'ok');
    """)
    conn.commit()
    conn.close()
    with db.koneksi(cfg) as c:
        assert c.execute("SELECT nilai FROM meta WHERE kunci = 'versi_skema'").fetchone()[0] == str(db.VERSI_SKEMA)
        assert "ringkasan_thinking" in db.kolom_tabel(c, "rating_bloom")
        assert c.execute("SELECT bloom FROM rating_bloom").fetchone()[0] == "C3"


@khusus_sqlite
def test_basis_data_asing_ditolak(cfg):
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(cfg.db_path)
    conn.execute("CREATE TABLE meta (kunci TEXT PRIMARY KEY, nilai TEXT NOT NULL)")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="tanpa versi skema"):
        with db.koneksi(cfg):
            pass


@khusus_sqlite
def test_migrasi_skema_v2_ke_v3(cfg):
    """v3: tabel turunan (transaksi/aturan/…) dibuat ulang dengan kolom sumber, ringkasan turunannya dihapus
    (pipeline harus diulang); rating, rencana, dan ringkasan lain tetap; kolom token ditambahkan."""
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(cfg.db_path)
    conn.executescript("""
        CREATE TABLE meta (kunci TEXT PRIMARY KEY, nilai TEXT NOT NULL);
        INSERT INTO meta VALUES ('versi_skema', '2');
        CREATE TABLE ringkasan (nama TEXT PRIMARY KEY, isi TEXT NOT NULL, diperbarui TEXT NOT NULL);
        INSERT INTO ringkasan VALUES ('preprocessing', '{}', 'x'), ('arm', '{}', 'x'), ('cl', '{}', 'x');
        CREATE TABLE transaksi (split TEXT NOT NULL, id_anon TEXT NOT NULL, angkatan INTEGER NOT NULL,
            semester_ke INTEGER NOT NULL, ip_semester REAL NOT NULL, items TEXT NOT NULL, gagal TEXT NOT NULL,
            PRIMARY KEY (id_anon, semester_ke));
        CREATE INDEX ix_transaksi_split ON transaksi (split);
        INSERT INTO transaksi VALUES ('latih', 'a', 2020, 1, 3.0, '[]', '[]');
        CREATE TABLE rating_bloom (sumber TEXT NOT NULL, id_cpmk TEXT NOT NULL, kode_mk TEXT NOT NULL,
            penilai TEXT NOT NULL, model TEXT NOT NULL DEFAULT '', versi_prompt TEXT NOT NULL DEFAULT '',
            bloom TEXT NOT NULL DEFAULT '', ranah TEXT, penalaran TEXT, alasan TEXT, keyakinan REAL, provider TEXT,
            model_pelayan TEXT, temperature REAL, thinking INTEGER, ringkasan_thinking TEXT, status TEXT NOT NULL,
            percobaan INTEGER, catatan TEXT, timestamp TEXT, PRIMARY KEY (sumber, model, versi_prompt, penilai, id_cpmk));
        INSERT INTO rating_bloom (sumber, id_cpmk, kode_mk, penilai, bloom, status)
            VALUES ('llm', 'COM60014-CPMK01', 'COM60014', '1', 'C3', 'ok');
        CREATE TABLE rencana (id TEXT PRIMARY KEY, dibuat TEXT NOT NULL, mode TEXT NOT NULL, label TEXT NOT NULL
            DEFAULT '', isi TEXT NOT NULL, status_dpa TEXT NOT NULL DEFAULT 'belum_ditinjau',
            catatan_dpa TEXT NOT NULL DEFAULT '', ditinjau_pada TEXT);
        INSERT INTO rencana (id, dibuat, mode, isi) VALUES ('r1', '2026-01-01', 'arm', '{}');
    """)
    conn.commit()
    conn.close()
    with db.koneksi(cfg) as c:
        assert c.execute("SELECT nilai FROM meta WHERE kunci = 'versi_skema'").fetchone()[0] == str(db.VERSI_SKEMA)
        assert "sumber" in db.kolom_tabel(c, "transaksi") and "sumber" in db.kolom_tabel(c, "aturan")
        assert c.execute("SELECT COUNT(*) FROM transaksi").fetchone()[0] == 0
        assert {"token_masuk", "token_keluar"} <= set(db.kolom_tabel(c, "rating_bloom"))
        assert c.execute("SELECT bloom FROM rating_bloom").fetchone()[0] == "C3"
        assert c.execute("SELECT id FROM rencana").fetchone()[0] == "r1"
        assert [r[0] for r in c.execute("SELECT nama FROM ringkasan")] == ["cl"]
        assert {"krs_asli", "ip_asli", "log_akses", "batch_llm"} <= {r[0] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")}
    tr = Transaksi(id_anon="M1", angkatan=2026, semester_ke=1, items=["COM60014@S1"], sumber="asli")
    simpan_transaksi(cfg, "latih", [tr])                     # KRS berjalan: ip_semester kosong boleh
    db.simpan_ringkasan(cfg, "preprocessing", {"sidik_data": "x"})
    assert load_transaksi(cfg, "latih") == [tr]
