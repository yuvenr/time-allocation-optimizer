"""Pindahkan isi basis data SQLite lokal (``data/crs.db``) ke PostgreSQL (hosting cloud).

Alur yang disarankan: impor data akademik, jalankan pipeline, dan penilaian Bloom di komputer lokal (SQLite),
lalu pindahkan hasilnya sekali ke PostgreSQL yang dipakai aplikasi di Streamlit Community Cloud / Azure:

    # .env atau terminal: DATABASE_URL=postgresql://user:pass@host/db?sslmode=require
    python -m scripts.migrasi_sqlite_ke_postgres                 # data/crs.db → DATABASE_URL
    python -m scripts.migrasi_sqlite_ke_postgres --timpa         # kosongkan tabel tujuan dulu

Yang dipindahkan hanya isi basis data (sudah anonim, tanpa NIM — KNF-04). Berkas akademik ber-NIM tidak pernah
ikut. Setiap tabel diperiksa jumlah barisnya sesudah dipindah; laporan dicetak dan dikembalikan.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys

import pandas as pd

from src import db
from src.config import Config, load_config


def _tabel_sqlite(path) -> list[str]:
    with sqlite3.connect(path) as c:
        return [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type = 'table' "
                                        "AND name NOT LIKE 'sqlite_%' ORDER BY name")]


def pindahkan(sumber: Config, tujuan: Config, *, timpa: bool = False) -> pd.DataFrame:
    """Salin semua tabel ``sumber`` (SQLite) ke ``tujuan`` (PostgreSQL) dalam SATU transaksi."""
    if db.backend(sumber) != "sqlite" or not sumber.db_path.exists():
        raise RuntimeError(f"Basis data SQLite sumber tidak ditemukan: {sumber.db_path}")
    if db.backend(tujuan) != "postgres":
        raise RuntimeError("Tujuan harus PostgreSQL: isi DATABASE_URL=postgresql://...")
    with db.koneksi(sumber):                     # pastikan skema sumber sudah versi terbaru (migrasi otomatis)
        pass
    tabel = _tabel_sqlite(sumber.db_path)
    tetap = [t for t in tabel if t != "meta" and not t.startswith("hasil_")]
    hasil = [t for t in tabel if t.startswith("hasil_")]
    laporan = []
    with sqlite3.connect(sumber.db_path) as src, db.koneksi(tujuan) as c:
        isi = [t for t in tetap if c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] > 0]
        if isi and not timpa:
            raise RuntimeError(f"Basis data tujuan tidak kosong ({', '.join(isi)}); jalankan dengan --timpa")
        for t in tetap:
            df = pd.read_sql_query(f"SELECT * FROM {t}", src)
            c.execute(f"DELETE FROM {t}")
            db.tulis(c, t, df.to_dict("records"))
            n = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            laporan.append({"tabel": t, "baris_sqlite": len(df), "baris_postgres": n, "cocok": n == len(df)})
        for t in hasil:
            df = pd.read_sql_query(f"SELECT * FROM {t}", src)
            c.ganti_tabel_df(t, df)
            n = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            laporan.append({"tabel": t, "baris_sqlite": len(df), "baris_postgres": n, "cocok": n == len(df)})
        if not all(r["cocok"] for r in laporan):
            raise RuntimeError("Jumlah baris tidak cocok, seluruh pemindahan dibatalkan")
    return pd.DataFrame(laporan)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sqlite", help="berkas SQLite sumber (bawaan: paths.db di config.yaml)")
    ap.add_argument("--url", help="URL PostgreSQL tujuan (bawaan: DATABASE_URL)")
    ap.add_argument("--timpa", action="store_true", help="kosongkan tabel tujuan yang sudah berisi")
    a = ap.parse_args(argv)
    tujuan = load_config()
    if a.url:
        tujuan.paths.db_url = a.url
    sumber = load_config(use_env=False)
    sumber.paths.db_url = ""
    if a.sqlite:
        sumber.paths.db = a.sqlite
    laporan = pindahkan(sumber, tujuan, timpa=a.timpa)
    print(laporan.to_string(index=False))
    print(f"\nSelesai: {len(laporan)} tabel → {db.keterangan(tujuan)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
