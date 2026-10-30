"""Salinan berkas di basis data untuk hosting tanpa disk persisten (Streamlit Community Cloud).

Aplikasi menulis beberapa berkas yang memengaruhi hasil: riwayat KRS multisemester beserta penanda asal
datanya di ``paths.raw`` dan parameter UI (``config.override.yaml``). Di Streamlit Community Cloud berkas itu
hilang setiap kali aplikasi dimulai ulang; tanpa riwayat, hasil proses di basis data dianggap basi (sidik data
berbeda) dan ARM/θ diam-diam kembali ke struktur kurikulum.

- Setiap kali berkas itu ditulis/dihapus aplikasi, salinannya diperbarui di tabel ``berkas`` (byte per byte).
- ``pulihkan`` dipanggil saat aplikasi mulai: berkas yang TIDAK ada di disk ditulis ulang dari basis data.
  Berkas yang sudah ada tidak pernah ditimpa, sehingga di komputer lokal berkas tetap menjadi acuan.

    python -m src.berkas simpan      # salin berkas lokal ke basis data (mis. setelah migrasi ke Neon)
    python -m src.berkas daftar
"""
from __future__ import annotations

import sys
from pathlib import Path

import src.config as konfigurasi
from src import db
from src.config import Config, load_config

BERKAS_RAW = ("histori_krs.csv", "SUMBER_DATA.txt", "kalibrasi_sintetis.json")
NAMA_OVERRIDE = "config.override.yaml"


def _lokasi(cfg: Config) -> dict[str, Path]:
    """nama di tabel → path di disk (path override dibaca saat dipanggil agar dapat diganti tes/hosting)."""
    peta = {f"raw/{f}": cfg.raw_dir / f for f in BERKAS_RAW}
    peta[NAMA_OVERRIDE] = Path(konfigurasi.OVERRIDE_PATH)
    return peta


def simpan(cfg: Config) -> dict[str, int]:
    """Samakan tabel ``berkas`` dengan disk: berkas yang ada disalin, yang tidak ada dihapus dari tabel.
    Mengembalikan ukuran (bita) berkas yang tersimpan."""
    hasil: dict[str, int] = {}
    with db.koneksi(cfg) as c:
        for nama, path in _lokasi(cfg).items():
            c.execute("DELETE FROM berkas WHERE nama = ?", (nama,))
            if path.exists():
                isi = path.read_bytes()
                db.tulis(c, "berkas", [{"nama": nama, "isi": isi, "diperbarui": db.sekarang()}])
                hasil[nama] = len(isi)
    return hasil


def pulihkan(cfg: Config) -> list[str]:
    """Tulis ulang berkas yang hilang dari disk menggunakan salinan di basis data. Mengembalikan nama berkas
    yang dipulihkan. Tidak pernah menimpa berkas yang sudah ada."""
    if not db.ada_db(cfg):
        return []
    lokasi = _lokasi(cfg)
    df = db.baca(cfg, "SELECT nama, isi FROM berkas")
    dipulihkan = []
    for nama, isi in zip(df["nama"], df["isi"]):
        path = lokasi.get(nama)
        if path is None or path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(isi))
        dipulihkan.append(nama)
    return dipulihkan


def daftar(cfg: Config):
    return db.baca(cfg, "SELECT nama, diperbarui FROM berkas ORDER BY nama")


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    cfg = load_config()
    if argv[:1] == ["simpan"]:
        for nama, n in simpan(cfg).items():
            print(f"tersimpan: {nama} ({n:,} bita)")
        print(f"Basis data: {db.keterangan(cfg)}")
    elif argv[:1] == ["daftar"]:
        print(daftar(cfg).to_string(index=False))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
