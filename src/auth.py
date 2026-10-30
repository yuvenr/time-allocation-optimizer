"""Autentikasi peran (KF-22): Administrator dan DPA dilindungi kata sandi; Mahasiswa terbuka.

- Kata sandi tidak pernah disimpan; ``.env`` hanya memuat hash PBKDF2-SHA256 bergaram
  (``CRS_HASH_ADMIN``, ``CRS_HASH_DPA``) dengan format ``pbkdf2_sha256$<iterasi>$<salt>$<hash>``.
- Perbandingan memakai ``hmac.compare_digest`` (waktu konstan). Percobaan masuk dicatat di tabel
  ``log_akses`` (tanpa kata sandi); UI mengunci peran setelah ``auth.maks_gagal`` kegagalan per sesi.
- Peran yang hash-nya belum diisi tidak dapat dibuka (aman secara bawaan).

Buat hash:  python -m src.auth          (kata sandi diminta tanpa ditampilkan)
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets

from src import db
from src.config import Config

ALGORITMA = "pbkdf2_sha256"
ITERASI = 600_000
VAR_HASH = {"Administrator": "CRS_HASH_ADMIN", "DPA": "CRS_HASH_DPA"}
MIN_PANJANG = 8


def buat_hash(kata_sandi: str, iterasi: int = ITERASI, salt: bytes | None = None) -> str:
    if len(kata_sandi) < MIN_PANJANG:
        raise ValueError(f"kata sandi minimal {MIN_PANJANG} karakter")
    salt = salt or secrets.token_bytes(16)
    turunan = hashlib.pbkdf2_hmac("sha256", kata_sandi.encode("utf-8"), salt, iterasi)
    b64 = lambda b: base64.b64encode(b).decode("ascii")  # noqa: E731
    return f"{ALGORITMA}${iterasi}${b64(salt)}${b64(turunan)}"


def cocok(kata_sandi: str, tersimpan: str) -> bool:
    try:
        algoritma, iterasi, salt, nilai = tersimpan.strip().split("$")
        if algoritma != ALGORITMA:
            return False
        turunan = hashlib.pbkdf2_hmac("sha256", kata_sandi.encode("utf-8"), base64.b64decode(salt), int(iterasi))
        return hmac.compare_digest(turunan, base64.b64decode(nilai))
    except (ValueError, TypeError):
        return False


def hash_peran(peran: str) -> str | None:
    var = VAR_HASH.get(peran)
    nilai = (os.getenv(var) or "").strip() if var else ""
    return nilai or None


def perlu_masuk(peran: str, cfg: Config) -> bool:
    return cfg.auth.aktif and peran in cfg.auth.peran_terproteksi


def masuk(peran: str, kata_sandi: str, cfg: Config) -> bool:
    """Periksa kata sandi peran lalu catat percobaannya (berhasil/gagal) di ``log_akses``."""
    tersimpan = hash_peran(peran)
    ok = bool(tersimpan) and cocok(kata_sandi, tersimpan)
    with db.koneksi(cfg) as c:
        db.tulis(c, "log_akses", [{"waktu": db.sekarang(), "peran": peran, "berhasil": int(ok),
                                   "keterangan": None if tersimpan else "hash peran belum dikonfigurasi"}])
    return ok


if __name__ == "__main__":
    import getpass
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pw = getpass.getpass("Kata sandi baru: ")
    if pw != getpass.getpass("Ulangi kata sandi: "):
        sys.exit("Kata sandi tidak sama.")
    try:
        h = buat_hash(pw)
    except ValueError as e:
        sys.exit(str(e))
    print("\nSalin ke .env (pilih peran):")
    print(f"CRS_HASH_ADMIN='{h}'")
    print(f"CRS_HASH_DPA='{h}'")
