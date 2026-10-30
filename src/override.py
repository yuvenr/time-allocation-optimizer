"""Override kurikulum oleh administrator (KF-01..KF-03, KF-11) — dataset tetap utuh.

Dataset Kurikulum 2024 hanya dibaca (CLAUDE.md §2). Koreksi yang diperlukan saat operasional disimpan
TERPISAH di basis data, wajib beralasan, dan diterapkan setiap kali dataset dimuat:
- ``override_cpmk``      : melengkapi/memperbaiki rumusan CPMK yang tidak lengkap atau ambigu (4.4.2);
  teks asli tetap tersimpan (kolom ``teks_dataset``) dan setiap penilaian Bloom berikutnya memakai teks baru;
- ``override_prasyarat`` : menambah atau menghapus relasi prasyarat "nilai MK" (KF-02); graf tetap harus
  asiklik dan kode harus dikenal.
Batas SKS per IP (KF-11/KNF-07) ditimpa lewat ``config.yaml`` (``aturan.BATAS_SKS_IP``), bukan di sini.
Kesalahan dataset yang permanen tetap diperbaiki di ``scripts/kurikulum/build_dataset.py`` + INK.
"""
from __future__ import annotations

import pandas as pd

from src import db
from src.config import Config
from src.schemas import Prasyarat

PREFIKS_ID = "OVR"


def _ada_db(cfg: Config) -> bool:
    return db.ada_db(cfg)


# ---------------------------------------------------------------------
# CPMK
# ---------------------------------------------------------------------


def daftar_override_cpmk(cfg: Config) -> pd.DataFrame:
    if not _ada_db(cfg):
        return pd.DataFrame(columns=["id_cpmk", "teks", "alasan", "diperbarui"])
    return db.baca(cfg, "SELECT * FROM override_cpmk ORDER BY id_cpmk")


def simpan_override_cpmk(cfg: Config, id_cpmk: str, teks: str, alasan: str, cpmk: pd.DataFrame) -> None:
    """``cpmk`` = tabel 02 (untuk memastikan ``id_cpmk`` dikenal)."""
    if id_cpmk not in set(cpmk["id_cpmk"]):
        raise ValueError(f"id_cpmk tidak dikenal: {id_cpmk}")
    if len(teks.strip()) < 10:
        raise ValueError("rumusan CPMK terlalu pendek (minimal 10 karakter)")
    if not alasan.strip():
        raise ValueError("alasan override wajib diisi")
    with db.koneksi(cfg) as c:
        db.tulis(c, "override_cpmk", [{"id_cpmk": id_cpmk, "teks": teks.strip(), "alasan": alasan.strip(),
                                        "diperbarui": db.sekarang()}], ganti=True)


def hapus_override_cpmk(cfg: Config, id_cpmk: str) -> None:
    with db.koneksi(cfg) as c:
        c.execute("DELETE FROM override_cpmk WHERE id_cpmk = ?", (id_cpmk,))


def terapkan_cpmk(cpmk: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Ganti teks CPMK sesuai override; tambahkan kolom ``teks_dataset`` & ``override``."""
    out = cpmk.copy()
    out["teks_dataset"] = out["teks"]
    out["override"] = False
    ovr = daftar_override_cpmk(cfg)
    if len(ovr):
        peta = dict(zip(ovr["id_cpmk"], ovr["teks"]))
        kena = out["id_cpmk"].isin(peta)
        out.loc[kena, "teks"] = out.loc[kena, "id_cpmk"].map(peta)
        out.loc[kena, "override"] = True
    return out


# ---------------------------------------------------------------------
# Prasyarat
# ---------------------------------------------------------------------


def daftar_override_prasyarat(cfg: Config) -> pd.DataFrame:
    if not _ada_db(cfg):
        return pd.DataFrame(columns=["kode_mk", "kode_mk_syarat", "aksi", "alasan", "diperbarui"])
    return db.baca(cfg, "SELECT * FROM override_prasyarat ORDER BY kode_mk, kode_mk_syarat")


def terapkan_prasyarat(prasyarat: list[Prasyarat], ovr: pd.DataFrame) -> list[Prasyarat]:
    """Hapus/tambah relasi ``nilai_minimal_mk`` sesuai override (urutan deterministik)."""
    hasil = list(prasyarat)
    for r in ovr.itertuples():
        if r.aksi == "hapus":
            hasil = [p for p in hasil if not (p.jenis == "nilai_minimal_mk" and p.kode_mk == r.kode_mk
                                              and p.kode_syarat == r.kode_mk_syarat)]
        else:
            hasil.append(Prasyarat.model_validate({
                "id_prasyarat": f"{PREFIKS_ID}-{r.kode_mk}-{r.kode_mk_syarat}", "kode_mk": r.kode_mk,
                "jenis_syarat": "nilai_minimal_mk", "kode_mk_syarat": r.kode_mk_syarat, "nilai_minimal": "E",
                "teks_asli": f"Prasyarat tambahan administrator: {r.kode_mk_syarat}",
                "sumber_ringkasan": f"override_prasyarat (basis data): {r.alasan}"}))
    return hasil


def simpan_override_prasyarat(cfg: Config, kode_mk: str, kode_syarat: str, aksi: str, alasan: str) -> None:
    """Validasi lewat pemuatan ulang kurikulum dengan override baru (kode dikenal, relasi yang dihapus ada,
    graf tetap asiklik); bila gagal, override tidak disimpan."""
    from src.data_loader import load_kurikulum

    if aksi not in ("tambah", "hapus"):
        raise ValueError(f"aksi tidak dikenal: {aksi}")
    if not alasan.strip():
        raise ValueError("alasan override wajib diisi")
    if kode_mk == kode_syarat:
        raise ValueError("MK tidak dapat menjadi prasyarat dirinya sendiri")
    kur = load_kurikulum(cfg, override=False)
    for k in (kode_mk, kode_syarat):
        if k not in kur:
            raise ValueError(f"kode MK tidak dikenal: {k}")
    ada = any(p.jenis == "nilai_minimal_mk" and p.kode_mk == kode_mk and p.kode_syarat == kode_syarat
              for p in kur.prasyarat)
    if aksi == "hapus" and not ada:
        raise ValueError(f"relasi {kode_syarat} → {kode_mk} tidak ada di dataset")
    if aksi == "tambah" and ada:
        raise ValueError(f"relasi {kode_syarat} → {kode_mk} sudah ada di dataset")
    baris = {"kode_mk": kode_mk, "kode_mk_syarat": kode_syarat, "aksi": aksi, "alasan": alasan.strip(),
             "diperbarui": db.sekarang()}
    calon = pd.concat([daftar_override_prasyarat(cfg).query("not (kode_mk == @kode_mk and kode_mk_syarat == "
                                                            "@kode_syarat)"), pd.DataFrame([baris])],
                      ignore_index=True)
    load_kurikulum(cfg, override=calon)          # galat (mis. siklus) dilempar sebelum disimpan
    with db.koneksi(cfg) as c:
        db.tulis(c, "override_prasyarat", [baris], ganti=True)


def hapus_override_prasyarat(cfg: Config, kode_mk: str, kode_syarat: str) -> None:
    with db.koneksi(cfg) as c:
        c.execute("DELETE FROM override_prasyarat WHERE kode_mk = ? AND kode_mk_syarat = ?", (kode_mk, kode_syarat))
