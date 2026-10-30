"""Data akademik nyata (KD-04, KD-10): KRS satu semester + IP/IPK semester sebelumnya.

Sumber: ekspor akademik FILKOM (xlsx dua sheet: "IP dan IPK <Genap|Ganjil> <tahun>" dan
"KRS <Ganjil|Genap> <tahun>"), atau folder hasil praproses anonim (``histori_krs.csv`` +
``ip_semester.csv``, format ``KAMUS_DATA.md``). Yang disimpan ke basis data (tabel ``krs_asli``,
``ip_asli``) hanya data ANONIM:

- NIM diganti ``id_anon = "M" + SHA-256(salt + NIM)[:8]`` (skema yang sama dengan skrip praproses
  akademik, sehingga ID konsisten bila salt sama); salt dari ``ANON_SALT`` di ``.env`` (KNF-04);
- kolom identitas lain (program studi, kelas) dibuang; NIM tidak pernah ditulis ke disk.

Data ini tidak memuat nilai per MK, sehingga transaksi ARM-nya berupa "MK yang diambil" dan profilnya
tidak memiliki riwayat nilai (lihat ``src/preprocessing.py`` dan ``src/simulate.py``).

Jalankan mandiri:  python -m src.akademik impor <folder | berkas.xlsx>
"""
from __future__ import annotations

import hashlib
import io
import os
import re
from pathlib import Path

import pandas as pd

from src import db
from src.config import Config, load_config
from src.logutil import catat_proses

KOLOM_KRS = ["id_anon", "angkatan", "semester_ke", "periode", "kode_mk", "nama_mk", "sks", "kurikulum_kode"]
KOLOM_IP = ["id_anon", "angkatan", "semester_ke", "periode", "ip_beban", "sks_beban", "ipk_beban", "sksk_beban",
            "ip_lulus", "sks_lulus", "ipk_lulus", "sksk_lulus", "sks_tidak_lulus"]
SALT_CONTOH = ("", "ganti-dengan-salt-acak", "crs-alpha-salt")

# kolom xlsx akademik → kolom anonim
PETA_XLSX_KRS = {"ANGKATAN": "angkatan", "SEMESTER": "semester_ke", "KODE MK": "kode_mk", "NAMA MK": "nama_mk",
                 "SKS": "sks", "KURIKULUM": "kurikulum_kode"}
PETA_XLSX_IP = {"ANGKATAN": "angkatan", "SEMESTER": "semester_ke", "IP BEBAN": "ip_beban", "SKS BEBAN": "sks_beban",
                "IPK BEBAN": "ipk_beban", "SKSK BEBAN": "sksk_beban", "IP LULUS": "ip_lulus",
                "SKS LULUS": "sks_lulus", "IPK LULUS": "ipk_lulus", "SKSK LULUS": "sksk_lulus"}


def id_anon(nim: str, salt: str) -> str:
    return "M" + hashlib.sha256((salt + str(nim).strip()).encode("utf-8")).hexdigest()[:8].upper()


def periode_dari_judul(judul: str) -> str:
    """'KRS Ganjil 2026' → 'Ganjil 2026/2027'."""
    m = re.search(r"(Ganjil|Genap)\s+(\d{4})", judul, flags=re.I)
    if not m:
        raise ValueError(f"periode tidak dapat dibaca dari nama sheet {judul!r} (harus memuat 'Ganjil/Genap <tahun>')")
    tahun = int(m.group(2))
    return f"{m.group(1).capitalize()} {tahun}/{tahun + 1}"


# ---------------------------------------------------------------------
# Membaca sumber
# ---------------------------------------------------------------------


def baca_xlsx(sumber: str | Path | bytes, salt: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Baca xlsx akademik lalu anonimkan di memori. Sheet dikenali dari kolomnya (KODE MK → KRS,
    IP BEBAN → IP); periode dari nama sheet."""
    salt = salt if salt is not None else os.getenv("ANON_SALT", "")
    if salt.strip() in SALT_CONTOH:
        raise ValueError("ANON_SALT di .env belum diisi (pakai salt yang sama dengan data sebelumnya agar id_anon "
                         "konsisten)")
    isi = io.BytesIO(sumber) if isinstance(sumber, bytes) else sumber
    sheet = pd.read_excel(isi, sheet_name=None, dtype={"NIM": str})
    krs = ip = None
    for judul, df in sheet.items():
        df = df.rename(columns=lambda c: str(c).strip())
        if "NIM" not in df.columns:
            continue
        df["NIM"] = df["NIM"].astype(str).str.strip()
        if "KODE MK" in df.columns:
            krs = df.rename(columns=PETA_XLSX_KRS).assign(periode=periode_dari_judul(judul))
            krs["id_anon"] = krs["NIM"].map(lambda n: id_anon(n, salt))
        elif "IP BEBAN" in df.columns:
            ip = df.rename(columns=PETA_XLSX_IP).assign(periode=periode_dari_judul(judul))
            ip["id_anon"] = ip["NIM"].map(lambda n: id_anon(n, salt))
            ip["sks_tidak_lulus"] = ip["sks_beban"] - ip["sks_lulus"]
    if krs is None:
        raise ValueError("sheet KRS (berkolom NIM, KODE MK) tidak ditemukan")
    # buang NIM & kolom identitas lain (program studi, kelas) sebelum data meninggalkan fungsi ini
    krs = krs.reindex(columns=KOLOM_KRS)
    ip = ip.reindex(columns=KOLOM_IP) if ip is not None else pd.DataFrame(columns=KOLOM_IP)
    return krs, ip


def baca_folder(folder: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Folder hasil praproses anonim: ``histori_krs.csv`` (wajib) + ``ip_semester.csv`` (opsional)."""
    folder = Path(folder)
    f_krs = folder / "histori_krs.csv"
    if not f_krs.exists():
        raise FileNotFoundError(f"{f_krs} tidak ditemukan")
    krs = pd.read_csv(f_krs, dtype={"id_anon": str, "kode_mk": str})
    f_ip = folder / "ip_semester.csv"
    ip = pd.read_csv(f_ip, dtype={"id_anon": str}) if f_ip.exists() else pd.DataFrame(columns=KOLOM_IP)
    return krs, ip


# ---------------------------------------------------------------------
# Validasi & penyimpanan
# ---------------------------------------------------------------------


def validasi(krs: pd.DataFrame, ip: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Periksa kolom, tipe, dan rentang; buang kolom asing (mis. NIM bila terbawa). Kode MK yang tidak
    dikenal Kurikulum 2024 (setelah normalisasi 11_penyetaraan) dilaporkan, bukan dibuang."""
    from src.data_loader import load_kurikulum
    from src.preprocessing import normalisasi_kode

    if "nim" in {c.lower() for c in krs.columns} | {c.lower() for c in ip.columns}:
        raise ValueError("data masih memuat kolom NIM, impor lewat xlsx (dianonimkan otomatis) atau anonimkan dulu")
    kurang = [k for k in KOLOM_KRS if k not in krs.columns and k not in ("nama_mk", "kurikulum_kode")]
    if kurang:
        raise ValueError(f"KRS: kolom wajib tidak ada: {kurang}")
    krs = krs.reindex(columns=KOLOM_KRS).copy()
    for k in ("angkatan", "semester_ke", "sks"):
        krs[k] = pd.to_numeric(krs[k], errors="raise").astype(int)
    for k in ("id_anon", "periode", "kode_mk"):
        krs[k] = krs[k].astype(str).str.strip()
    if (krs["semester_ke"] < 1).any() or (krs["sks"] < 1).any():
        raise ValueError("KRS: semester_ke dan sks harus ≥ 1")
    ganda = krs.duplicated(["id_anon", "periode", "kode_mk"])
    krs = krs[~ganda]

    if len(ip):
        kurang = [k for k in KOLOM_IP if k not in ip.columns]
        if kurang:
            raise ValueError(f"IP: kolom wajib tidak ada: {kurang}")
        ip = ip[KOLOM_IP].copy()
        for k in ("ip_beban", "ipk_beban", "ip_lulus", "ipk_lulus"):
            ip[k] = pd.to_numeric(ip[k], errors="raise").astype(float)
            if not ip[k].between(0, 4).all():
                raise ValueError(f"IP: {k} di luar rentang 0–4")
        for k in ("angkatan", "semester_ke"):
            ip[k] = pd.to_numeric(ip[k], errors="raise").astype(int)
        ip = ip.drop_duplicates(["id_anon", "periode"])
    else:
        ip = pd.DataFrame(columns=KOLOM_IP)

    kur = load_kurikulum(cfg)
    peta = {k: normalisasi_kode(k, kur) for k in krs["kode_mk"].unique()}
    asing = sorted(k for k, (baru, _) in peta.items() if not baru)
    lap = {
        "periode_krs": sorted(krs["periode"].unique()),
        "periode_ip": sorted(ip["periode"].unique()) if len(ip) else [],
        "n_baris_krs": int(len(krs)),
        "n_baris_ganda_dibuang": int(ganda.sum()),
        "n_mahasiswa_krs": int(krs["id_anon"].nunique()),
        "n_mahasiswa_ip": int(ip["id_anon"].nunique()) if len(ip) else 0,
        "n_mahasiswa_krs_dengan_ip": int(krs["id_anon"].drop_duplicates().isin(set(ip["id_anon"])).sum()),
        "n_kode_mk": len(peta),
        "kode_mk_tidak_dikenal": asing,
        "kode_mk_dipetakan": sorted(f"{k}→{','.join(baru)}" for k, (baru, _) in peta.items() if baru and baru != [k]),
        "mahasiswa_per_semester": {int(s): int(n) for s, n in
                                   krs.groupby("semester_ke")["id_anon"].nunique().items()},
    }
    if len(ip):
        lap["ip_ringkas"] = {"median_ip": round(float(ip["ip_beban"].median()), 2),
                             "median_ipk": round(float(ip["ipk_beban"].median()), 2)}
    return krs, ip, lap


def impor(cfg: Config, krs: pd.DataFrame, ip: pd.DataFrame, asal: str) -> dict:
    """Validasi lalu ganti isi ``krs_asli``/``ip_asli`` (periode yang sama ditimpa, periode lain dipertahankan)."""
    with catat_proses(cfg, "impor_akademik", {"asal": asal}) as log:
        krs, ip, lap = validasi(krs, ip, cfg)
        with db.koneksi(cfg) as c:
            c.executemany("DELETE FROM krs_asli WHERE periode = ?", [(p,) for p in lap["periode_krs"]])
            db.tulis(c, "krs_asli", krs.to_dict("records"))
            if len(ip):
                c.executemany("DELETE FROM ip_asli WHERE periode = ?", [(p,) for p in lap["periode_ip"]])
                db.tulis(c, "ip_asli", ip.to_dict("records"))
        lap["asal"] = asal
        log.ringkasan.update({k: v for k, v in lap.items() if not isinstance(v, (dict, list))})
        db.simpan_ringkasan(cfg, "akademik", {**lap, "diimpor": db.sekarang()})
    return lap


def impor_sumber(cfg: Config, sumber: str | Path) -> dict:
    p = Path(sumber)
    if p.is_dir():
        return impor(cfg, *baca_folder(p), asal=f"folder {p.name}")
    if p.suffix.lower() in (".xlsx", ".xls"):
        return impor(cfg, *baca_xlsx(p), asal=f"xlsx {p.name}")
    raise ValueError(f"sumber tidak dikenali: {p} (folder praproses atau berkas .xlsx)")


# ---------------------------------------------------------------------
# Membaca dari basis data
# ---------------------------------------------------------------------


def muat_krs(cfg: Config) -> pd.DataFrame:
    if not db.ada_db(cfg):
        return pd.DataFrame(columns=KOLOM_KRS)
    return db.baca(cfg, "SELECT * FROM krs_asli ORDER BY periode, id_anon, kode_mk")


def muat_ip(cfg: Config) -> pd.DataFrame:
    if not db.ada_db(cfg):
        return pd.DataFrame(columns=KOLOM_IP)
    return db.baca(cfg, "SELECT * FROM ip_asli ORDER BY periode, id_anon")


def ringkasan(cfg: Config) -> dict | None:
    return db.baca_ringkasan(cfg, "akademik")


def ada(cfg: Config) -> bool:
    return db.jumlah(cfg, "krs_asli") > 0


def sidik(cfg: Config) -> str:
    """Penanda isi data akademik (untuk mendeteksi hasil proses basi)."""
    r = ringkasan(cfg) or {}
    return f"{r.get('diimpor', '-')}:{db.jumlah(cfg, 'krs_asli')}:{db.jumlah(cfg, 'ip_asli')}"


def ip_sebelumnya(ip: pd.DataFrame) -> dict[str, tuple[float, float, int, int]]:
    """id_anon → (IP semester sebelumnya, IPK, SKS kumulatif lulus, semester IP) dari periode IP terbaru."""
    if ip.empty:
        return {}
    terbaru = ip.sort_values("periode").drop_duplicates("id_anon", keep="last")
    return {r.id_anon: (float(r.ip_beban), float(r.ipk_beban), int(r.sksk_lulus), int(r.semester_ke))
            for r in terbaru.itertuples()}


if __name__ == "__main__":
    import json
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) < 3 or sys.argv[1] != "impor":
        print(__doc__)
        sys.exit(1)
    print(json.dumps(impor_sumber(load_config(), sys.argv[2]), indent=2, ensure_ascii=False))
