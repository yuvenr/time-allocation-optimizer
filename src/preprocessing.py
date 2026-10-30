"""Preprocessing data KRS menjadi transaksi (UC-07) + normalisasi kode (CLAUDE.md §6).

Dua sumber, keduanya bermuara pada tabel ``transaksi`` (kolom ``sumber``):

A. KRS akademik nyata (``krs_asli``, ``src/akademik.py``): satu semester berjalan per mahasiswa, tanpa nilai.
   Mahasiswa pada semester > ``preprocessing.max_masa_studi`` dibuang (pola studi tepat waktu); kode
   dinormalisasi; satu transaksi per mahasiswa (``sumber = asli``).
B. Riwayat KRS multisemester bernilai (``paths.raw/histori_krs.csv``; ``sumber`` = status berkas:
   asli | sintetis | fixture):
   1. Anonimisasi ID (hash SHA-256 bergaram, ``ANON_SALT`` di .env) + buang kolom identitas.
   2. Filter alumni dengan masa studi ≤ ``preprocessing.max_masa_studi``.
   3. Penandaan lulus: nilai ≥ ``aturan.NILAI_LULUS_MINIMUM`` (nilai di luar skala = tidak lulus, dicatat).
   4. Normalisasi kode Kurikulum 2020 → 2024 lewat ``11_penyetaraan_mk_2020_2024.csv``:
      - ``identik``/``berubah_*`` → ``kode_baru`` (SKS mengikuti Kurikulum 2024; perubahan SKS/status dicatat);
      - ``dihapus`` → dibuang (default) atau item historis ``OLD:<kode>`` (``preprocessing.mk_dihapus``);
      - kode yang sudah berkode 2024 dipertahankan; kode tak dikenal dibuang dan dicatat;
      - ``mk_baru`` tidak punya riwayat → dilaporkan sebagai *cold start* (ditangani konten/struktur, bukan ARM).
      Pemetaan 1:n dan n:1 didukung (n:1 dalam satu semester digabung; lulus bila salah satu lulus).

Transaksi = himpunan MK yang DIAMBIL satu mahasiswa pada satu semester (keranjang per
mahasiswa-semester), item berlabel ``"{kode}@S{semester_ke}"``; MK yang tidak lulus dicatat di
``gagal``. Definisi "diambil" dipakai seragam karena KRS akademik tidak memuat nilai.
Split latih:uji per MAHASISWA (bukan transaksi) dengan seed tetap, terpisah untuk tiap sumber.

``normalisasi_riwayat`` + ``profil_dari_riwayat`` memakai pemetaan yang sama untuk riwayat nilai
seorang mahasiswa (AT21: prasyarat dapat dipenuhi MK lama yang disetarakan).

Jalankan mandiri:  python -m src.preprocessing
"""
from __future__ import annotations

import hashlib
import os
from collections import Counter

import numpy as np
import pandas as pd

from src import db
from src.config import Config, load_config
from src.data_loader import Kurikulum, load_histori, load_kurikulum, sidik_raw, simpan_transaksi, status_data_krs
from src.logutil import catat_proses
from src.schemas import Profile, Transaksi

SALT_DEFAULT = "crs-alpha-salt"
PREFIKS_LAMA = "OLD:"


def label_item(kode: str, semester: int) -> str:
    return f"{kode}@S{int(semester)}"


def pisah_item(item: str) -> tuple[str, int]:
    kode, s = item.rsplit("@S", 1)
    return kode, int(s)


# ---------------------------------------------------------------------
# 1. Anonimisasi
# ---------------------------------------------------------------------


def anonimisasi(df: pd.DataFrame, cfg: Config, salt: str | None = None) -> pd.DataFrame:
    """Ganti ID dengan hash SHA-256 bergaram dan buang kolom identitas (UC-07, KNF-04)."""
    salt = salt if salt is not None else os.getenv("ANON_SALT", SALT_DEFAULT)
    kol_id = next((k for k in cfg.preprocessing.kolom_id if k in df.columns), None)
    if kol_id is None:
        raise ValueError(f"Kolom ID tidak ditemukan (dicari: {cfg.preprocessing.kolom_id})")
    out = df.copy()
    out["id_anon"] = out[kol_id].astype(str).map(
        lambda x: "A" + hashlib.sha256((salt + "|" + x).encode("utf-8")).hexdigest()[:10]
    )
    buang = [k for k in set(cfg.preprocessing.kolom_identitas) | {kol_id} if k in out.columns and k != "id_anon"]
    return out.drop(columns=buang)


# ---------------------------------------------------------------------
# 2–3. Filter masa studi & status lulus
# ---------------------------------------------------------------------


def filter_masa_studi(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return df[df["masa_studi_semester"] <= cfg.preprocessing.max_masa_studi].copy()


def tandai_lulus(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Kolom ``lulus`` (nilai ≥ NILAI_LULUS_MINIMUM) dan ``nilai_dikenal`` (nilai ada di SKALA_NILAI)."""
    out = df.copy()
    nilai = out["nilai_huruf"].astype(str).str.strip().str.upper()
    out["lulus"] = nilai.map(cfg.aturan.lulus).astype(bool)
    out["nilai_dikenal"] = nilai.isin(cfg.aturan.SKALA_NILAI)
    return out


# ---------------------------------------------------------------------
# 4. Normalisasi kode lewat 11_penyetaraan
# ---------------------------------------------------------------------


def normalisasi_kode(kode: str, kur: Kurikulum) -> tuple[list[str], str]:
    """Kode MK (2020 atau 2024) → (kode Kurikulum 2024, jenis). ``jenis``: ``jenis_perubahan`` dari
    11, ``kurikulum_2024`` (sudah berkode 2024), atau ``tidak_dikenal``. ``dihapus`` → daftar kosong."""
    baris = [p for p in kur.penyetaraan if p.kode_lama == kode]
    if baris:
        if all(p.jenis_perubahan == "dihapus" for p in baris):
            return [], "dihapus"
        return sorted({p.kode_baru for p in baris if p.kode_baru and p.kode_baru in kur}), baris[0].jenis_perubahan
    if kode in kur:
        return [kode], "kurikulum_2024"
    return [], "tidak_dikenal"


def normalisasi_krs(df: pd.DataFrame, kur: Kurikulum, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Petakan ``kode_mk`` histori KRS ke Kurikulum 2024 (lihat docstring modul, langkah 4)."""
    simpan_lama = cfg.preprocessing.mk_dihapus == "prefiks_old"
    peta = {k: normalisasi_kode(k, kur) for k in df["kode_mk"].astype(str).unique()}

    def target(kode: str) -> list[str]:
        baru, jenis = peta[kode]
        if jenis == "dihapus" and simpan_lama:
            return [PREFIKS_LAMA + kode]
        return baru

    out = df.copy()
    out["kode_mk"] = out["kode_mk"].astype(str)
    out["kode_asal"] = out["kode_mk"]
    out["sks_asal"] = out["sks"]
    out["jenis_perubahan"] = out["kode_asal"].map(lambda k: peta[k][1])
    out["kode_mk"] = out["kode_asal"].map(target)
    terpetakan = out["kode_mk"].map(len) > 0
    dibuang = out.loc[~terpetakan]
    out = out.loc[terpetakan].explode("kode_mk")
    out["sks"] = [kur.sks(k) if k in kur else int(s) for k, s in zip(out["kode_mk"], out["sks_asal"])]
    if "lulus" in out.columns:
        # n:1 pada semester yang sama → satu baris; lulus bila salah satu komponen lulus
        out = (out.sort_values("lulus", ascending=False)
               .drop_duplicates(["id_anon", "semester_ke", "kode_mk"], keep="first"))
        # n:1 lintas semester: komponen yang diulang setelah MK barunya lulus tidak relevan lagi
        pertama_lulus = out[out["lulus"]].groupby(["id_anon", "kode_mk"])["semester_ke"].min().to_dict()
        s_lulus = [pertama_lulus.get((i, k)) for i, k in zip(out["id_anon"], out["kode_mk"])]
        out = out[[x is None or s <= x for s, x in zip(out["semester_ke"], s_lulus)]]
    out = out.sort_values(["id_anon", "semester_ke", "kode_mk"]).reset_index(drop=True)

    jenis_baris = df["kode_mk"].astype(str).map(lambda k: peta[k][1])
    kode_unik = list(peta)
    kode_ok = [k for k in kode_unik if peta[k][0]]
    lama = {p.kode_lama: p for p in kur.penyetaraan if p.kode_lama}
    perubahan = []
    for kode, n in Counter(df["kode_mk"].astype(str)).items():
        p = lama.get(kode)
        if p is None or not p.kode_baru:
            continue
        if p.sks_lama != p.sks_baru or (p.status_lama or "") != (p.status_baru or ""):
            perubahan.append({"kode_lama": kode, "kode_baru": p.kode_baru, "sks_lama": p.sks_lama,
                              "sks_baru": p.sks_baru, "status_lama": p.status_lama, "status_baru": p.status_baru,
                              "jenis_perubahan": p.jenis_perubahan, "jumlah_baris": int(n)})
    baru_2024 = sorted(p.kode_baru for p in kur.penyetaraan if p.jenis_perubahan == "mk_baru" and p.kode_baru)
    ada_riwayat = set(out["kode_mk"])
    laporan = {
        "n_baris": int(len(df)),
        "n_baris_terpetakan": int(terpetakan.sum()),
        "proporsi_baris_terpetakan": round(float(terpetakan.mean()) if len(df) else 1.0, 4),
        "n_kode_unik": len(kode_unik),
        "n_kode_terpetakan": len(kode_ok),
        "proporsi_kode_terpetakan": round(len(kode_ok) / len(kode_unik), 4) if kode_unik else 1.0,
        "baris_per_jenis": {k: int(v) for k, v in sorted(Counter(jenis_baris).items())},
        "kebijakan_mk_dihapus": cfg.preprocessing.mk_dihapus,
        "mk_dihapus": {k: int(v) for k, v in sorted(Counter(df.loc[jenis_baris == "dihapus", "kode_mk"]).items())},
        "mk_tidak_dikenal": {k: int(v) for k, v in sorted(Counter(dibuang.loc[dibuang["jenis_perubahan"]
                                                                                == "tidak_dikenal", "kode_asal"]).items())},
        "n_baris_berubah_sks": int(sum(x["jumlah_baris"] for x in perubahan if x["sks_lama"] != x["sks_baru"])),
        "n_baris_berubah_status": int(sum(x["jumlah_baris"] for x in perubahan
                                          if (x["status_lama"] or "") != (x["status_baru"] or ""))),
        "perubahan_sks_status": sorted(perubahan, key=lambda x: x["kode_lama"]),
        "mk_baru_cold_start": baru_2024,
        "mk_2024_tanpa_riwayat": sorted(set(kur.kode) - ada_riwayat),
    }
    if "nilai_dikenal" in df.columns:
        laporan["n_nilai_di_luar_skala"] = int((~df["nilai_dikenal"]).sum())
    return out, laporan


def normalisasi_riwayat(riwayat: dict[str, str], kur: Kurikulum, cfg: Config) -> tuple[dict[str, str], dict]:
    """Riwayat nilai mahasiswa (kode 2020/2024 → nilai huruf) → kode 2024 (AT21). Bila beberapa kode lama
    menjadi satu kode baru, nilai terbaik dipakai. Mengembalikan (riwayat_baru, laporan)."""
    skala = cfg.aturan.SKALA_NILAI

    def peringkat(n: str) -> int:
        n = str(n).strip().upper()
        return skala.index(n) if n in skala else len(skala)

    hasil: dict[str, str] = {}
    laporan: dict[str, list] = {"dipetakan": [], "dihapus": [], "tidak_dikenal": []}
    for kode, nilai in riwayat.items():
        baru, jenis = normalisasi_kode(str(kode).strip(), kur)
        if not baru:
            laporan["dihapus" if jenis == "dihapus" else "tidak_dikenal"].append(kode)
            continue
        for k in baru:
            if k not in hasil or peringkat(nilai) < peringkat(hasil[k]):
                hasil[k] = str(nilai).strip().upper()
            if k != kode:
                laporan["dipetakan"].append(f"{kode}→{k} ({jenis})")
    return hasil, laporan


def profil_dari_riwayat(riwayat: dict[str, str], kur: Kurikulum, cfg: Config, **kw) -> tuple[Profile, dict]:
    """Bangun ``Profile`` dari riwayat nilai (dinormalisasi ke kode 2024; lulus = ≥ NILAI_LULUS_MINIMUM)."""
    baru, laporan = normalisasi_riwayat(riwayat, kur, cfg)
    lulus = {k for k, n in baru.items() if cfg.aturan.lulus(n)}
    profil = Profile(mk_lulus=lulus, mk_pernah_gagal=set(baru) - lulus, riwayat_nilai=baru, **kw)
    return profil, laporan


# ---------------------------------------------------------------------
# 5. Transaksi
# ---------------------------------------------------------------------


def bentuk_transaksi(df: pd.DataFrame, sumber: str) -> list[Transaksi]:
    """Transaksi = himpunan MK yang diambil satu mahasiswa pada satu semester (item berlabel semester);
    ``gagal`` = MK yang tidak lulus pada semester itu."""
    hasil: list[Transaksi] = []
    for (id_anon, s), g in df.groupby(["id_anon", "semester_ke"], sort=True):
        lulus = set(g.loc[g["lulus"], "kode_mk"])
        hasil.append(Transaksi(
            id_anon=str(id_anon),
            angkatan=int(g["angkatan"].iloc[0]),
            semester_ke=int(s),
            ip_semester=float(g["ip_semester"].iloc[0]),
            items=[label_item(k, s) for k in sorted(set(g["kode_mk"]))],
            gagal=sorted(set(g.loc[~g["lulus"], "kode_mk"]) - lulus),
            sumber=sumber,
        ))
    return hasil


def transaksi_krs_asli(krs: pd.DataFrame, kur: Kurikulum, cfg: Config) -> tuple[list[Transaksi], dict]:
    """KRS akademik (satu semester, tanpa nilai) → satu transaksi per (mahasiswa, semester); hanya
    semester ≤ ``max_masa_studi`` (pola studi tepat waktu). Kode dinormalisasi lewat 11_penyetaraan."""
    if krs.empty:
        return [], {}
    d = krs.copy()
    d["lulus"] = True                      # tanpa nilai: dianggap diambil (gagal tidak diketahui)
    d["nilai_huruf"] = ""
    d["ip_semester"] = np.nan
    d, lap_norm = normalisasi_krs(d, kur, cfg)
    batas = cfg.preprocessing.max_masa_studi
    n_awal = d["id_anon"].nunique()
    lewat = d[d["semester_ke"] > batas]
    d = d[d["semester_ke"] <= batas]
    hasil = [Transaksi(id_anon=str(i), angkatan=int(g["angkatan"].iloc[0]), semester_ke=int(s), ip_semester=None,
                       items=[label_item(k, s) for k in sorted(set(g["kode_mk"]))], gagal=[], sumber="asli")
             for (i, s), g in d.groupby(["id_anon", "semester_ke"], sort=True)]
    lap = {
        "n_mahasiswa": int(n_awal),
        "n_mahasiswa_dibuang_semester": int(lewat["id_anon"].nunique()),
        "periode": sorted(krs["periode"].unique()),
        "proporsi_baris_terpetakan": lap_norm["proporsi_baris_terpetakan"],
        "mk_tidak_dikenal": lap_norm["mk_tidak_dikenal"],
        "transaksi_per_semester": {int(s): int(n) for s, n in Counter(t.semester_ke for t in hasil).items()},
    }
    return hasil, lap


# ---------------------------------------------------------------------
# 6. Split
# ---------------------------------------------------------------------


def split_alumni(ids: list[str], cfg: Config) -> tuple[list[str], list[str]]:
    """Bagi MAHASISWA (bukan transaksi) menjadi latih:uji dengan seed tetap."""
    urut = sorted(set(ids))
    rng = np.random.default_rng(cfg.seed)
    acak = list(rng.permutation(urut))
    n_uji = int(round(len(acak) * cfg.preprocessing.proporsi_uji))
    uji = sorted(acak[:n_uji])
    latih = sorted(acak[n_uji:])
    return latih, uji


# ---------------------------------------------------------------------
# Orkestrasi
# ---------------------------------------------------------------------


def _riwayat(cfg: Config, kur: Kurikulum) -> tuple[pd.DataFrame, dict]:
    """Langkah B1–B4 untuk ``histori_krs.csv``; (histori ter-normalisasi, laporan). Kosong bila tidak ada."""
    if status_data_krs(cfg) == "belum_tersedia":
        return pd.DataFrame(), {}
    mentah = load_histori(cfg)
    df = anonimisasi(mentah, cfg)
    n_alumni_awal = df["id_anon"].nunique()
    df = filter_masa_studi(df, cfg)
    n_alumni_filter = df["id_anon"].nunique()
    df = tandai_lulus(df, cfg)
    df, lap = normalisasi_krs(df, kur, cfg)
    return df, {"n_baris_mentah": int(len(mentah)), "n_alumni_awal": int(n_alumni_awal),
                "n_alumni_masa_studi_ok": int(n_alumni_filter),
                "n_alumni_dibuang_masa_studi": int(n_alumni_awal - n_alumni_filter),
                **lap, "n_baris_gagal": int((~df["lulus"]).sum())}


def jalankan_preprocessing(cfg: Config) -> dict:
    """UC-07: jalankan seluruh langkah lalu simpan histori anonim, transaksi, dan laporan ke basis data."""
    from src.akademik import muat_krs

    param = {
        "max_masa_studi": cfg.preprocessing.max_masa_studi,
        "NILAI_LULUS_MINIMUM": cfg.aturan.NILAI_LULUS_MINIMUM,
        "mk_dihapus": cfg.preprocessing.mk_dihapus,
        "proporsi_uji": cfg.preprocessing.proporsi_uji,
        "seed": cfg.seed,
    }
    with catat_proses(cfg, "preprocessing", param) as log:
        kurikulum = load_kurikulum(cfg)
        sumber_riwayat = status_data_krs(cfg)
        df, lap = _riwayat(cfg, kurikulum)
        tr_riwayat = bentuk_transaksi(df, sumber_riwayat) if len(df) else []
        tr_asli, lap_asli = transaksi_krs_asli(muat_krs(cfg), kurikulum, cfg)
        if not tr_riwayat and not tr_asli:
            raise ValueError("tidak ada data KRS: impor data akademik atau sediakan histori_krs.csv")

        set_uji: set[str] = set()
        jumlah: dict[str, dict] = {}
        for nama, tr in (("riwayat", tr_riwayat), ("asli", tr_asli)):
            latih_ids, uji_ids = split_alumni([t.id_anon for t in tr], cfg)
            set_uji |= set(uji_ids)
            jumlah[nama] = {"latih": len(latih_ids), "uji": len(uji_ids)}
        transaksi = tr_riwayat + tr_asli
        tr_latih = [t for t in transaksi if t.id_anon not in set_uji]
        tr_uji = [t for t in transaksi if t.id_anon in set_uji]

        db.hapus_ringkasan(cfg, "preprocessing")   # transaksi lama tidak sah selama penulisan
        simpan_transaksi(cfg, "latih", tr_latih)
        simpan_transaksi(cfg, "uji", tr_uji)
        kolom = ["id_anon", "angkatan", "semester_ke", "kode_mk", "kode_asal", "jenis_perubahan", "sks", "sks_asal",
                 "nilai_huruf", "lulus", "ip_semester", "kurikulum", "masa_studi_semester"]
        histori = df.reindex(columns=kolom).copy()
        histori["lulus"] = histori["lulus"].astype(int)
        histori["sumber"] = sumber_riwayat
        histori["split"] = np.where(histori["id_anon"].isin(set_uji), "uji", "latih")
        db.ganti_isi(cfg, "histori", histori)
        tanpa = [{"kode_mk": k, "jenis": "dihapus", "jumlah_baris": n} for k, n in lap.get("mk_dihapus", {}).items()]
        tanpa += [{"kode_mk": k, "jenis": "tidak_dikenal", "jumlah_baris": n}
                  for k, n in lap.get("mk_tidak_dikenal", {}).items()]
        db.simpan_hasil(cfg, "mk_tanpa_padanan", pd.DataFrame(tanpa, columns=["kode_mk", "jenis", "jumlah_baris"]))
        db.simpan_hasil(cfg, "perubahan_penyetaraan", pd.DataFrame(lap.get("perubahan_sks_status", [])))

        per_semester: dict[str, dict[str, int]] = {}
        for t in transaksi:
            d = per_semester.setdefault(str(t.semester_ke), {})
            d[t.sumber] = d.get(t.sumber, 0) + 1
        ringkasan = {
            **lap,
            "sumber_riwayat": sumber_riwayat,
            "krs_asli": lap_asli,
            "transaksi_per_semester_sumber": dict(sorted(per_semester.items(), key=lambda kv: int(kv[0]))),
            "n_transaksi": len(transaksi),
            "n_transaksi_asli": len(tr_asli),
            "n_transaksi_riwayat": len(tr_riwayat),
            "n_alumni_latih": jumlah["riwayat"]["latih"],
            "n_alumni_uji": jumlah["riwayat"]["uji"],
            "n_mahasiswa_asli_latih": jumlah["asli"]["latih"],
            "n_mahasiswa_asli_uji": jumlah["asli"]["uji"],
            "n_transaksi_latih": len(tr_latih),
            "n_transaksi_uji": len(tr_uji),
            "rata_item_per_transaksi": round(float(np.mean([len(t.items) for t in transaksi])), 2),
        }
        log.ringkasan.update({k: v for k, v in ringkasan.items() if not isinstance(v, (dict, list))})
        ringkasan["run_id"] = log.run_id
        ringkasan["sidik_data"] = sidik_raw(cfg)
        db.simpan_ringkasan(cfg, "preprocessing", ringkasan)
    return ringkasan


if __name__ == "__main__":
    import json
    import sys as _sys

    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(jalankan_preprocessing(load_config()), indent=2, ensure_ascii=False))
