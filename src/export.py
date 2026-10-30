"""Evaluasi & ekspor kesepakatan skor beban kognitif (UC-10, KF-10, KF-21).

Analisis UTAMA (tingkat mata kuliah; skor = CL per MK dari level Bloom CPMK, agregasi ``beban.agregasi_cl``):
- ICC(A,1) dua arah acak antara skor LLM dan rata-rata skor pakar, beserta ICC(C,1), komponen varians,
  CI 95%, dan kategori Koo & Li; pembanding: ICC(A,1) antarpakar; konsistensi internal: ICC(A,1)
  antarpengulangan LLM; pelengkap: MAE, Bland-Altman, uji t berpasangan (+ Shapiro-Wilk); baseline:
  heuristik KKO vs pakar. ``rm1_skor_mk.csv`` = data lebar per MK untuk JASP (hasil resmi dihitung di JASP;
  angka di sini adalah pratinjau & cek silang, ``src/statistik.py``).

Analisis PELENGKAP (tingkat CPMK):
- Kesepakatan dua penilai (mis. LLM vs pakar, heuristik vs pakar): weighted Cohen's kappa
  (linear & kuadratik) atas level ordinal C1–C6, akurasi eksak, dan akurasi ±1 level. CPMK afektif
  (``A``) dinilai terpisah (kecocokan A vs bukan-A), tidak dipaksa ke C1–C6. Subset ``kko_ambigu = Y``
  dilaporkan tersendiri sebagai kandidat utama pembahasan ketidaksepakatan.
- ``bloom_perbandingan.csv`` (``paths.ekspor``): satu baris per CPMK (heuristik, konsensus LLM/pakar,
  tiap pakar, setiap pengulangan LLM) — format lebar, siap untuk JASP.
- ``bloom_pakar_template.csv`` (``paths.rating``): formulir isian pakar; setelah diisi (kolom
  ``kode_pakar`` membedakan tiap pakar) diimpor ke basis data lewat UI atau ``--impor-pakar``.

Jalankan mandiri:  python -m src.export [--template] [--impor-pakar BERKAS.csv]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import db
from src.config import Config, load_config
from src.llm.beban import KOLOM_PAKAR, cl_mk, impor_pakar, konsensus_bloom, muat_rating
from src.llm.konteks import SumberKonteks
from src.logutil import catat_proses
from src.schemas import LEVEL_BLOOM

PASANGAN = [("llm", "pakar"), ("heuristik", "pakar"), ("llm", "heuristik"), ("llm_mock", "heuristik")]


def kappa_berbobot(a: list[int], b: list[int], bobot: str = "linear") -> float | None:
    """Weighted Cohen's kappa untuk kategori ordinal 1..6. None bila tidak terdefinisi."""
    if len(a) != len(b) or not a:
        return None
    k = len(LEVEL_BLOOM)
    o = np.zeros((k, k))
    for x, y in zip(a, b):
        o[x - 1, y - 1] += 1
    o /= o.sum()
    e = np.outer(o.sum(axis=1), o.sum(axis=0))
    i, j = np.indices((k, k))
    w = np.abs(i - j) if bobot == "linear" else (i - j) ** 2
    penyebut = float((w * e).sum())
    if penyebut == 0:
        return None
    return round(1 - float((w * o).sum()) / penyebut, 4)


def kesepakatan(r1: dict[str, str], r2: dict[str, str], subset: set[str] | None = None) -> dict:
    """Kesepakatan dua penilai atas CPMK bersama (dict id_cpmk → label C1..C6/A)."""
    ids = sorted(set(r1) & set(r2) & (subset if subset is not None else set(r1)))
    kog = [i for i in ids if r1[i] != "A" and r2[i] != "A"]
    a = [int(r1[i][1]) for i in kog]
    b = [int(r2[i][1]) for i in kog]
    selisih = [abs(x - y) for x, y in zip(a, b)]
    return {
        "n_cpmk": len(ids),
        "n_kognitif": len(kog),
        "kappa_linear": kappa_berbobot(a, b, "linear"),
        "kappa_kuadratik": kappa_berbobot(a, b, "kuadratik"),
        "akurasi_eksak": round(float(np.mean([d == 0 for d in selisih])), 4) if kog else None,
        "akurasi_pm1": round(float(np.mean([d <= 1 for d in selisih])), 4) if kog else None,
        "afektif_cocok": sum((r1[i] == "A") == (r2[i] == "A") for i in ids),
        "afektif_tidak_cocok": sum((r1[i] == "A") != (r2[i] == "A") for i in ids),
    }


def _label(rating: pd.DataFrame | None) -> dict[str, str]:
    if rating is None or rating.empty:
        return {}
    k = konsensus_bloom(rating)
    return dict(zip(k["id_cpmk"], k["bloom"]))


def tabel_perbandingan(cfg: Config, sk: SumberKonteks | None = None) -> tuple[pd.DataFrame, dict[str, dict]]:
    """(tabel lebar per CPMK, label konsensus per sumber)."""
    sk = sk or SumberKonteks.muat(cfg)
    tabel = sk.cpmk[["id_cpmk", "kode_mk", "teks", "kko_ambigu"]].copy()
    tabel["kko_ambigu"] = tabel["kko_ambigu"].eq("Y")
    label: dict[str, dict] = {}
    for sumber in ("heuristik", "llm", "llm_mock", "pakar"):
        r = muat_rating(cfg, sumber, sk)
        label[sumber] = _label(r)
        if not label[sumber]:
            continue
        tabel[f"bloom_{sumber}"] = tabel["id_cpmk"].map(label[sumber])
        awalan = "pakar_" if sumber == "pakar" else "run"
        if r is not None and r["penilai"].nunique() > 1:
            ok = r[r["status"] == "ok"]
            lebar = ok.pivot_table(index="id_cpmk", columns="penilai", values="bloom", aggfunc="first")
            for c in lebar.columns:
                tabel[f"bloom_{sumber}_{awalan}{c}"] = tabel["id_cpmk"].map(lebar[c])
    return tabel, label


def formulir_pakar(cfg: Config, sk: SumberKonteks | None = None) -> pd.DataFrame:
    """Formulir isian pakar (instrumen DPA) terisi daftar CPMK: satu baris per CPMK; pakar mengisi
    ``kode_pakar`` dan ``bloom`` (kolom ``nama_mk`` dan ``teks`` hanya bantuan membaca, diabaikan saat impor)."""
    sk = sk or SumberKonteks.muat(cfg)
    df = sk.cpmk[["id_cpmk", "kode_mk", "teks"]].copy()
    df.insert(2, "nama_mk", df["kode_mk"].map(lambda k: sk.kurikulum.mk[k].nama))
    for k in ("kode_pakar", "bloom", "catatan"):
        df[k] = ""
    return df[["id_cpmk", "kode_mk", "nama_mk", "teks"] + [c for c in KOLOM_PAKAR if c not in ("id_cpmk", "kode_mk")]]


def template_pakar(cfg: Config, sk: SumberKonteks | None = None) -> Path:
    """Tulis ``formulir_pakar`` ke ``paths.rating/bloom_pakar_template.csv`` (CLI)."""
    cfg.rating_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.rating_dir / "bloom_pakar_template.csv"
    formulir_pakar(cfg, sk).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------
# RM1 tingkat mata kuliah (ICC)
# ---------------------------------------------------------------------


def _skor(rating: pd.DataFrame, kur, metode: str) -> pd.Series:
    """CL per MK dari satu himpunan rating (konsensus bila >1 penilai); MK tanpa CPMK kognitif = NaN."""
    t = cl_mk(konsensus_bloom(rating), kur, metode)
    return pd.Series(t["cl"].values, index=t["kode_mk"], dtype=float)


def _per_penilai(rating: pd.DataFrame | None, kur, metode: str, awalan: str) -> pd.DataFrame:
    if rating is None or rating.empty:
        return pd.DataFrame(index=pd.Index(kur.kode, name="kode_mk"))
    ok = rating[rating["status"] == "ok"]
    kolom = {f"{awalan}{p}": _skor(g, kur, metode) for p, g in ok.groupby("penilai", sort=True)}
    return pd.DataFrame(kolom).reindex(kur.kode).rename_axis("kode_mk")


def _bulat(o):
    if isinstance(o, dict):
        return {k: _bulat(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_bulat(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    return o


def evaluasi_rm1(cfg: Config, sk: SumberKonteks | None = None) -> tuple[pd.DataFrame, dict]:
    """(tabel lebar per MK, ringkasan statistik RM1). LLM = rating ``llm`` (sungguhan) bila ada, selain itu
    ``llm_mock`` (DEMO — sama dengan heuristik, bukan hasil penelitian)."""
    from src.statistik import bland_altman, icc, mae, uji_t_berpasangan

    sk = sk or SumberKonteks.muat(cfg)
    kur, metode = sk.kurikulum, cfg.beban.agregasi_cl
    sumber_llm = "llm" if muat_rating(cfg, "llm", sk) is not None else "llm_mock"
    r_llm, r_pakar, r_heur = muat_rating(cfg, sumber_llm, sk), muat_rating(cfg, "pakar", sk), \
        muat_rating(cfg, "heuristik", sk)
    tabel = pd.DataFrame({"nama_mk": [kur.mk[k].nama for k in kur.kode]}, index=pd.Index(kur.kode, name="kode_mk"))
    tabel["skor_llm"] = _skor(r_llm, kur, metode) if r_llm is not None else np.nan
    run = _per_penilai(r_llm, kur, metode, "skor_llm_run")
    pakar = _per_penilai(r_pakar, kur, metode, "skor_pakar_")
    tabel = tabel.join(run).join(pakar)
    tabel["rata_pakar"] = pakar.mean(axis=1) if len(pakar.columns) else np.nan
    tabel["skor_heuristik"] = _skor(r_heur, kur, metode)
    tabel["selisih_llm_pakar"] = tabel["skor_llm"] - tabel["rata_pakar"]
    tabel["rata_llm_pakar"] = (tabel["skor_llm"] + tabel["rata_pakar"]) / 2
    tabel = tabel.reset_index()

    ringkasan: dict = {"sumber_llm": sumber_llm, "metode_agregasi": metode, "n_mk_skor_llm":
                       int(tabel["skor_llm"].notna().sum()), "n_pakar": len(pakar.columns),
                       "pakar": [c.removeprefix("skor_pakar_") for c in pakar.columns]}
    if sumber_llm == "llm_mock":
        ringkasan["peringatan"] = "LLM = mock (heuristik KKO), hanya demo, BUKAN hasil penelitian RM1"
    if len(run.columns) >= 2:
        ringkasan["antarpengulangan_llm"] = icc(run.dropna())
    if len(pakar.columns):
        pas = tabel[["skor_llm", "rata_pakar"]].dropna()
        ringkasan["utama_llm_vs_rata_pakar"] = {
            **icc(pas), "MAE": mae(pas["skor_llm"], pas["rata_pakar"]),
            "bland_altman": bland_altman(pas["skor_llm"], pas["rata_pakar"]),
            "uji_t_berpasangan": uji_t_berpasangan(pas["skor_llm"], pas["rata_pakar"])}
        base = tabel[["skor_heuristik", "rata_pakar"]].dropna()
        ringkasan["baseline_heuristik_vs_rata_pakar"] = {**icc(base),
                                                         "MAE": mae(base["skor_heuristik"], base["rata_pakar"])}
        if len(pakar.columns) >= 2:
            ringkasan["antarpakar"] = icc(pakar.dropna())
    else:
        ringkasan["catatan"] = CATATAN_TANPA_PAKAR_CPMK
    return tabel, _bulat(ringkasan)


# Instrumen pakar 10 CPMK (data/rating/instrumen_pakar.csv) TIDAK diimpor ke rating_bloom: rumusannya tidak identik
# dengan CPMK dataset 2024, sehingga RM1 utama dihitung terpisah oleh scripts/banding_llm_pakar.py.
CATATAN_TANPA_PAKAR_CPMK = ("penilaian pakar per CPMK dataset (UC-10) belum diimpor, evaluasi per CPMK/MK ini "
                            "dilewati. RM1 LLM vs instrumen pakar 10 CPMK: python -m scripts.banding_llm_pakar → "
                            "banding_pakar_llm_ringkasan.csv")


def ekspor_evaluasi(cfg: Config) -> dict:
    """RM1: ringkasan ICC tingkat MK (utama) + kesepakatan per CPMK (pelengkap) → basis data & ekspor CSV."""
    with catat_proses(cfg, "evaluasi_bloom", {"pasangan": [f"{a}~{b}" for a, b in PASANGAN]}) as log:
        sk = SumberKonteks.muat(cfg)
        tabel, label = tabel_perbandingan(cfg, sk)
        ambigu = set(tabel.loc[tabel["kko_ambigu"], "id_cpmk"])
        hasil = {}
        for x, y in PASANGAN:
            if label.get(x) and label.get(y):
                hasil[f"{x}~{y}"] = {"semua": kesepakatan(label[x], label[y]),
                                     "kko_ambigu": kesepakatan(label[x], label[y], ambigu)}
        db.simpan_hasil(cfg, "bloom_perbandingan", tabel)
        db.ekspor_csv(cfg, "bloom_perbandingan.csv", tabel)
        skor_mk, rm1 = evaluasi_rm1(cfg, sk)
        db.simpan_hasil(cfg, "rm1_skor_mk", skor_mk)
        db.ekspor_csv(cfg, "rm1_skor_mk.csv", skor_mk)          # data lebar per MK untuk ICC di JASP
        ringkasan = {"rm1": rm1, "sumber_tersedia": sorted(k for k, v in label.items() if v), "kesepakatan": hasil,
                     "catatan": None if label.get("pakar") else
                     CATATAN_TANPA_PAKAR_CPMK}
        db.simpan_ringkasan(cfg, "evaluasi", ringkasan)
        log.ringkasan["pasangan_dihitung"] = sorted(hasil)
        log.ringkasan["icc_utama"] = (rm1.get("utama_llm_vs_rata_pakar") or {}).get("ICC_A1")
    return ringkasan


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    c = load_config()
    if "--template" in sys.argv:
        print("Templat pakar:", template_pakar(c))
    if "--impor-pakar" in sys.argv:
        berkas = sys.argv[sys.argv.index("--impor-pakar") + 1]
        print("Impor pakar:", impor_pakar(c, pd.read_csv(berkas, dtype=str, keep_default_na=False)))
    print(json.dumps(ekspor_evaluasi(c), indent=2, ensure_ascii=False))
