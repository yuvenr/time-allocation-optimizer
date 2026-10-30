"""Riwayat KRS pelengkap (sintetis) yang dikalibrasi dari data akademik nyata.

Data akademik (``src/akademik.py``) hanya memuat KRS SATU semester berjalan (semester ganjil) dan IP/IPK
semester sebelumnya — tanpa nilai per MK dan tanpa riwayat multisemester. Riwayat lengkap bernilai tetap
dibutuhkan untuk: transaksi semester yang belum ada di ekspor KRS (semester genap) pada ARM & θ, serta
simulasi UC-12 (potong riwayat di semester k). Modul ini membangkitkan riwayat tersebut; hasilnya selalu
berlabel ``sintetis`` (``SUMBER_DATA.txt``, kolom ``sumber`` di basis data dan ekspor) dan tidak menimpa
data asli.

Cara kerja (kurikulum = ``data/kurikulum_2024``, hanya dibaca):

1. Setiap mahasiswa menempuh Jalur A (``08``) semester demi semester memakai constraint engine aplikasi
   (``src/rules.py``): paket semester 1–2, prasyarat, penawaran ganjil/genap, pita SKS ``09`` dari IP
   semester sebelumnya, korequisit & syarat PLP, agama tepat satu, larangan UBU60030. Urutan: MK wajib
   tertunda/mengulang → MK wajib semester itu → satu MK pilihan per slot tema → percepatan.
2. Kalibrasi dari data akademik (bila ada di basis data; selain itu nilai bawaan):
   - target IPK tiap mahasiswa = sampel acak IPK nyata (mahasiswa dengan SKS kumulatif ≥ 40), dipotong di
     ``generator.ipk_min`` (riwayat lulus tepat waktu yang wajar);
   - ketidaklulusan mengelompok per semester: peluang semester "bermasalah" = proporsi semester nyata yang
     memuat SKS tidak lulus; di semester itu tiap MK tidak lulus dengan peluang (proporsi SKS tidak lulus ÷
     proporsi semester bermasalah) — keduanya dari kelompok IPK yang sama;
   - preferensi MK pilihan = pangsa MK itu di dalam temanya pada KRS nyata (dengan penghalusan);
   - peluang percepatan (mengambil MK lebih awal) naik bersama IPK, sesuai SKS per semester nyata yang
     sering mencapai batas atas pita.
   Nilai MK lulus = target nilai + penyesuaian tingkat kesulitan (CL heuristik, dipusatkan) + derau → nilai
   huruf terdekat. Target nilai dicari secara numerik agar rata-rata bobot yang diharapkan (termasuk nilai E
   dari ketidaklulusan) sama dengan target IPK.
3. Laporan kalibrasi membandingkan statistik nyata vs sintetis (IPK, IP semester, SKS per semester,
   proporsi tidak lulus) dan disimpan di ``kalibrasi_sintetis.json`` serta ringkasan ``sintetis``.

Seed tetap (``config.seed``) → keluaran identik. Folder yang berisi riwayat asli (tanpa penanda
``sintetis``) tidak akan ditimpa.

Jalankan:  python -m src.sintetis [--out DIR] [--seed N] [--n-alumni N]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import db
from src.config import Config, load_config
from src.data_loader import Kurikulum, load_kurikulum
from src.rules import Keadaan, batas_sks, boleh_diambil, cek_mk, mk_paket_semester, semester_paket, unit_korequisit

JALUR = "A"
LABEL_KURIKULUM = "K2024"
PELUANG_AGAMA = {"Islam": 0.80, "Protestan": 0.08, "Katholik": 0.06, "Hindu": 0.04, "Budha": 0.02}
P_LAMBAT = 0.10            # mahasiswa yang kadang mengambil SKS di bawah pita
P_GAGAL_SKRIPSI = 0.10     # skripsi belum selesai dalam satu semester
SD_NILAI = 0.45            # derau nilai per MK (skala bobot 0–4)
EFEK_KESULITAN = 0.40      # selisih nilai MK termudah vs tersulit (CL heuristik 1..6)
IPK_BAWAAN = (3.60, 0.25)  # (rata-rata, SD) target IPK bila data akademik belum ada
P_BERMASALAH_BAWAAN, P_GAGAL_BAWAAN = 0.14, 0.045


@dataclass(frozen=True)
class Kalibrasi:
    ipk: np.ndarray                 # sampel target IPK
    p_bermasalah: float             # peluang semester memuat MK tidak lulus
    p_gagal: float                  # proporsi SKS tidak lulus (keseluruhan)
    preferensi: dict[str, float]    # kode MK pilihan → log pangsa dalam tema
    sumber: str                     # "data akademik (n=…)" | "bawaan"


@dataclass(frozen=True)
class Mahasiswa:
    target: float
    lambat: bool
    p_percepat: float
    agama_kode: str
    minat: dict[str, float]


# ---------------------------------------------------------------------
# Kalibrasi
# ---------------------------------------------------------------------


def kalibrasi(cfg: Config, kur: Kurikulum) -> Kalibrasi:
    """Statistik target dari data akademik di basis data (atau nilai bawaan bila belum diimpor)."""
    from src.akademik import muat_ip, muat_krs

    ip, krs = muat_ip(cfg), muat_krs(cfg)
    ipk_min = cfg.generator.ipk_min
    pilihan = sorted(k for k in kur.pilihan if kur.mk[k].tema)
    hitung = krs["kode_mk"].value_counts().to_dict() if len(krs) else {}
    preferensi = {}
    for tema in sorted({kur.mk[k].tema for k in pilihan}):
        anggota = [k for k in pilihan if kur.mk[k].tema == tema]
        total = sum(hitung.get(k, 0) for k in anggota)
        alfa = 1 + 0.1 * total / len(anggota)              # penghalusan: MK tak teramati tetap berpeluang
        for k in anggota:
            preferensi[k] = float(np.log((hitung.get(k, 0) + alfa) / (total + alfa * len(anggota))))
    if len(ip):
        acuan = ip[(ip["sksk_beban"] >= 40) & (ip["ipk_beban"] >= ipk_min)]
        if len(acuan) >= 20:
            sks = float(acuan["sks_beban"].sum())
            p_gagal = float(acuan["sks_tidak_lulus"].clip(lower=0).sum()) / sks if sks else P_GAGAL_BAWAAN
            p_bermasalah = max(float((acuan["sks_tidak_lulus"] > 0).mean()), p_gagal)
            return Kalibrasi(acuan["ipk_beban"].to_numpy(float), p_bermasalah, p_gagal, preferensi,
                             f"data akademik ({len(acuan)} mahasiswa, IPK ≥ {ipk_min:.2f}, SKS kumulatif ≥ 40)")
    rng = np.random.default_rng(cfg.seed)
    bawaan = np.clip(rng.normal(*IPK_BAWAAN, 1000), ipk_min, 4.0)
    return Kalibrasi(bawaan, P_BERMASALAH_BAWAAN, P_GAGAL_BAWAAN, preferensi, "bawaan (data akademik belum diimpor)")


def peta_target(bobot: dict[str, float], p_gagal: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(target IPK, target nilai lulus): E[bobot huruf terdekat dari clip(x + derau)] × (1 − p_gagal) untuk
    kisi x — dibalik dengan interpolasi di ``simulasi_mahasiswa``."""
    rng = np.random.default_rng(seed)
    derau = rng.normal(0, SD_NILAI, 4000)
    lulus = np.array(sorted(b for b in bobot.values() if b >= 1.0))
    kisi = np.linspace(1.0, 5.0, 401)
    g = np.clip(kisi[:, None] + derau[None, :], 0, 4)
    terdekat = lulus[np.abs(g[..., None] - lulus).argmin(axis=-1)]
    return terdekat.mean(axis=1) * (1 - p_gagal), kisi


# ---------------------------------------------------------------------
# Simulasi satu mahasiswa
# ---------------------------------------------------------------------


def _huruf(g: float, bobot: dict[str, float]) -> str:
    """Nilai huruf berbobot terdekat dengan g (lulus)."""
    lulus = {h: b for h, b in bobot.items() if b >= 1.0}
    return min(lulus, key=lambda h: (abs(lulus[h] - g), -lulus[h]))


def buat_mahasiswa(kur: Kurikulum, rng: np.random.Generator, kal: Kalibrasi, pilihan: list[str]) -> Mahasiswa:
    target = float(rng.choice(kal.ipk))
    agama = str(rng.choice(list(PELUANG_AGAMA), p=list(PELUANG_AGAMA.values())))
    # minat pada profil lulusan (PL) → MK pilihan ber-PL sama cenderung diambil bersama
    pl = sorted({m.profil_lulusan for k in pilihan if (m := kur.mk[k]).profil_lulusan})
    suka = str(rng.choice(pl)) if pl else None
    minat = {k: kal.preferensi.get(k, 0.0) + float(rng.normal(0, 0.6))
             + (1.0 if suka and suka in (kur.mk[k].profil_lulusan or "") else 0.0) for k in pilihan}
    p_percepat = 0.5 if target >= 3.5 else 0.25 if target >= 3.0 else 0.05
    return Mahasiswa(target=target, lambat=bool(rng.random() < P_LAMBAT), p_percepat=p_percepat,
                     agama_kode=kur.mk_agama(agama) or "", minat=minat)


def _tema_terpenuhi(lulus: set[str], kur: Kurikulum) -> set[str]:
    return {kur.mk[k].tema for k in lulus if kur.mk[k].sifat == "pilihan" and kur.mk[k].tema}


def _selesai(lulus: set[str], kur: Kurikulum) -> bool:
    agama = kur.grup("AGAMA")
    wajib_ok = all(k in lulus for k in kur.wajib - agama) and bool(agama & lulus)
    return wajib_ok and _tema_terpenuhi(lulus, kur) >= set(kur.tema) and kur.total_sks(lulus) >= \
        kur.param_int("AT01", "total_min")


def _kandidat_reguler(s: int, kd: Keadaan, mhs: Mahasiswa, kur: Kurikulum, pilihan: list[str],
                      rng: np.random.Generator) -> list[str]:
    """Urutan prioritas kandidat semester reguler (belum disaring aturan)."""
    agama = kur.grup("AGAMA")
    # sorted: urutan iterasi set bergantung PYTHONHASHSEED; rng.random() di bawah harus dipanggil dalam urutan tetap
    wajib = [k for k in sorted(kur.wajib - kd.lulus) if k not in agama or k == mhs.agama_kode]
    if agama & kd.lulus:
        wajib = [k for k in wajib if k not in agama]
    home = {k: kur.semester_jalur(k, JALUR) or kur.mk[k].semester_jalur_a or 8 for k in wajib}
    tertunda = sorted((k for k in wajib if home[k] < s), key=lambda k: (home[k], k))
    sekarang = sorted((k for k in wajib if home[k] == s), key=lambda k: k)
    percepat = sorted((k for k in wajib if home[k] > s and rng.random() < mhs.p_percepat), key=lambda k: (home[k], k))

    tema_slot = {sl.tema: sl.semester for t in range(1, 9) for sl in kur.slot(JALUR, t)
                 if sl.jenis_slot == "PILIHAN_TEMA" and sl.tema}
    belum = sorted(set(kur.tema) - _tema_terpenuhi(set(kd.lulus), kur))   # urutan tetap untuk rng.random()
    tema_wajib = sorted(t for t in belum if tema_slot.get(t, 8) <= s)
    tema_awal = sorted(t for t in belum if tema_slot.get(t, 8) > s and rng.random() < mhs.p_percepat)
    urut_minat = sorted(pilihan, key=lambda k: -mhs.minat[k])
    pil = [k for t in tema_wajib + tema_awal for k in urut_minat if kur.mk[k].tema == t]
    return tertunda + sekarang + pil + percepat


def simulasi_mahasiswa(kur: Kurikulum, cfg: Config, rng: np.random.Generator, kal: Kalibrasi,
                       kesulitan: dict[str, float], pilihan: list[str], maks_semester: int,
                       peta: tuple[np.ndarray, np.ndarray]) -> tuple[list[dict], int, float] | None:
    """Riwayat satu mahasiswa hingga lulus → (baris, masa studi, target IPK), atau ``None`` bila tidak lulus
    dalam ``maks_semester``."""
    mhs = buat_mahasiswa(kur, rng, kal, pilihan)
    bobot = cfg.aturan.BOBOT_NILAI
    lulus: set[str] = set()
    pernah: set[str] = set()
    ip_prev = mhs.target
    ipk_harap, kisi = peta
    target_nilai = float(np.interp(mhs.target, ipk_harap, kisi))
    p_mk = kal.p_gagal / kal.p_bermasalah if kal.p_bermasalah else 0.0
    riwayat: list[dict] = []
    paket = semester_paket(kur)
    kode_skripsi = cfg.aturan.SKRIPSI.kode
    for s in range(1, maks_semester + 1):
        kd = Keadaan(JALUR, mhs.agama_kode, frozenset(lulus), frozenset(pernah), ipk=mhs.target)
        ambil: list[str] = []
        if s in paket:
            ambil = [k for k in mk_paket_semester(s, kd, kur) if boleh_diambil(cek_mk(k, s, kd, kur, cfg))]
        else:
            _, sks_max, _ = batas_sks(s, JALUR, ip_prev, kur)
            if mhs.lambat and rng.random() < 0.5:   # kadang mengambil di bawah pita
                sks_max = max(12, sks_max - int(rng.choice([3, 6])))
            total = 0
            tema_diambil: set[str] = set()
            for kode in _kandidat_reguler(s, kd, mhs, kur, pilihan, rng):
                m = kur.mk[kode]
                if kode in ambil or (m.sifat == "pilihan" and m.tema in tema_diambil):
                    continue
                unit = sorted(unit_korequisit(kode, kd, kur) - set(ambil))
                if not all(boleh_diambil(cek_mk(u, s, kd, kur, cfg)) for u in unit):
                    continue
                sks_unit = kur.total_sks(unit)
                if total + sks_unit > sks_max:
                    continue
                ambil += unit
                total += sks_unit
                tema_diambil |= {kur.mk[u].tema for u in unit if kur.mk[u].sifat == "pilihan"}
        if not ambil:
            continue   # tidak ada MK yang ditawarkan/boleh diambil (mis. menunggu semester berparitas sama)
        nilai = {}
        bermasalah = rng.random() < kal.p_bermasalah
        for k in ambil:
            p_gagal = P_GAGAL_SKRIPSI if k == kode_skripsi else (p_mk if bermasalah else 0.0)
            if rng.random() < p_gagal:
                nilai[k] = "E"
            else:
                g = target_nilai - EFEK_KESULITAN * kesulitan.get(k, 0.0) + rng.normal(0, SD_NILAI)
                nilai[k] = _huruf(float(np.clip(g, 0, 4)), bobot)
        sks_sem = kur.total_sks(ambil)
        ip_prev = round(sum(bobot[nilai[k]] * kur.sks(k) for k in ambil) / sks_sem, 2)
        for k in ambil:
            riwayat.append({"semester_ke": s, "kode_mk": k, "sks": kur.sks(k), "nilai_huruf": nilai[k],
                            "ip_semester": ip_prev})
            pernah.add(k)
            if cfg.aturan.lulus(nilai[k]):
                lulus.add(k)
        if _selesai(lulus, kur):
            return riwayat, s, mhs.target
    return None


# ---------------------------------------------------------------------
# Orkestrasi
# ---------------------------------------------------------------------


def _kesulitan(cfg: Config, kur: Kurikulum) -> dict[str, float]:
    """CL heuristik KKO (C1..C6) → tingkat kesulitan relatif MK, dipusatkan di 0 (tidak bergantung hasil LLM)."""
    from src.llm.beban import muat_rating, cl_mk, konsensus_bloom
    from src.llm.konteks import SumberKonteks

    tabel = cl_mk(konsensus_bloom(muat_rating(cfg, "heuristik", SumberKonteks.muat(cfg, kur))), kur, "median")
    nilai = {r.kode_mk: (r.cl - 1) / 5 for r in tabel.itertuples() if pd.notna(r.cl)}
    tengah = float(np.mean(list(nilai.values()))) if nilai else 0.0
    return {k: v - tengah for k, v in nilai.items()}


def buat_histori(cfg: Config, kur: Kurikulum, rng: np.random.Generator, kal: Kalibrasi) -> pd.DataFrame:
    g = cfg.generator
    pilihan = sorted(k for k in kur.pilihan if kur.mk[k].tema)
    kesulitan = _kesulitan(cfg, kur)
    peta = peta_target(cfg.aturan.BOBOT_NILAI, kal.p_gagal, cfg.seed)
    maks = kur.param_int("AT04", "semester_max")
    angkatan_list = np.arange(g.angkatan_min, g.angkatan_max + 1)
    baris: list[dict] = []
    ids: set[str] = set()
    n = percobaan = 0
    while n < g.n_alumni:
        percobaan += 1
        if percobaan > g.n_alumni * 20:
            raise RuntimeError("Gagal membangkitkan cukup alumni, periksa parameter generator")
        hasil = simulasi_mahasiswa(kur, cfg, rng, kal, kesulitan, pilihan, maks, peta)
        if hasil is None:
            continue
        riwayat, masa, _ = hasil
        angkatan = int(angkatan_list[n % len(angkatan_list)])
        while (id_anon := f"SIN{int(rng.integers(10 ** 6, 10 ** 7))}") in ids:
            pass
        ids.add(id_anon)
        for b in riwayat:
            baris.append({"id_anon": id_anon, "angkatan": angkatan, **b, "kurikulum": LABEL_KURIKULUM,
                          "masa_studi_semester": masa})
        n += 1
    return pd.DataFrame(baris, columns=["id_anon", "angkatan", "semester_ke", "kode_mk", "sks", "nilai_huruf",
                                        "ip_semester", "kurikulum", "masa_studi_semester"])


def laporan_kalibrasi(histori: pd.DataFrame, kal: Kalibrasi, cfg: Config) -> dict:
    """Statistik nyata vs sintetis (median; SKS per semester)."""
    from src.akademik import muat_ip, muat_krs

    bobot = cfg.aturan.BOBOT_NILAI
    h = histori.assign(b=histori["nilai_huruf"].map(bobot), lulus=histori["nilai_huruf"].map(cfg.aturan.lulus))
    ipk = (h.assign(bs=h["b"] * h["sks"]).groupby("id_anon").apply(lambda g: g["bs"].sum() / g["sks"].sum(),
                                                                 include_groups=False))
    sem = h.groupby(["id_anon", "semester_ke"]).agg(ip=("ip_semester", "first"), sks=("sks", "sum"),
                                                    gagal=("lulus", lambda x: (~x).any())).reset_index()
    per = histori.groupby("id_anon")["masa_studi_semester"].first()
    sint = {"n_mahasiswa": int(per.shape[0]),
            "ipk_median": round(float(ipk.median()), 2), "ipk_rata": round(float(ipk.mean()), 2),
            "ipk_p10": round(float(ipk.quantile(0.1)), 2),
            "ip_semester_median": round(float(sem["ip"].median()), 2),
            "proporsi_semester_ada_tidak_lulus": round(float(sem["gagal"].mean()), 3),
            "sks_median_per_semester": {int(s): float(v) for s, v in sem.groupby("semester_ke")["sks"].median().items()},
            "proporsi_masa_studi_le_8": round(float((per <= cfg.preprocessing.max_masa_studi).mean()), 3)}
    nyata: dict = {}
    ip, krs = muat_ip(cfg), muat_krs(cfg)
    if len(ip):
        a = ip[ip["sksk_beban"] >= 40]
        nyata.update({"n_mahasiswa_ip": int(len(ip)), "ipk_median": round(float(a["ipk_beban"].median()), 2),
                      "ipk_rata": round(float(a["ipk_beban"].mean()), 2),
                      "ipk_p10": round(float(a["ipk_beban"].quantile(0.1)), 2),
                      "ip_semester_median": round(float(ip["ip_beban"].median()), 2),
                      "proporsi_semester_ada_tidak_lulus": round(float((ip["sks_tidak_lulus"] > 0).mean()), 3),
                      "sks_median_per_semester_ip": {int(s): float(v) for s, v in
                                                     ip.groupby("semester_ke")["sks_beban"].median().items()}})
    if len(krs):
        sks = krs.groupby(["id_anon", "semester_ke"])["sks"].sum().reset_index()
        nyata["sks_median_per_semester_krs"] = {int(s): float(v) for s, v in
                                                sks.groupby("semester_ke")["sks"].median().items()}
    return {"sumber_kalibrasi": kal.sumber, "ipk_min": cfg.generator.ipk_min,
            "p_semester_bermasalah": round(kal.p_bermasalah, 4), "proporsi_sks_tidak_lulus": round(kal.p_gagal, 4),
            "sintetis": sint, "nyata": nyata}


def _boleh_ditimpa(out: Path) -> bool:
    """Jangan pernah menimpa riwayat KRS asli: histori lama hanya boleh ditimpa bila berpenanda sintetis."""
    if not (out / "histori_krs.csv").exists():
        return True
    penanda = out / "SUMBER_DATA.txt"
    return penanda.exists() and penanda.read_text(encoding="utf-8").strip().lower().startswith("sintetis")


def generate(cfg: Config, out_dir: Path | None = None) -> dict:
    """Bangkitkan ``histori_krs.csv`` sintetis terkalibrasi (+ penanda & laporan) ke ``out_dir`` (default
    ``paths.raw``)."""
    out = Path(out_dir) if out_dir else cfg.raw_dir
    if not _boleh_ditimpa(out):
        raise FileExistsError(f"{out / 'histori_krs.csv'} adalah riwayat asli (tanpa penanda SUMBER_DATA.txt "
                              f"'sintetis'), tidak ditimpa")
    out.mkdir(parents=True, exist_ok=True)
    kur = load_kurikulum(cfg)
    kal = kalibrasi(cfg, kur)
    histori = buat_histori(cfg, kur, np.random.default_rng(cfg.seed), kal)
    histori.to_csv(out / "histori_krs.csv", index=False)
    laporan = laporan_kalibrasi(histori, kal, cfg)
    (out / "kalibrasi_sintetis.json").write_text(json.dumps(laporan, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "SUMBER_DATA.txt").write_text(
        f"sintetis\n\nRiwayat KRS pelengkap, dibangkitkan oleh src/sintetis.py (seed={cfg.seed}, "
        f"{cfg.generator.n_alumni} alumni, Kurikulum 2024 Jalur A).\nKalibrasi: {kal.sumber}.\n"
        f"Dipakai untuk semester yang belum ada di data KRS akademik dan untuk simulasi UC-12.\n"
        f"Hapus berkas ini hanya bila histori_krs.csv diganti riwayat alumni asli.\n", encoding="utf-8")
    if out.resolve() == cfg.raw_dir.resolve():   # keluaran ke folder lain (uji/perbandingan) tidak mencatat ringkasan
        db.simpan_ringkasan(cfg, "sintetis", {**laporan, "out_dir": str(out), "seed": cfg.seed,
                                             "n_baris": int(len(histori)), "dibangkitkan": db.sekarang()})
        from src.berkas import simpan as simpan_berkas

        simpan_berkas(cfg)            # salinan di basis data (hosting tanpa disk persisten)
    return {**laporan, "n_baris": int(len(histori)), "out_dir": str(out)}


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Bangkitkan riwayat KRS pelengkap (sintetis, terkalibrasi)")
    ap.add_argument("--out", type=Path, default=None, help="folder keluaran (default: paths.raw)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-alumni", type=int, default=None)
    args = ap.parse_args(argv)
    ov: dict = {}
    if args.seed is not None:
        ov["seed"] = args.seed
    if args.n_alumni is not None:
        ov["generator"] = {"n_alumni": args.n_alumni}
    print(json.dumps(generate(load_config(ov), args.out), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
