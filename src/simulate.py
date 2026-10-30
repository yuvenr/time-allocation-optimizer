"""Simulasi komparasi mode ``arm`` vs ``hybrid`` (UC-12) — dua evaluasi berpasangan.

1. ``jalankan_simulasi``: profil dibentuk dengan memotong riwayat multisemester data uji di semester k
   (``simulasi.k_potong``, default 2..6): IP = IP semester k, MK lulus = riwayat sampai k, rencana dimulai
   di semester k+1 dan dibandingkan dengan MK yang benar-benar diambil di semester k+1..8.
   → ``simulasi_metrik.csv`` (kolom ``sumber_data`` hanya di basis data, tidak diekspor).
2. ``jalankan_validasi_krs``: mahasiswa data uji KRS akademik nyata. Profil = IP & IPK semester sebelumnya
   (data akademik) + MK lulus yang DISIMPULKAN dari struktur Jalur A semester < s, dikurangi MK yang ada di
   KRS berjalan (diulang/tertunda). Hanya profil yang konsisten dengan SKS kumulatif lulus akademik
   (``simulasi.validasi_krs.toleransi_sks``) yang dipakai. Rekomendasi semester berjalan dibandingkan dengan
   KRS nyata. → ``validasi_krs.csv``. Agama mahasiswa tidak disimpulkan, sehingga AD-12/AD-01 tidak
   dinilai pada validasi ini.

Kedua keluaran berformat BERPASANGAN (satu baris per profil, kolom ``<metrik>_arm`` dan
``<metrik>_hybrid``) agar langsung dapat dipakai uji berpasangan (paired t-test / Wilcoxon) di JASP.

Jalankan mandiri:  python -m src.simulate
"""
from __future__ import annotations

from typing import Callable, Collection

import pandas as pd

from src import db
from src.config import Config, load_config
from src.data_loader import Kurikulum, histori_segar, transaksi_segar
from src.logutil import catat_proses
from src.metrics import METRIK_BOOLEAN, METRIK_NUMERIK, jaccard, metrik_rencana
from src.planner import KonteksPlanner, susun_rencana
from src.schemas import Profile, Transaksi

MODE = ("arm", "hybrid")


def kelompokkan(transaksi: list[Transaksi]) -> dict[str, list[Transaksi]]:
    per: dict[str, list[Transaksi]] = {}
    for t in transaksi:
        per.setdefault(t.id_anon, []).append(t)
    return {k: sorted(v, key=lambda t: t.semester_ke) for k, v in sorted(per.items())}


def bentuk_profil(riwayat: list[Transaksi], k: int, horizon: int = 8,
                  kode_dikenal: Collection[str] | None = None) -> tuple[Profile, dict[int, set[str]]] | None:
    """Potong riwayat di semester k → (profil, MK aktual per semester k+1..horizon).

    MK lulus = MK yang diambil sampai k dan tidak tercatat gagal pada semester pengambilannya.
    ``kode_dikenal`` (kode Kurikulum 2024) menyaring item yang tidak dapat direncanakan, mis.
    ``OLD:<kode>`` (MK 2020 yang dihapus, ``preprocessing.mk_dihapus = prefiks_old``).
    """
    sebelum = [t for t in riwayat if t.semester_ke <= k]
    sesudah = [t for t in riwayat if k < t.semester_ke <= horizon]
    if not sebelum or not sesudah or k + 1 > horizon or sebelum[-1].ip_semester is None:
        return None

    def saring(kode_set: set[str]) -> set[str]:
        return kode_set if kode_dikenal is None else {x for x in kode_set if x in kode_dikenal}

    lulus: set[str] = set()
    pernah: set[str] = set()
    for t in sebelum:
        pernah |= t.kode
        lulus |= t.kode - set(t.gagal)
    lulus, gagal = saring(lulus), saring(pernah - lulus)
    profil = Profile(semester_saat_ini=k + 1, ip_terakhir=sebelum[-1].ip_semester,
                     mk_lulus=lulus, mk_pernah_gagal=gagal)
    aktual = {t.semester_ke: saring(t.kode) for t in sesudah}
    return profil, aktual


def riwayat_alumni(histori: pd.DataFrame, k: int, kode_dikenal: Collection[str], cfg: Config) -> dict[str, str]:
    """Riwayat nilai (nilai terbaik per MK, kode Kurikulum 2024) satu alumni sampai semester k — untuk IPK
    (AD-10) dan pelacakan nilai akhir per MK. ``histori`` = baris tabel ``histori`` milik alumni itu."""
    skala = cfg.aturan.SKALA_NILAI
    h = histori[(histori["semester_ke"] <= k) & histori["kode_mk"].isin(kode_dikenal)]
    hasil: dict[str, str] = {}
    for kode, n in zip(h["kode_mk"], h["nilai_huruf"].astype(str).str.strip().str.upper()):
        if n in skala and (kode not in hasil or skala.index(n) < skala.index(hasil[kode])):
            hasil[kode] = n
    return hasil


def jalankan_simulasi(cfg: Config, k_list: list[int] | None = None,
                      progress: Callable[[int, int], None] | None = None) -> dict:
    """UC-12: jalankan kedua mode pada setiap profil uji lalu ekspor CSV berpasangan."""
    k_list = k_list or cfg.simulasi.k_potong
    with catat_proses(cfg, "simulasi", {"k_potong": k_list, "theta": cfg.beban.theta,
                                        "metode_beban": cfg.beban.metode}) as log:
        ctx = KonteksPlanner.muat(cfg)
        if ctx.theta is None:
            raise ValueError("θ belum tersedia, jalankan preprocessing dan ekstraksi LLM dulu")
        uji, alasan = transaksi_segar(cfg, "uji")
        if alasan:
            raise ValueError(f"{alasan}, jalankan ulang preprocessing")
        sumber = {t.id_anon: t.sumber for t in uji}
        kelompok = {i: r for i, r in kelompokkan(uji).items() if any(t.ip_semester is not None for t in r)}
        kode_dikenal = set(ctx.kurikulum.kode)
        histori, _ = histori_segar(cfg, "uji")
        per_alumni = dict(tuple(histori.groupby("id_anon"))) if len(histori) else {}
        tugas = [(id_anon, k) for id_anon in kelompok for k in k_list]
        baris = []
        for i, (id_anon, k) in enumerate(tugas, start=1):
            if progress:
                progress(i, len(tugas))
            hasil = bentuk_profil(kelompok[id_anon], k, cfg.kurikulum.horizon_semester, kode_dikenal)
            if hasil is None:
                continue
            profil, aktual = hasil
            if id_anon in per_alumni:
                profil = profil.model_copy(update={"riwayat_nilai": riwayat_alumni(per_alumni[id_anon], k,
                                                                                   kode_dikenal, cfg)})
            rec = {"id_anon": id_anon, "sumber_data": sumber[id_anon], "k": k,
                   "semester_mulai": profil.semester_saat_ini, "ip_terakhir": profil.ip_terakhir,
                   "sks_lulus_awal": ctx.kurikulum.total_sks(profil.mk_lulus),
                   "n_mk_gagal_belum_lulus": len(profil.mk_pernah_gagal), "theta": ctx.theta}
            metrik: dict[str, dict] = {}
            pelanggaran: dict[str, str] = {}
            for mode in MODE:
                plan = susun_rencana(profil, mode, ctx)
                metrik[mode] = metrik_rencana(plan, cfg, aktual, ctx.theta)
                pelanggaran[mode] = " | ".join(plan.validasi.pelanggaran)
            for m in METRIK_NUMERIK + METRIK_BOOLEAN:  # pasangan kolom bersebelahan
                for mode in MODE:
                    v = metrik[mode].get(m)
                    rec[f"{m}_{mode}"] = int(v) if isinstance(v, bool) else v
            for mode in MODE:
                rec[f"pelanggaran_{mode}"] = pelanggaran[mode]
            baris.append(rec)

        df = pd.DataFrame(baris)
        tabel = ringkas(df)
        db.simpan_hasil(cfg, "simulasi_metrik", df)
        db.simpan_hasil(cfg, "simulasi_ringkasan", tabel)
        # format berpasangan untuk JASP (KF-21); asal riwayat hanya disimpan di basis data
        db.ekspor_csv(cfg, "simulasi_metrik.csv", df.drop(columns="sumber_data", errors="ignore"))
        db.ekspor_csv(cfg, "simulasi_ringkasan.csv", tabel)
        gagal_keras = {
            mode: df.loc[(df[f"lolos_prasyarat_{mode}"] == 0) | (df[f"lolos_sks_{mode}"] == 0)
                         | (df[f"lolos_paritas_{mode}"] == 0), ["id_anon", "k", f"pelanggaran_{mode}"]]
            .to_dict("records") if len(df) else []
            for mode in MODE
        }
        ringkasan = {
            "n_profil": int(len(df)),
            "n_alumni_uji": len(kelompok),
            "sumber_data": sorted(set(sumber[i] for i in kelompok)),
            "k_potong": k_list,
            "theta": ctx.theta,
            "rata": {r["metrik"]: {"arm": r["rata_arm"], "hybrid": r["rata_hybrid"]} for r in tabel.to_dict("records")},
            "profil_gagal_aturan_keras": gagal_keras,
        }
        log.ringkasan.update({k: v for k, v in ringkasan.items() if k != "profil_gagal_aturan_keras"})
        log.ringkasan["n_gagal_aturan_keras"] = {m: len(v) for m, v in gagal_keras.items()}
        ringkasan["run_id"] = log.run_id
        db.simpan_ringkasan(cfg, "simulasi", ringkasan)
    ekspor_rm2(cfg)
    return ringkasan


# ---------------------------------------------------------------------
# Validasi terhadap KRS akademik nyata (data uji)
# ---------------------------------------------------------------------

METRIK_VALIDASI = ["jaccard", "presisi", "recall", "sks_rekomendasi", "beban_rekomendasi"]
METRIK_VALIDASI_BOOLEAN = ["lolos_prasyarat", "lolos_sks", "lolos_paritas"]


def mk_tersimpulkan(s: int, kur: Kurikulum) -> tuple[set[str], int]:
    """MK wajib struktur Jalur A semester < s (paket) dan SKS slot agama (agama tidak disimpulkan)."""
    kode, sks_agama = set(), 0
    for t in range(1, s):
        for slot in kur.slot("A", t):
            if slot.jenis_slot == "MK_WAJIB" and slot.kode_mk:
                kode.add(slot.kode_mk)
            elif slot.jenis_slot == "MK_WAJIB_PILIH_SATU":
                sks_agama += slot.sks
    return kode, sks_agama


def profil_krs_asli(s: int, diambil: set[str], info_ip: tuple[float, float, int, int] | None, kur: Kurikulum,
                    cfg: Config) -> tuple[Profile | None, str]:
    """Profil semester berjalan seorang mahasiswa (KRS akademik) atau (None, kategori pengecualian)."""
    if info_ip is None:
        return None, "tanpa_ip_semester_sebelumnya"
    ip, ipk, sksk_lulus, sem_ip = info_ip
    if sem_ip != s - 1:
        return None, "semester_ip_tidak_berurutan"
    struktur, sks_agama = mk_tersimpulkan(s, kur)
    lulus = struktur - diambil
    selisih = sksk_lulus - (kur.total_sks(lulus) + sks_agama)
    if abs(selisih) > cfg.simulasi.validasi_krs.toleransi_sks:
        # lebih: MK pilihan/percepatan yang tidak teramati; kurang: MK tidak lulus yang tidak diulang
        return None, "sks_lebih_dari_struktur" if selisih > 0 else "sks_kurang_dari_struktur"
    return Profile(semester_saat_ini=s, ip_terakhir=ip, mk_lulus=lulus, jalur="A", ipk=ipk), ""


def jalankan_validasi_krs(cfg: Config, progress: Callable[[int, int], None] | None = None) -> dict:
    """Bandingkan rekomendasi semester berjalan (kedua mode) dengan KRS akademik nyata mahasiswa data uji."""
    from src.akademik import ip_sebelumnya, muat_ip

    sv = cfg.simulasi.validasi_krs
    with catat_proses(cfg, "validasi_krs", {"semester": sv.semester, "toleransi_sks": sv.toleransi_sks}) as log:
        ctx = KonteksPlanner.muat(cfg)
        if ctx.theta is None:
            raise ValueError("θ belum tersedia, jalankan pipeline dulu")
        uji, alasan = transaksi_segar(cfg, "uji")
        if alasan:
            raise ValueError(f"{alasan}, jalankan ulang preprocessing")
        kur = ctx.kurikulum
        info = ip_sebelumnya(muat_ip(cfg))
        calon = [t for t in uji if t.sumber == "asli" and t.semester_ke in sv.semester]
        baris, dikecualikan = [], {}
        for i, t in enumerate(calon, start=1):
            if progress:
                progress(i, len(calon))
            diambil = {k for k in t.kode if k in kur}
            profil, sebab = profil_krs_asli(t.semester_ke, diambil, info.get(t.id_anon), kur, cfg)
            if profil is None:
                kunci = f"S{t.semester_ke}: {sebab}"
                dikecualikan[kunci] = dikecualikan.get(kunci, 0) + 1
                continue
            rec = {"id_anon": t.id_anon, "semester": t.semester_ke, "ip_terakhir": profil.ip_terakhir,
                   "ipk": profil.ipk, "sks_aktual": kur.total_sks(diambil), "beban_aktual": round(ctx.beban(diambil), 4),
                   "aktual_melebihi_theta": int(ctx.beban(diambil) > ctx.theta + 1e-9), "theta": ctx.theta}
            hasil: dict[str, dict] = {}
            for mode in MODE:
                plan = susun_rencana(profil, mode, ctx)
                rekom = {it.kode for it in plan.semester[0].items if it.jenis == "mk"} if plan.semester else set()
                irisan = len(rekom & diambil)
                per = plan.validasi.per_aturan if plan.validasi else {}
                hasil[mode] = {
                    "jaccard": round(jaccard(rekom, diambil), 4),
                    "presisi": round(irisan / len(rekom), 4) if rekom else None,
                    "recall": round(irisan / len(diambil), 4) if diambil else None,
                    "sks_rekomendasi": kur.total_sks(rekom),
                    "beban_rekomendasi": round(ctx.beban(rekom), 4),
                    "lolos_prasyarat": per.get("AD-03", False), "lolos_sks": per.get("AD-02", False),
                    "lolos_paritas": per.get("AD-04", False),
                }
            for m in METRIK_VALIDASI + METRIK_VALIDASI_BOOLEAN:
                for mode in MODE:
                    v = hasil[mode][m]
                    rec[f"{m}_{mode}"] = int(v) if isinstance(v, bool) else v
            baris.append(rec)
        df = pd.DataFrame(baris)
        ringkas_df = ringkas(df, METRIK_VALIDASI, METRIK_VALIDASI_BOOLEAN)
        db.simpan_hasil(cfg, "validasi_krs", df)
        db.simpan_hasil(cfg, "validasi_krs_ringkasan", ringkas_df)
        db.ekspor_csv(cfg, "validasi_krs.csv", df)
        db.ekspor_csv(cfg, "validasi_krs_ringkasan.csv", ringkas_df)
        ringkasan = {
            "n_profil": int(len(df)),
            "n_kandidat": len(calon),
            "dikecualikan": dikecualikan,
            "per_semester": {int(s): int(n) for s, n in df["semester"].value_counts().sort_index().items()}
            if len(df) else {},
            "theta": ctx.theta,
            "proporsi_aktual_melebihi_theta": round(float(df["aktual_melebihi_theta"].mean()), 4) if len(df) else None,
            "rata": {r["metrik"]: {"arm": r["rata_arm"], "hybrid": r["rata_hybrid"]}
                     for r in ringkas_df.to_dict("records")},
        }
        log.ringkasan.update({k: v for k, v in ringkasan.items() if not isinstance(v, dict)})
        db.simpan_ringkasan(cfg, "validasi_krs", ringkasan)
    return ringkasan


def ringkas(df: pd.DataFrame, numerik: list[str] | None = None, boolean: list[str] | None = None) -> pd.DataFrame:
    """Rata-rata tiap metrik per mode + selisih (hybrid − arm)."""
    numerik = METRIK_NUMERIK if numerik is None else numerik
    boolean = METRIK_BOOLEAN if boolean is None else boolean
    baris = []
    for m in numerik + boolean:
        if f"{m}_arm" not in df.columns:
            continue
        a = pd.to_numeric(df[f"{m}_arm"], errors="coerce")
        h = pd.to_numeric(df[f"{m}_hybrid"], errors="coerce")
        baris.append({"metrik": m, "rata_arm": round(float(a.mean()), 4), "rata_hybrid": round(float(h.mean()), 4),
                      "selisih_hybrid_minus_arm": round(float((h - a).mean()), 4),
                      "jenis": "proporsi" if m in boolean else "numerik"})
    return pd.DataFrame(baris, columns=["metrik", "rata_arm", "rata_hybrid", "selisih_hybrid_minus_arm", "jenis"])


DEFINISI_KEPATUHAN = {
    "lolos_semua": "validate_plan lolos seluruh aturan AD-01..AD-16; AD-01 dinilai atas seluruh horizon rencana "
                   "(hingga 14 semester, AD-09)",
    "lolos_prasyarat": "AD-03 prasyarat lulus di semester sebelumnya",
    "lolos_sks": "AD-02 SKS per semester tidak melebihi batas pita IP",
    "lolos_paritas": "AD-04 MK ditempatkan pada semester penawarannya",
    "lolos_145": "AD-01 lolos DAN 145 SKS sudah tercapai paling lambat semester 8 (lebih ketat dari lolos_semua)",
    "lolos_theta": "seluruh semester reguler B_s <= theta (diukur juga pada mode ARM murni)",
}


def tabel_kepatuhan(df: pd.DataFrame) -> pd.DataFrame:
    """RM2 kepatuhan: tabel kontingensi berpasangan + uji McNemar per metrik boolean (Pers. 2.20).
    b = lolos ARM tetapi tidak lolos hibrida, c = sebaliknya; p eksak = binomial dua sisi atas b+c."""
    from scipy.stats import binomtest

    baris = []
    for m in METRIK_BOOLEAN:
        if f"{m}_arm" not in df.columns:
            continue
        a, h = df[f"{m}_arm"].astype(int), df[f"{m}_hybrid"].astype(int)
        b, c = int(((a == 1) & (h == 0)).sum()), int(((a == 0) & (h == 1)).sum())
        n_diskordan = b + c
        chi2 = round((b - c) ** 2 / n_diskordan, 4) if n_diskordan else None
        p_eksak = round(float(binomtest(min(b, c), n_diskordan, 0.5).pvalue), 4) if n_diskordan else 1.0
        baris.append({
            "metrik": m, "n_profil": int(len(df)),
            "lolos_arm": int(a.sum()), "lolos_hybrid": int(h.sum()),
            "proporsi_arm": round(float(a.mean()), 4), "proporsi_hybrid": round(float(h.mean()), 4),
            "keduanya_lolos": int(((a == 1) & (h == 1)).sum()), "keduanya_gagal": int(((a == 0) & (h == 0)).sum()),
            "b_arm_lolos_hybrid_gagal": b, "c_arm_gagal_hybrid_lolos": c,
            "mcnemar_chi2": chi2, "mcnemar_p_eksak": p_eksak,
            "definisi": DEFINISI_KEPATUHAN.get(m, ""),
        })
    return pd.DataFrame(baris)


def tabel_parameter(cfg: Config) -> pd.DataFrame:
    """Parameter akhir yang dipakai simulasi RM2 (dari ringkasan proses di basis data, KNF-09)."""
    pre = db.baca_ringkasan(cfg, "preprocessing") or {}
    arm = db.baca_ringkasan(cfg, "arm") or {}
    th = db.baca_ringkasan(cfg, "theta") or {}
    sim = db.baca_ringkasan(cfg, "simulasi") or {}
    p = arm.get("parameter", {})
    dist = th.get("distribusi", {})
    baris = [
        ("arm", "min_support", p.get("min_support"), "ambang support per partisi semester (Pers. 4.1)"),
        ("arm", "min_confidence", p.get("min_confidence"), ""),
        ("arm", "min_lift", p.get("min_lift"), "aturan dipertahankan bila lift > min_lift"
         if p.get("lift_ketat") else "aturan dipertahankan bila lift >= min_lift"),
        ("arm", "max_len", p.get("max_len"), "panjang maksimum itemset"),
        ("arm", "partisi_semester", p.get("partisi_semester"), ""),
        ("arm", "buang_konsekuen_wajib", p.get("buang_konsekuen_wajib"), "aturan berkonsekuen MK wajib dibuang"),
        ("arm", "n_transaksi_latih_ditambang", arm.get("n_transaksi_latih"), ""),
        ("arm", "n_aturan", arm.get("n_aturan"), "setelah filter konsekuen wajib"),
        ("arm", "n_aturan_konsekuen_wajib_dibuang", arm.get("n_aturan_konsekuen_wajib_dibuang"), ""),
        ("arm", "rata_confidence", arm.get("rata_confidence"), ""),
        ("arm", "rata_lift", arm.get("rata_lift"), ""),
        ("arm", "run_id", arm.get("run_id"), "kode eksekusi pada log_proses"),
        ("theta", "theta", th.get("theta"), f"persentil ke-{th.get('persentil', 0):g} B_s semester transaksi data "
                                            "latih (Pers. 4.5)" if th.get("dikalibrasi_data") else th.get("sumber")),
        ("theta", "persentil", th.get("persentil"), ""),
        ("theta", "metode_beban", th.get("metode"), "B_s = sum(CL x SKS) (Pers. 4.4)"),
        ("theta", "n_semester_kalibrasi", th.get("n_semester"), "jumlah semester alumni data latih"),
        ("theta", "B_s_min", dist.get("min"), ""), ("theta", "B_s_p25", dist.get("p25"), ""),
        ("theta", "B_s_median", dist.get("median"), ""), ("theta", "B_s_p75", dist.get("p75"), ""),
        ("theta", "B_s_max", dist.get("max"), ""),
        ("theta", "mk_tanpa_cl", ";".join(th.get("mk_tanpa_cl", [])), "memakai cl_default"),
        ("praproses", "n_alumni_awal", pre.get("n_alumni_awal"), ""),
        ("praproses", "n_alumni_dibuang_masa_studi", pre.get("n_alumni_dibuang_masa_studi"), "masa studi > 8 semester"),
        ("praproses", "n_alumni_masa_studi_ok", pre.get("n_alumni_masa_studi_ok"), "masa studi <= 8 semester"),
        ("praproses", "n_alumni_latih", pre.get("n_alumni_latih"), "split 80:20 per alumni, seed tetap"),
        ("praproses", "n_alumni_uji", pre.get("n_alumni_uji"), ""),
        ("praproses", "n_mahasiswa_krs_latih", pre.get("n_mahasiswa_asli_latih"), "KRS semester berjalan"),
        ("praproses", "n_mahasiswa_krs_uji", pre.get("n_mahasiswa_asli_uji"), "KRS semester berjalan"),
        ("praproses", "n_transaksi_latih", pre.get("n_transaksi_latih"), ""),
        ("praproses", "n_transaksi_uji", pre.get("n_transaksi_uji"), ""),
        ("praproses", "proporsi_kode_terpetakan", pre.get("proporsi_kode_terpetakan"), ""),
        ("praproses", "run_id", pre.get("run_id"), ""),
        ("simulasi", "n_alumni_uji_dipakai", sim.get("n_alumni_uji"), ""),
        ("simulasi", "k_potong", ";".join(map(str, sim.get("k_potong", []))), "riwayat dipotong di semester k"),
        ("simulasi", "n_profil", sim.get("n_profil"), "alumni uji x jumlah k"),
        ("simulasi", "horizon_semester", cfg.kurikulum.horizon_semester, "batas lolos_145"),
        ("simulasi", "run_id", sim.get("run_id"), ""),
    ]
    return pd.DataFrame(baris, columns=["kelompok", "parameter", "nilai", "keterangan"])


def ekspor_rm2(cfg: Config) -> dict[str, str]:
    """Ekspor pendukung RM2 dari hasil simulasi tersimpan: parameter akhir + kepatuhan/McNemar."""
    df = db.baca_hasil(cfg, "simulasi_metrik")
    if df is None or df.empty:
        raise ValueError("hasil simulasi belum ada, jalankan simulasi dulu")
    kep = tabel_kepatuhan(df)
    par = tabel_parameter(cfg)
    db.simpan_hasil(cfg, "simulasi_kepatuhan", kep)
    return {"simulasi_kepatuhan": str(db.ekspor_csv(cfg, "simulasi_kepatuhan.csv", kep)),
            "rm2_parameter": str(db.ekspor_csv(cfg, "rm2_parameter.csv", par))}


if __name__ == "__main__":
    import json
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    c = load_config()
    r = jalankan_simulasi(c)
    print(json.dumps({k: v for k, v in r.items() if k != "profil_gagal_aturan_keras"}, indent=2, ensure_ascii=False))
    print("Profil gagal aturan keras:", {m: len(v) for m, v in r["profil_gagal_aturan_keras"].items()})
    print(json.dumps(jalankan_validasi_krs(c), indent=2, ensure_ascii=False))
