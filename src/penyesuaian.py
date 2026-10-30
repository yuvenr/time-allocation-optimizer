"""Penyesuaian manual rencana oleh mahasiswa (iterasi 3): menggeser MK antarsemester (drag & drop di UI).

Mahasiswa boleh memindahkan MK pada semester reguler — mis. PLP, PKM, atau Skripsi yang sering jatuh di
semester 7 — ke semester lain, selama susunan baru tidak menimbulkan pelanggaran BARU pada aturan keras
(``penyesuaian.aturan_keras``: prasyarat AD-03, penawaran AD-04, pita SKS AD-02, Skripsi AD-10, PLP AD-11, …).
Pemeriksaan memakai ``validate_plan`` yang sama dengan planner (UC-13), sehingga MK yang menjadi prasyarat MK
lain tidak dapat digeser melewati MK yang bergantung padanya. Korequisit (AT15: PLP 1 + PLP 2) ikut dipindahkan.

- Semester paket (AD-08) dan semester konversi merdeka belajar (AD-14) tidak dapat diubah.
- Pelanggaran baru aturan non-keras (mis. AD-07: MK wajib tertunda tidak diambil pada kesempatan pertama) dan
  beban semester > θ (mode hybrid, kecuali ``theta_keras``) diterima sebagai catatan untuk DPA.
- Susunan asli rekomendasi sistem disimpan di ``Plan.semester_sistem`` agar DPA dapat membandingkannya dan
  mahasiswa dapat mengembalikannya. Planner (kedua mode) tidak berubah.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.planner import PREFIKS_HORIZON, KonteksPlanner, peringatan_horizon
from src.rules import Keadaan, batas_sks, cek_mk, validate_plan
from src.schemas import Pemindahan, Plan, SemesterRencana

PREFIKS_ALASAN = "Digeser manual oleh mahasiswa"
PREFIKS_CATATAN = "Rencana disesuaikan manual oleh mahasiswa"


@dataclass
class HasilPenyesuaian:
    """``plan`` = rencana baru bila diterima, selain itu rencana semula; ``usulan`` = susunan yang diperiksa."""

    plan: Plan
    diterima: bool
    ditolak: list[str] = field(default_factory=list)      # pelanggaran baru aturan keras (alasan penolakan)
    catatan: list[str] = field(default_factory=list)      # pelanggaran baru aturan lain, θ, korequisit otomatis
    pemindahan: list[Pemindahan] = field(default_factory=list)
    usulan: Plan | None = None

    @property
    def berubah(self) -> bool:
        return bool(self.pemindahan)


def susunan_rencana(plan: Plan) -> dict[int, list[str]]:
    """Susunan MK yang dapat digeser: semester reguler → daftar kode MK (urutan rencana)."""
    return {sm.semester: [it.kode for it in sm.items if it.jenis == "mk"]
            for sm in plan.semester if sm.jenis == "reguler"}


def semester_terkunci(plan: Plan) -> set[int]:
    """Semester paket (AD-08) dan konversi (AD-14) — isinya ditetapkan struktur kurikulum."""
    return {sm.semester for sm in plan.semester if sm.jenis != "reguler"}


def semester_papan(plan: Plan, ctx: KonteksPlanner) -> list[int]:
    """Semester yang ditampilkan di papan drag & drop: semester reguler rencana + satu semester kosong
    sesudahnya (bila masih dalam masa studi maksimal AT04), agar MK dapat digeser ke semester berikutnya."""
    s_max = ctx.kurikulum.param_int("AT04", "semester_max")
    reguler = sorted(susunan_rencana(plan))
    if not reguler:
        return []
    terakhir = max(sm.semester for sm in plan.semester)
    tambahan = [terakhir + 1] if terakhir + 1 <= s_max and terakhir + 1 not in semester_terkunci(plan) else []
    return reguler + tambahan


def _semester_asal_sistem(plan: Plan) -> dict[str, int]:
    acuan = plan.semester_sistem if plan.semester_sistem is not None else plan.semester
    return {it.kode: sm.semester for sm in acuan for it in sm.items if it.jenis == "mk"}


def _tolak(plan: Plan, alasan: list[str]) -> HasilPenyesuaian:
    return HasilPenyesuaian(plan=plan, diterima=False, ditolak=alasan)


def sesuaikan_rencana(plan: Plan, susunan: dict[int, list[str]], ctx: KonteksPlanner) -> HasilPenyesuaian:
    """Terapkan susunan baru (semester → kode MK) bila tidak menimbulkan pelanggaran baru aturan keras.

    ``susunan`` harus memuat tepat MK yang dapat digeser pada ``plan`` (lihat ``susunan_rencana``); semester
    yang tidak disebut dianggap kosong. ``ValueError`` bila himpunan MK berbeda (masukan tidak sah).
    """
    kur, cfg = ctx.kurikulum, ctx.cfg
    awal = susunan_rencana(plan)
    asal = {k: s for s, ks in awal.items() for k in ks}
    susunan = {int(s): list(ks) for s, ks in susunan.items()}
    daftar = [k for ks in susunan.values() for k in ks]
    if sorted(daftar) != sorted(asal):
        raise ValueError("susunan harus memuat tepat MK yang dapat digeser pada rencana (tanpa tambahan/kehilangan)")

    s_awal, s_max = plan.profil.semester_saat_ini, kur.param_int("AT04", "semester_max")
    terkunci = semester_terkunci(plan)
    salah = []
    for s, ks in sorted(susunan.items()):
        if not ks:
            continue
        if s in terkunci:
            jenis = next(sm.jenis for sm in plan.semester if sm.semester == s)
            salah.append(f"S{s} adalah semester {'paket (AD-08)' if jenis == 'paket' else 'konversi (AD-14)'}, "
                         f"isinya mengikuti struktur kurikulum dan tidak dapat ditambah MK lain")
        elif s < s_awal or s > s_max:
            salah.append(f"S{s} di luar rentang rencana S{s_awal}–S{s_max} (AD-09)")
    if salah:
        return _tolak(plan, salah)

    # korequisit (AT15): MK yang tidak ikut digeser menyusul pasangannya
    tujuan = {k: s for s, ks in susunan.items() for k in ks}
    otomatis: set[str] = set()
    for k in sorted(k for k in asal if tujuan[k] != asal[k]):
        for q in sorted(kur.korequisit(k)):
            if q in asal and tujuan[q] == asal[q] and tujuan[q] != tujuan[k]:
                susunan[tujuan[q]].remove(q)
                susunan.setdefault(tujuan[k], []).append(q)
                tujuan[q] = tujuan[k]
                otomatis.add(q)
    waktu = datetime.now()
    pemindahan = [Pemindahan(kode=k, dari=asal[k], ke=tujuan[k], waktu=waktu, otomatis=k in otomatis)
                  for k in sorted(asal, key=lambda x: (asal[x], x)) if tujuan[k] != asal[k]]
    if not pemindahan:
        return HasilPenyesuaian(plan=plan, diterima=True, usulan=plan)

    usulan = _bangun(plan, susunan, awal, ctx)
    lama = set(plan.validasi.pelanggaran) if plan.validasi else set()
    baru = [h for h in usulan.validasi.rincian if h.melanggar and h.teks() not in lama]
    keras = set(cfg.penyesuaian.aturan_keras)

    def teks(h) -> str:   # sebut MK yang terdampak (alasan aturan kadang hanya menyebut MK prasyaratnya)
        return f"{kur.label(h.kode_mk)}: {h.teks()}" if h.kode_mk in kur else h.teks()

    ditolak = [teks(h) for h in baru if h.kode in keras]
    catatan = [f"{kur.label(p.kode)} ikut dipindahkan ke S{p.ke} (korequisit, diambil bersamaan, AT15)"
               for p in pemindahan if p.otomatis]
    catatan += [f"Menyimpang dari anjuran: {teks(h)}" for h in baru if h.kode not in keras]
    if plan.mode == "hybrid" and plan.theta is not None:
        beban_lama = {sm.semester: sm.total_beban for sm in plan.semester}
        for sm in usulan.semester:
            if sm.total_beban > plan.theta + 1e-9 and beban_lama.get(sm.semester, 0.0) <= plan.theta + 1e-9:
                teks = f"S{sm.semester}: beban kognitif {sm.total_beban:.1f} > θ {plan.theta:.1f}"
                (ditolak if cfg.penyesuaian.theta_keras else catatan).append(teks)
    if ditolak:
        return HasilPenyesuaian(plan=plan, diterima=False, ditolak=ditolak, catatan=catatan, pemindahan=pemindahan,
                                usulan=usulan)
    usulan.penyesuaian = list(plan.penyesuaian) + pemindahan
    if plan.status_dpa != "belum_ditinjau":
        usulan.status_dpa = "belum_ditinjau"
        catatan.append("Status validasi DPA kembali 'belum ditinjau' karena rencana berubah")
    return HasilPenyesuaian(plan=usulan, diterima=True, catatan=catatan, pemindahan=pemindahan, usulan=usulan)


def _bangun(plan: Plan, susunan: dict[int, list[str]], awal: dict[int, list[str]], ctx: KonteksPlanner) -> Plan:
    """Rencana baru dari susunan: total SKS/beban, sumber aturan per MK (cek ulang di semester barunya),
    alasan pemindahan, peringatan horizon, dan ``validate_plan``."""
    kur, cfg = ctx.kurikulum, ctx.cfg
    prof = plan.profil
    sistem = _semester_asal_sistem(plan)
    item = {it.kode: it for sm in plan.semester for it in sm.items if it.jenis == "mk"}
    lama = {sm.semester: sm for sm in plan.semester}
    horizon = cfg.kurikulum.horizon_semester
    kd = Keadaan.dari_profil(prof, kur, cfg)
    semester: list[SemesterRencana] = []
    for s in sorted(set(lama) | {s for s, ks in susunan.items() if ks}):
        sm_lama = lama.get(s)
        if sm_lama is not None and sm_lama.jenis != "reguler":
            semester.append(sm_lama.model_copy(deep=True))
            kd = kd.tambah([it.kode for it in sm_lama.items if it.jenis == "mk"],
                           sum(it.sks for it in sm_lama.items if it.jenis == "konversi"))
            continue
        kode_s = susunan.get(s, [])
        items = []
        for k in kode_s:
            it = item[k].model_copy(deep=True)
            hasil = cek_mk(k, s, kd, kur, cfg)
            it.sumber = [h.teks() for h in hasil if h.status == "lolos"]
            it.perlu_verifikasi = [h.teks() for h in hasil if h.status == "perlu_verifikasi_manual"]
            it.alasan = [a for a in it.alasan if not a.startswith(PREFIKS_ALASAN)]
            if sistem.get(k, s) != s:
                it.alasan.insert(0, f"{PREFIKS_ALASAN}: rekomendasi sistem S{sistem[k]} → S{s}")
            items.append(it)
        if sm_lama is not None:
            sks_min, sks_max, sumber = sm_lama.sks_min, sm_lama.batas_sks, sm_lama.sumber_batas
        else:
            sks_min, sks_max, sumber = batas_sks(s, prof.jalur, prof.ip_terakhir, kur)
        tetap = sm_lama is not None and kode_s == awal.get(s)
        semester.append(SemesterRencana(
            semester=s, jenis="reguler", batas_sks=sks_max, sks_min=sks_min, sumber_batas=sumber, items=items,
            total_sks=kur.total_sks(kode_s), total_beban=round(ctx.beban(kode_s), 4),
            dilewati_theta=list(sm_lama.dilewati_theta) if tetap else [],
            catatan=list(sm_lama.catatan) if sm_lama is not None else []))
        kd = kd.tambah(kode_s)
    while semester and not semester[-1].items and semester[-1].semester > horizon:
        semester.pop()

    p = plan.model_copy(deep=True)
    p.semester = semester
    if p.semester_sistem is None:
        p.semester_sistem = [sm.model_copy(deep=True) for sm in plan.semester]
    s_max = kur.param_int("AT04", "semester_max")
    p.peringatan = [w for w in p.peringatan if not w.startswith((PREFIKS_HORIZON, PREFIKS_CATATAN))]
    if (w := peringatan_horizon(semester, horizon, s_max, kur)) is not None:
        p.peringatan.append(w)
    p.peringatan.append(f"{PREFIKS_CATATAN}; susunan asli rekomendasi sistem tetap tersimpan untuk ditinjau DPA.")
    p.validasi = validate_plan(p, prof, kur, cfg, beban=ctx.beban)
    return p


def pindahkan(plan: Plan, kode: str, ke: int, ctx: KonteksPlanner) -> HasilPenyesuaian:
    """Pindahkan satu MK (beserta korequisitnya) ke semester ``ke``."""
    susunan = susunan_rencana(plan)
    dari = next((s for s, ks in susunan.items() if kode in ks), None)
    if dari is None:
        return _tolak(plan, [f"{ctx.kurikulum.label(kode) if kode in ctx.kurikulum else kode} tidak dapat digeser "
                             f"(bukan MK pada semester reguler rencana ini)"])
    susunan[dari].remove(kode)
    susunan.setdefault(int(ke), []).append(kode)
    return sesuaikan_rencana(plan, susunan, ctx)


def semester_tujuan_sah(plan: Plan, kode: str, ctx: KonteksPlanner) -> list[int]:
    """Semester tujuan yang DITERIMA untuk memindahkan ``kode`` (untuk menu pindah yang aksesibel)."""
    susunan = susunan_rencana(plan)
    dari = next((s for s, ks in susunan.items() if kode in ks), None)
    if dari is None:
        return []
    s_max = ctx.kurikulum.param_int("AT04", "semester_max")
    terkunci = semester_terkunci(plan)
    return [s for s in range(plan.profil.semester_saat_ini, s_max + 1)
            if s != dari and s not in terkunci and pindahkan(plan, kode, s, ctx).diterima]


def kembalikan_rekomendasi(plan: Plan, ctx: KonteksPlanner) -> Plan:
    """Kembalikan susunan rekomendasi sistem (membatalkan semua pemindahan manual)."""
    if plan.semester_sistem is None:
        return plan
    kur, cfg = ctx.kurikulum, ctx.cfg
    p = plan.model_copy(deep=True)
    p.semester, p.semester_sistem, p.penyesuaian = p.semester_sistem, None, []
    p.peringatan = [w for w in p.peringatan if not w.startswith((PREFIKS_HORIZON, PREFIKS_CATATAN))]
    if (w := peringatan_horizon(p.semester, cfg.kurikulum.horizon_semester, kur.param_int("AT04", "semester_max"),
                                kur)) is not None:
        p.peringatan.append(w)
    p.validasi = validate_plan(p, p.profil, kur, cfg, beban=ctx.beban)
    if p.status_dpa != "belum_ditinjau":
        p.status_dpa = "belum_ditinjau"
    return p
