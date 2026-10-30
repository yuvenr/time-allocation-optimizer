"""Penyusun rencana studi (UC-02) — mode ``arm`` dan ``hybrid`` — Kurikulum 2024.

Kedua mode memakai algoritma yang SAMA. Satu-satunya perbedaan: pada mode ``hybrid``
kandidat yang membuat beban semester > θ dilewati. Jangan menambah logika lain yang
hanya berlaku di salah satu mode.

Per semester s, mulai semester berjalan sampai horizon (``kurikulum.horizon_semester``;
diperpanjang ≤ AT04 bila belum lulus):

1. Semester paket (AT06: 1–2): MK paket jalur (08) yang boleh diambil; slot agama = input.
2. Slot konversi merdeka belajar (Jalur B–F, semester 6) bila AT12 terpenuhi: satu item
   konversi (AT13) — tema yang terpenuhi ditandai perlu verifikasi DPA.
3. Semester reguler:
   a. batas SKS = pita 09 (jalur → SEMUA; tanpa baris → pita seragam AT05) dengan IP terakhir
      sebagai asumsi IP semester mendatang;
   b. kandidat = MK yang DIBUTUHKAN (wajib belum lulus yang sudah tiba waktunya menurut
      struktur jalur, atau disarankan "tambah" oleh 09; pilihan dari tema yang belum terpenuhi)
      dan lolos seluruh aturan per MK (``src.rules.cek_mk``);
   c. urutan (lexicographic): tingkat prioritas → saran "kurangi" 09 paling akhir → skor ARM
      (support(c@Ss) + max confidence×lift aturan X → c@Ss dengan X ⊆ MK terpilih) → pemecah
      seri deterministik. Tingkat: (0) wajib tertunda/mengulang (AT09) → (1) wajib paket
      semester s / saran "tambah" 09 → (2) pilihan untuk slot tema 08 / saran 09 / AT08 /
      slot sisa tema → (3) pilihan tema lain yang belum terpenuhi;
   d. greedy: ambil unit teratas (MK + korequisitnya, mis. PLP 1 + PLP 2) yang muat batas SKS
      — dan θ pada mode hybrid — lalu hitung ulang urutan (skor ARM bergantung MK terpilih).
4. MK terpilih dianggap lulus; lanjut ke s+1.

Tanpa data KRS alumni, ARM kosong sehingga urutan hanya ditentukan struktur kurikulum resmi
(08, 09) dan aturan — hal ini dinyatakan di asumsi setiap rencana.
"""
from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Literal

import numpy as np

from src import db
from src.arm import ModelARM, pilih_dari_cfg
from src.config import Config, load_config
from src.data_loader import Kurikulum, ada_transaksi, label_sumber_data, load_kurikulum, transaksi_segar
from src.llm.beban import muat_beban_mk
from src.rules import (Keadaan, _daftar_param, aturan_penawaran, batas_sks, boleh_diambil, cek_mk, hitung_ipk,
                       mk_paket_semester, semester_paket, syarat_konversi, tema_terpenuhi, unit_korequisit,
                       validate_plan)
from src.schemas import (JALUR, STATUS_DPA, HasilAturan, ItemRencana, PitaSKS, Plan, Profile, SemesterRencana,
                         Transaksi)

Mode = Literal["arm", "hybrid"]
KODE_KONVERSI = "KONVERSI-MB"
TINGKAT = {
    0: "MK wajib tertunda/mengulang, diprioritaskan (AT09)",
    1: "MK wajib paket semester ini",
    2: "MK pilihan untuk slot tema semester ini",
    3: "MK pilihan untuk tema yang belum terpenuhi",
}

# ---------------------------------------------------------------------
# Beban kognitif semester & θ
# ---------------------------------------------------------------------


def hitung_beban(pasangan: Iterable[tuple[float, int]], metode: str = "jumlah") -> float:
    """B_s = Σ cl×sks (``jumlah``) atau Σ cl×sks / Σ sks (``rata_tertimbang``)."""
    pasangan = list(pasangan)
    if not pasangan:
        return 0.0
    total = sum(cl * sks for cl, sks in pasangan)
    if metode == "rata_tertimbang":
        return total / sum(sks for _, sks in pasangan)
    return total


def _ringkas_theta(b: list[float], cfg: Config, sumber: str, **extra) -> dict:
    arr = np.array([x for x in b if x > 0], dtype=float)
    if len(arr) == 0:
        return {"theta": None, "sumber": f"{sumber} (tidak ada data beban)", "n_semester": 0, **extra}
    q = np.percentile(arr, [0, 25, 50, 75, 100])
    return {
        "theta": round(float(np.percentile(arr, cfg.beban.theta_persentil)), 4),
        "sumber": sumber, "persentil": cfg.beban.theta_persentil, "metode": cfg.beban.metode,
        "n_semester": int(len(arr)),
        "distribusi": {"min": round(float(q[0]), 2), "p25": round(float(q[1]), 2), "median": round(float(q[2]), 2),
                       "p75": round(float(q[3]), 2), "max": round(float(q[4]), 2)},
        **extra,
    }


# ---------------------------------------------------------------------
# Konteks
# ---------------------------------------------------------------------


@dataclass
class KonteksPlanner:
    """Semua masukan planner yang dimuat sekali lalu dipakai berulang (UI & simulasi)."""

    cfg: Config
    kurikulum: Kurikulum
    arm: ModelARM
    cl: dict[str, float]
    keterangan_cl: dict[str, str]
    cl_default: float | None
    theta: float | None = None
    info_theta: dict = field(default_factory=dict)
    info_cl: dict = field(default_factory=dict)
    status_krs: str = "belum_tersedia"
    justifikasi_cpmk: dict[str, list[str]] = field(default_factory=dict)
    _penawaran: dict = field(default_factory=dict, repr=False, compare=False)

    def ditawarkan(self, kode: str, s: int, kd: "Keadaan") -> bool:
        """AD-04 untuk (MK, semester) — di-cache; penawaran tidak bergantung pada riwayat mahasiswa."""
        kunci = (kode, s)
        if kunci not in self._penawaran:
            self._penawaran[kunci] = aturan_penawaran(kode, s, kd, self.kurikulum, self.cfg).lolos
        return self._penawaran[kunci]

    @classmethod
    def muat(cls, cfg: Config | None = None) -> "KonteksPlanner":
        cfg = cfg or load_config()
        kur = load_kurikulum(cfg)
        beban = muat_beban_mk(cfg, kur)
        cl, ket, info_cl = beban.cl, beban.keterangan, beban.info
        default = cfg.beban.cl_default
        if default is None and cl:
            default = float(np.median(list(cl.values())))
        ctx = cls(cfg, kur, ModelARM.dari_cfg(cfg, kur), cl, ket, default, info_cl=info_cl,
                  status_krs=label_sumber_data(cfg), justifikasi_cpmk=beban.justifikasi)
        ctx.info_theta = tentukan_theta(ctx)
        ctx.theta = ctx.info_theta.get("theta")
        return ctx

    @property
    def mk_tanpa_cl(self) -> list[str]:
        return sorted(k for k in self.kurikulum.kode if k not in self.cl)

    def cl_mk(self, kode: str) -> float | None:
        return self.cl.get(kode, self.cl_default)

    def dihitung_beban(self, kode: str) -> bool:
        """Kegiatan pada masa transisi (16: PLP/PKL/PKM/Capstone) tidak masuk B_s bila dikonfigurasi."""
        return not (self.cfg.beban.kecualikan_kegiatan_transisi and kode in self.kurikulum.mk_kegiatan_transisi)

    def beban(self, kode_list: Iterable[str]) -> float:
        kur = self.kurikulum
        pasangan = []
        for k in dict.fromkeys(kode_list):
            if k in kur and self.dihitung_beban(k):
                c = self.cl_mk(k)
                if c is not None:
                    pasangan.append((c, kur.sks(k)))
        return hitung_beban(pasangan, self.cfg.beban.metode)


def theta_data(transaksi: list[Transaksi], ctx: KonteksPlanner) -> dict:
    """θ = persentil B_s semester-semester KRS data latih (MK yang diambil), dengan pemilihan sumber per
    semester yang sama dengan ARM (``arm.sumber_data``)."""
    pilih, sumber = pilih_dari_cfg(transaksi, ctx.cfg)
    b = [ctx.beban(t.kode) for t in pilih]
    return _ringkas_theta(b, ctx.cfg, f"otomatis P{ctx.cfg.beban.theta_persentil:g} B_s semester KRS data latih",
                          dikalibrasi_data=any("asli" in x for x in sumber.values()),
                          sumber_per_semester={str(s): x for s, x in sumber.items()})


def theta_struktur(ctx: KonteksPlanner) -> dict:
    """θ = persentil B_s semester struktur resmi Jalur A–F (08); komposisi semester yang identik
    dihitung sekali. Slot tema/agama/sisa tema memakai median CL MK opsinya; slot konversi diabaikan."""
    kur = ctx.kurikulum

    def median_cl(kode: Iterable[str]) -> float | None:
        v = [ctx.cl_mk(k) for k in kode if ctx.cl_mk(k) is not None]
        return float(np.median(v)) if v else None

    semua_tema = {k for t in kur.tema for k in kur.mk_tema(t)}
    sudah: set[tuple] = set()
    b = []
    for j in JALUR:
        for s in range(1, 9):
            pasangan, tanda = [], []
            for slot in kur.slot(j, s):
                if slot.jenis_slot == "MK_WAJIB":
                    if not ctx.dihitung_beban(slot.kode_mk):
                        continue
                    c, t = ctx.cl_mk(slot.kode_mk), slot.kode_mk
                elif slot.jenis_slot == "MK_WAJIB_PILIH_SATU":
                    c, t = median_cl(kur.grup("AGAMA")), "AGAMA"
                elif slot.jenis_slot == "PILIHAN_TEMA":
                    c, t = median_cl(kur.mk_tema(slot.tema)), f"TEMA_{slot.tema}"
                elif slot.jenis_slot == "PILIHAN_SISA_TEMA":
                    c, t = median_cl(semua_tema), "SISA_TEMA"
                else:
                    continue
                if c is not None:
                    pasangan.append((c, slot.sks))
                    tanda.append(t)
            kunci = tuple(sorted(tanda))
            if kunci in sudah:
                continue
            sudah.add(kunci)
            b.append(hitung_beban(pasangan, ctx.cfg.beban.metode))
    return _ringkas_theta(b, ctx.cfg, f"otomatis P{ctx.cfg.beban.theta_persentil:g} B_s semester struktur "
                                      f"resmi Jalur A–F (08)", dikalibrasi_data=False)


def tentukan_theta(ctx: KonteksPlanner) -> dict:
    """θ: manual (config) → data KRS (bila transaksi latih ada) → struktur kurikulum resmi."""
    bc = ctx.cfg.beban
    if bc.theta is not None:
        return {"theta": float(bc.theta), "sumber": "config beban.theta (ditimpa manual)", "metode": bc.metode,
                "dikalibrasi_data": False}
    if not ctx.cl and ctx.cl_default is None:
        return {"theta": None, "sumber": "tidak tersedia, skor CL belum ada", "dikalibrasi_data": False}
    sumber = bc.theta_sumber
    catatan = None
    if sumber in ("otomatis", "data"):
        latih, catatan = transaksi_segar(ctx.cfg, "latih")
        if latih:
            info = theta_data(latih, ctx)
            if info.get("theta") is not None:
                return info
            catatan = "transaksi latih tidak memuat MK Kurikulum 2024 yang dapat dihitung bebannya"
    info = theta_struktur(ctx)
    # tanpa data KRS sama sekali tidak perlu catatan; transaksi basi/kosong dicatat
    if catatan and (sumber == "data" or ada_transaksi(ctx.cfg)):
        info["catatan"] = f"{catatan}, memakai struktur kurikulum"
    return info


# ---------------------------------------------------------------------
# Penyusun rencana
# ---------------------------------------------------------------------


@dataclass
class _Semester:
    """Konteks rekomendasi satu semester reguler (09, 08, AT08)."""

    s: int
    sks_min: int
    sks_max: int
    sumber_batas: str
    pita: PitaSKS | None
    tambah_kode: set[str] = field(default_factory=set)
    tambah_tema: set[str] = field(default_factory=set)
    kurangi_kode: set[str] = field(default_factory=set)
    kurangi_tema: set[str] = field(default_factory=set)
    kurangi_sisa: bool = False
    slot_tema: set[str] = field(default_factory=set)
    slot_sisa: bool = False
    catatan: list[str] = field(default_factory=list)
    target_lulus: int = 8          # perkiraan semester lulus tercepat (K5)
    preferensi: dict[str, str] = field(default_factory=dict)   # tema → MK pilihan pilihan mahasiswa


@dataclass
class _Calon:
    kode: str
    unit: tuple[str, ...]
    tingkat: int
    alasan: list[str]
    kurangi: bool
    support: float
    conf_lift: float
    aturan: str | None
    seri: tuple
    kritis: bool = False

    @property
    def kunci(self) -> tuple:
        """Urutan: MK wajib tertunda (AT09) → kesempatan terakhir sebelum perkiraan lulus (K5) → tingkat
        kebutuhan → saran 'kurangi' 09 → skor ARM (support + max conf×lift) → seri deterministik."""
        return (self.tingkat != 0, not self.kritis, self.tingkat, self.kurangi,
                -(self.support + self.conf_lift)) + self.seri


def _tema_dari_kode(kode: str) -> str | None:
    return kode[len("TEMA_"):] if kode.startswith("TEMA_") else None


def _konteks_semester(s: int, kd: Keadaan, ip: float, tema_selesai: set[str], ctx: KonteksPlanner) -> _Semester:
    kur = ctx.kurikulum
    sks_min, sks_max, sumber = batas_sks(s, kd.jalur, ip, kur)
    pita = kur.pita(kd.jalur, s, ip)
    ks = _Semester(s, sks_min, sks_max, sumber, pita)
    if pita is not None:
        for kode in pita.kode_terdampak:
            tema = _tema_dari_kode(kode)
            if pita.aksi in ("tambah", "tambah_jika_ada"):
                if tema:
                    ks.tambah_tema.add(tema)
                elif kode in kur:
                    ks.tambah_kode.add(kode)
            elif pita.aksi == "kurangi":
                if tema:
                    ks.kurangi_tema.add(tema)
                elif kode == "PILIHAN_SISA_TEMA":
                    ks.kurangi_sisa = True
                elif kode in kur:
                    ks.kurangi_kode.add(kode)
        ks.catatan.append(f"Rekomendasi 09 (IP {pita.ip_label}, {pita.sks_min}–{pita.sks_max} SKS): "
                          f"{pita.teks_asli}")
        if pita.catatan:
            ks.catatan.append(f"Catatan dataset: {pita.catatan}")
    for slot in kur.slot(kd.jalur, s):
        if slot.jenis_slot == "PILIHAN_TEMA" and slot.tema:
            ks.slot_tema.add(slot.tema)
        elif slot.jenis_slot == "PILIHAN_SISA_TEMA":
            ks.slot_sisa = True
    # AT08: di semester 5, slot Tema A diganti Tema C bila Tema A sudah terpenuhi
    at08 = kur.aturan.get("AT08")
    if at08 is not None and str(s) == at08.parameter.get("sem"):
        tema_ganti = re.findall(r"Tema ([A-Z])", at08.aturan)
        if len(tema_ganti) >= 2:
            asal, ganti = tema_ganti[0], tema_ganti[1]
            if asal in ks.slot_tema and asal in tema_selesai:
                ks.slot_tema.discard(asal)
                ks.slot_tema.add(ganti)
                ks.catatan.append(f"AT08: Tema {asal} sudah terpenuhi → slot diganti Tema {ganti} "
                                  f"[{at08.sumber}]")
    return ks


def _perlu(kode: str, s: int, kd: Keadaan, tema_selesai: set[str], gagal: set[str], ks: _Semester,
           ctx: KonteksPlanner) -> tuple[int, list[str], bool] | None:
    """Apakah MK dibutuhkan di semester s? → (tingkat, alasan, kurangi) atau None."""
    kur = ctx.kurikulum
    m = kur.mk[kode]
    if m.sifat == "wajib":
        agama = kur.grup("AGAMA")
        if kode in agama and (kode != kd.agama_kode or agama & kd.lulus):
            return None
        home = kur.semester_jalur(kode, kd.jalur) or m.semester_jalur_a
        kurangi = kode in ks.kurangi_kode
        if kode in gagal or home < s:
            sebab = "pernah tidak lulus" if kode in gagal else f"semester paketnya S{home} sudah lewat"
            return 0, [f"{TINGKAT[0]}: {sebab}"], kurangi
        if home == s:
            return 1, [f"{TINGKAT[1]} (08 Jalur {kd.jalur} S{home})"], kurangi
        if kode in ks.tambah_kode:
            return 1, [f"Rekomendasi 09 menambah {kode} pada pita IP ini (paket asal S{home})"], kurangi
        return None
    tema = m.tema
    if not tema or tema in tema_selesai:
        return None
    pref = ks.preferensi.get(tema)
    if pref is not None and kode != pref:   # mahasiswa memilih MK lain untuk tema ini
        return None
    tambahan = [f"Dipilih mahasiswa untuk Tema {tema} (panel MK pilihan per tema)"] if pref else []
    if tema in ks.slot_tema:
        return 2, [f"{TINGKAT[2]}: Tema {tema} (08 Jalur {kd.jalur} S{s})"] + tambahan, tema in ks.kurangi_tema
    if tema in ks.tambah_tema:
        return 2, [f"Rekomendasi 09 menambah MK pilihan Tema {tema} pada pita IP ini"] + tambahan, \
            tema in ks.kurangi_tema
    if ks.slot_sisa:
        return 2, [f"Slot pilihan sisa tema (08 Jalur {kd.jalur} S{s}): Tema {tema}"] + tambahan, ks.kurangi_sisa
    return 3, [f"{TINGKAT[3]}: Tema {tema}"] + tambahan, tema in ks.kurangi_tema


def _urutkan(kd: Keadaan, ks: _Semester, cek: dict[str, list[HasilAturan]], terpilih: list[str],
             tema_konv: set[str], gagal: set[str], ctx: KonteksPlanner) -> list[_Calon]:
    kur, s = ctx.kurikulum, ks.s
    tema_selesai = tema_terpenuhi(set(kd.lulus) | set(terpilih), tema_konv, kur)
    item_terpilih = {f"{k}@S{s}" for k in terpilih}
    calon: dict[tuple[str, ...], _Calon] = {}
    for kode, hasil in cek.items():
        if kode in terpilih or not boleh_diambil(hasil):
            continue
        perlu = _perlu(kode, s, kd, tema_selesai, gagal, ks, ctx)
        if perlu is None:
            continue
        unit = tuple(sorted(unit_korequisit(kode, kd, kur)))
        if unit in calon or any(u in terpilih or u not in cek or not boleh_diambil(cek[u]) for u in unit):
            continue
        tingkat, alasan, kurangi = perlu
        kritis = _kesempatan_terakhir(kode, s, kd, ks.target_lulus, ctx, ks.preferensi)
        if kritis:
            alasan = alasan + [f"Kesempatan terakhir sebelum perkiraan lulus S{ks.target_lulus}: "
                               f"{'MK ini' if kur.mk[kode].sifat == 'wajib' else f'Tema {kur.mk[kode].tema}'} tidak "
                               f"ditawarkan lagi hingga semester itu, diutamakan agar studi tidak molor"]
        item = f"{kode}@S{s}"
        sup = ctx.arm.support.get(item, 0.0)
        cl_score, aturan = ctx.arm.skor_aturan(item, item_terpilih)
        m = kur.mk[kode]
        home = kur.semester_jalur(kode, kd.jalur) or m.semester_jalur_a or 99
        persis = 0 if (m.sifat == "wajib" or s in m.semester_tabel11) else 1
        calon[unit] = _Calon(kode, unit, tingkat, alasan, kurangi, sup, cl_score, aturan, (home, persis, kode),
                             kritis)
    return sorted(calon.values(), key=lambda c: c.kunci)


def _kesempatan_terakhir(kode: str, s: int, kd: Keadaan, target: int, ctx: KonteksPlanner,
                         preferensi: dict[str, str] | None = None) -> bool:
    """K5 (minimalkan keterlambatan): True bila semester s adalah kesempatan TERAKHIR sebelum perkiraan
    semester lulus ``target`` untuk mengambil MK wajib ini (atau MK mana pun dari tema pilihannya — hanya
    MK yang dipilih mahasiswa bila tema itu punya preferensi), menurut penawaran AD-04. Tidak berlaku
    di/sesudah ``target`` (semua MK sama mendesaknya)."""
    if s >= target:
        return False
    kur = ctx.kurikulum
    m = kur.mk[kode]
    if m.sifat == "wajib" or not m.tema:
        calon = [kode]
    elif (preferensi or {}).get(m.tema):
        calon = [preferensi[m.tema]]
    else:
        calon = sorted(kur.mk_tema(m.tema))
    return not any(ctx.ditawarkan(k, t, kd) for t in range(s + 1, target + 1) for k in calon)


def perkiraan_lulus(s: int, kd: Keadaan, tema_konv: set[str], sks_max: int, ctx: KonteksPlanner) -> int:
    """Batas bawah semester lulus dari semester s: sisa SKS yang masih dibutuhkan (MK wajib belum lulus +
    tema pilihan belum terpenuhi + kekurangan terhadap total minimal) dibagi kapasitas pita SKS (asumsi
    IP tetap). Minimal ``horizon_semester`` (8) agar rencana normal tetap mengikuti struktur kurikulum."""
    kur = ctx.kurikulum
    agama = kur.grup("AGAMA")
    wajib = kur.total_sks(kur.wajib - agama - kd.lulus)
    if not agama & kd.lulus:
        wajib += min((kur.sks(k) for k in agama), default=0)
    tema = sum(tp.sks_minimal for t, tp in kur.tema.items() if t not in tema_terpenuhi(kd.lulus, tema_konv, kur))
    sisa = max(wajib + tema, kur.param_int("AT01", "total_min") - kd.sks_lulus(kur))
    return max(ctx.cfg.kurikulum.horizon_semester, s + math.ceil(sisa / max(sks_max, 1)) - 1)


def _item(kode: str, s: int, calon: _Calon | None, alasan_awal: list[str], hasil: list[HasilAturan],
          ctx: KonteksPlanner) -> ItemRencana:
    kur = ctx.kurikulum
    m = kur.mk[kode]
    alasan = list(alasan_awal)
    if calon is not None:
        if calon.kurangi:
            alasan.append("09 menyarankan mengurangi MK ini pada pita IP ini; tetap diambil karena kapasitas SKS cukup")
        if ctx.arm.kosong:
            alasan.append("ARM belum aktif (data KRS belum tersedia), urutan dari struktur kurikulum")
        else:
            alasan.append(f"Support {kode}@S{s}: {calon.support:.2f}")
            if calon.aturan:
                alasan.append(f"Aturan asosiasi pendukung: {calon.aturan}")
    cl = ctx.cl_mk(kode)
    if cl is None:
        alasan.append("CL belum tersedia")
    else:
        ket = ctx.keterangan_cl.get(kode) or ("default: median CL MK lain" if kode not in ctx.cl else "")
        alasan.append(f"Beban kognitif CL {cl:.2f}" + (f" ({ket})" if ket else ""))
        if not ctx.dihitung_beban(kode):
            alasan.append("Kegiatan masa transisi (16): tidak dihitung dalam beban semester")
    jadwal = [f"{j.kegiatan}: dilaksanakan pada transisi semester {j.transisi_semester} ({j.bulan_pelaksanaan}), "
              f"diklaim di KRS semester {j.semester_krs} [16_jadwal_kegiatan_lapangan.csv]" for j in kur.jadwal(kode)]
    return ItemRencana(
        kode=kode, nama=m.nama, sks=m.sks, sifat=m.sifat, semester_anjuran=m.semester_acuan, tema=m.tema,
        prioritas=TINGKAT[calon.tingkat] if calon else None, cl=cl, justifikasi_cl=ctx.keterangan_cl.get(kode),
        justifikasi_cpmk=list(ctx.justifikasi_cpmk.get(kode, [])),
        skor=round(calon.support + calon.conf_lift, 4) if calon else 0.0,
        support=round(calon.support, 4) if calon else 0.0, conf_lift=round(calon.conf_lift, 4) if calon else 0.0,
        aturan_pendukung=calon.aturan if calon else None, alasan=alasan,
        sumber=[h.teks() for h in hasil if h.status == "lolos"],
        perlu_verifikasi=[h.teks() for h in hasil if h.status == "perlu_verifikasi_manual"],
        jadwal=jadwal,
    )


def _selesai(kd: Keadaan, tema_konv: set[str], kur: Kurikulum) -> bool:
    agama = kur.grup("AGAMA")
    wajib_ok = kur.wajib - agama <= kd.lulus
    agama_ok = bool(agama & kd.lulus) or kd.agama_kode is None
    return wajib_ok and agama_ok and tema_terpenuhi(kd.lulus, tema_konv, kur) == set(kur.tema)


def _tema_untuk_konversi(s: int, kd: Keadaan, tema_konv: set[str], sks_konv: int, profile: Profile,
                         ctx: KonteksPlanner) -> tuple[list[str], str]:
    """Tema yang dipenuhi konversi: input mahasiswa, atau asumsi — tema belum terpenuhi sebanyak
    kapasitas SKS konversi; sisanya (yang MK-nya ditawarkan pada slot sisa tema) untuk semester berikut."""
    kur = ctx.kurikulum
    belum = sorted(set(kur.tema) - tema_terpenuhi(kd.lulus, tema_konv, kur))
    if profile.tema_konversi is not None:
        return sorted(set(profile.tema_konversi) & set(belum)), "sesuai input mahasiswa (tema_konversi)"
    kapasitas = sks_konv // min(t.sks_minimal for t in kur.tema.values())
    if len(belum) <= kapasitas:
        return belum, f"asumsi planner: konversi {sks_konv} SKS mencakup seluruh {len(belum)} tema yang belum terpenuhi"
    s_sisa = next((x for x in range(s + 1, 15) if any(sl.jenis_slot == "PILIHAN_SISA_TEMA"
                                                        for sl in kur.slot(kd.jalur, x))), s + 1)

    def ditawarkan(t: str) -> bool:
        return any(aturan_penawaran(k, s_sisa, kd, kur, ctx.cfg).lolos for k in kur.mk_tema(t))

    def support(t: str) -> float:
        return max((ctx.arm.support.get(f"{k}@S{s_sisa}", 0.0) for k in kur.mk_tema(t)), default=0.0)

    urut = sorted(belum, key=lambda t: (not ditawarkan(t), -support(t), t))
    sisa = set(urut[:len(belum) - kapasitas])
    tercakup = [t for t in belum if t not in sisa]
    return tercakup, (f"asumsi planner: konversi {sks_konv} SKS mencakup {len(tercakup)} tema; tema "
                      f"{', '.join(sorted(sisa))} disisakan untuk slot pilihan sisa tema S{s_sisa}")


def susun_rencana(profile: Profile, mode: Mode, ctx: KonteksPlanner, label: str = "") -> Plan:
    """UC-02: susun rencana dari ``profile.semester_saat_ini`` sampai horizon."""
    if mode not in ("arm", "hybrid"):
        raise ValueError(f"mode tidak dikenal: {mode}")
    cfg, kur = ctx.cfg, ctx.kurikulum
    theta = ctx.theta if mode == "hybrid" else None
    if mode == "hybrid" and theta is None:
        raise ValueError("θ belum tersedia, skor beban kognitif (CL) belum ada, atau isi beban.theta di config")

    tak_dikenal = sorted((profile.mk_lulus | profile.mk_pernah_gagal) - set(kur.kode))
    tak_dikenal = sorted(set(tak_dikenal) | {k for k in profile.riwayat_nilai if k not in kur})
    profile = profile.model_copy(update={"mk_lulus": {k for k in profile.mk_lulus if k in kur},
                                         "mk_pernah_gagal": {k for k in profile.mk_pernah_gagal if k in kur},
                                         "riwayat_nilai": {k: n for k, n in profile.riwayat_nilai.items()
                                                           if k in kur}})
    kd = Keadaan.dari_profil(profile, kur, cfg)
    ip = profile.ip_terakhir
    gagal = set(profile.mk_pernah_gagal)
    paket = semester_paket(kur)
    s_max = kur.param_int("AT04", "semester_max")
    horizon = cfg.kurikulum.horizon_semester
    s_konv_slot = next((sl.semester for sl in kur.struktur if sl.jalur == profile.jalur
                        and sl.jenis_slot == "PILIHAN_KONVERSI"), None)
    konversi_dipakai = s_konv_slot is not None and profile.semester_saat_ini > s_konv_slot
    tema_konv: set[str] = set(profile.tema_konversi or ()) if konversi_dipakai else set()
    peringatan: list[str] = []
    asumsi_tambahan: list[str] = []
    if konversi_dipakai and profile.tema_konversi is None:
        peringatan.append(f"Jalur {profile.jalur}: tema yang dipenuhi konversi semester {s_konv_slot} belum diisi; "
                          f"semua tema yang belum lulus direncanakan secara reguler")
    if kd.agama_kode is None and not kur.grup("AGAMA") & kd.lulus:
        peringatan.append("Agama mahasiswa belum diisi: MK agama tidak direncanakan (sistem tidak menyimpulkan agama)")
    pref, catatan_pref = preferensi_sah(profile.pilihan_tema, kur)
    peringatan += catatan_pref
    tema_awal = tema_terpenuhi(kd.lulus, tema_konv, kur)
    pref = {t: k for t, k in pref.items() if t not in tema_awal}   # tema yang sudah lulus tidak direncanakan

    semester: list[SemesterRencana] = []
    s = profile.semester_saat_ini
    kosong_beruntun = 0
    while s <= s_max:
        if s > horizon and (not cfg.kurikulum.perpanjang_melewati_horizon or _selesai(kd, tema_konv, kur)):
            break
        slot_konv = [sl for sl in kur.slot(kd.jalur, s) if sl.jenis_slot == "PILIHAN_KONVERSI"]
        if s in paket:
            sem = _semester_paket(s, kd, theta, ctx, peringatan)
        elif slot_konv and not konversi_dipakai and syarat_konversi(kd, kur)[0]:
            sem, tercakup = _semester_konversi(s, kd, slot_konv[0], tema_konv, profile, ctx, asumsi_tambahan)
            tema_konv |= set(tercakup)
            konversi_dipakai = True
        else:
            if slot_konv and not konversi_dipakai:
                _, kurang = syarat_konversi(kd, kur)
                peringatan.append(f"S{s}: konversi merdeka belajar Jalur {kd.jalur} belum dapat diambil, syarat AT12 "
                                  f"belum terpenuhi ({', '.join(kurang)}); semester direncanakan reguler")
                konversi_dipakai = True
            sem = _semester_reguler(s, kd, ip, gagal, tema_konv, theta, ctx, pref)
        semester.append(sem)
        kode_sem = [it.kode for it in sem.items if it.jenis == "mk"]
        kd = kd.tambah(kode_sem, sum(it.sks for it in sem.items if it.jenis == "konversi"))
        gagal -= set(kode_sem)
        kosong_beruntun = kosong_beruntun + 1 if (s > horizon and not sem.items) else 0
        if kosong_beruntun >= 2:
            break
        s += 1

    akhir = semester[-1].semester if semester else profile.semester_saat_ini
    if not _selesai(kd, tema_konv, kur):
        wajib_kurang = sorted(kur.wajib - kur.grup("AGAMA") - kd.lulus)
        tema_kurang = sorted(set(kur.tema) - tema_terpenuhi(kd.lulus, tema_konv, kur))
        bagian = []
        if wajib_kurang:
            bagian.append(f"MK wajib belum dapat direncanakan: {', '.join(wajib_kurang)}")
        if tema_kurang:
            bagian.append(f"tema pilihan belum terpenuhi: {', '.join(tema_kurang)}")
        if bagian:
            peringatan.append(f"Syarat lulus belum tercapai hingga S{akhir}: " + "; ".join(bagian))
        for t in tema_kurang:
            if t in pref:
                peringatan.append(f"Tema {t}: MK pilihan yang dipilih ({kur.label(pref[t])}) tidak dapat "
                                  f"dijadwalkan (semester penawaran/prasyarat/kapasitas SKS). Pilih MK lain atau "
                                  f"kembalikan ke rekomendasi sistem di panel MK pilihan per tema.")
    if (w := peringatan_horizon(semester, horizon, s_max, kur)) is not None:
        peringatan.append(w)
    if tak_dikenal:
        peringatan.append(f"Kode MK riwayat tidak dikenal Kurikulum 2024 diabaikan (normalisasi dulu lewat "
                          f"penyetaraan): {', '.join(tak_dikenal)}")
    tanpa_cl = sorted({it.kode for sm in semester for it in sm.items if it.jenis == "mk" and it.kode not in ctx.cl})
    if theta is not None and tanpa_cl and ctx.cl:
        peringatan.append(f"MK tanpa skor CL memakai default {ctx.cl_default:.2f}: {', '.join(tanpa_cl)}")

    plan = Plan(
        id=f"{datetime.now():%Y%m%d-%H%M%S}-{mode}-{uuid.uuid4().hex[:4]}",
        mode=mode, dibuat=datetime.now(), profil=profile, theta=ctx.theta, metode_beban=cfg.beban.metode,
        semester=semester, peringatan=peringatan, asumsi=_asumsi(profile, theta, ctx) + asumsi_tambahan,
        sumber_data=ctx.status_krs, label=label, sks_lulus_awal=Keadaan.dari_profil(profile, kur).sks_lulus(kur),
        sks_minimal_lulus=kur.param_int("AT01", "total_min"),
    )
    plan.validasi = validate_plan(plan, profile, kur, cfg, beban=ctx.beban)
    return plan


PREFIKS_HORIZON = "Rencana melewati semester"


def peringatan_horizon(semester: list[SemesterRencana], horizon: int, s_max: int, kur: Kurikulum) -> str | None:
    """Peringatan bila rencana melewati horizon batas masa studi — dipakai planner & penyesuaian manual."""
    akhir = max((sm.semester for sm in semester if sm.items), default=0)
    if akhir <= horizon:
        return None
    sesudah = [f"{it.kode} (S{sm.semester})" for sm in semester if sm.semester > horizon for it in sm.items]
    return (f"{PREFIKS_HORIZON} {horizon} (hingga S{akhir}); masa studi maksimal {s_max} semester "
            f"[{kur.aturan['AT04'].sumber}]. MK yang belum terjadwal hingga S{horizon}: "
            f"{', '.join(sesudah)}")


def preferensi_sah(pilihan_tema: dict[str, str], kur: Kurikulum) -> tuple[dict[str, str], list[str]]:
    """Saring preferensi MK pilihan per tema: kode harus MK pilihan dari tema itu dan boleh direkomendasikan
    (AT10). Mengembalikan (preferensi sah, catatan untuk entri yang diabaikan)."""
    sah, catatan = {}, []
    dilarang = set(_daftar_param(kur, "AT10", "kode")) if "AT10" in kur.aturan else set()
    for tema, kode in sorted(pilihan_tema.items()):
        if not kode:
            continue
        if tema not in kur.tema or kode not in kur.mk_tema(tema) or kode in dilarang:
            catatan.append(f"Preferensi Tema {tema} → {kode} diabaikan: bukan MK pilihan tema tersebut")
            continue
        sah[tema] = kode
    return sah, catatan


def _semester_paket(s: int, kd: Keadaan, theta: float | None, ctx: KonteksPlanner,
                    peringatan: list[str]) -> SemesterRencana:
    kur = ctx.kurikulum
    sks_paket = semester_paket(kur)[s]
    sumber = f"{kur.aturan['AT06'].sumber}; 08_struktur_kurikulum_jalur.csv (Jalur {kd.jalur}, S{s})"
    pilih, items = [], []
    for k in mk_paket_semester(s, kd, kur):
        hasil = cek_mk(k, s, kd, kur, ctx.cfg)
        if boleh_diambil(hasil):
            pilih.append(k)
            items.append(_item(k, s, None, [f"MK paket semester {s} (AT06; 08 Jalur {kd.jalur})"], hasil, ctx))
    beban = ctx.beban(pilih)
    if theta is not None and beban > theta:
        peringatan.append(f"S{s} (paket): beban {beban:.1f} > θ {theta:.1f}; paket AT06 tetap diikuti")
    return SemesterRencana(semester=s, paket=True, jenis="paket", batas_sks=sks_paket, sks_min=0,
                           sumber_batas=sumber, items=items, total_sks=kur.total_sks(pilih),
                           total_beban=round(beban, 4))


def _semester_konversi(s: int, kd: Keadaan, slot, tema_konv: set[str], profile: Profile, ctx: KonteksPlanner,
                       asumsi: list[str]) -> tuple[SemesterRencana, list[str]]:
    kur = ctx.kurikulum
    sks = min(slot.sks, kur.param_int("AT13", "konversi_max_per_semester"))
    tercakup, dasar = _tema_untuk_konversi(s, kd, tema_konv, sks, profile, ctx)
    asumsi.append(f"S{s} konversi merdeka belajar Jalur {kd.jalur}: {dasar}")
    item = ItemRencana(
        kode=KODE_KONVERSI, nama=slot.nama_slot, sks=sks, sifat="pilihan", jenis="konversi",
        tema_konversi=tercakup, prioritas="Slot konversi merdeka belajar (08)",
        alasan=[f"Jalur {kd.jalur} ({slot.nama_jalur}): slot konversi semester {s}",
                f"Tema yang dipenuhi: {', '.join(tercakup) or '-'} ({dasar})",
                "Tidak dihitung dalam beban kognitif (isi kegiatan belum diketahui)"],
        sumber=[slot.sumber, kur.aturan["AT12"].sumber, kur.aturan["AT13"].sumber, kur.aturan["AT14"].sumber],
        perlu_verifikasi=["AD-14: kesetaraan SKS dan tema konversi merdeka belajar perlu disetujui DPA"],
    )
    return SemesterRencana(semester=s, jenis="konversi", batas_sks=sks, sks_min=0,
                           sumber_batas=kur.aturan["AT13"].sumber, items=[item], total_sks=sks,
                           total_beban=0.0), tercakup


def _semester_reguler(s: int, kd: Keadaan, ip: float, gagal: set[str], tema_konv: set[str], theta: float | None,
                      ctx: KonteksPlanner, preferensi: dict[str, str] | None = None) -> SemesterRencana:
    kur, cfg = ctx.kurikulum, ctx.cfg
    tema_awal = tema_terpenuhi(kd.lulus, tema_konv, kur)
    ks = _konteks_semester(s, kd, ip, tema_awal, ctx)
    ks.target_lulus = perkiraan_lulus(s, kd, tema_konv, ks.sks_max, ctx)
    ks.preferensi = dict(preferensi or {})
    # aturan per MK hanya bergantung pada keadaan awal semester → cukup dihitung sekali
    cek = {k: cek_mk(k, s, kd, kur, cfg) for k in kur.kode if k not in kd.lulus}
    terpilih: list[str] = []
    dipilih: list[_Calon] = []
    dilewati: list[str] = []
    sks = 0
    while True:
        pilihan = None
        for c in _urutkan(kd, ks, cek, terpilih, tema_konv, gagal, ctx):
            tambah = kur.total_sks(c.unit)
            if sks + tambah > ks.sks_max:
                continue
            if theta is not None and ctx.beban(terpilih + list(c.unit)) > theta + 1e-9:  # satu-satunya beda mode
                if c.kode not in dilewati:
                    dilewati.append(c.kode)
                continue
            pilihan = c
            break
        if pilihan is None:
            break
        terpilih += list(pilihan.unit)
        dipilih.append(pilihan)
        sks += kur.total_sks(pilihan.unit)

    items = []
    for c in dipilih:
        for k in c.unit:
            alasan = list(c.alasan) if k == c.kode else [f"Korequisit {c.kode}: diambil bersamaan (AT15)"]
            items.append(_item(k, s, c, alasan, cek[k], ctx))
    catatan = list(ks.catatan)
    if sks < ks.sks_min:
        catatan.append(f"Total {sks} SKS di bawah rentang pita {ks.sks_min}–{ks.sks_max}: tidak ada MK lain yang "
                       f"dibutuhkan dan ditawarkan pada semester ini")
    return SemesterRencana(semester=s, jenis="reguler", batas_sks=ks.sks_max, sks_min=ks.sks_min,
                           sumber_batas=ks.sumber_batas, items=items, total_sks=sks,
                           total_beban=round(ctx.beban(terpilih), 4),
                           dilewati_theta=[k for k in dilewati if k not in terpilih], catatan=catatan)


def _asumsi_ipk(profile: Profile, ctx: KonteksPlanner) -> str:
    ipk, ket = hitung_ipk(profile.riwayat_nilai, ctx.kurikulum, ctx.cfg)
    if ipk is None and profile.ipk is not None:
        ipk, ket = profile.ipk, "IPK yang diisikan / data akademik"
    sk = ctx.cfg.aturan.SKRIPSI
    if ipk is None:
        return f"IPK tidak dapat dihitung ({ket}); syarat IPK ≥ {sk.min_ipk:.2f} Skripsi (AD-10) perlu diverifikasi."
    return f"IPK {ipk:.2f} ({ket}) diasumsikan tetap untuk syarat Skripsi AD-10 (IPK ≥ {sk.min_ipk:.2f})."


def _asumsi(profile: Profile, theta: float | None, ctx: KonteksPlanner) -> list[str]:
    cfg = ctx.cfg
    ac = cfg.aturan
    asumsi = [
        f"IP semester mendatang diasumsikan tetap = IP terakhir {profile.ip_terakhir:.2f} (pita SKS 09/AT05).",
        f"Semua MK yang direncanakan diasumsikan lulus (nilai ≥ {ac.NILAI_LULUS_MINIMUM}; AD-07).",
        f"Prasyarat dianggap terpenuhi bila MK prasyarat "
        f"{'pernah ditempuh (tafsiran literal ≥ E)' if ac.PRASYARAT_E_BERARTI_PERNAH_MENEMPUH else 'telah lulus'} "
        f"di semester sebelumnya (AD-03).",
        _asumsi_ipk(profile, ctx),
        "Urutan pengambilan memprioritaskan MK wajib tertunda (AT09), lalu MK/tema yang kesempatan penawarannya "
        "habis sebelum perkiraan semester lulus, agar masa studi tidak molor (K5).",
        f"Penawaran MK pilihan: mode {ac.MODE_PENAWARAN_PILIHAN} (Tabel 11; INK07/INK08).",
    ]
    if ac.PENAWARAN_DARI_REKOMENDASI_09:
        asumsi.append("MK yang disarankan 'tambah' oleh 09 di semester berparitas lain (Skripsi/PKM di S7, Capstone "
                      "di S6) dianggap ditawarkan pada paritas tersebut (PENAWARAN_DARI_REKOMENDASI_09).")
    if ctx.arm.kosong:
        sebab = ctx.arm.catatan or "data KRS belum tersedia"
        asumsi.append(f"ARM belum aktif ({sebab}); urutan memakai struktur kurikulum resmi (08, 09) dan aturan "
                      f"domain.")
    else:
        asumsi.append("Skor ARM dari data KRS data latih.")
    if cfg.beban.kecualikan_kegiatan_transisi:
        asumsi.append("PLP/PKL/PKM/Capstone dilaksanakan pada masa transisi (16): SKS dihitung di semester KRS, "
                      "tetapi tidak masuk beban kognitif semester.")
    if theta is not None:
        cat = "" if ctx.info_theta.get("dikalibrasi_data") else ", BELUM dikalibrasi dari data KRS"
        asumsi.append(f"θ = {theta:.2f} ({ctx.info_theta.get('sumber', '')}{cat}); beban = {cfg.beban.metode}.")
    if ctx.info_cl.get("sumber"):
        teks = f"CL per MK: {ctx.info_cl['sumber']} (asumsi agregasi; lihat beban.agregasi_cl)"
        if ctx.info_cl.get("catatan"):
            teks += f", {ctx.info_cl['catatan']}"
        asumsi.append(teks + ".")
    return asumsi


# ---------------------------------------------------------------------
# Penyimpanan rencana (UC-03..UC-05)
# ---------------------------------------------------------------------


def simpan_rencana(plan: Plan, cfg: Config) -> str:
    """Simpan (atau perbarui) rencana ke tabel ``rencana`` (KD-11). Mengembalikan id rencana."""
    with db.koneksi(cfg) as c:
        db.tulis(c, "rencana", [{
            "id": plan.id, "dibuat": plan.dibuat.isoformat(timespec="seconds"), "mode": plan.mode,
            "label": plan.label, "isi": plan.model_dump_json(), "status_dpa": plan.status_dpa,
            "catatan_dpa": plan.catatan_dpa,
            "ditinjau_pada": plan.catatan_dpa_diperbarui.isoformat(timespec="seconds")
            if plan.catatan_dpa_diperbarui else None,
        }], ganti=True)
    return plan.id


def _dari_baris(r) -> Plan:
    plan = Plan.model_validate_json(r["isi"])
    plan.status_dpa = r["status_dpa"]
    plan.catatan_dpa = r["catatan_dpa"]
    waktu = r["ditinjau_pada"]
    plan.catatan_dpa_diperbarui = datetime.fromisoformat(waktu) if isinstance(waktu, str) and waktu else None
    return plan


def muat_rencana(cfg: Config, plan_id: str) -> Plan:
    df = db.baca(cfg, "SELECT * FROM rencana WHERE id = ?", (plan_id,))
    if df.empty:
        raise KeyError(f"rencana {plan_id!r} tidak ditemukan")
    return _dari_baris(df.iloc[0])


def daftar_rencana(cfg: Config) -> list[Plan]:
    """Rencana tersimpan, terbaru lebih dulu; baris berformat lama/rusak dilewati."""
    hasil = []
    for _, r in db.baca(cfg, "SELECT * FROM rencana ORDER BY dibuat DESC, id DESC").iterrows():
        try:
            hasil.append(_dari_baris(r))
        except ValueError:   # isi JSON berformat lama (ValidationError turunan ValueError)
            continue
    return hasil


def perbarui_catatan_dpa(cfg: Config, plan_id: str, catatan: str, status: str | None = None) -> Plan:
    """UC-05 / KF-19: simpan catatan bimbingan dan validasi akhir DPA (disetujui | perlu_revisi)."""
    if status is not None and status not in STATUS_DPA:
        raise ValueError(f"status DPA tidak dikenal: {status}")
    plan = muat_rencana(cfg, plan_id)
    plan.catatan_dpa = catatan
    if status is not None:
        plan.status_dpa = status
    plan.catatan_dpa_diperbarui = datetime.now()
    simpan_rencana(plan, cfg)
    return plan


if __name__ == "__main__":
    import sys as _sys

    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ctx = KonteksPlanner.muat()
    prof = Profile(semester_saat_ini=1, ip_terakhir=3.2, agama="Islam", jalur="A")
    for m in ("arm", "hybrid") if ctx.theta is not None else ("arm",):
        p = susun_rencana(prof, m, ctx)
        print(f"== {m} (θ={ctx.theta}) lolos={p.validasi.lolos}")
        for sem in p.semester:
            print(f"  S{sem.semester} [{sem.jenis}] {sem.total_sks:>2}/{sem.batas_sks} SKS beban {sem.total_beban:6.1f} | "
                  + ", ".join(i.kode for i in sem.items))
        for w in p.peringatan + p.validasi.pelanggaran:
            print("  !", w)
