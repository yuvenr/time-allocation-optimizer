"""Simulasi skenario mahasiswa berisiko studi molor (> 8 semester) — demo UC-02 / K5.

Setiap skenario adalah profil FIKTIF Jalur A (bukan data mahasiswa) yang riwayatnya dibentuk dari struktur
kurikulum resmi (08) semester 1..s-1, lalu diberi "permasalahan": MK tidak lulus (nilai E), MK/tema yang
belum diambil, dan IP rendah (pita SKS kecil, AT05).

Dua pola dibandingkan pada profil yang sama dan dengan constraint engine yang sama (``cek_mk``: prasyarat,
penawaran, syarat Skripsi/PLP; pita SKS ``batas_sks``):

- **Pola konvensional** (``rencana_konvensional``): tiap semester mahasiswa mengambil paket struktur
  Jalur A semester itu terlebih dahulu, lalu MK/tema tertunda menurut urutan tertundanya bila SKS masih
  cukup; yang tidak muat atau belum boleh diambil ditunda lagi.
- **Rekomendasi sistem** (``susun_rencana``, mode ``arm`` dan ``hybrid``): MK wajib tertunda (AT09) dan
  MK/tema yang kesempatan penawarannya habis sebelum perkiraan lulus (K5) diutamakan.

Hasil utama: perkiraan semester lulus masing-masing pola.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.config import Config
from src.data_loader import Kurikulum
from src.planner import KonteksPlanner, _selesai, susun_rencana
from src.rules import Keadaan, batas_sks, boleh_diambil, cek_mk, semester_paket, unit_korequisit
from src.schemas import Plan, Profile

JALUR_SKENARIO = "A"
AGAMA_SKENARIO = "Islam"   # profil fiktif; pada profil nyata agama selalu input mahasiswa


@dataclass(frozen=True)
class Skenario:
    id: str
    nama: str
    deskripsi: str
    semester: int                      # semester berjalan (yang akan direncanakan)
    ip: float                          # IP semester sebelumnya
    nilai: str = "B"                   # nilai MK yang lulus
    gagal: tuple[str, ...] = ()        # MK yang ditempuh tetapi tidak lulus (nilai E)
    belum: tuple[str, ...] = ()        # MK / "TEMA:X" dari paket lalu yang belum diambil


SKENARIO: list[Skenario] = [
    Skenario(
        "prasyarat_plp", "Gagal MK prasyarat berantai",
        "Tidak lulus Strategi/Perencanaan Pembelajaran semester 4 (CIE62104, CIE62106). Keduanya prasyarat MK "
        "semester 5 yang hanya ditawarkan di semester ganjil, dan MK semester 5 itu prasyarat PLP semester 7.",
        semester=5, ip=2.30, gagal=("CIE62104", "CIE62106")),
    Skenario(
        "ip_rendah", "IP rendah (pita SKS kecil)",
        "IP semester lalu 1,60 sehingga hanya boleh mengambil 12–15 SKS, dan dua MK wajib tidak lulus.",
        semester=5, ip=1.60, nilai="C+", gagal=("CIE61102", "CIE62209")),
    Skenario(
        "tertinggal", "Tertinggal paket (beban ringan / cuti)",
        "Di semester 4 hanya mengambil sebagian paket: CIE62107, CIE62303, dan COM60052 belum diambil.",
        semester=5, ip=2.75, belum=("CIE62107", "CIE62303", "COM60052")),
    Skenario(
        "menumpuk", "Tunggakan menumpuk di semester akhir",
        "Tidak lulus CIE62303 dan CIE61109, serta baru mengambil 3 dari 7 MK pilihan tema di semester 6.",
        semester=7, ip=2.40, gagal=("CIE62303", "CIE61109"),
        belum=("TEMA:F", "TEMA:G", "TEMA:H", "TEMA:I")),
]


def cari(id_skenario: str) -> Skenario:
    return next(s for s in SKENARIO if s.id == id_skenario)


def _mk_tema(tema: str, kur: Kurikulum, sudah: set[str]) -> str | None:
    """MK pertama (urut kode) dari tema yang belum dipakai — untuk riwayat paket yang lalu."""
    return next((k for k in sorted(kur.mk_tema(tema)) if k not in sudah), None)


def profil_skenario(sk: Skenario, kur: Kurikulum, cfg: Config) -> Profile:
    """Riwayat nilai = paket struktur Jalur A semester 1..s-1, dikurangi ``belum``, ``gagal`` bernilai E."""
    riwayat: dict[str, str] = {}
    for s in range(1, sk.semester):
        for slot in kur.slot(JALUR_SKENARIO, s):
            if slot.jenis_slot == "MK_WAJIB" and slot.kode_mk:
                kode = slot.kode_mk
            elif slot.jenis_slot == "MK_WAJIB_PILIH_SATU":
                kode = kur.mk_agama(AGAMA_SKENARIO)
            elif slot.jenis_slot == "PILIHAN_TEMA" and slot.tema and f"TEMA:{slot.tema}" not in sk.belum:
                kode = _mk_tema(slot.tema, kur, set(riwayat))
            else:
                continue
            if kode and kode in kur and kode not in sk.belum:
                riwayat[kode] = "E" if kode in sk.gagal else sk.nilai
    lulus = {k for k, n in riwayat.items() if cfg.aturan.lulus(n)}
    return Profile(semester_saat_ini=sk.semester, ip_terakhir=sk.ip, mk_lulus=lulus,
                   mk_pernah_gagal=set(riwayat) - lulus, riwayat_nilai=riwayat, jalur=JALUR_SKENARIO,
                   agama=AGAMA_SKENARIO)


# ---------------------------------------------------------------------
# Pola konvensional (pembanding)
# ---------------------------------------------------------------------


@dataclass
class HasilKonvensional:
    semester: dict[int, list[str]] = field(default_factory=dict)   # semester → kode MK diambil
    semester_lulus: int | None = None                              # None = belum lulus hingga AT04


def _tertunda_awal(profil: Profile, kur: Kurikulum) -> list[str]:
    """Antrean awal: MK wajib gagal/terlewat (urut semester paket) lalu tema paket lalu yang belum diambil."""
    s0 = profil.semester_saat_ini
    agama = kur.grup("AGAMA")
    antre = []
    for s in range(1, s0):
        for slot in kur.slot(JALUR_SKENARIO, s):
            if slot.jenis_slot == "MK_WAJIB" and slot.kode_mk and slot.kode_mk not in profil.mk_lulus:
                antre.append(slot.kode_mk)
            elif slot.jenis_slot == "MK_WAJIB_PILIH_SATU" and not agama & profil.mk_lulus:
                antre.append("AGAMA")
            elif slot.jenis_slot == "PILIHAN_TEMA" and slot.tema \
                    and not kur.mk_tema(slot.tema) & profil.mk_lulus:
                antre.append(f"TEMA:{slot.tema}")
    return list(dict.fromkeys(antre))


def rencana_konvensional(profil: Profile, ctx: KonteksPlanner) -> HasilKonvensional:
    """Paket struktur semester berjalan dulu, lalu antrean tertunda (FIFO); constraint sama dengan planner."""
    kur, cfg = ctx.kurikulum, ctx.cfg
    kd = Keadaan.dari_profil(profil, kur, cfg)
    s_max = kur.param_int("AT04", "semester_max")
    antre = _tertunda_awal(profil, kur)
    hasil = HasilKonvensional()
    paket = semester_paket(kur)
    for s in range(profil.semester_saat_ini, s_max + 1):
        if _selesai(kd, set(), kur):
            break
        sks_max = paket.get(s) or batas_sks(s, JALUR_SKENARIO, profil.ip_terakhir, kur)[1]
        slot_s = []
        for slot in kur.slot(JALUR_SKENARIO, s):
            if slot.jenis_slot == "MK_WAJIB" and slot.kode_mk:
                slot_s.append(slot.kode_mk)
            elif slot.jenis_slot == "MK_WAJIB_PILIH_SATU":
                slot_s.append("AGAMA")
            elif slot.jenis_slot in ("PILIHAN_TEMA", "PILIHAN_SISA_TEMA"):
                slot_s.append(f"TEMA:{slot.tema}" if slot.tema else "TEMA:*")
        ambil: list[str] = []
        tunda: list[str] = []
        for unit_id in slot_s + antre:
            kode = _pilih_kode(unit_id, s, kd, ambil, ctx)
            if kode is None:
                if not _sudah_beres(unit_id, kd, kur):
                    tunda.append(unit_id)
                continue
            unit = sorted(unit_korequisit(kode, kd, kur) - set(ambil))
            if kur.total_sks(ambil) + kur.total_sks(unit) > sks_max:
                tunda.append(unit_id)
                continue
            ambil += unit
        antre = list(dict.fromkeys(u for u in tunda if not _sudah_beres(u, kd.tambah(ambil), kur)))
        hasil.semester[s] = ambil
        kd = kd.tambah(ambil)
        if _selesai(kd, set(), kur):
            hasil.semester_lulus = s
            break
    return hasil


def _sudah_beres(unit_id: str, kd: Keadaan, kur: Kurikulum) -> bool:
    if unit_id == "AGAMA":
        return bool(kur.grup("AGAMA") & kd.lulus)
    if unit_id == "TEMA:*":
        return True       # slot sisa tema: kebutuhan tema yang nyata sudah ada di antrean
    if unit_id.startswith("TEMA:"):
        return bool(kur.mk_tema(unit_id[5:]) & kd.lulus)
    return unit_id in kd.lulus


def _pilih_kode(unit_id: str, s: int, kd: Keadaan, ambil: list[str], ctx: KonteksPlanner) -> str | None:
    """Kode MK yang boleh diambil di semester s untuk satu unit paket/antrean, atau None."""
    kur, cfg = ctx.kurikulum, ctx.cfg
    if _sudah_beres(unit_id, kd, kur) and unit_id != "TEMA:*":
        return None
    if unit_id == "AGAMA":
        calon = [kd.agama_kode] if kd.agama_kode else []
    elif unit_id == "TEMA:*":
        diambil_tema = {kur.mk[k].tema for k in ambil if k in kur}
        calon = sorted(k for t in kur.tema if not kur.mk_tema(t) & kd.lulus and t not in diambil_tema
                       for k in kur.mk_tema(t))
    elif unit_id.startswith("TEMA:"):
        calon = sorted(kur.mk_tema(unit_id[5:]))
    else:
        calon = [unit_id]
    for k in calon:
        if k and k not in ambil and k not in kd.lulus and boleh_diambil(cek_mk(k, s, kd, kur, cfg)):
            return k
    return None


# ---------------------------------------------------------------------
# Perbandingan
# ---------------------------------------------------------------------


def semester_lulus(plan: Plan) -> int | None:
    """Semester terakhir berisi MK bila rencana memenuhi syarat lulus; None bila tidak tercapai ≤ AT04."""
    if any(p.startswith("Syarat lulus belum tercapai") for p in plan.peringatan):
        return None
    return max((s.semester for s in plan.semester if s.items), default=None)


def jalankan_skenario(sk: Skenario, ctx: KonteksPlanner) -> dict:
    """Profil skenario → pola konvensional vs rekomendasi sistem (arm, dan hybrid bila θ tersedia)."""
    profil = profil_skenario(sk, ctx.kurikulum, ctx.cfg)
    konv = rencana_konvensional(profil, ctx)
    rencana = {m: susun_rencana(profil, m, ctx, label=f"skenario-{sk.id}")
               for m in (("arm", "hybrid") if ctx.theta is not None else ("arm",))}
    return {"profil": profil, "konvensional": konv, "rencana": rencana,
            "lulus": {"konvensional": konv.semester_lulus, **{m: semester_lulus(p) for m, p in rencana.items()}}}
