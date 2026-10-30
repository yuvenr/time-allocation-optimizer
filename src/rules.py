"""Aturan domain constraint engine dan validator rencana (UC-13).

Setiap aturan adalah fungsi MURNI atas (kurikulum, config, keadaan akademik) yang
mengembalikan ``HasilAturan(kode, status, alasan, sumber)`` — tanpa I/O. ``status``:
``lolos`` | ``gagal`` | ``perlu_verifikasi_manual`` (syarat yang isinya tidak ada di
dokumen, mis. pedoman PLP/Skripsi — tidak dianggap otomatis terpenuhi).

Semua angka aturan dibaca dari dataset (``src.data_loader.Kurikulum``); tafsiran yang
belum dikonfirmasi prodi ada di ``config.yaml`` bagian ``aturan``. Planner (mode ``arm``
maupun ``hybrid``) dan ``validate_plan`` memakai fungsi yang SAMA. Batasan beban θ bukan
aturan domain; kepatuhannya dihitung terpisah sebagai metrik.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable

from src.config import Config
from src.data_loader import Kurikulum
from src.schemas import HasilAturan, Plan, Profile, ValidationResult

# Kode aturan (AD-xx) → (nama, rujukan). AD-01..AD-10 = aturan akademik standar; AD-11..AD-16 = aturan
# kurikulum 2024 tambahan (dokumen [R]/[D]) agar rencana tidak melanggar kurikulum yang berlaku.
DEFINISI_ATURAN: dict[str, tuple[str, str]] = {
    "AD-01": ("Total SKS minimal 145", "AT01"),
    "AD-02": ("Beban SKS per semester sesuai IP semester sebelumnya", "AT05; AT06; AT13; 09"),
    "AD-03": ("Prasyarat MK telah lulus", "05 nilai_minimal_mk; AT21"),
    "AD-04": ("Semester penawaran ganjil/genap", "01 semester_rekomendasi_jalur_A / semester_ditawarkan_tabel11; 09"),
    "AD-05": ("Seluruh MK wajib + MK pilihan minimal 27 SKS", "AT02"),
    "AD-06": ("Total beban studi paling banyak 160 SKS", "AT03; Pedoman UB 8.2"),
    "AD-07": ("MK lulus tidak diulang; MK wajib belum lulus/E diprioritaskan; D/D+ ≤ 10% beban kredit",
              "AT09; Pedoman UB 8.4"),
    "AD-08": ("Semester 1–2 berupa paket (≤ 20 SKS)", "AT06; 08"),
    "AD-09": ("Masa studi 7–14 semester (normal 8)", "AT04; Pedoman UB 8.6"),
    "AD-10": ("Skripsi: lulus ≥ 120 SKS dan IPK ≥ 2,00", "Pedoman UB 8.5"),
    "AD-11": ("PLP: diambil bersamaan & lulus seluruh MK Kependidikan", "AT15; AT16; 05"),
    "AD-12": ("MK agama: tepat satu sesuai agama mahasiswa", "01 grup_pilih_satu"),
    "AD-13": ("Sembilan tema pilihan terpenuhi (≥ 3 SKS per tema)", "AT07; AT08; 06"),
    "AD-14": ("Jalur merdeka belajar: slot & syarat konversi", "AT11–AT14; 08"),
    "AD-15": ("MK yang tidak direkomendasikan bagi mahasiswa PTI", "AT10"),
    "AD-16": ("Syarat pedoman eksternal (PLP, PKL, Capstone, Skripsi, PKM)", "05 pedoman_eksternal"),
}


def ganjil(semester: int) -> bool:
    return semester % 2 == 1


def paritas(semester: int) -> str:
    return "ganjil" if ganjil(semester) else "genap"


@dataclass(frozen=True)
class Keadaan:
    """Status akademik pada AWAL suatu semester (masukan fungsi aturan).

    ``lulus`` = MK dengan nilai ≥ NILAI_LULUS_MINIMUM (MK yang direncanakan dianggap lulus);
    ``pernah`` = ``lulus`` ∪ MK yang pernah ditempuh tanpa lulus; ``sks_konversi`` = SKS konversi
    merdeka belajar yang sudah diakui; ``ipk`` = IPK dari riwayat nilai (``None`` = tidak dapat
    dihitung), diasumsikan tetap untuk semester mendatang seperti asumsi IP pada AD-02.
    """

    jalur: str
    agama_kode: str | None
    lulus: frozenset[str]
    pernah: frozenset[str]
    ipk: float | None = None
    sks_konversi: int = 0

    @classmethod
    def dari_profil(cls, profil: Profile, kur: Kurikulum, cfg: Config | None = None) -> "Keadaan":
        ipk = hitung_ipk(profil.riwayat_nilai, kur, cfg)[0] if cfg is not None else None
        if ipk is None:
            ipk = profil.ipk                    # IPK yang diketahui (data akademik / input mahasiswa)
        return cls(profil.jalur, kur.mk_agama(profil.agama), frozenset(profil.mk_lulus),
                   frozenset(profil.pernah_menempuh), ipk, profil.sks_konversi)

    def tambah(self, kode_list: Iterable[str], sks_konversi: int = 0) -> "Keadaan":
        k = frozenset(kode_list)
        return replace(self, lulus=self.lulus | k, pernah=self.pernah | k,
                       sks_konversi=self.sks_konversi + sks_konversi)

    def sks_lulus(self, kur: Kurikulum) -> int:
        return kur.total_sks(self.lulus & set(kur.mk)) + self.sks_konversi


def hitung_ipk(riwayat: dict[str, str], kur: Kurikulum, cfg: Config | None) -> tuple[float | None, str]:
    """IPK = Σ(bobot × SKS) / Σ SKS atas riwayat nilai (kode Kurikulum 2024; nilai terbaik per MK).

    Mengembalikan (IPK dibulatkan 2 desimal, keterangan). ``None`` bila riwayat kosong, ada nilai di
    luar skala, atau kode di luar kurikulum (IPK tidak dapat dihitung dengan benar).
    """
    if cfg is None or not riwayat:
        return None, "riwayat nilai tidak diisi, IPK tidak dapat dihitung"
    asing = sorted(k for k in riwayat if k not in kur)
    tak_sah = sorted(k for k, n in riwayat.items() if cfg.aturan.bobot(n) is None)
    if asing or tak_sah:
        return None, (f"IPK tidak dihitung: kode di luar kurikulum {asing}" if asing
                      else f"IPK tidak dihitung: nilai di luar skala pada {tak_sah}")
    sks = sum(kur.sks(k) for k in riwayat)
    total = sum(cfg.aturan.bobot(n) * kur.sks(k) for k, n in riwayat.items())
    return round(total / sks, 2), f"IPK dari {len(riwayat)} MK ({sks} SKS)"


def _h(kode: str, status: str, alasan: str, sumber: str, kode_mk: str | None = None,
       semester: int | None = None) -> HasilAturan:
    return HasilAturan(kode=kode, status=status, alasan=alasan, sumber=sumber, kode_mk=kode_mk, semester=semester)


def _daftar_param(kur: Kurikulum, id_aturan: str, kunci: str) -> list[str]:
    nilai = kur.param(id_aturan, kunci)
    if ":" in nilai:
        nilai = nilai.split(":", 1)[1]
    return [x.strip() for x in nilai.split(",") if x.strip()]


# ---------------------------------------------------------------------
# Aturan per MK (dipanggil untuk setiap kandidat pada semester s)
# ---------------------------------------------------------------------


def aturan_sudah_lulus(kode: str, s: int, kd: Keadaan, kur: Kurikulum, cfg: Config) -> HasilAturan:
    """AD-07: MK yang sudah lulus tidak direkomendasikan lagi; MK yang pernah gagal boleh diulang."""
    if kode in kd.lulus:
        return _h("AD-07", "gagal", f"{kur.label(kode)} sudah lulus; tidak direkomendasikan lagi",
                  "riwayat nilai mahasiswa", kode, s)
    status = "pernah ditempuh tanpa lulus → boleh diulang" if kode in kd.pernah else "belum pernah ditempuh"
    return _h("AD-07", "lolos", status, "riwayat nilai mahasiswa", kode, s)


def aturan_tidak_direkomendasikan(kode: str, s: int, kd: Keadaan, kur: Kurikulum,
                                  cfg: Config) -> HasilAturan | None:
    """AD-15 (§4.10): MK yang dilarang direkomendasikan (AT10: UBU60030)."""
    if kode not in _daftar_param(kur, "AT10", "kode"):
        return None
    a = kur.aturan["AT10"]
    return _h("AD-15", "gagal", f"{kur.label(kode)}: {a.aturan} ({a.keterangan})", a.sumber, kode, s)


def _saran_tambah_09(kode: str, kur: Kurikulum) -> list:
    return [p for p in kur.pita_sks if p.aksi == "tambah" and kode in p.kode_terdampak]


def aturan_penawaran(kode: str, s: int, kd: Keadaan, kur: Kurikulum, cfg: Config) -> HasilAturan:
    """AD-04 (§4.5): semester penawaran.

    - MK wajib: paritas ``semester_rekomendasi_jalur_A``. Dengan ``PENAWARAN_DARI_REKOMENDASI_09``,
      MK yang disarankan 09 untuk "ditambah" di semester berparitas lain (Skripsi/PKM di sem. 7,
      Capstone di sem. 6) juga dianggap ditawarkan pada paritas itu mulai semester tersebut.
    - MK pilihan: ``semester_ditawarkan_tabel11`` (``tabel11_ketat``) atau paritas yang sama dan
      ≥ semester Tabel 11 terkecil (``paritas``, default; INK07/INK08). ``semester_paritas_deskripsi``
      tidak dipakai (INK05).
    """
    m = kur.mk[kode]
    ac = cfg.aturan
    if m.sifat == "wajib":
        home = m.semester_jalur_a
        sumber = f"01_mata_kuliah.csv#{kode} semester_rekomendasi_jalur_A={home}"
        if ganjil(home) == ganjil(s):
            return _h("AD-04", "lolos", f"MK wajib semester {home} ({paritas(home)}) ditawarkan di semester "
                                        f"{paritas(s)}", sumber, kode, s)
        if ac.PENAWARAN_DARI_REKOMENDASI_09:
            for p in _saran_tambah_09(kode, kur):
                if ganjil(p.semester) == ganjil(s) and s >= p.semester:
                    return _h("AD-04", "lolos",
                              f"09 menyarankan menambah {kode} di semester {p.semester}; dianggap ditawarkan di "
                              f"semester {paritas(s)} ≥ {p.semester} (tafsiran PENAWARAN_DARI_REKOMENDASI_09)",
                              p.sumber, kode, s)
        return _h("AD-04", "gagal", f"MK wajib semester {home} ({paritas(home)}) tidak ditawarkan di semester "
                                    f"{s} ({paritas(s)})", sumber, kode, s)
    t11 = m.semester_tabel11
    if not t11:
        return _h("AD-04", "gagal", f"{kur.label(kode)}: semester penawaran tidak tercantum di dokumen",
                  f"01_mata_kuliah.csv#{kode} semester_ditawarkan_tabel11 kosong (CLAUDE.md §7a)", kode, s)
    mode = ac.MODE_PENAWARAN_PILIHAN
    sumber = (f"01_mata_kuliah.csv#{kode} semester_ditawarkan_tabel11={';'.join(map(str, t11))}; "
              f"[R] Tabel 11 hal. 24; mode {mode}")
    if mode == "tabel11_ketat":
        ok = s in t11
        syarat = f"semester {', '.join(map(str, t11))}"
    else:
        ok = any(ganjil(x) == ganjil(s) for x in t11) and s >= min(t11)
        syarat = f"semester {paritas(t11[0])} ≥ {min(t11)}"
    teks = "ditawarkan" if ok else "tidak ditawarkan"
    return _h("AD-04", "lolos" if ok else "gagal",
              f"MK pilihan Tema {m.tema} {teks} di semester {s} (penawaran: {syarat})", sumber, kode, s)


def aturan_prasyarat(kode: str, s: int, kd: Keadaan, kur: Kurikulum, cfg: Config) -> HasilAturan | None:
    """AD-03 (§4.1): 'Nilai MK X ≥ E'. Dengan ``PRASYARAT_E_BERARTI_PERNAH_MENEMPUH`` = pernah menempuh
    X dengan nilai apa pun (di semester sebelumnya); selain itu X harus lulus."""
    syarat = kur.prasyarat_nilai(kode)
    if not syarat:
        return None
    pernah = cfg.aturan.PRASYARAT_E_BERARTI_PERNAH_MENEMPUH

    def terpenuhi(p) -> bool:
        if p.nilai_minimal == "E" and pernah:
            return p.kode_syarat in kd.pernah
        return p.kode_syarat in kd.lulus

    kurang = [p for p in syarat if not terpenuhi(p)]
    tafsir = "pernah menempuh" if pernah else "lulus"
    if kurang:
        daftar = ", ".join(kur.label(p.kode_syarat) for p in kurang)
        return _h("AD-03", "gagal", f"prasyarat belum terpenuhi: {daftar} (≥ E = {tafsir}, di semester sebelumnya)",
                  "; ".join(p.sumber for p in kurang), kode, s)
    daftar = ", ".join(p.kode_syarat for p in syarat)
    return _h("AD-03", "lolos", f"prasyarat terpenuhi: {daftar} (≥ E = {tafsir})",
              "; ".join(p.sumber for p in syarat), kode, s)


def aturan_syarat_plp(kode: str, s: int, kd: Keadaan, kur: Kurikulum, cfg: Config) -> HasilAturan | None:
    """AD-11 (§4.2): PLP hanya boleh diambil bila LULUS seluruh MK Kependidikan (INK30: 10 MK)."""
    syarat = kur.syarat_kelompok(kode)
    if not syarat:
        return None
    ambang = cfg.aturan.NILAI_LULUS_MINIMUM
    kurang = [p.kode_syarat for p in syarat if p.kode_syarat not in kd.lulus]
    sumber = f"{kur.aturan['AT16'].sumber}; 05_prasyarat.csv#{syarat[0].id}–{syarat[-1].id}"
    if kurang:
        return _h("AD-11", "gagal", f"PLP mensyaratkan lulus {len(syarat)} MK Kependidikan (ambang lulus "
                                    f"{ambang}); belum lulus: {', '.join(sorted(kurang))}", sumber, kode, s)
    return _h("AD-11", "lolos", f"seluruh {len(syarat)} MK Kependidikan lulus (ambang {ambang})", sumber, kode, s)


def aturan_pedoman_eksternal(kode: str, s: int, kd: Keadaan, kur: Kurikulum,
                             cfg: Config) -> HasilAturan | None:
    """AD-16 (§4.3): syarat 'sesuai pedoman …' tidak ada isinya di dokumen → perlu verifikasi manual."""
    p = kur.pedoman_eksternal(kode)
    if p is None:
        return None
    return _h("AD-16", "perlu_verifikasi_manual",
              f"{p.teks_asli}; isi pedoman tidak termuat di dokumen, verifikasi manual oleh DPA", p.sumber, kode, s)


def aturan_agama(kode: str, s: int, kd: Keadaan, kur: Kurikulum, cfg: Config) -> HasilAturan | None:
    """AD-12 (§4.7): tepat satu MK agama sesuai agama mahasiswa (input pengguna, tidak disimpulkan)."""
    grup = kur.grup("AGAMA")
    if kode not in grup:
        return None
    sumber = "01_mata_kuliah.csv grup_pilih_satu=AGAMA; 08 slot MK_WAJIB_PILIH_SATU"
    sudah = sorted(grup & kd.lulus)
    if sudah and kode not in sudah:
        return _h("AD-12", "gagal", f"sudah lulus {kur.label(sudah[0])}; MK agama cukup satu", sumber, kode, s)
    if kd.agama_kode is None:
        return _h("AD-12", "gagal", "agama mahasiswa belum diisi; sistem tidak menyimpulkan agama", sumber, kode, s)
    if kode != kd.agama_kode:
        return _h("AD-12", "gagal", f"bukan MK agama mahasiswa ({kur.label(kd.agama_kode)})", sumber, kode, s)
    return _h("AD-12", "lolos", "MK agama sesuai input mahasiswa", sumber, kode, s)


def aturan_skripsi(kode: str, s: int, kd: Keadaan, kur: Kurikulum, cfg: Config) -> HasilAturan | None:
    """AD-10 (Tabel 4.5; Pedoman UB 8.5): Skripsi hanya setelah lulus ≥ ``min_sks_lulus`` SKS dengan
    IPK ≥ ``min_ipk``. SKS lulus dihitung pada AWAL semester s (MK yang direncanakan sebelumnya dianggap
    lulus). IPK diasumsikan tetap; bila tidak dapat dihitung dari riwayat → perlu verifikasi manual."""
    sk = cfg.aturan.SKRIPSI
    if kode != sk.kode:
        return None
    sumber = "Pedoman Pendidikan UB Subbab 8.5 (AD-10); config aturan.SKRIPSI"
    sks = kd.sks_lulus(kur)
    if sks < sk.min_sks_lulus:
        return _h("AD-10", "gagal", f"SKS lulus di awal semester {s} = {sks} < {sk.min_sks_lulus}", sumber, kode, s)
    if kd.ipk is None:
        return _h("AD-10", "perlu_verifikasi_manual",
                  f"SKS lulus {sks} ≥ {sk.min_sks_lulus}; IPK ≥ {sk.min_ipk:.2f} perlu diverifikasi (riwayat nilai "
                  f"tidak lengkap)", sumber, kode, s)
    if kd.ipk < sk.min_ipk:
        return _h("AD-10", "gagal", f"IPK {kd.ipk:.2f} < {sk.min_ipk:.2f}", sumber, kode, s)
    return _h("AD-10", "lolos", f"SKS lulus {sks} ≥ {sk.min_sks_lulus} dan IPK {kd.ipk:.2f} ≥ {sk.min_ipk:.2f} "
                                f"(IPK diasumsikan tetap)", sumber, kode, s)


ATURAN_MK: tuple[Callable[..., HasilAturan | None], ...] = (
    aturan_sudah_lulus, aturan_tidak_direkomendasikan, aturan_penawaran, aturan_prasyarat,
    aturan_syarat_plp, aturan_skripsi, aturan_pedoman_eksternal, aturan_agama,
)


def cek_mk(kode: str, s: int, kd: Keadaan, kur: Kurikulum, cfg: Config) -> list[HasilAturan]:
    """Jalankan seluruh aturan per MK untuk ``kode`` di semester ``s`` (aturan yang tidak berlaku dilewati)."""
    if kode not in kur:
        return [_h("AD-05", "gagal", f"{kode} tidak ada di kurikulum 2024", "01_mata_kuliah.csv", kode, s)]
    return [h for f in ATURAN_MK if (h := f(kode, s, kd, kur, cfg)) is not None]


def boleh_diambil(hasil: Iterable[HasilAturan]) -> bool:
    return not any(h.melanggar for h in hasil)


# ---------------------------------------------------------------------
# Aturan per semester
# ---------------------------------------------------------------------


def semester_paket(kur: Kurikulum) -> dict[int, int]:
    """AT06: semester paket → SKS paket (``sks_sem1=20;sks_sem2=20``)."""
    return {int(k[len("sks_sem"):]): int(v) for k, v in kur.aturan["AT06"].parameter.items()
            if k.startswith("sks_sem")}


def batas_sks(s: int, jalur: str, ip: float, kur: Kurikulum) -> tuple[int, int, str]:
    """(sks_min, sks_max, sumber) untuk semester s. Semester paket: SKS paket (AT06); ada baris 09:
    pita jalur → SEMUA; tanpa baris (mis. semester > 8): pita seragam AT05."""
    paket = semester_paket(kur)
    if s in paket:
        return 0, paket[s], f"{kur.aturan['AT06'].sumber} (paket {paket[s]} SKS, tanpa pita IP)"
    p = kur.pita(jalur, s, ip)
    if p is not None:
        return p.sks_min, p.sks_max, p.sumber
    u = kur.pita_umum(ip)
    if u is None:
        raise ValueError(f"Tidak ada pita SKS untuk jalur {jalur} semester {s} IP {ip}")
    return u[0], u[1], f"{kur.aturan['AT05'].sumber} (pita seragam 09; semester {s} tanpa baris rekomendasi)"


def aturan_beban_sks(s: int, kode_list: list[str], konversi_sks: int, kd: Keadaan, ip: float,
                     kur: Kurikulum, cfg: Config) -> HasilAturan:
    """AD-02 (§4.4): total SKS semester ≤ sks_max pita IP (IP terakhir sebagai asumsi); semester
    konversi merdeka belajar dibatasi AT13 (09 tidak memuat baris untuk semester konversi)."""
    total = kur.total_sks(kode_list) + konversi_sks
    if konversi_sks:
        maks = kur.param_int("AT13", "konversi_max_per_semester")
        sumber, konteks = kur.aturan["AT13"].sumber, "semester konversi"
    else:
        _, maks, sumber = batas_sks(s, kd.jalur, ip, kur)
        konteks = f"IP {ip:.2f}"
    if total > maks:
        return _h("AD-02", "gagal", f"{total} SKS > batas {maks} SKS ({konteks})", sumber, semester=s)
    return _h("AD-02", "lolos", f"{total} SKS ≤ batas {maks} SKS ({konteks})", sumber, semester=s)


def aturan_korequisit(s: int, kode_list: list[str], kd: Keadaan, kur: Kurikulum,
                      cfg: Config) -> list[HasilAturan]:
    """AD-11 (§4.2, AT15): MK berkorequisit (PLP 1 & PLP 2) diambil pada semester yang sama."""
    hasil = []
    for k in kode_list:
        for q in sorted(kur.korequisit(k)):
            if q not in kode_list and q not in kd.lulus:
                sumber = f"{kur.aturan['AT15'].sumber}; " + "; ".join(p.sumber for p in kur.syarat(k, "korequisit"))
                hasil.append(_h("AD-11", "gagal", f"{kur.label(k)} harus diambil bersamaan dengan {kur.label(q)}",
                                sumber, k, s))
    return hasil


def mk_paket_semester(s: int, kd: Keadaan, kur: Kurikulum) -> list[str]:
    """AD-08: MK paket semester s pada jalur mahasiswa (slot agama → MK agama mahasiswa), belum lulus."""
    hasil = []
    for slot in kur.slot(kd.jalur, s):
        if slot.jenis_slot == "MK_WAJIB":
            k = slot.kode_mk
        elif slot.jenis_slot == "MK_WAJIB_PILIH_SATU":
            if kur.grup("AGAMA") & kd.lulus:
                continue
            k = kd.agama_kode
        else:
            continue
        if k and k not in kd.lulus:
            hasil.append(k)
    return hasil


def aturan_paket(s: int, kode_list: list[str], kd: Keadaan, kur: Kurikulum, cfg: Config) -> list[HasilAturan]:
    """AD-08 (AT06): semester paket memuat tepat MK paket yang boleh diambil — tidak kurang, tidak lebih."""
    if s not in semester_paket(kur):
        return []
    sumber = f"{kur.aturan['AT06'].sumber}; 08_struktur_kurikulum_jalur.csv (Jalur {kd.jalur}, sem. {s})"
    harus = {k for k in mk_paket_semester(s, kd, kur) if boleh_diambil(cek_mk(k, s, kd, kur, cfg))}
    hasil = [_h("AD-08", "gagal", f"{kur.label(k)} (paket) tidak diambil", sumber, k, s)
             for k in sorted(harus - set(kode_list))]
    hasil += [_h("AD-08", "gagal", f"{kur.label(k)} bukan MK paket semester {s}", sumber, k, s)
              for k in sorted(set(kode_list) - harus)]
    return hasil or [_h("AD-08", "lolos", f"paket semester {s} sesuai struktur", sumber, semester=s)]


def syarat_konversi(kd: Keadaan, kur: Kurikulum) -> tuple[bool, list[str]]:
    """AT12: sudah lulus semua MK wajib kecuali MK lapangan, Capstone, dan Skripsi (agama cukup satu)."""
    kecuali = set(_daftar_param(kur, "AT12", "wajib_lulus"))
    agama = kur.grup("AGAMA")
    kurang = sorted(kur.wajib - kecuali - agama - kd.lulus)
    if not agama & kd.lulus:
        kurang.append("MK agama")
    return not kurang, kurang


def aturan_konversi(s: int, konversi_sks: int, kd: Keadaan, kur: Kurikulum, cfg: Config) -> list[HasilAturan]:
    """AD-14 (§4.8): konversi merdeka belajar hanya pada slot PILIHAN_KONVERSI jalur mahasiswa,
    ≤ AT13 SKS, dan setelah syarat AT12 terpenuhi. Kesetaraan tema selalu perlu verifikasi DPA."""
    if not konversi_sks:
        return []
    slot = [x for x in kur.slot(kd.jalur, s) if x.jenis_slot == "PILIHAN_KONVERSI"]
    if not slot:
        return [_h("AD-14", "gagal", f"Jalur {kd.jalur} tidak memiliki slot konversi di semester {s}",
                   f"08_struktur_kurikulum_jalur.csv (Jalur {kd.jalur})", semester=s)]
    hasil = []
    maks = kur.param_int("AT13", "konversi_max_per_semester")
    if konversi_sks > maks:
        hasil.append(_h("AD-14", "gagal", f"konversi {konversi_sks} SKS > maksimal {maks} SKS",
                        kur.aturan["AT13"].sumber, semester=s))
    ok, kurang = syarat_konversi(kd, kur)
    if not ok:
        hasil.append(_h("AD-14", "gagal", f"syarat konversi belum terpenuhi; MK wajib belum lulus: "
                                          f"{', '.join(kurang)}", kur.aturan["AT12"].sumber, semester=s))
    if not hasil:
        hasil.append(_h("AD-14", "perlu_verifikasi_manual",
                        f"konversi kegiatan merdeka belajar {konversi_sks} SKS: kesetaraan SKS dan tema yang "
                        f"terpenuhi perlu disetujui DPA", f"{slot[0].sumber}; {kur.aturan['AT14'].sumber}",
                        semester=s))
    return hasil


# ---------------------------------------------------------------------
# Aturan akhir rencana
# ---------------------------------------------------------------------


def tema_terpenuhi(kode_set: Iterable[str], tema_konversi: Iterable[str], kur: Kurikulum) -> set[str]:
    """Tema dengan SKS MK pilihan (lulus/direncanakan) ≥ sks_minimal, atau dipenuhi lewat konversi."""
    kode_set, konv = set(kode_set), set(tema_konversi)
    return {t for t, tp in kur.tema.items()
            if t in konv or kur.total_sks(kode_set & kur.mk_tema(t)) >= tp.sks_minimal}


def aturan_total_sks(kode_akhir: set[str], konversi_sks: int, kur: Kurikulum,
                     s_akhir: int) -> list[HasilAturan]:
    """AD-01 (total SKS ≥ AT01 total_min) dan AD-06 (total SKS ≤ AT03 total_max)."""
    total = kur.total_sks(kode_akhir) + konversi_sks
    mn, mx = kur.param_int("AT01", "total_min"), kur.param_int("AT03", "total_max")
    s1, s3 = kur.aturan["AT01"].sumber, kur.aturan["AT03"].sumber
    return [
        _h("AD-01", "gagal", f"total SKS hingga semester {s_akhir} = {total} < {mn}", s1) if total < mn
        else _h("AD-01", "lolos", f"total SKS {total} ≥ {mn}", s1),
        _h("AD-06", "gagal", f"total SKS {total} > maksimal {mx}", s3) if total > mx
        else _h("AD-06", "lolos", f"total SKS {total} ≤ {mx}", s3),
    ]


def aturan_kelengkapan(kode_akhir: set[str], tema_konversi: set[str], konversi_sks: int,
                       kur: Kurikulum) -> list[HasilAturan]:
    """AD-05: seluruh MK wajib (grup agama dinilai AD-12) + SKS pilihan ≥ AT02 pilihan_min (SKS konversi
    merdeka belajar dihitung sebagai pilihan); AD-13: sembilan tema pilihan terpenuhi."""
    hasil = []
    s2 = kur.aturan["AT02"].sumber
    wajib_kurang = sorted(kur.wajib - kur.grup("AGAMA") - kode_akhir)
    if wajib_kurang:
        hasil.append(_h("AD-05", "gagal", f"MK wajib belum tercakup: {', '.join(wajib_kurang)}", s2))
    pilihan = kur.total_sks(kode_akhir & kur.pilihan) + konversi_sks
    minimal = kur.param_int("AT02", "pilihan_min")
    if pilihan < minimal:
        hasil.append(_h("AD-05", "gagal", f"SKS MK pilihan {pilihan} < {minimal}", s2))
    if not hasil:
        hasil.append(_h("AD-05", "lolos", f"seluruh MK wajib tercakup; SKS pilihan {pilihan} ≥ {minimal}", s2))
    kurang_tema = sorted(set(kur.tema) - tema_terpenuhi(kode_akhir, tema_konversi, kur))
    s7 = f"{kur.aturan['AT07'].sumber}; 06_tema_pilihan.csv"
    hasil.append(_h("AD-13", "gagal", f"tema pilihan belum terpenuhi: {', '.join(kurang_tema)}", s7) if kurang_tema
                 else _h("AD-13", "lolos", "sembilan tema pilihan terpenuhi", s7))
    return hasil


def aturan_agama_akhir(kode_akhir: set[str], kd_awal: Keadaan, kur: Kurikulum) -> HasilAturan:
    """AD-12: pada akhir rencana tepat satu MK agama tercakup (sesuai agama mahasiswa)."""
    sumber = "01_mata_kuliah.csv grup_pilih_satu=AGAMA"
    ada = sorted(kur.grup("AGAMA") & kode_akhir)
    if len(ada) > 1:
        return _h("AD-12", "gagal", f"lebih dari satu MK agama: {', '.join(ada)}", sumber)
    if not ada:
        if kd_awal.agama_kode is None:
            return _h("AD-12", "perlu_verifikasi_manual",
                      "agama mahasiswa belum diisi; slot MK agama tidak dapat direncanakan", sumber)
        return _h("AD-12", "gagal", f"{kur.label(kd_awal.agama_kode)} belum direncanakan", sumber)
    if kd_awal.agama_kode and ada[0] != kd_awal.agama_kode and ada[0] not in kd_awal.lulus:
        return _h("AD-12", "gagal", f"{ada[0]} tidak sesuai agama mahasiswa", sumber)
    return _h("AD-12", "lolos", f"MK agama: {kur.label(ada[0])}", sumber)


def aturan_nilai(profile: Profile, kode_akhir: set[str], konversi_sks: int, kur: Kurikulum,
                 cfg: Config) -> list[HasilAturan]:
    """AD-07 (Pedoman UB 8.4): kelulusan mensyaratkan tidak ada nilai E dan SKS bernilai D/D+ paling
    banyak 10% total beban kredit. Dinilai dari riwayat nilai (nilai MK mendatang belum diketahui):
    - proporsi D/D+ > batas → perlu verifikasi + daftar MK yang disarankan diulang untuk perbaikan;
    - MK pilihan bernilai E yang tidak direncanakan ulang → perlu verifikasi (status nilai E di transkrip).
    """
    a = cfg.aturan
    sumber = "Pedoman Pendidikan UB Subbab 8.4 (AD-07); config aturan.NILAI_DIBATASI"
    riwayat = {k: str(n).strip().upper() for k, n in profile.riwayat_nilai.items() if k in kur}
    hasil = []
    total = kur.total_sks(kode_akhir) + konversi_sks
    dibatasi = {k: n for k, n in riwayat.items() if n in a.NILAI_DIBATASI}
    sks_d = kur.total_sks(dibatasi)
    if total and sks_d / total > a.BATAS_PROPORSI_NILAI_DIBATASI + 1e-9:
        urut = sorted(dibatasi, key=lambda k: (a.SKALA_NILAI.index(dibatasi[k]) * -1, -kur.sks(k), k))
        perlu, sisa = [], sks_d
        for k in urut:   # nilai terendah & SKS terbesar lebih dulu sampai proporsi ≤ batas
            if sisa / total <= a.BATAS_PROPORSI_NILAI_DIBATASI + 1e-9:
                break
            perlu.append(k)
            sisa -= kur.sks(k)
        hasil.append(_h("AD-07", "perlu_verifikasi_manual",
                        f"SKS bernilai {'/'.join(a.NILAI_DIBATASI)} = {sks_d} dari {total} SKS "
                        f"({sks_d / total:.1%} > {a.BATAS_PROPORSI_NILAI_DIBATASI:.0%}); pertimbangkan mengulang "
                        f"untuk perbaikan: {', '.join(f'{kur.label(k)} ({dibatasi[k]})' for k in perlu)}", sumber))
    e_pilihan = sorted(k for k, n in riwayat.items()
                       if not a.lulus(n) and k in kur.pilihan and k not in kode_akhir)
    if e_pilihan:
        hasil.append(_h("AD-07", "perlu_verifikasi_manual",
                        f"MK pilihan belum lulus yang tidak diulang: {', '.join(e_pilihan)}; pastikan nilai E tidak "
                        f"tercantum pada transkrip kelulusan", sumber))
    return hasil


def aturan_masa_studi(semester_list: list[int], semester_awal: int, kur: Kurikulum) -> list[HasilAturan]:
    """AD-09 (AT04): rencana dimulai di semester berjalan, tidak melewati masa studi maksimal, dan
    tidak menyelesaikan studi sebelum masa studi minimal (7 semester)."""
    mn, mx = kur.param_int("AT04", "semester_min"), kur.param_int("AT04", "semester_max")
    sumber = kur.aturan["AT04"].sumber
    hasil = [_h("AD-09", "gagal", f"semester {s} di luar rentang {semester_awal}–{mx}", sumber, semester=s)
             for s in semester_list if s < semester_awal or s > mx]
    if semester_list and max(semester_list) < mn:
        hasil.append(_h("AD-09", "gagal", f"rencana selesai di semester {max(semester_list)} < masa studi "
                                          f"minimal {mn} semester", sumber))
    return hasil or [_h("AD-09", "lolos", f"rencana di dalam masa studi {mn}–{mx} semester", sumber)]


# ---------------------------------------------------------------------
# Validator rencana (UC-13) — identik untuk mode arm dan hybrid
# ---------------------------------------------------------------------


def unit_korequisit(kode: str, kd: Keadaan, kur: Kurikulum) -> set[str]:
    """MK beserta korequisitnya yang belum lulus (diambil sebagai satu unit, mis. PLP 1 + PLP 2)."""
    return {kode} | {q for q in kur.korequisit(kode) if q not in kd.lulus}


def mk_tertunda(profile: Profile, kd: Keadaan, kur: Kurikulum) -> set[str]:
    """AT09: MK wajib yang pernah gagal atau semester paketnya sudah lewat, dan belum lulus."""
    hasil = set()
    agama = kur.grup("AGAMA")
    for k in kur.wajib - kd.lulus:
        if k in agama and (k != kd.agama_kode or agama & kd.lulus):
            continue
        home = kur.semester_jalur(k, kd.jalur) or kur.mk[k].semester_jalur_a
        if k in profile.mk_pernah_gagal or home < profile.semester_saat_ini:
            hasil.add(k)
    return hasil


def validate_plan(plan: Plan, profile: Profile, kur: Kurikulum, cfg: Config,
                  beban: Callable[[Iterable[str]], float] | None = None) -> ValidationResult:
    """Periksa rencana terhadap seluruh aturan (AD-01..AD-16: batasan akademik kurikulum).

    Aturan dievaluasi berurutan per semester: MK yang direncanakan pada semester s dianggap
    lulus mulai semester s+1. θ (bila rencana ``hybrid`` membawa θ) beserta fungsi ``beban``
    (beban kognitif sekumpulan MK, sama dengan yang dipakai planner) hanya dipakai untuk menilai
    apakah penundaan MK wajib (AT09) memang terpaksa — bukan aturan domain.
    """
    kd0 = Keadaan.dari_profil(profile, kur, cfg)
    kd = kd0
    ip = profile.ip_terakhir
    semua: list[HasilAturan] = []
    direncanakan: dict[str, int] = {}
    konversi_total = profile.sks_konversi          # SKS konversi yang sudah diakui sebelum rencana
    tema_konv: set[str] = set(profile.tema_konversi or ())
    info_semester: dict[int, tuple[Keadaan, list[str], int, float]] = {}

    for sem in sorted(plan.semester, key=lambda x: x.semester):
        s = sem.semester
        mk = [it.kode for it in sem.items if it.jenis == "mk"]
        konv = sum(it.sks for it in sem.items if it.jenis == "konversi")
        for it in sem.items:
            if it.jenis == "konversi":
                tema_konv |= set(it.tema_konversi)
        for k in mk:
            if k in direncanakan:
                semua.append(_h("AD-07", "gagal", f"{kur.label(k)} sudah direncanakan di S{direncanakan[k]}",
                                "rencana", k, s))
                continue
            semua += cek_mk(k, s, kd, kur, cfg)
        unik = [k for k in dict.fromkeys(mk) if k in kur]
        semua.append(aturan_beban_sks(s, unik, konv, kd, ip, kur, cfg))
        semua += aturan_korequisit(s, unik, kd, kur, cfg)
        semua += aturan_paket(s, unik, kd, kur, cfg)
        semua += aturan_konversi(s, konv, kd, kur, cfg)
        info_semester[s] = (kd, unik, konv, sem.total_beban)
        for k in unik:
            direncanakan.setdefault(k, s)
        konversi_total += konv
        kd = kd.tambah(unik, konv)

    kode_akhir = set(kd.lulus)
    s_akhir = max(info_semester, default=profile.semester_saat_ini)
    semua += aturan_total_sks(kode_akhir, konversi_total, kur, s_akhir)
    semua += aturan_kelengkapan(kode_akhir, tema_konv, konversi_total, kur)
    semua.append(aturan_agama_akhir(kode_akhir, kd0, kur))
    semua += aturan_nilai(profile, kode_akhir, konversi_total, kur, cfg)
    semua += aturan_masa_studi(sorted(info_semester), profile.semester_saat_ini, kur)
    theta = plan.theta if plan.mode == "hybrid" else None
    semua += _cek_prioritas_tertunda(profile, kd0, info_semester, direncanakan, ip, kur, cfg, beban, theta)

    per_aturan = {kode: not any(h.melanggar for h in semua if h.kode == kode) for kode in DEFINISI_ATURAN}
    urut = {kode: i for i, kode in enumerate(DEFINISI_ATURAN)}
    gagal = sorted((h for h in semua if h.melanggar), key=lambda h: (urut.get(h.kode, 99), h.semester or 0))
    verifikasi = [h for h in semua if h.status == "perlu_verifikasi_manual"]
    return ValidationResult(lolos=not gagal, pelanggaran=[h.teks() for h in gagal], per_aturan=per_aturan,
                            perlu_verifikasi=[h.teks() for h in verifikasi], rincian=gagal + verifikasi)


def _cek_prioritas_tertunda(profile: Profile, kd0: Keadaan, info: dict, direncanakan: dict[str, int], ip: float,
                            kur: Kurikulum, cfg: Config, beban: Callable[[Iterable[str]], float] | None,
                            theta: float | None) -> list[HasilAturan]:
    """AD-07 (AT09): MK wajib tertunda harus diambil pada semester non-paket PERTAMA yang
    memungkinkan (ditawarkan, syarat terpenuhi, kapasitas SKS — dan θ bila rencana hybrid — cukup)."""
    hasil = []
    paket = semester_paket(kur)
    sumber = kur.aturan["AT09"].sumber
    for k in sorted(mk_tertunda(profile, kd0, kur)):
        for s in sorted(info):
            kd, mk, konv, _ = info[s]
            if s in paket or konv:
                continue
            if direncanakan.get(k) == s:
                break
            unit = unit_korequisit(k, kd, kur)
            if not all(boleh_diambil(cek_mk(u, s, kd, kur, cfg)) for u in unit):
                continue
            _, maks, _ = batas_sks(s, kd.jalur, ip, kur)
            muat_sks = kur.total_sks(mk) + kur.total_sks(unit) <= maks
            muat_theta = theta is None or beban is None or beban(list(mk) + sorted(unit)) <= theta + 1e-9
            if muat_sks and muat_theta:
                lokasi = f"direncanakan di S{direncanakan[k]}" if k in direncanakan else "tidak direncanakan"
                hasil.append(_h("AD-07", "gagal", f"{kur.label(k)} (wajib tertunda) seharusnya diprioritaskan di "
                                                  f"S{s}; {lokasi}", sumber, k, s))
            break
    return hasil
