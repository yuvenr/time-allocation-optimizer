"""Model pydantic untuk seluruh entitas data.

Dipakai untuk memvalidasi dataset Kurikulum 2024, input mentah KRS (KD-xx), keluaran
LLM terstruktur (UC-09: penilaian Bloom per CPMK), profil
mahasiswa, rencana studi (UC-02), dan hasil validasi aturan (UC-13).
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------
# Data mentah (data/raw)
# ---------------------------------------------------------------------


class KRS(BaseModel):
    """Baris ``histori_krs.csv``."""

    id_anon: str
    angkatan: int
    semester_ke: int = Field(ge=1)
    kode_mk: str
    sks: int = Field(ge=1)
    nilai_huruf: str
    ip_semester: float = Field(ge=0, le=4)
    kurikulum: str
    masa_studi_semester: int = Field(ge=1)


# ---------------------------------------------------------------------
# Kurikulum 2024 (data/kurikulum_2024 — hasil scripts/kurikulum/build_dataset.py)
# Nama kolom CSV dipertahankan; model memetakan kolom → field (alias) dan
# mengubah sel kosong menjadi None. Kode MK tidak pernah diubah.
# ---------------------------------------------------------------------

JALUR = ("A", "B", "C", "D", "E", "F")


def _kosong_jadi_none(v):
    if v is None:
        return None
    if isinstance(v, float) and v != v:  # NaN
        return None
    if isinstance(v, str) and not v.strip():
        return None
    return v


def _pisah(v, sep: str = ";") -> tuple[str, ...]:
    v = _kosong_jadi_none(v)
    if v is None:
        return ()
    if isinstance(v, (list, tuple)):
        return tuple(str(x).strip() for x in v if str(x).strip())
    return tuple(x.strip() for x in str(v).split(sep) if x.strip())


class _BarisDataset(BaseModel):
    """Basis model baris CSV Kurikulum 2024: sel kosong → None."""

    model_config = {"populate_by_name": True, "frozen": True}

    @field_validator("*", mode="before")
    @classmethod
    def _kosong(cls, v):
        return _kosong_jadi_none(v)


class MataKuliah(_BarisDataset):
    """Baris ``01_mata_kuliah.csv`` (master MK)."""

    kode: str = Field(alias="kode_mk", min_length=1)
    nama: str = Field(alias="nama_mk", min_length=1)
    sks: int = Field(ge=1, le=20)
    status: Literal["Wajib", "Pilihan"]
    kategori: str
    kelompok_prodi: Literal["Kependidikan", "Non-Kependidikan"] | None = None
    profil_lulusan: str | None = None
    tema: str | None = Field(default=None, alias="tema_pilihan")
    semester_paritas_deskripsi: Literal["Ganjil", "Genap"] | None = None
    semester_jalur_a: int | None = Field(default=None, alias="semester_rekomendasi_jalur_A", ge=1, le=8)
    semester_slot_tema_jalur_a: int | None = Field(default=None, alias="semester_slot_tema_jalur_A", ge=1, le=8)
    semester_tabel11: tuple[int, ...] = Field(default=(), alias="semester_ditawarkan_tabel11")
    grup_pilih_satu: str | None = None
    is_mk_lapangan: bool = False
    deskripsi: str | None = None
    halaman_pdf_deskripsi: str | None = None
    catatan: str | None = None

    @field_validator("semester_tabel11", mode="before")
    @classmethod
    def _semester_list(cls, v):
        return tuple(int(x) for x in _pisah(v))

    @property
    def sifat(self) -> Literal["wajib", "pilihan"]:
        return "wajib" if self.status == "Wajib" else "pilihan"

    @property
    def semester_acuan(self) -> int | None:
        """Semester acuan Jalur A: semester paket (MK wajib) atau slot tema (MK pilihan prodi)."""
        return self.semester_jalur_a or self.semester_slot_tema_jalur_a


class Prasyarat(_BarisDataset):
    """Baris ``05_prasyarat.csv`` (edge list MK → syarat)."""

    id: str = Field(alias="id_prasyarat")
    kode_mk: str
    jenis: Literal["nilai_minimal_mk", "lulus_kelompok_mk_kependidikan", "korequisit", "pedoman_eksternal"] = \
        Field(alias="jenis_syarat")
    kode_syarat: str | None = Field(default=None, alias="kode_mk_syarat")
    nilai_minimal: str | None = None
    teks_asli: str | None = None
    sumber_deskripsi: str | None = None
    sumber_ringkasan: str | None = None
    status_konsistensi: str | None = None

    @property
    def sumber(self) -> str:
        bag = [f"05_prasyarat.csv#{self.id}"]
        if self.sumber_ringkasan:
            bag.append(f"[R] {self.sumber_ringkasan}")
        if self.sumber_deskripsi:
            bag.append(f"[D] {self.sumber_deskripsi}")
        return "; ".join(bag)


JENIS_SLOT = Literal["MK_WAJIB", "MK_WAJIB_PILIH_SATU", "PILIHAN_TEMA", "PILIHAN_KONVERSI", "PILIHAN_SISA_TEMA"]


class SlotStruktur(_BarisDataset):
    """Baris ``08_struktur_kurikulum_jalur.csv`` (slot paket per jalur × semester)."""

    jalur: Literal["A", "B", "C", "D", "E", "F"]
    nama_jalur: str
    semester: int = Field(ge=1, le=8)
    urutan: int = Field(ge=1)
    kode_mk: str | None = None
    nama_slot: str
    jenis_slot: JENIS_SLOT
    tema: str | None = None
    sks: int = Field(ge=1, le=24)
    mode_pelaksanaan: str
    total_sks_semester: int
    halaman_pdf: str | None = None

    @property
    def sumber(self) -> str:
        return f"08_struktur_kurikulum_jalur.csv (Jalur {self.jalur}, sem. {self.semester}, urutan {self.urutan}); " \
               f"[R] hal. {self.halaman_pdf}"


class PitaSKS(_BarisDataset):
    """Baris ``09_rekomendasi_beban_sks.csv``: pita IP inklusif → rentang SKS + aksi rekomendasi."""

    jalur: str
    semester: int = Field(ge=1, le=14)
    ip_label: str
    ip_min: float = Field(ge=0, le=4)
    ip_max: float = Field(ge=0, le=4)
    sks_min: int = Field(ge=0)
    sks_max: int = Field(ge=0)
    aksi: Literal["tambah", "kurangi", "tetap", "tambah_jika_ada"]
    kode_terdampak: tuple[str, ...] = ()
    target_sks: int | None = None
    teks_asli: str | None = None
    halaman_pdf: str | None = None
    catatan: str | None = None

    @field_validator("kode_terdampak", mode="before")
    @classmethod
    def _daftar(cls, v):
        return _pisah(v)

    def cocok(self, ip: float) -> bool:
        """Pita inklusif; IP dibulatkan 2 desimal seperti di dokumen (2,995 → 3,00)."""
        ip = round(float(ip) + 1e-9, 2)
        return self.ip_min - 1e-9 <= ip <= self.ip_max + 1e-9

    @property
    def sumber(self) -> str:
        return f"09_rekomendasi_beban_sks.csv (jalur {self.jalur}, sem. {self.semester}, IP {self.ip_label}); " \
               f"[R] hal. {self.halaman_pdf}"


class AturanAkademik(_BarisDataset):
    """Baris ``10_aturan_akademik.csv``; ``parameter`` diurai menjadi dict ``kunci → nilai``."""

    id: str = Field(alias="id_aturan")
    kategori: str
    aturan: str
    parameter_teks: str | None = Field(default=None, alias="parameter")
    keterangan: str | None = None
    halaman_pdf: str | None = Field(default=None, alias="halaman_pdf_ringkasan")

    @property
    def parameter(self) -> dict[str, str]:
        """``"wajib_min=118;pilihan_min=27"`` → ``{"wajib_min": "118", "pilihan_min": "27"}``."""
        hasil: dict[str, str] = {}
        for bagian in _pisah(self.parameter_teks):
            if "=" in bagian:
                k, v = bagian.split("=", 1)
                hasil[k.strip()] = v.strip()
        return hasil

    @property
    def sumber(self) -> str:
        return f"10_aturan_akademik.csv#{self.id}; [R] hal. {self.halaman_pdf}"


class TemaPilihan(_BarisDataset):
    """Baris ``06_tema_pilihan.csv``."""

    kode: str = Field(alias="kode_tema")
    nama: str = Field(alias="nama_tema")
    jumlah_mk_opsi: int
    sks_minimal: int = Field(alias="sks_minimal_ditempuh", ge=1)
    profil_lulusan: str | None = None
    semester_slot_jalur_a: int | None = Field(default=None, alias="semester_slot_jalur_A")


class Penyetaraan(_BarisDataset):
    """Baris ``11_penyetaraan_mk_2020_2024.csv`` (Kurikulum 2020 → 2024)."""

    kode_lama: str | None = None
    nama_lama: str | None = None
    sks_lama: int | None = None
    status_lama: str | None = None
    kode_baru: str | None = None
    nama_baru: str | None = None
    sks_baru: int | None = None
    status_baru: str | None = None
    jenis_perubahan: str
    halaman_pdf: str | None = None
    catatan: str | None = None

    @field_validator("jenis_perubahan")
    @classmethod
    def _jenis(cls, v: str) -> str:
        if v not in ("identik", "dihapus", "mk_baru") and not v.startswith("berubah_"):
            raise ValueError(f"jenis_perubahan tidak dikenal: {v}")
        return v


class JadwalLapangan(_BarisDataset):
    """Baris ``16_jadwal_kegiatan_lapangan.csv`` (kegiatan pada masa transisi semester)."""

    kode_mk: tuple[str, ...]
    kegiatan: str
    durasi_bulan: int
    transisi_semester: str
    bulan_pelaksanaan: str
    semester_krs: int = Field(alias="semester_diprogram_di_KRS", ge=1, le=14)
    sumber: str | None = None

    @field_validator("kode_mk", mode="before")
    @classmethod
    def _kode(cls, v):
        return _pisah(v)


# ---------------------------------------------------------------------
# Hasil proses (disimpan di basis data SQLite, src/db.py)
# ---------------------------------------------------------------------


class Transaksi(BaseModel):
    """Satu transaksi = himpunan MK yang diambil satu mahasiswa pada satu semester (UC-07)."""

    id_anon: str
    angkatan: int
    semester_ke: int
    ip_semester: float | None = None  # IP semester itu (None: KRS semester berjalan, nilai belum ada)
    items: list[str]  # "{kode}@S{semester_ke}", seluruh MK yang diambil
    gagal: list[str] = Field(default_factory=list)  # kode MK tidak lulus pada semester ini
    sumber: str = "asli"  # asli | sintetis | fixture

    @property
    def kode(self) -> set[str]:
        return {i.rsplit("@S", 1)[0] for i in self.items}


# ---------------------------------------------------------------------
# Keluaran LLM terstruktur (JSON) — kontrak yang sama untuk mock & provider sungguhan
# ---------------------------------------------------------------------

LEVEL_BLOOM = ("C1", "C2", "C3", "C4", "C5", "C6")
LABEL_BLOOM = {"C1": "Mengingat", "C2": "Memahami", "C3": "Menerapkan", "C4": "Menganalisis",
               "C5": "Mengevaluasi", "C6": "Mencipta", "A": "Afektif"}


class PenilaianBloom(BaseModel):
    """Penilaian satu CPMK (UC-09; CLAUDE.md §7) — label identik dengan instrumen pakar DPA.

    ``bloom`` = C1–C6 untuk ranah kognitif, ``A`` untuk CPMK afektif (tidak dipaksa ke C1–C6).
    """

    id_cpmk: str = Field(min_length=1)
    penalaran: str | None = Field(default=None, max_length=2000)   # chain-of-thought singkat (prompt v2)
    bloom: Literal["C1", "C2", "C3", "C4", "C5", "C6", "A"]
    ranah: Literal["kognitif", "afektif", "afektif+kognitif"]
    alasan: str = Field(min_length=1)
    keyakinan: float | None = Field(default=None, ge=0, le=1)

    @field_validator("bloom", mode="before")
    @classmethod
    def _huruf_besar(cls, v):
        return v.strip().upper() if isinstance(v, str) else v

    @model_validator(mode="after")
    def _ranah_konsisten(self) -> "PenilaianBloom":
        if (self.bloom == "A") != (self.ranah == "afektif"):
            raise ValueError("bloom 'A' hanya untuk ranah afektif (dan ranah afektif murni harus 'A')")
        return self


# ---------------------------------------------------------------------
# Profil, rencana, validasi (UC-01..UC-05, UC-13)
# ---------------------------------------------------------------------


class HasilAturan(BaseModel):
    """Keluaran satu fungsi aturan domain (CLAUDE.md §4): status + alasan + sumber.

    ``perlu_verifikasi_manual`` dipakai untuk syarat yang isinya tidak ada di dokumen
    (mis. pedoman PLP/PKL/Skripsi): tidak dianggap pelanggaran, tetapi juga TIDAK
    dianggap terpenuhi otomatis.
    """

    kode: str                    # label aturan (AD-xx)
    status: Literal["lolos", "gagal", "perlu_verifikasi_manual"]
    alasan: str
    sumber: str                  # nama CSV + id baris / halaman PDF
    kode_mk: str | None = None
    semester: int | None = None

    @property
    def lolos(self) -> bool:
        return self.status == "lolos"

    @property
    def melanggar(self) -> bool:
        return self.status == "gagal"

    def teks(self) -> str:
        lokasi = f"S{self.semester}: " if self.semester else ""
        return f"{self.kode}: {lokasi}{self.alasan} [{self.sumber}]"


class Profile(BaseModel):
    """Profil mahasiswa untuk penyusun rencana (UC-01).

    ``semester_saat_ini`` adalah semester yang akan direncanakan (semester berjalan);
    ``ip_terakhir`` adalah IP semester sebelumnya; ``mk_lulus`` berisi MK yang lulus
    (nilai ≥ ``aturan.NILAI_LULUS_MINIMUM``) sampai semester sebelumnya;
    ``mk_pernah_gagal`` berisi MK yang pernah ditempuh tetapi belum lulus.
    ``agama`` adalah input pengguna (tidak pernah disimpulkan sistem); ``tema_konversi``
    (jalur B–F) = tema pilihan yang (akan) dipenuhi lewat konversi merdeka belajar,
    ``None`` = serahkan pada asumsi planner; ``sks_konversi`` = SKS hasil konversi yang
    sudah diakui (bukan kode MK, sehingga tidak tercatat di ``mk_lulus``).
    ``ipk`` = IPK yang diketahui (mis. dari data akademik); dipakai untuk AD-10 bila IPK tidak
    dapat dihitung dari ``riwayat_nilai``.
    ``pilihan_tema`` (iterasi 3) = preferensi mahasiswa: tema → kode MK pilihan yang diminati; tema
    yang tidak diisi diserahkan pada rekomendasi sistem. Input profil sehingga berlaku sama di kedua mode.
    """

    semester_saat_ini: int = Field(ge=1, le=14)
    ip_terakhir: float = Field(ge=0, le=4)
    mk_lulus: set[str] = Field(default_factory=set)
    mk_pernah_gagal: set[str] = Field(default_factory=set)
    jalur: Literal["A", "B", "C", "D", "E", "F"] = "A"
    agama: str | None = None
    tema_konversi: set[str] | None = None
    sks_konversi: int = Field(0, ge=0, le=60)
    riwayat_nilai: dict[str, str] = Field(default_factory=dict)
    ipk: float | None = Field(None, ge=0, le=4)
    pilihan_tema: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _gagal_bukan_lulus(self) -> "Profile":
        self.mk_pernah_gagal = set(self.mk_pernah_gagal) - set(self.mk_lulus)
        if self.agama is not None and not self.agama.strip():
            self.agama = None
        return self

    @property
    def pernah_menempuh(self) -> set[str]:
        return set(self.mk_lulus) | set(self.mk_pernah_gagal)


class ItemRencana(BaseModel):
    """Satu MK dalam rencana beserta alasannya."""

    kode: str
    nama: str
    sks: int
    sifat: str
    semester_anjuran: int | None = None
    jenis: Literal["mk", "konversi"] = "mk"   # konversi = slot merdeka belajar (bukan MK prodi)
    tema: str | None = None
    tema_konversi: list[str] = Field(default_factory=list)
    prioritas: str | None = None
    cl: float | None = None
    justifikasi_cl: str | None = None
    justifikasi_cpmk: list[str] = Field(default_factory=list)   # KF-17: "id_cpmk: label — alasan" per CPMK
    skor: float = 0.0
    support: float = 0.0
    conf_lift: float = 0.0
    aturan_pendukung: str | None = None
    alasan: list[str] = Field(default_factory=list)
    sumber: list[str] = Field(default_factory=list)
    perlu_verifikasi: list[str] = Field(default_factory=list)
    jadwal: list[str] = Field(default_factory=list)   # kegiatan lapangan pada masa transisi (16)


class SemesterRencana(BaseModel):
    semester: int
    paket: bool = False
    jenis: Literal["paket", "reguler", "konversi"] = "reguler"
    batas_sks: int | None = None
    sks_min: int | None = None
    sumber_batas: str | None = None
    items: list[ItemRencana] = Field(default_factory=list)
    total_sks: int = 0
    total_beban: float = 0.0
    dilewati_theta: list[str] = Field(default_factory=list)  # kandidat yang dilewati karena θ
    catatan: list[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """Hasil ``validate_plan`` (UC-13)."""

    lolos: bool
    pelanggaran: list[str] = Field(default_factory=list)
    per_aturan: dict[str, bool] = Field(default_factory=dict)
    perlu_verifikasi: list[str] = Field(default_factory=list)
    rincian: list[HasilAturan] = Field(default_factory=list)   # hanya status gagal / perlu_verifikasi


STATUS_DPA = ("belum_ditinjau", "disetujui", "perlu_revisi")


class Pemindahan(BaseModel):
    """Satu pemindahan MK antarsemester oleh mahasiswa (iterasi 3, ``src/penyesuaian.py``).
    ``otomatis`` = ikut dipindahkan sistem karena korequisit (AT15, mis. PLP 1 + PLP 2)."""

    kode: str
    dari: int
    ke: int
    waktu: datetime
    otomatis: bool = False


class Plan(BaseModel):
    """Rencana studi dari semester berjalan sampai horizon (default semester 8; diperpanjang ≤ AT04
    bila belum lulus) (UC-02). ``sumber_data`` = status data KRS saat rencana disusun."""

    id: str
    mode: Literal["arm", "hybrid"]
    dibuat: datetime
    profil: Profile
    theta: float | None = None
    metode_beban: str = "jumlah"
    semester: list[SemesterRencana] = Field(default_factory=list)
    validasi: ValidationResult | None = None
    peringatan: list[str] = Field(default_factory=list)
    asumsi: list[str] = Field(default_factory=list)
    sumber_data: str = "belum_tersedia"
    label: str = ""
    sks_lulus_awal: int = 0           # SKS lulus (termasuk konversi) sebelum rencana — untuk lolos_145 dalam horizon
    sks_minimal_lulus: int = 145      # AT01
    status_dpa: Literal["belum_ditinjau", "disetujui", "perlu_revisi"] = "belum_ditinjau"   # validasi akhir DPA (KF-19)
    catatan_dpa: str = ""
    catatan_dpa_diperbarui: datetime | None = None
    # penyesuaian manual (iterasi 3): riwayat pemindahan + susunan asli rekomendasi sistem (untuk DPA & reset)
    penyesuaian: list[Pemindahan] = Field(default_factory=list)
    semester_sistem: list[SemesterRencana] | None = None

    @property
    def disesuaikan(self) -> bool:
        return self.semester_sistem is not None

    @property
    def total_sks_rencana(self) -> int:
        return sum(s.total_sks for s in self.semester)
